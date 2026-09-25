"""JSONL persistence and construction for canonical training manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from open_badger.data.readers.base import DatasetReader
from open_badger.data.sampling import (
    build_training_examples,
    iter_training_examples,
)
from open_badger.data.schema import (
    DatasetMetadata,
    SamplePointer,
    TrainingExampleRef,
)

_MANIFEST_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("revision", pa.string()),
        ("episode_index", pa.int64()),
        ("episode_frame_index", pa.int64()),
        ("global_frame_index", pa.int64()),
        ("timestamp", pa.float64()),
        ("fps", pa.float64()),
        ("action_horizon", pa.int64()),
        ("action_frame_offsets", pa.list_(pa.int64())),
        ("observation_history_seconds", pa.float64()),
        ("camera_keys", pa.list_(pa.string())),
        ("task", pa.string()),
        ("control_mode", pa.string()),
        ("action_dim", pa.int64()),
        ("state_dim", pa.int64()),
    ]
)


class SampleManifest:
    """A reproducible list of canonical training example references."""

    def __init__(self, metadata: DatasetMetadata, examples: Iterable[TrainingExampleRef] = ()):
        self.metadata = metadata
        self.examples = list(examples)
        self._validate_examples()

    def _validate_examples(self) -> None:
        for example in self.examples:
            if example.sample.dataset_id != self.metadata.dataset_id:
                raise ValueError("Manifest example dataset_id does not match metadata.")
            if example.sample.revision != self.metadata.revision:
                raise ValueError("Manifest example revision does not match metadata.")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @classmethod
    def build(
        cls,
        reader: DatasetReader,
        *,
        action_horizon: int,
        action_spacing_seconds: float | None = None,
        observation_history_seconds: float = 0.0,
        drop_incomplete_horizons: bool = True,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> SampleManifest:
        metadata = reader.metadata()
        examples = build_training_examples(
            metadata,
            list(reader.iter_pointers()),
            action_horizon=action_horizon,
            action_spacing_seconds=action_spacing_seconds,
            observation_history_seconds=observation_history_seconds,
            drop_incomplete_horizons=drop_incomplete_horizons,
            shuffle=shuffle,
            seed=seed,
        )
        return cls(metadata, examples)

    @classmethod
    def build_parquet(
        cls,
        reader: DatasetReader,
        directory: str | Path,
        *,
        action_horizon: int,
        action_spacing_seconds: float | None = None,
        observation_history_seconds: float = 0.0,
        drop_incomplete_horizons: bool = True,
        write_batch_size: int = 4096,
    ) -> ParquetManifest:
        """Stream validated references into a durable Parquet manifest."""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = reader.metadata()
        examples_path = output_dir / "examples.parquet"
        temporary_examples_path = output_dir / "examples.parquet.tmp"
        rows: list[dict[str, Any]] = []
        row_count = 0
        writer: pq.ParquetWriter | None = None

        try:
            for example in iter_training_examples(
                metadata,
                reader.iter_pointers(),
                action_horizon=action_horizon,
                action_spacing_seconds=action_spacing_seconds,
                observation_history_seconds=observation_history_seconds,
                drop_incomplete_horizons=drop_incomplete_horizons,
            ):
                rows.append(_example_to_row(example))
                if len(rows) < write_batch_size:
                    continue
                writer = writer or pq.ParquetWriter(temporary_examples_path, _MANIFEST_SCHEMA)
                writer.write_table(pa.Table.from_pylist(rows, schema=_MANIFEST_SCHEMA))
                row_count += len(rows)
                rows.clear()

            writer = writer or pq.ParquetWriter(temporary_examples_path, _MANIFEST_SCHEMA)
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=_MANIFEST_SCHEMA))
                row_count += len(rows)
            writer.close()
            writer = None
            temporary_examples_path.replace(examples_path)
            _write_json(
                output_dir / "metadata.json",
                metadata.to_dict(),
            )
            _write_json(
                output_dir / "manifest.json",
                {"format": "parquet", "row_count": row_count},
            )
        except Exception:
            if writer is not None:
                writer.close()
            temporary_examples_path.unlink(missing_ok=True)
            raise

        return ParquetManifest(metadata, output_dir, row_count)

class ParquetManifest:
    """Lazy, disk-backed access to a Parquet training-reference manifest."""

    def __init__(self, metadata: DatasetMetadata, directory: str | Path, row_count: int):
        self.metadata = metadata
        self.directory = Path(directory)
        self.row_count = row_count

    def __len__(self) -> int:
        return self.row_count

    def __iter__(self):
        parquet_file = pq.ParquetFile(self.directory / "examples.parquet")
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                yield _row_to_example(row)

    @classmethod
    def read(cls, directory: str | Path) -> ParquetManifest:
        input_dir = Path(directory)
        metadata = DatasetMetadata.from_dict(
            json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
        )
        descriptor = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
        if descriptor.get("format") != "parquet":
            raise ValueError("Unsupported manifest format.")
        return cls(metadata, input_dir, int(descriptor["row_count"]))


def _example_to_row(example: TrainingExampleRef) -> dict[str, Any]:
    pointer = example.sample
    return {
        "dataset_id": pointer.dataset_id,
        "revision": pointer.revision,
        "episode_index": pointer.episode_index,
        "episode_frame_index": pointer.episode_frame_index,
        "global_frame_index": pointer.global_frame_index,
        "timestamp": pointer.timestamp,
        "fps": pointer.fps,
        "action_horizon": example.action_horizon,
        "action_frame_offsets": list(example.action_frame_offsets),
        "observation_history_seconds": example.observation_history_seconds,
        "camera_keys": list(example.camera_keys),
        "task": example.task,
        "control_mode": example.control_mode,
        "action_dim": example.action_dim,
        "state_dim": example.state_dim,
    }


def _row_to_example(row: dict[str, Any]) -> TrainingExampleRef:
    return TrainingExampleRef(
        sample=SamplePointer(
            dataset_id=row["dataset_id"],
            revision=row["revision"],
            episode_index=row["episode_index"],
            episode_frame_index=row["episode_frame_index"],
            global_frame_index=row["global_frame_index"],
            timestamp=row["timestamp"],
            fps=row["fps"],
        ),
        action_horizon=row["action_horizon"],
        action_frame_offsets=tuple(row["action_frame_offsets"]),
        observation_history_seconds=row["observation_history_seconds"],
        camera_keys=tuple(row["camera_keys"]),
        task=row["task"],
        control_mode=row["control_mode"],
        action_dim=row["action_dim"],
        state_dim=row["state_dim"],
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)