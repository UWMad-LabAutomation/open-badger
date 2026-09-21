"""JSONL persistence and construction for canonical training manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from open_badger.data.readers.base import DatasetReader
from open_badger.data.sampling import build_training_examples
from open_badger.data.schema import DatasetMetadata, TrainingExampleRef


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
    ) -> "SampleManifest":
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

    def write(self, directory: str | Path) -> None:
        """Write metadata and examples as inspectable JSON/JSONL files."""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for example in self.examples:
                stream.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def read(cls, directory: str | Path) -> "SampleManifest":
        input_dir = Path(directory)
        metadata = DatasetMetadata.from_dict(
            json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
        )
        examples: list[TrainingExampleRef] = []
        with (input_dir / "samples.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    examples.append(TrainingExampleRef.from_dict(json.loads(line)))
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid manifest entry at line {line_number}.") from exc
        return cls(metadata, examples)
