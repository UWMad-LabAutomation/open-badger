"""LeRobot-backed implementation of the canonical dataset reader."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_badger.data.schema import DatasetMetadata, FeatureMetadata, SamplePointer


class LeRobotReader:
    """Adapt a pinned LeRobotDataset to Open Badger's data contracts."""

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        revision: str | None = None,
        video_backend: str | None = None,
        download_videos: bool = True,
        dataset: Any | None = None,
    ) -> None:
        if dataset is None:
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                revision=revision,
                video_backend=video_backend,
                download_videos=download_videos,
            )

        self.dataset = dataset
        self._metadata: DatasetMetadata | None = None

    def metadata(self) -> DatasetMetadata:
        """Return LeRobot metadata normalized to the canonical schema."""
        if self._metadata is None:
            meta = self.dataset.meta
            features = {
                name: self._feature_metadata(name, feature)
                for name, feature in meta.features.items()
            }
            self._metadata = DatasetMetadata(
                dataset_id=str(self.dataset.repo_id),
                revision=str(self.dataset.revision),
                format_version=str(getattr(meta.info, "codebase_version", "v3.0")),
                fps=float(meta.fps),
                total_frames=int(meta.total_frames),
                total_episodes=int(meta.total_episodes),
                features=features,
                episode_lengths=self._episode_lengths(meta.episodes),
                tasks=self._tasks_by_episode(meta.episodes, meta.tasks),
                stats=getattr(meta, "stats", None),
            )
        return self._metadata

    def iter_pointers(self) -> Iterator[SamplePointer]:
        """Yield frame pointers lazily without caching a dataset-sized index."""
        metadata = self.metadata()
        for dataset_index in range(len(self.dataset)):
            row = self._raw_row(dataset_index)
            yield SamplePointer(
                dataset_id=metadata.dataset_id,
                revision=metadata.revision,
                episode_index=int(_scalar(row["episode_index"])),
                episode_frame_index=int(_scalar(row["frame_index"])),
                global_frame_index=int(_scalar(row["index"])),
                timestamp=float(_scalar(row["timestamp"])),
                fps=metadata.fps,
                dataset_index=dataset_index,
            )

    def get_frame(self, pointer: SamplePointer) -> Mapping[str, Any]:
        """Resolve a canonical pointer to LeRobot's decoded frame mapping."""
        self._validate_pointer(pointer)
        if pointer.dataset_index is None:
            raise ValueError("LeRobot frame lookup requires dataset_index.")
        if pointer.dataset_index >= len(self.dataset):
            raise KeyError(f"Dataset index {pointer.dataset_index} is not available.")

        row = self._raw_row(pointer.dataset_index)
        expected = (
            pointer.global_frame_index,
            pointer.episode_index,
            pointer.episode_frame_index,
        )
        actual = (
            int(_scalar(row["index"])),
            int(_scalar(row["episode_index"])),
            int(_scalar(row["frame_index"])),
        )
        if actual != expected:
            raise ValueError("Pointer dataset_index does not match its frame identifiers.")
        return dict(self.dataset[pointer.dataset_index])

    def _validate_pointer(self, pointer: SamplePointer) -> None:
        metadata = self.metadata()
        if pointer.dataset_id != metadata.dataset_id or pointer.revision != metadata.revision:
            raise ValueError("Pointer dataset identity does not match the reader dataset.")

    def _raw_row(self, relative_index: int) -> Mapping[str, Any]:
        if hasattr(self.dataset, "get_raw_item"):
            return self.dataset.get_raw_item(relative_index)
        return self.dataset.hf_dataset[relative_index]

    @staticmethod
    def _feature_metadata(name: str, feature: Mapping[str, Any]) -> FeatureMetadata:
        dtype = str(feature.get("dtype", ""))
        if dtype in {"image", "video"}:
            kind = "image"
        elif name == "action" or name.endswith(".action"):
            kind = "action"
        elif name == "state" or name.endswith(".state"):
            kind = "state"
        else:
            kind = "other"
        return FeatureMetadata(
            name=name,
            kind=kind,
            shape=tuple(int(dimension) for dimension in feature.get("shape", ())),
            dtype=dtype,
            names=tuple(str(value) for value in feature.get("names", ())),
            units=tuple(str(value) for value in feature.get("units", ())),
        )

    @staticmethod
    def _episode_lengths(episodes: Any) -> dict[int, int]:
        return {
            int(_scalar(episode["episode_index"])): int(_scalar(episode["length"]))
            for episode in _rows(episodes)
        }

    @staticmethod
    def _tasks_by_episode(episodes: Any, tasks: Any) -> dict[int, str]:
        task_names = list(getattr(tasks, "index", ()))
        result: dict[int, str] = {}
        for episode in _rows(episodes):
            episode_index = int(_scalar(episode["episode_index"]))
            task_value = episode.get("tasks")
            if task_value is None:
                continue
            if isinstance(task_value, (list, tuple)):
                task_value = task_value[0] if task_value else None
            if isinstance(task_value, int) and 0 <= task_value < len(task_names):
                task_value = task_names[task_value]
            if task_value is not None:
                result[episode_index] = str(task_value)
        return result


def _rows(table: Any) -> list[Mapping[str, Any]]:
    if hasattr(table, "to_pylist"):
        return table.to_pylist()
    if hasattr(table, "to_dict"):
        records = table.to_dict(orient="records")
        return records
    return list(table)


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
