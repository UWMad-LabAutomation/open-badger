"""Canonical data contracts for version-independent dataset indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureMetadata:
    """Schema information for one dataset feature."""

    # `name` is the canonical dataset key; `kind` identifies how the feature
    # is used, while `shape` and `dtype` describe the stored tensor values.
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    # Optional per-dimension labels and physical units preserve robot-specific
    # meaning such as joint order, gripper position, or radians versus meters.
    names: tuple[str, ...] = ()
    units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"image", "state", "action", "other"}:
            raise ValueError(f"Unsupported feature kind: {self.kind!r}")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("Feature dimensions must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "names": list(self.names),
            "units": list(self.units),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureMetadata":
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            shape=tuple(int(dimension) for dimension in value.get("shape", ())),
            dtype=str(value["dtype"]),
            names=tuple(str(name) for name in value.get("names", ())),
            units=tuple(str(unit) for unit in value.get("units", ())),
        )


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset-level metadata normalized across LeRobot storage versions."""

    dataset_id: str
    revision: str
    format_version: str
    fps: float
    total_frames: int
    total_episodes: int
    features: dict[str, FeatureMetadata]
    episode_lengths: dict[int, int]
    tasks: dict[int, str] = field(default_factory=dict)
    stats: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty.")
        if self.fps <= 0:
            raise ValueError("fps must be greater than zero.")
        if self.total_frames < 0 or self.total_episodes < 0:
            raise ValueError("Dataset counts must be non-negative.")
        if any(index < 0 or length < 0 for index, length in self.episode_lengths.items()):
            raise ValueError("Episode indices and lengths must be non-negative.")

    @property
    def action_feature(self) -> FeatureMetadata | None:
        return next((feature for feature in self.features.values() if feature.kind == "action"), None)

    @property
    def state_feature(self) -> FeatureMetadata | None:
        return next((feature for feature in self.features.values() if feature.kind == "state"), None)

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features.values() if feature.kind == "image")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "format_version": self.format_version,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "total_episodes": self.total_episodes,
            "features": {name: feature.to_dict() for name, feature in self.features.items()},
            "episode_lengths": {str(index): length for index, length in self.episode_lengths.items()},
            "tasks": {str(index): task for index, task in self.tasks.items()},
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetMetadata":
        return cls(
            dataset_id=str(value["dataset_id"]),
            revision=str(value.get("revision", "")),
            format_version=str(value["format_version"]),
            fps=float(value["fps"]),
            total_frames=int(value["total_frames"]),
            total_episodes=int(value["total_episodes"]),
            features={
                str(name): FeatureMetadata.from_dict(feature)
                for name, feature in value.get("features", {}).items()
            },
            episode_lengths={int(index): int(length) for index, length in value.get("episode_lengths", {}).items()},
            tasks={int(index): str(task) for index, task in value.get("tasks", {}).items()},
            stats=value.get("stats"),
        )


@dataclass(frozen=True)
class SamplePointer:
    """Stable reference to one frame in a canonical dataset."""

    dataset_id: str
    revision: str
    episode_index: int
    episode_frame_index: int
    global_frame_index: int | None
    timestamp: float
    fps: float

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.episode_frame_index < 0:
            raise ValueError("Episode and frame indices must be non-negative.")
        if self.global_frame_index is not None and self.global_frame_index < 0:
            raise ValueError("global_frame_index must be non-negative.")
        if self.timestamp < 0 or self.fps <= 0:
            raise ValueError("timestamp must be non-negative and fps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "episode_index": self.episode_index,
            "episode_frame_index": self.episode_frame_index,
            "global_frame_index": self.global_frame_index,
            "timestamp": self.timestamp,
            "fps": self.fps,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SamplePointer":
        return cls(
            dataset_id=str(value["dataset_id"]),
            revision=str(value.get("revision", "")),
            episode_index=int(value["episode_index"]),
            episode_frame_index=int(value["episode_frame_index"]),
            global_frame_index=(
                None if value.get("global_frame_index") is None else int(value["global_frame_index"])
            ),
            timestamp=float(value["timestamp"]),
            fps=float(value["fps"]),
        )


@dataclass(frozen=True)
class TrainingExampleRef:
    """Manifest entry describing one future-action training example."""

    sample: SamplePointer
    action_horizon: int
    action_frame_offsets: tuple[int, ...]
    observation_history_seconds: float = 0.0
    camera_keys: tuple[str, ...] = ()
    task: str | None = None
    control_mode: str | None = None
    action_dim: int | None = None
    state_dim: int | None = None

    def __post_init__(self) -> None:
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be at least one.")
        if len(self.action_frame_offsets) != self.action_horizon:
            raise ValueError("action_frame_offsets must match action_horizon.")
        if not self.action_frame_offsets or self.action_frame_offsets[0] != 0:
            raise ValueError("The first action frame offset must be zero.")
        if any(offset < 0 for offset in self.action_frame_offsets):
            raise ValueError("Action frame offsets must be non-negative.")
        if tuple(sorted(set(self.action_frame_offsets))) != self.action_frame_offsets:
            raise ValueError("Action frame offsets must be strictly increasing.")
        if self.observation_history_seconds < 0:
            raise ValueError("observation_history_seconds must be non-negative.")
        if self.action_dim is not None and self.action_dim < 1:
            raise ValueError("action_dim must be positive when provided.")
        if self.state_dim is not None and self.state_dim < 0:
            raise ValueError("state_dim must be non-negative when provided.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "action_horizon": self.action_horizon,
            "action_frame_offsets": list(self.action_frame_offsets),
            "observation_history_seconds": self.observation_history_seconds,
            "camera_keys": list(self.camera_keys),
            "task": self.task,
            "control_mode": self.control_mode,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingExampleRef":
        return cls(
            sample=SamplePointer.from_dict(value["sample"]),
            action_horizon=int(value["action_horizon"]),
            action_frame_offsets=tuple(int(offset) for offset in value["action_frame_offsets"]),
            observation_history_seconds=float(value.get("observation_history_seconds", 0.0)),
            camera_keys=tuple(str(key) for key in value.get("camera_keys", ())),
            task=None if value.get("task") is None else str(value["task"]),
            control_mode=(None if value.get("control_mode") is None else str(value["control_mode"])),
            action_dim=(None if value.get("action_dim") is None else int(value["action_dim"])),
            state_dim=(None if value.get("state_dim") is None else int(value["state_dim"])),
        )
