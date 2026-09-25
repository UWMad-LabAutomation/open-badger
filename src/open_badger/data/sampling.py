"""Deterministic temporal sampling for manifest references."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator

from open_badger.data.schema import DatasetMetadata, SamplePointer, TrainingExampleRef


def action_frame_offsets(
    *,
    action_horizon: int,
    fps: float,
    action_spacing_seconds: float | None = None,
) -> tuple[int, ...]:
    """Convert a control horizon and time spacing into frame offsets."""
    if action_horizon < 1:
        raise ValueError("action_horizon must be at least one.")
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")

    spacing = 1.0 / fps if action_spacing_seconds is None else action_spacing_seconds
    if spacing <= 0:
        raise ValueError("action_spacing_seconds must be greater than zero.")

    offsets = tuple(round(index * spacing * fps) for index in range(action_horizon))
    if len(set(offsets)) != action_horizon:
        raise ValueError(
            "action_spacing_seconds is too small for the dataset FPS and produces duplicate frames."
        )
    return offsets


def build_training_examples(
    metadata: DatasetMetadata,
    pointers: Iterable[SamplePointer],
    *,
    action_horizon: int,
    action_spacing_seconds: float | None = None,
    observation_history_seconds: float = 0.0,
    drop_incomplete_horizons: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[TrainingExampleRef]:
    return list(
        iter_training_examples(
            metadata,
            pointers,
            action_horizon=action_horizon,
            action_spacing_seconds=action_spacing_seconds,
            observation_history_seconds=observation_history_seconds,
            drop_incomplete_horizons=drop_incomplete_horizons,
            shuffle=shuffle,
            seed=seed,
        )
    )


def iter_training_examples(
    metadata: DatasetMetadata,
    pointers: Iterable[SamplePointer],
    *,
    action_horizon: int,
    action_spacing_seconds: float | None = None,
    observation_history_seconds: float = 0.0,
    drop_incomplete_horizons: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[TrainingExampleRef]:
    """Build canonical example references from frame pointers.

    This function does not read tensors or decode video. It only validates
    temporal references against episode metadata and yields manifest-ready
    references. Shuffling is supported for the in-memory compatibility path;
    disk-backed manifests should be shuffled by the training sampler.
    """
    if shuffle:
        pointers = list(pointers)
        random.Random(seed).shuffle(pointers)

    if observation_history_seconds < 0:
        raise ValueError("observation_history_seconds must be non-negative.")

    offsets = action_frame_offsets(
        action_horizon=action_horizon,
        fps=metadata.fps,
        action_spacing_seconds=action_spacing_seconds,
    )
    for pointer in pointers:
        if pointer.dataset_id != metadata.dataset_id or pointer.revision != metadata.revision:
            raise ValueError("Pointer dataset identity does not match metadata.")

        episode_length = metadata.episode_lengths.get(pointer.episode_index)
        if episode_length is None:
            raise KeyError(f"Missing metadata for episode {pointer.episode_index}.")
        last_required_frame = pointer.episode_frame_index + offsets[-1]
        if last_required_frame >= episode_length:
            if drop_incomplete_horizons:
                continue
            raise ValueError(
                f"Frame {pointer.episode_frame_index} in episode {pointer.episode_index} "
                f"cannot provide horizon ending at frame {last_required_frame}."
            )

        action_feature = metadata.action_feature
        state_feature = metadata.state_feature
        yield TrainingExampleRef(
            sample=pointer,
            action_horizon=action_horizon,
            action_frame_offsets=offsets,
            observation_history_seconds=observation_history_seconds,
            camera_keys=metadata.camera_keys,
            task=metadata.tasks.get(pointer.episode_index),
            action_dim=(None if action_feature is None else action_feature.shape[-1]),
            state_dim=(None if state_feature is None else state_feature.shape[-1]),
        )
