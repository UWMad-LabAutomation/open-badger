"""Manifest-backed, lazy training-example resolution."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from open_badger.data.readers.base import DatasetReader
from open_badger.data.schema import TrainingExampleRef


class ManifestTrainingDataset(IterableDataset):
    """Resolve manifest references into raw observation/action tensor mappings.

    Image and task formatting are intentionally left to a later model-specific
    preprocessing stage. Only feature access and action-chunk assembly happen here.
    """

    def __init__(
        self,
        reader: DatasetReader,
        manifest: Iterable[TrainingExampleRef],
        *,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1024,
        seed: int = 0,
        include_action: bool = True,
    ) -> None:
        super().__init__()
        if shuffle and shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive when shuffle is enabled.")
        self.reader = reader
        self.manifest = manifest
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.include_action = include_action
        self._epoch = 0

        metadata = reader.metadata()
        manifest_metadata = getattr(manifest, "metadata", None)
        if manifest_metadata is not None and (
            metadata.dataset_id != manifest_metadata.dataset_id
            or metadata.revision != manifest_metadata.revision
        ):
            raise ValueError("Manifest dataset identity does not match the reader.")

    def __len__(self) -> int:
        try:
            return len(self.manifest)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("The manifest must provide a length.") from exc

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        epoch = self._epoch
        self._epoch += 1

        iter_shard = getattr(self.manifest, "iter_shard", None)
        if callable(iter_shard):
            examples = iter_shard(worker_id, worker_count)
        elif hasattr(self.manifest, "examples"):
            stored_examples = self.manifest.examples
            examples = (
                stored_examples[index]
                for index in range(worker_id, len(stored_examples), worker_count)
            )
        elif isinstance(self.manifest, Sequence):
            examples = (
                self.manifest[index]
                for index in range(worker_id, len(self.manifest), worker_count)
            )
        else:
            if worker_count > 1:
                raise ValueError(
                    "Parallel loading requires a manifest with iter_shard() or "
                    "random-access examples."
                )
            examples = (
                example for example in self.manifest
            )
        if self.shuffle:
            rng = random.Random(self.seed + epoch * worker_count + worker_id)
            examples = _shuffle_buffer(examples, self.shuffle_buffer_size, rng)

        for example in examples:
            yield self._resolve_example(example)

    def _resolve_example(self, example: TrainingExampleRef) -> dict[str, Any]:
        frame = dict(self.reader.get_frame(example.sample))
        for feature in self.reader.metadata().features.values():
            if feature.kind == "state" and feature.name in frame:
                frame[feature.name] = torch.as_tensor(frame[feature.name])
        if self.include_action:
            action_frames = [
                torch.as_tensor(
                    self.reader.get_feature(
                        example.sample,
                        "action",
                        frame_offset=offset,
                    )
                )
                for offset in example.action_frame_offsets
            ]
            frame["action"] = torch.stack(action_frames)
        else:
            frame.pop("action", None)
        frame["task"] = example.task or ""
        return frame


def _shuffle_buffer(
    examples: Iterable[TrainingExampleRef],
    buffer_size: int,
    rng: random.Random,
) -> Iterator[TrainingExampleRef]:
    source = iter(examples)
    buffer = list(islice(source, buffer_size))
    for example in source:
        selected = rng.randrange(len(buffer))
        yield buffer[selected]
        buffer[selected] = example
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))
