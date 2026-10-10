from types import SimpleNamespace
from typing import ClassVar

import pytest

from open_badger.data.readers.lerobot import LeRobotReader


class FakeTasks:
    index: ClassVar[list[str]] = ["pick", "place"]


class FakeDataset:
    repo_id = "demo/dataset"
    revision = "rev-1"

    def __init__(self):
        self.meta = SimpleNamespace(
            info=SimpleNamespace(codebase_version="v3.0"),
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["x", "y"],
                },
                "action": {"dtype": "float32", "shape": [3]},
                "observation.images.front": {
                    "dtype": "video",
                    "shape": [3, 64, 64],
                },
            },
            fps=10,
            total_frames=3,
            total_episodes=1,
            episodes=[
                {"episode_index": 0, "length": 3, "tasks": ["pick"]},
            ],
            tasks=FakeTasks(),
            stats={"action": {"mean": [0.0, 0.0, 0.0]}},
        )
        self.rows = [
            {
                "index": 100,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
            },
            {
                "index": 101,
                "episode_index": 0,
                "frame_index": 1,
                "timestamp": 0.1,
            },
            {
                "index": 102,
                "episode_index": 0,
                "frame_index": 2,
                "timestamp": 0.2,
            },
        ]
        self.frames = [
            {**row, "observation.state": [row["frame_index"], 0], "action": [1, 2, 3]}
            for row in self.rows
        ]
        self.raw_read_indices = []
        self.frame_read_indices = []

    def __len__(self):
        return len(self.rows)

    def get_raw_item(self, index):
        self.raw_read_indices.append(index)
        return self.rows[index]

    def __getitem__(self, index):
        self.frame_read_indices.append(index)
        return self.frames[index]


def test_lerobot_reader_normalizes_metadata_and_pointers():
    reader = LeRobotReader("demo/dataset", dataset=FakeDataset())

    metadata = reader.metadata()
    pointers = list(reader.iter_pointers())

    assert metadata.dataset_id == "demo/dataset"
    assert metadata.revision == "rev-1"
    assert metadata.episode_lengths == {0: 3}
    assert metadata.tasks == {0: "pick"}
    assert metadata.action_feature is not None
    assert metadata.action_feature.shape == (3,)
    assert metadata.camera_keys == ("observation.images.front",)
    assert [pointer.global_frame_index for pointer in pointers] == [100, 101, 102]
    assert [pointer.dataset_index for pointer in pointers] == [0, 1, 2]


def test_lerobot_reader_resolves_frame_by_global_pointer():
    reader = LeRobotReader("demo/dataset", dataset=FakeDataset())
    pointer = next(pointer for pointer in reader.iter_pointers() if pointer.global_frame_index == 101)

    frame = reader.get_frame(pointer)

    assert frame["observation.state"] == [1, 0]
    assert frame["action"] == [1, 2, 3]
    assert reader.dataset.frame_read_indices == [1]
    assert reader.dataset.raw_read_indices == [0, 1, 1]


def test_lerobot_pointer_iteration_is_lazy():
    dataset = FakeDataset()
    reader = LeRobotReader("demo/dataset", dataset=dataset)

    pointers = reader.iter_pointers()

    assert dataset.raw_read_indices == []
    assert next(pointers).dataset_index == 0
    assert dataset.raw_read_indices == [0]
    assert not hasattr(reader, "_pointers")
    assert not hasattr(reader, "_relative_indices")


def test_lerobot_reader_rejects_pointer_from_another_dataset():
    reader = LeRobotReader("demo/dataset", dataset=FakeDataset())
    pointer = next(reader.iter_pointers())
    foreign_pointer = pointer.__class__(
        "other/dataset",
        pointer.revision,
        pointer.episode_index,
        pointer.episode_frame_index,
        pointer.global_frame_index,
        pointer.timestamp,
        pointer.fps,
    )

    with pytest.raises(ValueError, match="dataset identity"):
        reader.get_frame(foreign_pointer)
