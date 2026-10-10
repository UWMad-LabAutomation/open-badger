from dataclasses import dataclass

import torch

from open_badger.data.dataset import ManifestTrainingDataset
from open_badger.data.schema import (
    DatasetMetadata,
    FeatureMetadata,
    SamplePointer,
    TrainingExampleRef,
)


def make_metadata():
    return DatasetMetadata(
        dataset_id="demo/dataset",
        revision="rev-1",
        format_version="v3",
        fps=10.0,
        total_frames=3,
        total_episodes=1,
        features={
            "observation.state": FeatureMetadata(
                "observation.state", "state", (2,), "float32"
            ),
            "action": FeatureMetadata("action", "action", (2,), "float32"),
        },
        episode_lengths={0: 3},
        tasks={0: "pick"},
    )


@dataclass
class FakeReader:
    dataset_metadata: DatasetMetadata

    def metadata(self):
        return self.dataset_metadata

    def get_frame(self, pointer):
        return {
            "observation.state": [pointer.episode_frame_index, 1.0],
            "observation.images.front": torch.zeros(3, 8, 8),
        }

    def get_feature(self, pointer, feature_name, *, frame_offset=0):
        assert feature_name == "action"
        return [pointer.episode_frame_index + frame_offset, 2.0]


def test_manifest_dataset_resolves_observation_and_action_chunk():
    reader = FakeReader(make_metadata())
    pointer = SamplePointer(
        "demo/dataset",
        "rev-1",
        0,
        0,
        0,
        0.0,
        10.0,
        dataset_index=0,
    )
    example = TrainingExampleRef(
        sample=pointer,
        action_horizon=2,
        action_frame_offsets=(0, 1),
        camera_keys=("observation.images.front",),
        task="pick",
        action_dim=2,
        state_dim=2,
    )
    dataset = ManifestTrainingDataset(reader, [example])

    resolved = next(iter(dataset))

    assert resolved["observation.state"].tolist() == [0.0, 1.0]
    assert resolved["observation.images.front"].shape == (3, 8, 8)
    assert resolved["action"].shape == (2, 2)
    assert resolved["action"].tolist() == [[0, 2.0], [1, 2.0]]
    assert resolved["task"] == "pick"


def test_manifest_dataset_shuffles_with_bounded_buffer_reproducibly():
    reader = FakeReader(make_metadata())
    examples = [
        TrainingExampleRef(
            sample=SamplePointer(
                "demo/dataset",
                "rev-1",
                0,
                index,
                index,
                index / 10.0,
                10.0,
                dataset_index=index,
            ),
            action_horizon=1,
            action_frame_offsets=(0,),
            action_dim=2,
            state_dim=2,
        )
        for index in range(3)
    ]

    first = ManifestTrainingDataset(
        reader, examples, shuffle=True, shuffle_buffer_size=2, seed=7
    )
    second = ManifestTrainingDataset(
        reader, examples, shuffle=True, shuffle_buffer_size=2, seed=7
    )

    first_order = [row["observation.state"][0].item() for row in first]
    second_order = [row["observation.state"][0].item() for row in second]

    assert first_order == second_order
    assert sorted(first_order) == [0.0, 1.0, 2.0]
