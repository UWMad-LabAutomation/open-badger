from dataclasses import dataclass

import pytest

from open_badger.data import (
    DatasetMetadata,
    FeatureMetadata,
    SampleManifest,
    SamplePointer,
)
from open_badger.data.sampling import action_frame_offsets


@dataclass
class FakeReader:
    dataset_metadata: DatasetMetadata
    pointers: list[SamplePointer]

    def metadata(self):
        return self.dataset_metadata

    def iter_pointers(self):
        return iter(self.pointers)

    def get_frame(self, pointer):
        return {"pointer": pointer}


def make_metadata():
    return DatasetMetadata(
        dataset_id="demo/dataset",
        revision="rev-1",
        format_version="v3",
        fps=10.0,
        total_frames=8,
        total_episodes=2,
        features={
            "observation.state": FeatureMetadata(
                "observation.state", "state", (2,), "float32", names=("x", "y")
            ),
            "action": FeatureMetadata("action", "action", (3,), "float32", names=("x", "y", "gripper")),
            "observation.images.front": FeatureMetadata(
                "observation.images.front", "image", (3, 64, 64), "uint8"
            ),
        },
        episode_lengths={0: 5, 1: 3},
        tasks={0: "pick", 1: "place"},
    )


def test_action_offsets_follow_dataset_fps():
    assert action_frame_offsets(action_horizon=3, fps=10.0, action_spacing_seconds=0.2) == (0, 2, 4)


def test_manifest_drops_incomplete_horizons_and_preserves_metadata():
    metadata = make_metadata()
    pointers = [
        SamplePointer("demo/dataset", "rev-1", 0, 0, 0, 0.0, 10.0),
        SamplePointer("demo/dataset", "rev-1", 0, 2, 2, 0.2, 10.0),
        SamplePointer("demo/dataset", "rev-1", 1, 0, 5, 0.0, 10.0),
    ]
    manifest = SampleManifest.build(
        FakeReader(metadata, pointers),
        action_horizon=3,
        action_spacing_seconds=0.1,
        shuffle=True,
        seed=7,
    )

    assert len(manifest) == 2
    assert {example.sample.episode_frame_index for example in manifest} == {0, 2}
    assert manifest.examples[0].task in {"pick", "place"}
    assert manifest.examples[0].action_dim == 3
    assert manifest.examples[0].state_dim == 2


def test_manifest_round_trip(tmp_path):
    metadata = make_metadata()
    pointer = SamplePointer("demo/dataset", "rev-1", 1, 0, 5, 0.0, 10.0)
    manifest = SampleManifest.build(FakeReader(metadata, [pointer]), action_horizon=1)

    manifest.write(tmp_path / "manifest")
    restored = SampleManifest.read(tmp_path / "manifest")

    assert restored.metadata == manifest.metadata
    assert restored.examples == manifest.examples


def test_manifest_rejects_mismatched_pointer_revision():
    metadata = make_metadata()
    pointer = SamplePointer("demo/dataset", "other-rev", 0, 0, 0, 0.0, 10.0)

    with pytest.raises(ValueError, match="Pointer dataset identity"):
        SampleManifest.build(FakeReader(metadata, [pointer]), action_horizon=1)
