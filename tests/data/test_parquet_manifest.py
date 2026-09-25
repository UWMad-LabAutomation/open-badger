from dataclasses import dataclass

from open_badger.data import (
    DatasetMetadata,
    FeatureMetadata,
    SampleManifest,
    SamplePointer,
)
from open_badger.data.manifest import ParquetManifest


@dataclass
class FakeReader:
    dataset_metadata: DatasetMetadata
    pointers: list[SamplePointer]

    def metadata(self):
        return self.dataset_metadata

    def iter_pointers(self):
        return iter(self.pointers)

    def get_frame(self, pointer):
        raise AssertionError("Manifest generation must not decode frames.")


def make_metadata():
    return DatasetMetadata(
        dataset_id="demo/dataset",
        revision="rev-1",
        format_version="v3",
        fps=10.0,
        total_frames=3,
        total_episodes=1,
        features={
            "observation.state": FeatureMetadata("observation.state", "state", (2,), "float32"),
            "action": FeatureMetadata("action", "action", (3,), "float32"),
        },
        episode_lengths={0: 3},
        tasks={0: "pick"},
    )


def test_build_parquet_streams_and_round_trips_references(tmp_path):
    metadata = make_metadata()
    pointers = [
        SamplePointer("demo/dataset", "rev-1", 0, 0, 0, 0.0, 10.0),
        SamplePointer("demo/dataset", "rev-1", 0, 1, 1, 0.1, 10.0),
        SamplePointer("demo/dataset", "rev-1", 0, 2, 2, 0.2, 10.0),
    ]

    manifest = SampleManifest.build_parquet(
        FakeReader(metadata, pointers),
        tmp_path / "manifest",
        action_horizon=2,
        write_batch_size=1,
    )
    restored = ParquetManifest.read(tmp_path / "manifest")

    assert len(manifest) == 2
    assert len(restored) == 2
    assert [example.sample.global_frame_index for example in restored] == [0, 1]
    assert all(example.action_frame_offsets == (0, 1) for example in restored)
    assert (tmp_path / "manifest" / "examples.parquet").exists()
