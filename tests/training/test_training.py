import torch

from open_badger.data.schema import (
    DatasetMetadata,
    FeatureMetadata,
    SamplePointer,
    TrainingExampleRef,
)
from open_badger.training.training import (
    build_training_dataloader,
    train,
    train_from_manifest,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.device = torch.device("cpu")

    def training_step(self, batch):
        loss = (self.weight * batch["input"] - batch["target"]).square().mean()
        return {"loss": loss, "metrics": {"train_loss": loss.detach()}}


class FakeReader:
    def __init__(self):
        self.dataset_metadata = DatasetMetadata(
            dataset_id="demo/dataset",
            revision="rev-1",
            format_version="v3",
            fps=10.0,
            total_frames=2,
            total_episodes=1,
            features={
                "observation.state": FeatureMetadata(
                    "observation.state", "state", (2,), "float32"
                ),
                "action": FeatureMetadata("action", "action", (2,), "float32"),
            },
            episode_lengths={0: 2},
        )

    def metadata(self):
        return self.dataset_metadata

    def get_frame(self, pointer):
        return {"observation.state": [0.0, 1.0]}

    def get_feature(self, pointer, feature_name, *, frame_offset=0):
        return [float(frame_offset), 2.0]


def test_train_runs_optimizer_steps_with_gradient_accumulation():
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        {"input": torch.tensor([1.0]), "target": torch.tensor([1.0])},
        {"input": torch.tensor([2.0]), "target": torch.tensor([2.0])},
    ]

    result = train(
        model,
        batches,
        optimizer,
        max_steps=2,
        gradient_accumulation_steps=2,
    )

    assert result.steps == 2
    assert result.loss >= 0
    assert result.metrics["train_loss"] >= 0
    assert model.weight.item() > 0


def test_train_rejects_empty_batches():
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    try:
        train(model, [], optimizer, max_steps=1)
    except ValueError as exc:
        assert "empty batch iterable" in str(exc)
    else:
        raise AssertionError("Expected an empty iterable to be rejected.")


def test_build_training_dataloader_resolves_manifest_and_batches():
    reader = FakeReader()
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
        task="pick",
        action_dim=2,
        state_dim=2,
    )
    loader = build_training_dataloader(
        reader,
        [example],
        {
            "sampling": {"shuffle": False},
            "loader": {
                "batch_size": 1,
                "num_workers": 0,
                "persistent_workers": False,
            },
        },
    )

    batch = next(iter(loader))

    assert batch["observation.state"].shape == (1, 2)
    assert batch["action"].shape == (1, 2, 2)


def test_train_from_manifest_passes_resolved_action_chunk_to_model():
    class ChunkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.device = torch.device("cpu")
            self.action_shape = None

        def training_step(self, batch):
            self.action_shape = batch["action"].shape
            loss = (batch["action"] * self.scale).square().mean()
            return {"loss": loss}

    reader = FakeReader()
    example = TrainingExampleRef(
        sample=SamplePointer(
            "demo/dataset",
            "rev-1",
            0,
            0,
            0,
            0.0,
            10.0,
            dataset_index=0,
        ),
        action_horizon=2,
        action_frame_offsets=(0, 1),
        action_dim=2,
        state_dim=2,
    )
    model = ChunkModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_from_manifest(
        model,
        reader,
        [example],
        optimizer,
        data_config={
            "sampling": {"shuffle": False},
            "loader": {
                "batch_size": 1,
                "num_workers": 0,
                "persistent_workers": False,
            },
        },
        max_steps=1,
    )

    assert result.steps == 1
    assert model.action_shape == (1, 2, 2)
