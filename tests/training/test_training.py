from pathlib import Path

import pytest
import torch

from open_badger.data.schema import (
    DatasetMetadata,
    FeatureMetadata,
    SamplePointer,
    TrainingExampleRef,
)
from open_badger.training.training import (
    build_optimizer,
    build_training_dataloader,
    load_training_config,
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


def test_build_training_dataloader_caps_workers_to_available_examples():
    reader = FakeReader()
    example = TrainingExampleRef(
        sample=SamplePointer(
            "demo/dataset", "rev-1", 0, 0, 0, 0.0, 10.0, dataset_index=0
        ),
        action_horizon=1,
        action_frame_offsets=(0,),
        action_dim=2,
        state_dim=2,
    )

    with pytest.warns(UserWarning, match="fewer independent work shards"):
        loader = build_training_dataloader(
            reader,
            [example],
            {
                "sampling": {"shuffle": False},
                "loader": {
                    "batch_size": 1,
                    "num_workers": 4,
                    "persistent_workers": False,
                },
            },
        )

    assert loader.num_workers == 1


def test_build_training_dataloader_serializes_unsplittable_iterables():
    class SizedIterable:
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __iter__(self):
            return iter(self.examples)

    reader = FakeReader()
    example = TrainingExampleRef(
        sample=SamplePointer(
            "demo/dataset", "rev-1", 0, 0, 0, 0.0, 10.0, dataset_index=0
        ),
        action_horizon=1,
        action_frame_offsets=(0,),
        action_dim=2,
        state_dim=2,
    )
    with pytest.warns(UserWarning, match="fewer independent work shards"):
        loader = build_training_dataloader(
            reader,
            SizedIterable([example]),
            {
                "sampling": {"shuffle": False},
                "loader": {
                    "batch_size": 1,
                    "num_workers": 4,
                    "persistent_workers": False,
                },
            },
        )

    assert loader.num_workers == 1


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


def test_load_training_config_composes_referenced_model_and_data_configs():
    repository_root = Path(__file__).resolve().parents[2]

    model_config, data_config = load_training_config(
        repository_root / "configs/training/train.yaml"
    )

    assert model_config["model"]["name"] == "molmoact2"
    assert model_config["finetuning"]["selected_strategy"] == "lora"
    assert model_config["data"]["sampling"]["action_horizon"] == 10
    assert data_config["manifest"]["rebuild"] is False


def test_load_training_config_supports_strategy_override():
    repository_root = Path(__file__).resolve().parents[2]

    model_config, _ = load_training_config(
        repository_root / "configs/training/train.yaml",
        strategy="action_expert_only",
    )

    assert model_config["finetuning"]["selected_strategy"] == "action_expert_only"


def test_build_optimizer_uses_model_parameter_groups():
    model = ToyModel()
    model.policy = object()
    model.optimizer_parameter_groups = lambda: [{"params": [model.weight], "lr": 0.03}]

    optimizer = build_optimizer(
        model,
        {
            "finetuning": {
                "optimizer": {
                    "learning_rate": 0.1,
                    "betas": [0.8, 0.9],
                    "eps": 1e-6,
                    "weight_decay": 0.01,
                }
            }
        },
    )

    assert optimizer.param_groups[0]["lr"] == 0.03
    assert optimizer.defaults["lr"] == 0.1
    assert optimizer.defaults["betas"] == (0.8, 0.9)
