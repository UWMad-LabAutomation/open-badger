import torch
import pytest

from open_badger.training.checkpointing import TrainingCheckpointManager


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)


def test_checkpoint_round_trips_model_optimizer_and_latest_pointer(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model.layer(torch.ones(1, 2)).square().mean()
    loss.backward()
    optimizer.step()
    expected_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    manager = TrainingCheckpointManager(
        tmp_path / "checkpoints",
        keep_last_n=2,
        identity={"model": "tiny", "strategy": "full_finetune"},
    )

    path = manager.save(model, optimizer, step=1)

    restored_model = TinyModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    step = manager.load("latest", restored_model, restored_optimizer)

    assert path.name == "checkpoint-00000001"
    assert (path / "model.safetensors").is_file()
    assert (path / "training_state.pt").is_file()
    assert step == 1
    assert all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in expected_state.items()
    )
    assert restored_optimizer.state_dict()["state"]


def test_checkpoint_retention_keeps_only_newest_checkpoints(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    manager = TrainingCheckpointManager(
        tmp_path / "checkpoints",
        keep_last_n=2,
        identity={"model": "tiny"},
    )

    for step in range(1, 4):
        manager.save(model, optimizer, step=step)

    assert sorted(path.name for path in manager.directory.glob("checkpoint-*")) == [
        "checkpoint-00000002",
        "checkpoint-00000003",
    ]
    assert manager.load("latest", model, optimizer) == 3


def test_checkpoint_rejects_identity_mismatch(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    manager = TrainingCheckpointManager(
        tmp_path / "checkpoints",
        keep_last_n=1,
        identity={"strategy": "lora"},
    )
    manager.save(model, optimizer, step=1)
    other_run = TrainingCheckpointManager(
        tmp_path / "checkpoints",
        keep_last_n=1,
        identity={"strategy": "full_finetune"},
    )
    other_model = TinyModel()

    with pytest.raises(ValueError, match="does not match"):
        other_run.load("latest", other_model, torch.optim.AdamW(other_model.parameters()))
