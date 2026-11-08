import pytest
import torch

from scripts.train import _run_vla_inference_smoke_test, _run_vla_smoke_test


class ToyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch):
        return {
            "loss": self.weight * batch["loss"],
            "metrics": {"metric": 2.0},
        }

    def predict_action_chunk_from_batch(self, batch):
        return batch["actions"]


def test_vla_smoke_test_uses_generic_training_step(capsys):
    model = ToyVLA()

    _run_vla_smoke_test(model, [{"loss": torch.tensor(1.0)}])

    assert "VLA forward/loss smoke test passed" in capsys.readouterr().out


def test_vla_inference_smoke_test_accepts_action_chunks(capsys):
    model = ToyVLA()
    actions = torch.zeros(1, 10, 6)

    _run_vla_inference_smoke_test(model, [{"actions": actions}])

    assert "VLA inference smoke test passed" in capsys.readouterr().out


def test_vla_inference_smoke_test_requires_model_prediction_hook():
    with pytest.raises(TypeError, match="implement predict_action_chunk_from_batch"):
        _run_vla_inference_smoke_test(torch.nn.Linear(1, 1), [{"input": torch.ones(1)}])


def test_vla_inference_smoke_test_rejects_non_finite_action_values():
    model = ToyVLA()
    actions = torch.full((1, 1, 1), float("nan"))

    with pytest.raises(FloatingPointError, match="VLA produced non-finite"):
        _run_vla_inference_smoke_test(model, [{"actions": actions}])
