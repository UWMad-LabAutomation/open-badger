import torch

from open_badger.models import Model, ModelRegistry


class DummyModel(Model):
    def build_network(self) -> None:
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, inputs, **kwargs):
        return self.linear(inputs)

    def compute_loss(self, batch, outputs):
        return torch.mean(outputs)


@ModelRegistry.register("demo")
class RegisteredDummyModel(DummyModel):
    pass


def test_registry_builds_registered_model():
    model = ModelRegistry.build("demo", {"device": "cpu", "dtype": "float32"})

    assert isinstance(model, DummyModel)
    assert model.model_name == "demo"
    assert model.device.type == "cpu"
    assert model.dtype == torch.float32


def test_registry_raises_for_unknown_model():
    try:
        ModelRegistry.get("missing-model")
    except KeyError:
        return

    raise AssertionError("Expected KeyError for an unknown model name.")


def test_model_training_step_runs_default_pipeline():
    model = RegisteredDummyModel({"device": "cpu", "dtype": "float32"})
    batch = torch.randn(4, 2)

    result = model.training_step(batch)

    assert "loss" in result
    assert "outputs" in result
    assert torch.is_tensor(result["loss"])
