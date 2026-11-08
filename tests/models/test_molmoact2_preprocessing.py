from types import SimpleNamespace

import pytest
import torch

from open_badger.data.schema import DatasetMetadata, FeatureMetadata
from open_badger.models.registry import MolmoAct2


def make_metadata():
    return DatasetMetadata(
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
            "observation.images.front": FeatureMetadata(
                "observation.images.front", "image", (3, 2, 2), "uint8"
            ),
        },
        episode_lengths={0: 2},
        stats={"observation.state": {}, "action": {}},
    )


def make_uninitialized_model():
    model = MolmoAct2.__new__(MolmoAct2)
    torch.nn.Module.__init__(model)
    model.policy = SimpleNamespace(
        config=SimpleNamespace(
            normalization_mapping={
                "STATE": SimpleNamespace(name="QUANTILES"),
                "ACTION": SimpleNamespace(name="QUANTILES"),
            }
        ),
        predict_action_chunk=lambda inputs, **kwargs: torch.ones(1, 10, 2),
    )
    model.image_keys = ["observation.images.front"]
    model.state_dim = 2
    model.action_dim = 2
    model.max_action_dim = 32
    model.action_horizon = 2
    model.history_length = 1
    model.norm_tag = None
    model.preprocessor = None
    model.postprocessor = None
    model._configured_dataset_metadata = None
    model._configured_native_metadata = None
    model.num_flow_matching_steps = 8
    return model


def test_preprocess_batch_uses_official_pipeline_and_normalizes_uint8_images(monkeypatch):
    import lerobot.policies

    observed = {}

    def processor(batch):
        observed.update(batch)
        return batch

    def make_processors(policy_config, *, dataset_stats, dataset_meta):
        assert policy_config.normalization_mapping["STATE"].name == "QUANTILES"
        assert dataset_stats == make_metadata().stats
        assert dataset_meta == "native-metadata"
        return processor, object()

    monkeypatch.setattr(lerobot.policies, "make_pre_post_processors", make_processors)
    model = make_uninitialized_model()
    metadata = make_metadata()
    model.configure_dataset(metadata, native_metadata="native-metadata")

    batch = {
        "observation.state": torch.tensor([[0.1, 0.2]]),
        "observation.images.front": torch.tensor(
            [[[[0, 255], [128, 64]], [[0, 255], [128, 64]], [[0, 255], [128, 64]]]],
            dtype=torch.uint8,
        ),
        "action": torch.ones(1, 2, 2),
        "task": ["pick"],
    }

    processed = model.preprocess_batch(batch)

    image = observed["observation.images.front"]
    assert image.dtype == torch.float32
    assert image[0, 0, 0, 1].item() == 1.0
    assert image[0, 0, 1, 0].item() == pytest.approx(128 / 255)
    assert observed["observation.state"].shape == (1, 2)
    assert observed["action"].shape == (1, 2, 2)
    assert processed["task"] == ["pick"]


def test_preprocess_batch_requires_dataset_processor():
    model = make_uninitialized_model()

    try:
        model.preprocess_batch(
            {
                "observation.state": torch.zeros(1, 2),
                "observation.images.front": torch.zeros(1, 3, 2, 2),
                "action": torch.zeros(1, 2, 2),
            }
        )
    except RuntimeError as exc:
        assert "configure_dataset" in str(exc)
    else:
        raise AssertionError("Expected preprocessing without dataset setup to fail.")


def test_observation_preprocessing_and_action_prediction_need_no_target(monkeypatch):
    import lerobot.policies

    observed = {}

    def processor(batch):
        observed.update(batch)
        return batch

    monkeypatch.setattr(
        lerobot.policies,
        "make_pre_post_processors",
        lambda *args, **kwargs: (processor, object()),
    )
    model = make_uninitialized_model()
    model.configure_dataset(make_metadata())
    batch = {
        "observation.state": torch.tensor([[0.1, 0.2]]),
        "observation.images.front": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "task": ["pick"],
    }

    actions = model.predict_action_chunk_from_batch(batch)

    assert actions.shape == (1, 10, 2)
    assert "action" not in observed
    assert observed["observation.images.front"].dtype == torch.float32
    assert torch.all(observed["observation.images.front"] == 1.0)
