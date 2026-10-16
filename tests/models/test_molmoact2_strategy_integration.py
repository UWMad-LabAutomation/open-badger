from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

from open_badger.models.registry import MolmoAct2
from open_badger.training.checkpointing import TrainingCheckpointManager


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wq = nn.Linear(4, 4)
        self.transformer.other = nn.Linear(4, 4)
        self.vision_backbone = nn.Module()
        self.vision_backbone.patch_embedding = nn.Linear(4, 4)
        self.image_projector = nn.Linear(4, 4)
        self.multi_modal_projector = nn.Linear(4, 4)
        self.action_expert = nn.Module()
        self.action_expert.linear = nn.Linear(4, 4)


class TinyHFMolmoAct2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()


class PolicyStrategyProbe(nn.Module):
    get_optim_params = MolmoAct2Policy.get_optim_params
    _apply_lora_adapters = MolmoAct2Policy._apply_lora_adapters
    _build_inner_lora_config = MolmoAct2Policy._build_inner_lora_config
    _get_inner_peft_targets = MolmoAct2Policy._get_inner_peft_targets
    _lora_target_modules = MolmoAct2Policy._lora_target_modules
    _validate_peft_config = MolmoAct2Policy._validate_peft_config
    _unfreeze_action_expert_parameters = MolmoAct2Policy._unfreeze_action_expert_parameters
    _freeze_vlm_parameters = MolmoAct2Policy._freeze_vlm_parameters
    _freeze_input_embeddings = MolmoAct2Policy._freeze_input_embeddings
    _enable_gradient_checkpointing = MolmoAct2Policy._enable_gradient_checkpointing

    def __init__(self, train_mode_vlm):
        super().__init__()
        self.model = TinyHFMolmoAct2()
        self.config = SimpleNamespace(
            train_mode_vlm=train_mode_vlm,
            checkpoint_path="base-checkpoint",
            lora_rank=2,
            lora_alpha=4,
            lora_dropout=0.0,
            lora_bias="none",
            optimizer_lr=1.0e-5,
            optimizer_vit_lr=5.0e-6,
            optimizer_connector_lr=4.0e-6,
            optimizer_action_expert_lr=5.0e-5,
        )

    def _hf_model(self):
        return self.model

    def _backbone(self):
        return self.model.model


def test_pinned_lerobot_action_expert_only_mode_freezes_non_expert_parameters():
    policy = PolicyStrategyProbe("freeze")

    policy._freeze_vlm_parameters()

    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all("action_expert" in name for name in trainable_names)


def test_pinned_lerobot_lora_mode_trains_adapters_and_action_expert():
    policy = PolicyStrategyProbe("lora")

    policy._apply_lora_adapters()

    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    assert any("transformer.wq.lora_" in name for name in trainable_names)
    assert any("vision_backbone.patch_embedding.lora_" in name for name in trainable_names)
    assert any("action_expert" in name for name in trainable_names)
    assert not any("transformer.other" in name for name in trainable_names)


def test_full_finetune_mode_leaves_non_embedding_backbone_parameters_trainable():
    policy = PolicyStrategyProbe("fft")

    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }

    assert any("transformer.other.weight" in name for name in trainable_names)
    assert any("vision_backbone.patch_embedding.weight" in name for name in trainable_names)
    assert any("action_expert" in name for name in trainable_names)


def test_pinned_lerobot_freezes_input_embeddings_for_full_finetuning():
    policy = PolicyStrategyProbe("fft")
    policy.model.embedding = nn.Embedding(8, 4)
    policy.model.lm_head = nn.Linear(4, 8, bias=False)
    policy.model.get_input_embeddings = lambda: policy.model.embedding
    policy._backbone().get_input_embeddings = lambda: policy.model.embedding

    policy._freeze_input_embeddings()

    assert not policy.model.embedding.weight.requires_grad
    assert policy.model.lm_head.weight.requires_grad


def test_pinned_lerobot_gradient_checkpointing_enables_backbone_components():
    policy = PolicyStrategyProbe("fft")
    policy.config.gradient_checkpointing = True
    calls = []

    def enable_gradient_checkpointing(**kwargs):
        calls.append(kwargs)

    policy.model.gradient_checkpointing_enable = enable_gradient_checkpointing

    policy._enable_gradient_checkpointing()

    assert calls == [{"gradient_checkpointing_kwargs": {"use_reentrant": False}}]
    assert policy._backbone().transformer.gradient_checkpointing
    assert policy._backbone().vision_backbone.gradient_checkpointing


def _make_checkpoint_model(strategy, *, action_expert_only=False):
    model = MolmoAct2.__new__(MolmoAct2)
    nn.Module.__init__(model)
    model.config = {
        "finetuning": {
            "selected_strategy": strategy,
            "action_expert_only": action_expert_only,
        }
    }
    model.policy = nn.Module()
    model.policy.base_vlm = nn.Linear(2, 2)
    model.policy.lora_adapter = nn.Linear(2, 2)
    model.policy.action_expert = nn.Linear(2, 2)
    for name, parameter in model.named_parameters():
        if action_expert_only:
            parameter.requires_grad = "action_expert" in name
        else:
            parameter.requires_grad = strategy == "full_finetune" or (
                strategy == "lora"
                and ("lora_adapter" in name or "action_expert" in name)
            )
    return model


def test_checkpoint_artifacts_match_each_finetuning_strategy():
    lora_model = _make_checkpoint_model("lora")
    lora_filename, lora_state = lora_model.training_checkpoint_artifact()
    assert lora_filename == "adapter_model.safetensors"
    assert all("base_vlm" not in name for name in lora_state)
    assert any("lora_adapter" in name for name in lora_state)
    assert any("action_expert" in name for name in lora_state)

    expert_model = _make_checkpoint_model("lora", action_expert_only=True)
    expert_filename, expert_state = expert_model.training_checkpoint_artifact()
    assert expert_filename == "action_expert.safetensors"
    assert all("action_expert" in name for name in expert_state)
    full_expert_model = _make_checkpoint_model("full_finetune", action_expert_only=True)
    full_expert_filename, full_expert_state = full_expert_model.training_checkpoint_artifact()
    assert full_expert_filename == "action_expert.safetensors"
    assert all("action_expert" in name for name in full_expert_state)

    full_model = _make_checkpoint_model("full_finetune")
    full_filename, full_state = full_model.training_checkpoint_artifact()
    assert full_filename == "model.safetensors"
    assert len(full_state) == len(full_model.state_dict())


def test_lora_checkpoint_round_trips_adapters_and_action_expert(tmp_path):
    source = _make_checkpoint_model("lora")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in source.parameters() if parameter.requires_grad]
    )
    manager = TrainingCheckpointManager(
        tmp_path / "checkpoints",
        keep_last_n=2,
        identity={"strategy": "lora", "dataset_id": "demo/data"},
    )
    parameters = dict(source.named_parameters())
    expected_trainable = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if name in parameters and parameters[name].requires_grad
    }
    checkpoint_path = manager.save(source, optimizer, step=1)

    restored = _make_checkpoint_model("lora")
    restored_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored.parameters() if parameter.requires_grad]
    )
    manager.load("latest", restored, restored_optimizer)

    assert (checkpoint_path / "adapter_model.safetensors").is_file()
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in expected_trainable.items()
    )


def test_lora_policy_uses_open_badger_learning_rates_for_actual_le_robot_groups():
    policy = PolicyStrategyProbe("lora")
    policy._apply_lora_adapters()
    adapter = MolmoAct2.__new__(MolmoAct2)
    nn.Module.__init__(adapter)
    adapter.config = {
        "finetuning": {
            "optimizer": {
                "language_learning_rate": 1.0e-5,
                "vision_learning_rate": 5.0e-6,
                "connector_learning_rate": 4.0e-6,
                "action_expert_learning_rate": 5.0e-5,
            }
        }
    }
    adapter.policy = policy

    groups = adapter.optimizer_parameter_groups()
    rates = {
        id(parameter): group["lr"]
        for group in groups
        for parameter in group["params"]
    }
    names_by_id = {id(parameter): name for name, parameter in policy.named_parameters()}
    lora_rate = next(
        rates[parameter_id]
        for parameter_id, name in names_by_id.items()
        if "transformer.wq.lora_A" in name
    )
    vision_rate = next(
        rates[parameter_id]
        for parameter_id, name in names_by_id.items()
        if "vision_backbone.patch_embedding.lora_A" in name
    )
    action_expert_rate = next(
        rates[parameter_id]
        for parameter_id, name in names_by_id.items()
        if "action_expert" in name and name.endswith("weight")
    )

    assert lora_rate == 1.0e-5
    assert vision_rate == 5.0e-6
    assert action_expert_rate == 5.0e-5


def test_full_finetune_optimizer_groups_separate_language_vision_connectors_and_expert():
    policy = PolicyStrategyProbe("fft")
    adapter = MolmoAct2.__new__(MolmoAct2)
    nn.Module.__init__(adapter)
    adapter.config = {
        "finetuning": {
            "optimizer": {
                "language_learning_rate": 1.0e-5,
                "vision_learning_rate": 5.0e-6,
                "connector_learning_rate": 4.0e-6,
                "action_expert_learning_rate": 5.0e-5,
            }
        }
    }
    adapter.policy = policy

    groups = adapter.optimizer_parameter_groups()
    rates = {
        id(parameter): group["lr"]
        for group in groups
        for parameter in group["params"]
    }

    assert rates[id(policy.model.model.transformer.other.weight)] == 1.0e-5
    assert rates[id(policy.model.model.vision_backbone.patch_embedding.weight)] == 5.0e-6
    assert rates[id(policy.model.model.image_projector.weight)] == 4.0e-6
    assert rates[id(policy.model.model.multi_modal_projector.weight)] == 4.0e-6
    assert rates[id(policy.model.model.action_expert.linear.weight)] == 5.0e-5


def test_molmoact2_optimizer_groups_use_configured_learning_rates():
    model = MolmoAct2.__new__(MolmoAct2)
    nn.Module.__init__(model)
    model.config = {
        "finetuning": {
            "optimizer": {
                "language_learning_rate": 1e-5,
                "vision_learning_rate": 5e-6,
                "connector_learning_rate": 4e-6,
                "action_expert_learning_rate": 5e-5,
            }
        }
    }

    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_backbone = nn.Linear(2, 2)
            self.vision_backbone = nn.Linear(2, 2)
            self.connector = nn.Linear(2, 2)
            self.action_expert = nn.Linear(2, 2)

        def get_optim_params(self):
            return [
                {"params": [parameter], "lr": 5e-5}
                for parameter in self.parameters()
            ]

    model.policy = FakePolicy()

    groups = model.optimizer_parameter_groups()
    configured_lrs = {
        id(parameter): group["lr"]
        for group in groups
        for parameter in group["params"]
    }

    assert configured_lrs[id(model.policy.language_backbone.weight)] == 1e-5
    assert configured_lrs[id(model.policy.vision_backbone.weight)] == 5e-6
    assert configured_lrs[id(model.policy.connector.weight)] == 4e-6
    assert configured_lrs[id(model.policy.action_expert.weight)] == 5e-5
