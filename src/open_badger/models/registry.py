"""Model registry and shared abstractions for policy implementations.

The registry is intentionally narrow: it resolves a logical model identifier such as
"molmoact2" to a concrete implementation class, while the training/inference layers
remain responsible for backend-specific orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


class Model(nn.Module, ABC):
    """Base class shared by all PyTorch policy implementations.

    Child models define how to build their specific backbone and encode Lerobot2
    batches into model inputs/outputs. The trainer works with the generic model
    interface instead of model-specific implementation details.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = dict(config or {})
        model_config = self.config.get("model", {}) or {}
        runtime_config = self.config.get("runtime", {}) or {}
        self.model_name = (
            self.config.get("name")
            or self.config.get("model_name")
            or model_config.get("name")
            or self.__class__.__name__
        )
        self.device = self._resolve_device(self.config.get("device", runtime_config.get("device")))
        self.dtype = self._resolve_dtype(self.config.get("dtype", runtime_config.get("dtype")))
        self.checkpoint_path = self.config.get("checkpoint") or self.config.get("checkpoint_path")

        self.build_network()

    # cpu is not supported as a default; must have CUDA available
    @staticmethod
    def _resolve_device(value: Any) -> torch.device | str:
        if value is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                raise RuntimeError("No CUDA device available.")
        return torch.device(value) if not isinstance(value, torch.device) else value

    @staticmethod
    def _resolve_dtype(value: Any) -> torch.dtype:
        if value is None:
            return torch.float32

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
            "half": torch.float16,
            "full": torch.float32,
        }

        # support direct torch.dtype instances as well as string representations
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("-", "")
            if normalized in dtype_map:
                return dtype_map[normalized]
        raise ValueError(f"Unsupported dtype config: {value!r}")

    @property
    def name(self) -> str:
        return self.model_name

    @abstractmethod
    def build_network(self) -> None:
        """Create and attach the actual model modules."""

    @abstractmethod
    def forward(self, inputs: Any, **kwargs: Any) -> Any:
        """Run the model on a single batch or batch-like structure."""

    @abstractmethod
    def preprocess_batch(self, batch: Any) -> Any:
        """Normalize a Lerobot2-like batch into the backend's input contract."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement preprocess_batch().")

    @abstractmethod
    def compute_loss(self, batch: Any, outputs: Any) -> torch.Tensor:
        """Compute the training objective for the given outputs."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute_loss().")

    def training_step(self, batch: Any) -> dict[str, Any]:
        """Apply the default training-step workflow for a model-specific policy."""
        inputs = self.preprocess_batch(batch)
        outputs = self(inputs)
        loss = self.compute_loss(batch, outputs)
        return {"loss": loss, "outputs": outputs}

    def eval_step(self, batch: Any) -> dict[str, Any]:
        """Evaluation step. Defaults to the same workflow as training."""
        return self.training_step(batch)

    def set_device(self, device: str | torch.device) -> None:
        self.device = self._resolve_device(device)
        self.to(self.device)

    def set_dtype(self, dtype: str | torch.dtype) -> None:
        self.dtype = self._resolve_dtype(dtype)
        self.to(dtype=self.dtype)

    def load_checkpoint(self, path: str | None = None) -> None:
        checkpoint_path = path or self.checkpoint_path
        if checkpoint_path is None:
            return

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def save_checkpoint(self, path: str | None = None) -> None:
        checkpoint_path = path or self.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("A checkpoint path must be provided to save a model state.")

        torch.save({"model_state_dict": self.state_dict()}, checkpoint_path)

    def freeze_parameters(self, freeze: bool = True, exclude: set[str] | None = None) -> None:
        exclude = exclude or set()
        for name, parameter in self.named_parameters():
            if name in exclude:
                continue
            parameter.requires_grad = not freeze

    def unfreeze_parameters(self, exclude: set[str] | None = None) -> None:
        self.freeze_parameters(freeze=False, exclude=exclude)

    @staticmethod
    def select_history_window(value: Any, *, history_length: int) -> tuple[Any, Any | None]:
        """Split one example's observation value into (current_frame, full_window).

        Shared across model implementations so frame-stacking is a config toggle
        rather than a per-model reimplementation. `history_length=1` is the
        Markovian default: `value` is a single frame and `full_window` is `None`.
        `history_length>1` expects `value` as a sequence of that many frames
        ordered oldest-to-newest; the most recent frame is `current_frame`.
        """
        if history_length == 1:
            return value, None
        if len(value) != history_length:
            raise ValueError(f"Expected {history_length} stacked frames, got {len(value)}.")
        return value[-1], value


class MolmoAct2(Model):
    """Open Badger adapter around the official LeRobot MolmoAct2 policy."""

    model_name = "molmoact2"

    def build_network(self) -> None:
        try:
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
            from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        except ImportError as exc:
            raise ImportError(
                "MolmoAct2 requires the official LeRobot MolmoAct2 policy. "
                "Build the Docker image or install the pinned lerobot[molmoact2] dependency."
            ) from exc

        model_cfg = self.config.get("model", self.config) or {}
        checkpoint_cfg = model_cfg.get("checkpoint", {}) or {}
        runtime_cfg = self.config.get("runtime", {}) or {}
        inference_cfg = self.config.get("inference", {}) or {}
        training_cfg = self.config.get("training", {}) or {}

        repo_id = checkpoint_cfg.get("repo_id", "allenai/MolmoAct2")
        revision = checkpoint_cfg.get("revision")
        self.norm_tag = inference_cfg.get("norm_tag")
        self.action_mode = training_cfg.get("action_mode", "continuous")
        self.image_keys = list(inference_cfg.get("image_keys") or [])
        if not self.image_keys:
            raise ValueError(
                "MolmoAct2 requires inference.image_keys so the official LeRobot "
                "policy can build its visual input features."
            )
        self.n_action_steps = inference_cfg.get("n_action_steps")
        self.num_flow_matching_steps = int(
            training_cfg.get("flow_matching_steps", inference_cfg.get("flow_matching_steps", 8))
        )
        self.action_dim = int(inference_cfg.get("action_dim", model_cfg.get("max_action_dim", 32)))
        self.action_horizon = int(inference_cfg.get("action_horizon", 1))
        max_action_horizon = int(
            model_cfg.get("max_action_horizon", 30)
        )
        if self.action_horizon < 1 or self.action_horizon > max_action_horizon:
            raise ValueError(
                "`inference.action_horizon` must be between 1 and the checkpoint "
                f"maximum ({max_action_horizon}), got {self.action_horizon}."
            )
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        if int(self.n_action_steps) > self.action_horizon:
            raise ValueError(
                "`inference.n_action_steps` cannot exceed `inference.action_horizon`."
            )
        self.history_length = int(inference_cfg.get("history_length", 1))
        if self.history_length < 1:
            raise ValueError("`inference.history_length` must be >= 1.")

        state_dim = int(inference_cfg.get("state_dim", self.action_dim))
        input_features = {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for key in self.image_keys
        }
        input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,),
        )
        output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        }

        policy_config = MolmoAct2Config(
            checkpoint_path=repo_id,
            checkpoint_revision=revision,
            chunk_size=self.action_horizon,
            n_action_steps=int(self.n_action_steps),
            action_mode=self.action_mode,
            inference_action_mode=inference_cfg.get("action_mode", "continuous"),
            norm_tag=self.norm_tag,
            image_keys=self.image_keys,
            num_flow_timesteps=self.num_flow_matching_steps,
            expected_max_action_dim=int(model_cfg.get("max_action_dim", 32)),
            model_dtype=str(runtime_cfg.get("dtype", "bfloat16")),
            train_mode_vlm=training_cfg.get("train_mode_vlm", "lora"),
            gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
            input_features=input_features,
            output_features=output_features,
            device=str(self.device),
        )
        self.policy = MolmoAct2Policy(policy_config)
        self.policy.to(self.device)

    def preprocess_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a batch prepared for the official LeRobot policy.

        Universal Open Badger data preprocessing owns conversion from raw
        LeRobot observations to this tensor contract. The official policy then
        owns image/token processing, action normalization, and padding masks.
        """
        if not isinstance(batch, Mapping):
            raise TypeError("MolmoAct2 expects a mapping batch.")
        return batch

    def forward(self, inputs: Mapping[str, Any], **kwargs: Any) -> torch.Tensor:
        """Delegate inference-time action generation to the official policy."""
        if not isinstance(inputs, Mapping):
            raise TypeError("MolmoAct2.forward() expects an official tensor batch mapping.")
        return self.policy.predict_action_chunk(
            dict(inputs),
            inference_action_mode=kwargs.get("inference_action_mode"),
            num_steps=kwargs.get("num_steps", self.num_flow_matching_steps),
        )

    def training_step(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Delegate training and loss computation to official LeRobot code."""
        prepared_batch = self.preprocess_batch(batch)
        loss, metrics = self.policy(dict(prepared_batch), reduction="mean")
        return {"loss": loss, "metrics": metrics}

    def compute_loss(self, batch: Any, outputs: Any) -> torch.Tensor:
        """Return a loss produced by the official policy forward method."""
        if isinstance(outputs, tuple):
            return outputs[0]
        if torch.is_tensor(outputs):
            return outputs
        raise TypeError("Official MolmoAct2 outputs must be a (loss, metrics) tuple or tensor.")


class ModelRegistry:
    """Factory and registry for concrete model implementations."""

    _registry: dict[str, type[Model]] = {}

    @classmethod
    def register(
        cls,
        name: str | None = None,
    ) -> Callable[[type[Model]], type[Model]]:
        def decorator(model_cls: type[Model]) -> type[Model]:
            if not issubclass(model_cls, Model):
                raise TypeError(f"{model_cls.__name__} must subclass Model.")

            registry_name = cls._normalize_name(name or getattr(model_cls, "model_name", model_cls.__name__))
            cls._registry[registry_name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def _normalize_name(cls, name: str) -> str:
        return name.strip().lower().replace("-", "").replace("_", "")

    @classmethod
    def available_models(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def get(cls, name: str) -> type[Model]:
        normalized_name = cls._normalize_name(name)
        if normalized_name not in cls._registry:
            available = ", ".join(cls.available_models()) or "none"
            raise KeyError(f"Unknown model '{name}'. Available models: {available}.")
        return cls._registry[normalized_name]

    @classmethod
    def build(cls, name: str, config: dict[str, Any] | None = None, **kwargs: Any) -> Model:
        model_cls = cls.get(name)
        model_config = dict(config or {})
        model_config.setdefault("name", name)

        instance = model_cls(model_config, **kwargs)
        if not isinstance(instance, Model):
            raise TypeError(f"Resolved model '{name}' did not return a Model instance.")
        return instance

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> Model:
        if not isinstance(config, dict):
            raise TypeError("Model config must be a dictionary.")

        model_config = config.get("model", config)
        name = config.get("name") or config.get("model_name") or model_config.get("name")
        if name is None:
            raise KeyError("Model config must define a 'name', 'model', or 'model_name'.")

        return cls.build(str(name), config=config)


ModelRegistry.register("molmoact2")(MolmoAct2)


__all__ = ["Model", "ModelRegistry", "MolmoAct2"]