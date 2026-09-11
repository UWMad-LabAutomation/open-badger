"""Model registry and shared abstractions for policy implementations.

The registry is intentionally narrow: it resolves a logical model identifier such as
"molmoact2" to a concrete implementation class, while the training/inference layers
remain responsible for backend-specific orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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
        self.model_name = self.config.get("name") or self.config.get("model_name") or self.__class__.__name__
        self.device = self._resolve_device(self.config.get("device"))
        self.dtype = self._resolve_dtype(self.config.get("dtype"))
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

    def preprocess_batch(self, batch: Any) -> Any:
        """Normalize a Lerobot2-like batch into the backend's input contract."""
        return batch

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

        name = config.get("name") or config.get("model") or config.get("model_name")
        if name is None:
            raise KeyError("Model config must define a 'name', 'model', or 'model_name'.")

        return cls.build(str(name), config=config)


__all__ = ["Model", "ModelRegistry"]
