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


class MolmoAct2(Model):
    """MolmoAct2 vision-language-action policy (allenai/MolmoAct2).

    Wraps the official Hugging Face checkpoint: a Molmo2-ER vision-language
    backbone connected to a flow-matching continuous action expert via
    per-layer KV conditioning. Outputs are continuous, absolute joint/encoder
    position targets.

    `forward()` wraps the checkpoint's documented `predict_action()` API,
    which is inference-only (`@torch.no_grad()` internally) and is the
    correct, robot-safe way to produce actions in the checkpoint's native
    format. It is not differentiable. The training objective in
    `compute_loss()` will need a separate, gradient-carrying path through the
    flow-matching action expert and is deferred until that work starts.
    """

    model_name = "molmoact2"

    def build_network(self) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "MolmoAct2 requires the 'transformers' package. Install the "
                "project's 'model' extra to use this checkpoint."
            ) from exc

        checkpoint_cfg = self.config.get("checkpoint", {}) or {}
        repo_id = checkpoint_cfg.get("repo_id", "allenai/MolmoAct2")
        revision = checkpoint_cfg.get("revision")
        cache_dir = checkpoint_cfg.get("cache_dir")
        local_files_only = bool(checkpoint_cfg.get("local_files_only", False))
        attn_implementation = (self.config.get("optimization", {}) or {}).get("attention")

        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
        )
        self.hf_model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )

        inference_cfg = self.config.get("inference", {}) or {}
        self.norm_tag = inference_cfg.get("norm_tag")
        self.action_mode = inference_cfg.get("action_mode", "continuous")
        self.image_keys = list(inference_cfg.get("image_keys") or [])
        self.n_action_steps = inference_cfg.get("n_action_steps")
        self.num_flow_matching_steps = inference_cfg.get("flow_matching_steps")
        self.action_dim = int(inference_cfg.get("action_dim", 32))

    def preprocess_batch(self, batch: Any) -> Any:
        """Convert a Lerobot2-like batch into per-example MolmoAct2 inputs.

        Expects `batch` to provide `task` (instruction strings),
        `observation.state` (robot state vectors), and one
        `observation.images.*` entry per key configured in
        `inference.image_keys` (order matters; it must match the checkpoint's
        expected camera order). Returns a list of per-example dicts with
        `images`, `task`, and `state`, ready for `forward()`.
        """
        if not self.image_keys:
            raise ValueError(
                "MolmoAct2 requires `inference.image_keys` to be configured with "
                "the ordered observation image keys for this checkpoint."
            )

        tasks = batch["task"]
        states = batch["observation.state"]
        batch_size = len(tasks)

        examples = []
        for idx in range(batch_size):
            images = [batch[image_key][idx] for image_key in self.image_keys]
            state = states[idx]
            if torch.is_tensor(state):
                state = state.detach().cpu().numpy()
            examples.append({"images": images, "task": tasks[idx], "state": state})
        return examples

    def forward(self, inputs: Any, **kwargs: Any) -> Any:
        """Run MolmoAct2's continuous flow-matching action head.

        `inputs` is the list of per-example dicts produced by
        `preprocess_batch()`. Returns a `(batch, n_action_steps, action_dim)`
        tensor of absolute joint/encoder position targets, already
        unnormalized to the robot's native units by the checkpoint's
        `norm_stats.json`.
        """
        if self.norm_tag is None:
            raise ValueError(
                "MolmoAct2 requires `inference.norm_tag` to be set in the model "
                "config (selects normalization stats from the checkpoint's "
                "norm_stats.json)."
            )

        action_chunks = []
        for example in inputs:
            output = self.hf_model.predict_action(
                processor=self.processor,
                images=example["images"],
                task=example["task"],
                state=example["state"],
                norm_tag=self.norm_tag,
                inference_action_mode=self.action_mode,
                num_steps=self.num_flow_matching_steps,
                n_action_steps=self.n_action_steps,
                return_dict=True,
            )
            action_chunks.append(output.actions[..., : self.action_dim])

        return torch.cat(action_chunks, dim=0)

    def compute_loss(self, batch: Any, outputs: Any) -> torch.Tensor:
        raise NotImplementedError(
            "MolmoAct2 compute_loss() is intentionally left for the flow-matching "
            "training objective, which needs a separate gradient-carrying path "
            "through the action expert."
        )


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


ModelRegistry.register("molmoact2")(MolmoAct2)


__all__ = ["Model", "ModelRegistry", "MolmoAct2"]
