"""Model-agnostic PyTorch training orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import warnings
import torch
from torch import nn
import yaml
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from open_badger.data.dataset import ManifestTrainingDataset
from open_badger.data.readers.base import DatasetReader
from open_badger.data.schema import DatasetMetadata, TrainingExampleRef
from open_badger.models.registry import Model, ModelRegistry
from open_badger.logging import MetricLogger


@dataclass(frozen=True)
class TrainingResult:
    """Summary of the most recent optimizer step."""

    steps: int
    loss: float
    metrics: dict[str, float]


def load_training_config(
    config_path: str | Path,
    *,
    strategy: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and compose the job, model, and data YAML configurations."""
    job_path = Path(config_path).expanduser().resolve()
    job_config = _read_yaml_mapping(job_path)
    model_section = job_config.get("model")
    data_section = job_config.get("data")
    if not isinstance(model_section, Mapping) or not model_section.get("config"):
        raise ValueError("Training config must define model.config.")
    if not isinstance(data_section, Mapping) or not data_section.get("config"):
        raise ValueError("Training config must define data.config.")

    model_path = (job_path.parent / str(model_section["config"])).resolve()
    data_path = (job_path.parent / str(data_section["config"])).resolve()
    model_config = _read_yaml_mapping(model_path)
    data_config = _read_yaml_mapping(data_path)

    finetuning = dict(model_config.get("finetuning", {}) or {})
    selected_strategy = strategy or str(model_section.get("strategy", "lora"))
    strategies = finetuning.get("strategies", {}) or {}
    if selected_strategy not in strategies:
        available = ", ".join(sorted(str(strat) for strat in strategies)) or "none"
        raise ValueError(
            f"Unknown fine-tuning strategy {selected_strategy!r}; available strategies: {available}."
        )
    finetuning["selected_strategy"] = selected_strategy
    model_config["finetuning"] = finetuning
    job_training = job_config.get("training", {}) or {}
    if not isinstance(job_training, Mapping):
        raise TypeError("Training config's training section must be a mapping.")
    model_training = dict(model_config.get("training", {}) or {})
    model_logging = dict(model_training.get("logging", {}) or {})
    job_logging = job_training.get("logging", {}) or {}
    if not isinstance(job_logging, Mapping):
        raise TypeError("Training config's training.logging section must be a mapping.")
    for key, value in job_logging.items():
        if key in {"tensorboard", "wandb"} and isinstance(value, Mapping):
            backend_config = dict(model_logging.get(key, {}) or {})
            backend_config.update(value)
            model_logging[key] = backend_config
        else:
            model_logging[key] = value
    if job_logging:
        model_training["logging"] = model_logging
        model_config["training"] = model_training
    model_config["data"] = data_config
    return model_config, data_config


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file does not exist: {path}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration file must contain a YAML mapping: {path}")
    return loaded


def build_model(
    config: Mapping[str, Any],
    *,
    dataset_metadata: DatasetMetadata | None = None,
    native_metadata: Any | None = None,
) -> Model:
    """Construct a registered model, inferring feature dimensions when possible."""
    model_config = dict(config)
    if dataset_metadata is not None:
        data_config = dict(model_config.get("data", {}) or {})
        feature_config = dict(data_config.get("features", {}) or {})
        if not feature_config.get("image_keys"):
            feature_config["image_keys"] = list(dataset_metadata.camera_keys)
        action_feature = dataset_metadata.action_feature
        if feature_config.get("action_dim") is None and action_feature and action_feature.shape:
            feature_config["action_dim"] = action_feature.shape[-1]
        state_feature = dataset_metadata.state_feature
        if feature_config.get("state_dim") is None and state_feature and state_feature.shape:
            feature_config["state_dim"] = state_feature.shape[-1]
        data_config["features"] = feature_config
        model_config["data"] = data_config

    model = ModelRegistry.build_from_config(model_config)
    if dataset_metadata is not None:
        model.configure_dataset(dataset_metadata, native_metadata=native_metadata)
    return model


def build_optimizer(model: nn.Module, model_config: Mapping[str, Any]) -> Optimizer:
    """Build the configured AdamW optimizer from model-provided parameter groups."""
    finetuning = model_config.get("finetuning", {}) or {}
    optimizer_config = finetuning.get("optimizer", {}) or {}
    get_parameter_groups = getattr(model, "optimizer_parameter_groups", None)
    if callable(get_parameter_groups):
        parameter_groups = get_parameter_groups()
    else:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        parameter_groups = [{"params": trainable_parameters}]
    if not parameter_groups or not any(group.get("params") for group in parameter_groups):
        raise ValueError("The model has no trainable parameters.")
    return torch.optim.AdamW(
        parameter_groups,
        lr=float(optimizer_config.get("learning_rate", 1.0e-4)),
        betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
        eps=float(optimizer_config.get("eps", 1.0e-8)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )


def build_training_dataloader(
    reader: DatasetReader,
    manifest: Iterable[TrainingExampleRef],
    data_config: Mapping[str, Any],
    *,
    include_action: bool = True,
) -> DataLoader:
    """Compose manifest resolution and PyTorch batching from data config."""
    sampling_config = data_config.get("sampling", {}) or {}
    loader_config = data_config.get("loader", {}) or {}
    num_workers = int(loader_config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("loader.num_workers must be non-negative.")
    try:
        manifest_size = len(manifest)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("Training manifests must provide a length for worker allocation.") from exc
    if manifest_size < 1:
        raise ValueError("Cannot build a training loader from an empty manifest.")
    manifest_shards = getattr(manifest, "num_shards", None)
    if manifest_shards is not None:
        available_shards = int(manifest_shards)
    elif hasattr(manifest, "examples") or isinstance(manifest, (list, tuple)):
        available_shards = manifest_size
    else:
        available_shards = 1
    if available_shards < 1:
        raise ValueError("The training manifest has no readable Parquet row groups.")
    worker_limit = min(manifest_size, available_shards)
    if num_workers > worker_limit:
        warnings.warn(
            f"Reducing DataLoader workers from {num_workers} to {worker_limit}: "
            "the manifest has fewer independent work shards.",
            stacklevel=2,
        )
        num_workers = worker_limit
    batch_size = int(loader_config.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("loader.batch_size must be positive.")
    persistent_workers = bool(loader_config.get("persistent_workers", False))
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires loader.num_workers greater than zero.")

    dataset = ManifestTrainingDataset(
        reader,
        manifest,
        shuffle=bool(sampling_config.get("shuffle", False)),
        shuffle_buffer_size=int(loader_config.get("shuffle_buffer_size", 1024)),
        seed=int(sampling_config.get("seed", 0)),
        include_action=include_action,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", False)),
        persistent_workers=persistent_workers,
    )


def train(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    *,
    max_steps: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    device: str | torch.device | None = None,
    log_every_steps: int = 10,
    metric_logger: MetricLogger | None = None,
) -> TrainingResult:
    """Run optimizer steps over an iterable of already-collated raw batches."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided.")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be positive.")

    target_device = (
        torch.device(device)
        if device is not None
        else getattr(model, "device", None)
    )
    model.train()
    batch_iterator = iter(batches)
    optimizer.zero_grad(set_to_none=True)
    last_loss = 0.0
    last_metrics: dict[str, float] = {}

    for step in range(1, max_steps + 1):
        step_started = perf_counter()
        accumulated_loss = 0.0
        accumulated_metrics: dict[str, float] = {}
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(batches)
                try:
                    batch = next(batch_iterator)
                except StopIteration as exc:
                    raise ValueError("Cannot train from an empty batch iterable.") from exc

            if target_device is not None and not getattr(
                model, "preprocessor_handles_device", False
            ):
                batch = _move_to_device(batch, target_device)
            result = model.training_step(batch)
            if not isinstance(result, Mapping) or "loss" not in result:
                raise TypeError("model.training_step() must return a mapping containing 'loss'.")

            loss = result["loss"]
            if not torch.is_tensor(loss) or loss.numel() != 1:
                raise TypeError("model.training_step()['loss'] must be a scalar tensor.")
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("Training loss is not finite.")

            (loss / gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach().item()) / gradient_accumulation_steps
            metrics = result.get("metrics", {})
            if isinstance(metrics, Mapping):
                for name, value in metrics.items():
                    if torch.is_tensor(value):
                        value = value.detach().item()
                    if isinstance(value, (int, float)):
                        accumulated_metrics[str(name)] = (
                            accumulated_metrics.get(str(name), 0.0)
                            + float(value) / gradient_accumulation_steps
                        )

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = accumulated_loss
        last_metrics = accumulated_metrics
        if metric_logger is not None and (
            step % log_every_steps == 0 or step == max_steps
        ):
            logged_metrics = {
                "train/loss": last_loss,
                "train/step_time_seconds": perf_counter() - step_started,
                **{
                    f"train/{name}": value
                    for name, value in last_metrics.items()
                },
            }
            metric_logger.log_metrics(logged_metrics, step=step)

    return TrainingResult(steps=max_steps, loss=last_loss, metrics=last_metrics)


def train_from_manifest(
    model: nn.Module,
    reader: DatasetReader,
    manifest: Iterable[TrainingExampleRef],
    optimizer: Optimizer,
    *,
    data_config: Mapping[str, Any],
    max_steps: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    device: str | torch.device | None = None,
    log_every_steps: int = 10,
    metric_logger: MetricLogger | None = None,
) -> TrainingResult:
    """Resolve manifest examples into batches and train on them."""
    if isinstance(model, Model):
        reader_dataset = getattr(reader, "dataset", None)
        model.configure_dataset(
            reader.metadata(),
            native_metadata=getattr(reader_dataset, "meta", None),
        )
    batches = build_training_dataloader(reader, manifest, data_config)
    return train(
        model,
        batches,
        optimizer,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        device=device,
        log_every_steps=log_every_steps,
        metric_logger=metric_logger,
    )


"""Move batch tensors from system memory to GPU memory"""
def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value
