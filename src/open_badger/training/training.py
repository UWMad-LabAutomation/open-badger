"""Model-agnostic PyTorch training orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from open_badger.data.dataset import ManifestTrainingDataset
from open_badger.data.readers.base import DatasetReader
from open_badger.data.schema import DatasetMetadata, TrainingExampleRef
from open_badger.models.registry import Model, ModelRegistry


@dataclass(frozen=True)
class TrainingResult:
    """Summary of the most recent optimizer step."""

    steps: int
    loss: float
    metrics: dict[str, float]


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


def build_training_dataloader(
    reader: DatasetReader,
    manifest: Iterable[TrainingExampleRef],
    data_config: Mapping[str, Any],
) -> DataLoader:
    """Compose manifest resolution and PyTorch batching from data config."""
    sampling_config = data_config.get("sampling", {}) or {}
    loader_config = data_config.get("loader", {}) or {}
    num_workers = int(loader_config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("loader.num_workers must be non-negative.")
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
) -> TrainingResult:
    """Run optimizer steps over an iterable of already-collated raw batches."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided.")

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

    for _ in range(max_steps):
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
