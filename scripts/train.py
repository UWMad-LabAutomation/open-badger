"""Run local OpenBadger training or a one-batch model smoke test."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from open_badger.data import LeRobotReader, ParquetManifest, SampleManifest
from open_badger.training.training import (
    build_model,
    build_optimizer,
    load_training_config,
    train_from_manifest,
)


def _manifest_directory(
    configured_directory: str | None,
    dataset_id: str,
    revision: str,
    *,
    repository_root: Path,
) -> Path:
    if configured_directory:
        return Path(configured_directory).expanduser().resolve()
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_id)
    safe_revision = re.sub(r"[^A-Za-z0-9._-]+", "_", revision)
    return repository_root / "outputs" / "manifests" / safe_id / safe_revision


def _load_or_build_manifest(
    reader: LeRobotReader,
    data_config: Mapping[str, Any],
    directory: Path,
) -> ParquetManifest:
    manifest_config = data_config.get("manifest", {}) or {}
    if directory.joinpath("manifest.json").is_file() and not manifest_config.get("rebuild", False):
        manifest = ParquetManifest.read(directory)
        metadata = reader.metadata()
        if (
            manifest.metadata.dataset_id != metadata.dataset_id
            or manifest.metadata.revision != metadata.revision
        ):
            raise ValueError(
                "Cached manifest dataset identity does not match the opened dataset. "
                "Set manifest.rebuild=true or choose another manifest directory."
            )
        if len(manifest):
            first_example = next(iter(manifest))
            if first_example.sample.dataset_index is None:
                raise ValueError(
                    "Cached manifest predates direct dataset indices. "
                    "Set manifest.rebuild=true to regenerate it."
                )
        return manifest

    sampling = data_config.get("sampling", {}) or {}
    return SampleManifest.build_parquet(
        reader,
        directory,
        action_horizon=int(sampling.get("action_horizon", 1)),
        action_spacing_seconds=sampling.get("action_spacing_seconds"),
        observation_history_seconds=float(sampling.get("observation_history_seconds", 0.0)),
        drop_incomplete_horizons=bool(sampling.get("drop_incomplete_horizons", True)),
        write_batch_size=int(manifest_config.get("write_batch_size", 4096)),
    )


def _run_vla_smoke_test(model: torch.nn.Module, batches: Any) -> None:
    model.train()
    try:
        batch = next(iter(batches))
    except StopIteration as exc:
        raise ValueError("The training manifest produced no batches.") from exc

    with torch.no_grad():
        result = model.training_step(batch)
    loss = result.get("loss") if isinstance(result, Mapping) else None
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise TypeError("VLA smoke test expected training_step() to return a scalar loss.")
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("VLA smoke-test loss is not finite.")
    metrics = result.get("metrics", {})
    print(f"VLA forward/loss smoke test passed: loss={loss.detach().float().item():.6f}")
    if isinstance(metrics, Mapping):
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value}")


def _run_vla_inference_smoke_test(model: torch.nn.Module, batches: Any) -> None:
    predict_from_batch = getattr(model, "predict_action_chunk_from_batch", None)
    if not callable(predict_from_batch):
        raise TypeError("VLA models must implement predict_action_chunk_from_batch().")
    model.eval()
    try:
        batch = next(iter(batches))
    except StopIteration as exc:
        raise ValueError("The VLA inference manifest produced no batches.") from exc

    with torch.inference_mode():
        actions = predict_from_batch(batch)
    if not torch.is_tensor(actions) or actions.ndim != 3:
        raise TypeError(
            "VLA inference smoke test expected an action tensor with shape "
            "[batch, horizon, action_dim]."
        )
    if actions.shape[0] != 1 or actions.shape[1] < 1 or actions.shape[2] < 1:
        raise ValueError(f"Unexpected predicted action chunk shape: {tuple(actions.shape)}.")
    if not bool(torch.isfinite(actions).all()):
        raise FloatingPointError("VLA produced non-finite actions.")
    print(
        "VLA inference smoke test passed: "
        f"action_shape={tuple(actions.shape)}, "
        f"range=[{actions.min().item():.4f}, {actions.max().item():.4f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/training/train.yaml",
        help="High-level training YAML config.",
    )
    parser.add_argument("--dataset-id", help="Override data.source.dataset_id.")
    parser.add_argument(
        "--backend",
        choices=("huggingface", "local"),
        help="Override data.source.backend.",
    )
    parser.add_argument("--data-root", type=Path, help="Override data.source.root.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        help="Override data.source.episodes with selected episode indices.",
    )
    parser.add_argument(
        "--image-keys",
        nargs="+",
        help="Override data.features.image_keys with selected camera features.",
    )
    parser.add_argument(
        "--normalize-gripper",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override data.embodiment.normalize_gripper.",
    )
    parser.add_argument("--revision", help="Override data.source.revision.")
    parser.add_argument("--manifest-dir", type=Path, help="Override data.manifest.directory.")
    parser.add_argument("--strategy", help="Override the configured fine-tuning strategy.")
    parser.add_argument("--batch-size", type=int, help="Override data.loader.batch_size.")
    parser.add_argument("--num-workers", type=int, help="Override data.loader.num_workers.")
    parser.add_argument("--max-steps", type=int, help="Override training.training.max_steps.")
    smoke_tests = parser.add_mutually_exclusive_group()
    smoke_tests.add_argument(
        "--vla-smoke-test",
        "--smoke-test",
        dest="vla_smoke_test",
        action="store_true",
        help="Run one VLA forward/loss batch without backward or optimizer updates.",
    )
    smoke_tests.add_argument(
        "--vla-inference-smoke-test",
        "--inference-smoke-test",
        dest="vla_inference_smoke_test",
        action="store_true",
        help="Predict one VLA action chunk from a real observation batch.",
    )
    args = parser.parse_args()

    model_config, data_config = load_training_config(args.config, strategy=args.strategy)
    source = dict(data_config.get("source", {}) or {})
    if args.backend is not None:
        source["backend"] = args.backend
    if args.data_root is not None:
        source["root"] = str(args.data_root)
    if args.episodes is not None:
        source["episodes"] = args.episodes
    data_config["source"] = source
    if args.image_keys is not None:
        data_config.setdefault("features", {})["image_keys"] = args.image_keys
    if args.normalize_gripper is not None:
        embodiment_config = dict(data_config.get("embodiment", {}) or {})
        embodiment_config["normalize_gripper"] = args.normalize_gripper
        data_config["embodiment"] = embodiment_config

    backend = str(source.get("backend", "huggingface")).lower()
    if backend not in {"huggingface", "local"}:
        raise ValueError(f"Data source backend {backend!r} is not implemented.")
    dataset_id = args.dataset_id or source.get("dataset_id")
    if not dataset_id:
        raise ValueError(
            "Set data.source.dataset_id in the data config or pass --dataset-id."
        )
    revision = args.revision or source.get("revision")
    reader = LeRobotReader(
        str(dataset_id),
        root=source.get("root"),
        episodes=source.get("episodes"),
        revision=revision,
        video_backend=source.get("video_backend"),
        download_videos=bool(source.get("download_videos", True)),
    )
    metadata = reader.metadata()

    manifest_config = dict(data_config.get("manifest", {}) or {})
    manifest_directory = args.manifest_dir or _manifest_directory(
        manifest_config.get("directory"),
        metadata.dataset_id,
        metadata.revision,
        repository_root=Path(__file__).resolve().parents[1],
    )
    manifest_config["directory"] = str(manifest_directory)
    data_config["manifest"] = manifest_config
    manifest = _load_or_build_manifest(reader, data_config, manifest_directory)
    if len(manifest) == 0:
        raise ValueError(
            "The manifest contains no valid training examples. Check action horizon, "
            "episode lengths, and action spacing."
        )

    model_config["data"] = data_config
    model = build_model(
        model_config,
        dataset_metadata=metadata,
        native_metadata=reader.dataset.meta,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The model registry must return a torch.nn.Module.")

    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive.")
        data_config.setdefault("loader", {})["batch_size"] = args.batch_size
    elif args.vla_smoke_test or args.vla_inference_smoke_test:
        data_config.setdefault("loader", {})["batch_size"] = 1
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative.")
        loader_config = data_config.setdefault("loader", {})
        loader_config["num_workers"] = args.num_workers
        if args.num_workers == 0:
            loader_config["persistent_workers"] = False

    print(
        f"Dataset: {metadata.dataset_id} @ {metadata.revision}; "
        f"{metadata.total_frames} frames, {metadata.total_episodes} episodes"
    )
    print(f"Manifest: {manifest_directory} ({len(manifest)} examples)")

    if args.vla_smoke_test:
        from open_badger.training.training import build_training_dataloader

        _run_vla_smoke_test(
            model,
            build_training_dataloader(reader, manifest, data_config),
        )
        return
    if args.vla_inference_smoke_test:
        from open_badger.training.training import build_training_dataloader

        _run_vla_inference_smoke_test(
            model,
            build_training_dataloader(
                reader,
                manifest,
                data_config,
                include_action=False,
            ),
        )
        return

    optimizer = build_optimizer(model, model_config)
    model_training = model_config.get("training", {}) or {}
    finetuning = model_config.get("finetuning", {}) or {}
    optimizer_config = finetuning.get("optimizer", {}) or {}
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(model_training.get("max_steps", 1))
    )
    result = train_from_manifest(
        model,
        reader,
        manifest,
        optimizer,
        data_config=data_config,
        max_steps=max_steps,
        gradient_accumulation_steps=int(
            model_training.get("gradient_accumulation_steps", 1)
        ),
        max_grad_norm=optimizer_config.get("grad_clip_norm"),
        device=model_config.get("runtime", {}).get("device"),
    )
    print(f"Training complete: steps={result.steps}, loss={result.loss:.6f}")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
