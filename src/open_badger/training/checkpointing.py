"""Local, resumable checkpoints for model training."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.optim import Optimizer

_CHECKPOINT_NAME = re.compile(r"checkpoint-\d{8}(?:-[0-9a-f]{8})?$")


class TrainingCheckpointManager:
    """Persist strategy-specific safe-tensor weights and optimizer/RNG state."""

    def __init__(
        self,
        directory: str | Path,
        *,
        keep_last_n: int,
        identity: Mapping[str, Any],
    ) -> None:
        if keep_last_n < 1:
            raise ValueError("keep_last_n must be positive.")
        self.directory = Path(directory).expanduser().resolve()
        self.keep_last_n = keep_last_n
        self.identity = dict(identity)

    def save(self, model: nn.Module, optimizer: Optimizer, *, step: int) -> Path:
        if step < 1:
            raise ValueError("Checkpoint step must be positive.")
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"checkpoint-{step:08d}"
        checkpoint_path = self.directory / checkpoint_name
        if checkpoint_path.exists():
            suffix = f"{time.time_ns() & 0xFFFFFFFF:08x}"
            checkpoint_name = f"{checkpoint_name}-{suffix}"
            checkpoint_path = self.directory / checkpoint_name
            if checkpoint_path.exists():
                raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

        artifact_method = getattr(model, "training_checkpoint_artifact", None)
        if callable(artifact_method):
            artifact_name, state = artifact_method()
        else:
            artifact_name, state = "model.safetensors", dict(model.state_dict())
        if Path(artifact_name).name != artifact_name or not artifact_name.endswith(".safetensors"):
            raise ValueError(f"Invalid safe-tensor checkpoint filename: {artifact_name!r}.")
        if not state or not all(torch.is_tensor(value) for value in state.values()):
            raise TypeError("Checkpoint model state must contain only tensors.")

        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{checkpoint_name}-", dir=self.directory)
        )
        # dettach from vram and store tensors in system mem before safely writing to cold storage
        try:
            safe_state = {
                name: tensor.detach().to(device="cpu").contiguous().clone()
                for name, tensor in state.items()
            }
            save_file(safe_state, str(temporary_path / artifact_name))
            torch.save(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    ),
                },
                temporary_path / "training_state.pt",
            )
            (temporary_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "step": step,
                        "artifact": artifact_name,
                        "identity": self.identity,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(checkpoint_path)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

        latest_temp = self.directory / "latest.json.tmp"
        latest_temp.write_text(
            json.dumps({"checkpoint": checkpoint_name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        latest_temp.replace(self.directory / "latest.json")
        self._prune_checkpoints()
        return checkpoint_path

    def load(
        self,
        checkpoint: str | Path,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> int:
        checkpoint_path = self._resolve_checkpoint(checkpoint)
        metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported training checkpoint format.")
        self._validate_identity(metadata.get("identity", {}))

        artifact_name = str(metadata["artifact"])
        artifact_path = checkpoint_path / artifact_name
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model checkpoint artifact is missing: {artifact_path}")
        model_state = load_file(str(artifact_path), device="cpu")
        load_artifact = getattr(model, "load_training_checkpoint_artifact", None)
        if callable(load_artifact):
            load_artifact(artifact_name, model_state)
        else:
            if artifact_name != "model.safetensors":
                raise ValueError(
                    f"Model does not support checkpoint artifact {artifact_name!r}."
                )
            model.load_state_dict(model_state, strict=True)

        training_state = torch.load(
            checkpoint_path / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        torch.set_rng_state(training_state["cpu_rng_state"])
        cuda_rng_states = training_state.get("cuda_rng_states", [])
        if cuda_rng_states:
            if not torch.cuda.is_available():
                warnings.warn(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable; "
                    "continuing with the current device RNG.",
                    stacklevel=2,
                )
            elif len(cuda_rng_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_rng_states)
            else:
                warnings.warn(
                    "CUDA device count differs from the checkpoint; restoring RNG state "
                    "for the current device only.",
                    stacklevel=2,
                )
                torch.cuda.set_rng_state(
                    cuda_rng_states[0], device=torch.cuda.current_device()
                )

        step = metadata.get("step")
        if not isinstance(step, int) or step < 1:
            raise ValueError("Checkpoint metadata has an invalid training update count.")
        return step

    def _resolve_checkpoint(self, checkpoint: str | Path) -> Path:
        if str(checkpoint) == "latest":
            latest_file = self.directory / "latest.json"
            if not latest_file.is_file():
                raise FileNotFoundError(f"No latest checkpoint exists in {self.directory}.")
            checkpoint_name = json.loads(latest_file.read_text(encoding="utf-8")).get("checkpoint")
            if not isinstance(checkpoint_name, str) or not _CHECKPOINT_NAME.fullmatch(checkpoint_name):
                raise ValueError(f"Invalid latest checkpoint pointer in {latest_file}.")
            checkpoint_path = self.directory / checkpoint_name
        else:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Training checkpoint directory does not exist: {checkpoint_path}")
        return checkpoint_path

    def _validate_identity(self, identity: Any) -> None:
        if not isinstance(identity, Mapping):
            raise TypeError("Checkpoint identity must be a mapping.")
        mismatches = {
            key: (identity.get(key), expected)
            for key, expected in self.identity.items()
            if identity.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Checkpoint does not match this run configuration: {mismatches}.")

    def _prune_checkpoints(self) -> None:
        checkpoints = sorted(
            (
                path
                for path in self.directory.iterdir()
                if path.is_dir() and _CHECKPOINT_NAME.fullmatch(path.name)
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for old_checkpoint in checkpoints[self.keep_last_n :]:
            shutil.rmtree(old_checkpoint)
