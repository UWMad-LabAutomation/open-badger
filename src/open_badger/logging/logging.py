"""Optional console, TensorBoard, and Weights & Biases training metrics."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml


class MetricLogger(Protocol):
    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None: ...

    def close(self) -> None: ...


class ExperimentLogger:
    """Fan out scalar training metrics to configured experiment backends."""

    def __init__(
        self,
        config: Mapping[str, Any],
        run_config: Mapping[str, Any],
        *,
        repository_root: Path,
        default_run_name: str,
    ) -> None:
        self.console = bool(config.get("console", True))
        tensorboard_config = config.get("tensorboard", {}) or {}
        wandb_config = config.get("wandb", {}) or {}
        self.writer = None
        self.wandb_run = None
        with ExitStack() as resources:
            if tensorboard_config.get("enabled", False):
                from torch.utils.tensorboard import SummaryWriter

                log_dir = Path(
                    tensorboard_config.get("log_dir")
                    or Path("outputs/tensorboard") / default_run_name
                ).expanduser()
                if not log_dir.is_absolute():
                    log_dir = repository_root / log_dir
                self.writer = SummaryWriter(log_dir=str(log_dir))
                resources.callback(self.writer.close)
                self.writer.add_text(
                    "run/config",
                    yaml.safe_dump(dict(run_config), sort_keys=True),
                )

            if wandb_config.get("enabled", False):
                import wandb

                self.wandb_run = wandb.init(
                    project=str(wandb_config.get("project", "open-badger")),
                    entity=wandb_config.get("entity"),
                    name=wandb_config.get("run_name") or default_run_name,
                    mode=str(wandb_config.get("mode", "online")),
                    config=dict(run_config),
                )
                resources.callback(self.wandb_run.finish)
            self._resources = resources.pop_all()

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        if self.console:
            values = " ".join(f"{name}={value:.6g}" for name, value in metrics.items())
            print(f"step={step} {values}", flush=True)
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, global_step=step)
        if self.wandb_run is not None:
            self.wandb_run.log(dict(metrics), step=step)

    def close(self) -> None:
        self._resources.close()


def default_run_name(model_name: str, dataset_id: str) -> str:
    """Create a timestamped, filesystem-safe run name."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe_model = _safe_name(model_name)
    safe_dataset = _safe_name(dataset_id)
    return f"{safe_model}-{safe_dataset}-{timestamp}"


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_." else "-" for character in value)
    return safe.strip("-.") or "run"
