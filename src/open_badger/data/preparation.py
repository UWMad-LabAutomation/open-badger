"""LeRobot dataset preparation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets import write_stats
from lerobot.datasets.compute_stats import RunningQuantileStats


def ensure_quantile_stats(
    dataset: Any,
    feature_names: Sequence[str] = ("observation.state", "action"),
    *,
    batch_size: int = 8192,
) -> tuple[str, ...]:
    """Compute and persist missing quantile statistics from raw tabular features."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    stats = dataset.meta.stats or {}
    missing = [
        name
        for name in feature_names
        if not isinstance(stats.get(name), dict)
        or stats[name].get("q01") is None
        or stats[name].get("q99") is None
    ]
    if not missing:
        return ()

    available_features = set(dataset.features)
    unknown = set(missing).difference(available_features)
    if unknown:
        raise KeyError(f"Dataset is missing requested features: {sorted(unknown)}.")

    accumulators = {name: RunningQuantileStats() for name in missing}
    rows = dataset.hf_dataset.select_columns(missing).with_format("numpy")
    for batch in rows.iter(batch_size=batch_size):
        for name, accumulator in accumulators.items():
            values = np.asarray(batch[name])
            if values.dtype.kind not in "biuf":
                raise TypeError(f"Feature {name!r} must contain numeric values.")
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            accumulator.update(values)

    updated_stats = deepcopy(stats)
    for name, accumulator in accumulators.items():
        computed = accumulator.get_statistics()
        feature_stats = updated_stats.setdefault(name, {})
        feature_stats.update(
            {
                key: value
                for key, value in computed.items()
                if key.startswith("q") and key[1:].isdigit()
            }
        )

    metadata_root = Path(dataset.meta.root)
    write_stats(updated_stats, metadata_root)
    dataset.meta.stats = updated_stats
    return tuple(missing)
