from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from open_badger.data.preparation import ensure_quantile_stats


class FakeRows:
    def __init__(self, values, selected=None):
        self.values = values
        self.selected = selected or list(values)

    def select_columns(self, names):
        return FakeRows(self.values, names)

    def with_format(self, _format):
        return self

    def iter(self, batch_size):
        for start in range(0, len(next(iter(self.values.values()))), batch_size):
            yield {
                name: np.asarray(self.values[name][start : start + batch_size])
                for name in self.selected
            }


class FakeLeRobotDataset:
    def __init__(self):
        self.features = {
            "observation.state": {},
            "action": {},
            "observation.images.top": {},
        }
        self.hf_dataset = FakeRows(
            {
                "observation.state": [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]],
                "action": [[1.0], [2.0], [3.0]],
                "observation.images.top": [[255], [0], [128]],
            }
        )
        self.meta = SimpleNamespace(
            root=Path("/tmp/lerobot-test"),
            stats={
                "observation.state": {"mean": [10.0, 20.0], "std": [1.0, 1.0]},
                "observation.images.top": {"mean": [0.5]},
            },
        )


def test_ensure_quantile_stats_computes_selected_raw_features(monkeypatch):
    import lerobot.datasets

    writes = []
    monkeypatch.setattr(
        lerobot.datasets,
        "write_stats",
        lambda stats, root: writes.append((stats, root)),
    )
    dataset = FakeLeRobotDataset()

    updated = ensure_quantile_stats(dataset, batch_size=2)

    assert updated == ("observation.state", "action")
    assert "q01" in dataset.meta.stats["observation.state"]
    assert "q99" in dataset.meta.stats["action"]
    assert dataset.meta.stats["observation.state"]["mean"] == [10.0, 20.0]
    assert dataset.meta.stats["observation.state"]["std"] == [1.0, 1.0]
    assert dataset.meta.stats["observation.images.top"] == {"mean": [0.5]}
    assert writes[0][1] == Path("/tmp/lerobot-test")


def test_ensure_quantile_stats_skips_complete_stats(monkeypatch):
    import lerobot.datasets

    monkeypatch.setattr(
        lerobot.datasets,
        "write_stats",
        lambda *_args: pytest.fail("Complete quantile stats must not be rewritten."),
    )
    dataset = FakeLeRobotDataset()
    dataset.meta.stats.update(
        {
            "observation.state": {"q01": [0.0], "q99": [1.0]},
            "action": {"q01": [0.0], "q99": [1.0]},
        }
    )

    assert ensure_quantile_stats(dataset) == ()


def test_ensure_quantile_stats_rejects_unknown_features():
    with pytest.raises(KeyError, match="missing requested features"):
        ensure_quantile_stats(FakeLeRobotDataset(), ("missing",))
