import sys
from types import ModuleType, SimpleNamespace

from open_badger.logging import ExperimentLogger


def test_experiment_logger_fans_metrics_out_to_tensorboard_and_wandb(
    monkeypatch,
    tmp_path,
):
    writer_events = []

    class FakeWriter:
        def __init__(self, *, log_dir):
            writer_events.append(("init", log_dir))

        def add_text(self, tag, value):
            writer_events.append(("text", tag, value))

        def add_scalar(self, tag, value, *, global_step):
            writer_events.append(("scalar", tag, value, global_step))

        def close(self):
            writer_events.append(("close",))

    wandb_events = []
    fake_run = SimpleNamespace(
        log=lambda metrics, *, step: wandb_events.append(("log", metrics, step)),
        finish=lambda: wandb_events.append(("finish",)),
    )
    fake_wandb = ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: (
        wandb_events.append(("init", kwargs)) or fake_run
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.tensorboard",
        SimpleNamespace(SummaryWriter=FakeWriter),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = ExperimentLogger(
        {
            "console": False,
            "tensorboard": {"enabled": True, "log_dir": str(tmp_path / "tb")},
            "wandb": {"enabled": True, "project": "test-project"},
        },
        {"training": {"max_steps": 2}},
        repository_root=tmp_path,
        default_run_name="test-run",
    )

    logger.log_metrics({"train/loss": 0.5}, step=1)
    logger.close()

    assert ("init", str(tmp_path / "tb")) in writer_events
    assert ("scalar", "train/loss", 0.5, 1) in writer_events
    assert ("close",) in writer_events
    assert wandb_events[0][0] == "init"
    assert ("log", {"train/loss": 0.5}, 1) in wandb_events
    assert ("finish",) in wandb_events


def test_experiment_logger_uses_timestamped_run_name_for_tensorboard(
    monkeypatch,
    tmp_path,
):
    writer_events = []

    class FakeWriter:
        def __init__(self, *, log_dir):
            writer_events.append(log_dir)

        def add_text(self, *_args):
            pass

        def add_scalar(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "torch.utils.tensorboard",
        SimpleNamespace(SummaryWriter=FakeWriter),
    )
    logger = ExperimentLogger(
        {"console": False, "tensorboard": {"enabled": True}},
        {},
        repository_root=tmp_path,
        default_run_name="model-dataset-run",
    )
    logger.close()

    assert writer_events == [
        str(tmp_path / "outputs/tensorboard/model-dataset-run")
    ]
