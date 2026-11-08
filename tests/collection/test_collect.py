"""Collection command contracts; no robot or training dependencies required."""
import json
from pathlib import Path
import subprocess

import pytest

from scripts import collect


def test_record_preserves_task_as_one_argument_and_uses_absolute_paths(tmp_path):
    station = tmp_path / "station with spaces.yaml"
    station.write_text("robot: {}\n")
    task = 'move cup; $(echo should-not-execute)'
    args = collect.parser().parse_args([
        "record", "--station", str(station), "--task", task, "--seconds", "2",
    ])
    command = collect.collection_command(args, tmp_path)
    assert command[0] == str(tmp_path / ".venv/bin/yam-abc-teleop")
    assert command[command.index("--record") + 1] == task
    assert command[command.index("--station") + 1] == str(station)


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected(tmp_path, duration):
    station = tmp_path / "station.yaml"
    station.touch()
    args = collect.parser().parse_args([
        "record", "--station", str(station), "--task", "test", "--seconds", duration,
    ])
    with pytest.raises(ValueError, match="duration"):
        collect.collection_command(args, tmp_path)


def test_conversion_is_lerobot_and_refuses_existing_output(tmp_path):
    source = tmp_path / "episodes"
    source.mkdir()
    out = tmp_path / "dataset"
    args = collect.parser().parse_args([
        "convert", str(source), "--repo-id", "local/test", "--out", str(out),
    ])
    command = collect.collection_command(args, tmp_path)
    assert command[1:] == [str(source), "--to", "lerobot", "--repo-id", "local/test", "--out", str(out)]
    out.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        collect.collection_command(args, tmp_path)


def test_dry_run_never_executes(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run launched a subprocess")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    collect.main(["--checkout", str(tmp_path / "missing"), "--dry-run", "cameras"])
    collect.main(["--checkout", str(tmp_path / "missing"), "--dry-run", "setup"])


def test_revision_mismatch_rejected(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "wrong\n")
    pin = json.loads(collect.LOCK.read_text())
    with pytest.raises(ValueError, match="Revision mismatch"):
        collect.verify_checkout(tmp_path, pin)


def test_uninitialized_submodule_rejected(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    pin = json.loads(collect.LOCK.read_text())
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw:
                        pin["commit"] if "rev-parse" in cmd else "")
    with pytest.raises(ValueError, match="Missing checkout"):
        collect.verify_checkout(tmp_path, pin)


def test_tracked_changes_rejected(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    pin = json.loads(collect.LOCK.read_text())
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw:
                        pin["commit"] if "rev-parse" in cmd else " M runtime.py")
    with pytest.raises(ValueError, match="Tracked changes"):
        collect.verify_checkout(tmp_path, pin)


def test_station_overrides_preserve_original_and_handle_camera_roster(tmp_path):
    station = tmp_path / "station.yaml"
    original = "control_hz: 30\ncameras: []\nrobot:\n  robots:\n    - channel: can0\nsave_root: old\n"
    station.write_text(original)
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text("cameras:\n  - role: top\n    fps: 30\n    serial: '1234'\n")
    args = collect.parser().parse_args([
        "record", "--station", str(station), "--task", "pick cup",
        "--cameras", str(cameras), "--control-hz", "20", "--save-root", str(tmp_path / "raw"),
        "--set", "robot.robots.0.channel=can2", "--set", "cameras.0.fps=15",
    ])
    config = collect.resolve_station(args)
    assert config["control_hz"] == 20
    assert config["robot"]["robots"][0]["channel"] == "can2"
    assert config["cameras"][0]["fps"] == 15
    assert config["cameras"][0]["serial"] == "1234"
    assert config["task_name"] == "pick cup"
    assert config["save_root"] == str(tmp_path / "raw")
    assert station.read_text() == original


@pytest.mark.parametrize("override", ["typo=20", "robot.robots.5.channel=can3", "control_hz=0"])
def test_invalid_station_override_rejected(override):
    args = collect.parser().parse_args([
        "gui", "--station", str(collect.ROOT / "configs/collection/yam_single.yaml"),
        "--set", override,
    ])
    with pytest.raises(ValueError):
        collect.resolve_station(args)


def test_record_duration_alias():
    args = collect.parser().parse_args([
        "record", "--station", "station.yaml", "--task", "test", "--episode-seconds", "45",
    ])
    assert args.seconds == 45


def test_effective_station_is_passed_to_process(monkeypatch, tmp_path):
    import yaml
    station = collect.ROOT / "configs/collection/yam_single.yaml"
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    args = collect.parser().parse_args([
        "gui", "--station", str(station), "--task", "cup", "--control-hz", "25",
    ])
    command = collect.collection_command(args, tmp_path)
    collect.apply_station_overrides(args, command)
    effective = Path(command[command.index("--station") + 1])
    assert effective != station
    assert yaml.safe_load(effective.read_text())["control_hz"] == 25


def test_record_preview_does_not_write_files_or_launch(monkeypatch, tmp_path, capsys):
    station = collect.ROOT / "configs/collection/yam_single.yaml"
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("launched"))
    collect.main([
        "--dry-run", "record", "--station", str(station), "--task", "cup", "--control-hz", "20",
    ])
    assert "control_hz: 20.0" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()
