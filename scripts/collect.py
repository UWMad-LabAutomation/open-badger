"""Launch the pinned YAM collection environment without importing training dependencies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "third_party/yam-abc-reproduce.lock.json"
DEFAULT_CHECKOUT = ROOT / "third_party/yam-abc-reproduce"


def run(command: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print(f"[{cwd}] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def verify_checkout(checkout: Path, pin: dict) -> None:
    for path, expected in ((checkout, pin["commit"]),
                           (checkout / "third_party/i2rt", pin["i2rt_commit"])):
        # An empty submodule directory otherwise resolves to its parent's Git HEAD.
        if not (path / ".git").exists():
            raise ValueError(f"Missing checkout at {path}; run setup first.")
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"Revision mismatch at {path}: expected {expected}, got {actual}")
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True
        ).strip()
        if dirty:
            raise ValueError(f"Tracked changes at {path}; use external station configs to preserve the pin.")


def setup(checkout: Path, pin: dict, *, dry_run: bool) -> None:
    if not checkout.exists():
        run(["git", "clone", "--no-checkout", "--single-branch", "--branch",
             pin["branch"], pin["repository"], str(checkout)], cwd=ROOT, dry_run=dry_run)
        run(["git", "checkout", "--detach", pin["commit"]], cwd=checkout, dry_run=dry_run)
    else:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        if actual != pin["commit"]:
            raise ValueError("Existing checkout has a different revision; refusing to reset it.")
    # Override the upstream SSH transport for this command only; preserve its gitlink pin.
    run(["git", "-c", "submodule.third_party/i2rt.url=https://github.com/i2rt-robotics/i2rt.git",
         "submodule", "update", "--init", "third_party/i2rt"], cwd=checkout, dry_run=dry_run)
    if not dry_run:
        verify_checkout(checkout, pin)
    run(["uv", "sync", "--frozen", "--python", "3.12", "--no-default-groups",
         "--extra", "camera", "--extra", "gui", "--extra", "lerobot"],
        cwd=checkout, dry_run=dry_run)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Fetch pinned sources and install their isolated uv environment.")
    commands.add_parser("cameras", help="Discover camera serials.")
    for name in ("record", "gui"):
        child = commands.add_parser(name)
        child.add_argument("--station", type=Path, required=True)
        child.add_argument("--control-hz", type=float, help="Control/recording loop rate (independent of camera FPS).")
        child.add_argument("--save-root", type=Path, help="Directory for raw episodes.")
        child.add_argument("--cameras", type=Path, help="Camera roster YAML; replaces station cameras.")
        child.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                           help="Override an existing station field using YAML values, e.g. robot.robots.0.channel=can2.")
        if name == "record":
            child.add_argument("--task", required=True)
            child.add_argument("--seconds", "--episode-seconds", type=float, default=10.0,
                               help="Recorded episode duration in seconds (default: 10).")
        else:
            child.add_argument("--task", help="Initial task label in the GUI.")
            child.add_argument("--host", default="127.0.0.1")
            child.add_argument("--port", type=int, default=8042)
    convert = commands.add_parser("convert", help="Export completed raw episodes to LeRobot v3.")
    convert.add_argument("source", type=Path)
    convert.add_argument("--repo-id", required=True)
    convert.add_argument("--out", type=Path, required=True)
    return p


def collection_command(args: argparse.Namespace, checkout: Path) -> list[str]:
    entry = {"record": "teleop", "gui": "gui", "cameras": "cameras", "convert": "convert"}[args.command]
    command = [str(checkout / ".venv/bin" / f"yam-abc-{entry}")]
    if args.command in {"record", "gui"}:
        station = args.station.expanduser().resolve()
        if not station.is_file():
            raise ValueError(f"Station config does not exist: {station}")
        command += ["--station", str(station)]
        if args.command == "record":
            if not args.task.strip() or not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError("Recording requires a nonempty task and a finite positive duration.")
            command += ["--record", args.task, "--seconds", str(args.seconds)]
        else:
            command += ["--host", args.host, "--port", str(args.port)]
    elif args.command == "convert":
        source = args.source.expanduser().resolve()
        out = args.out.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Episode directory does not exist: {source}")
        if out.exists():
            raise ValueError(f"Export destination already exists: {out}; choose a new directory.")
        command += [str(source), "--to", "lerobot", "--repo-id", args.repo_id, "--out", str(out)]
    return command


def resolve_station(args: argparse.Namespace) -> dict | None:
    """Apply explicit CLI overrides without editing the operator's station file."""
    if not (args.overrides or args.cameras or args.save_root or
            args.control_hz is not None or args.task is not None):
        return None
    import yaml

    def parse_yaml(text: str):
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc

    config = parse_yaml(args.station.expanduser().resolve().read_text())
    if not isinstance(config, dict):
        raise ValueError("Station YAML must contain a mapping.")
    if args.cameras:
        roster = parse_yaml(args.cameras.expanduser().resolve().read_text())
        cameras = roster.get("cameras") if isinstance(roster, dict) else roster
        if not isinstance(cameras, list):
            raise ValueError("Camera YAML must contain a list or a cameras list.")
        config["cameras"] = cameras
        config.pop("cameras_config", None)
    for override in args.overrides:
        path, separator, value = override.partition("=")
        if not separator or not path:
            raise ValueError("--set requires PATH=VALUE.")
        node = config
        parts = path.split(".")
        try:
            for part in parts[:-1]:
                if isinstance(node, list):
                    if not part.isdecimal():
                        raise ValueError("List indices must be non-negative integers.")
                    node = node[int(part)]
                else:
                    node = node[part]
            key = parts[-1]
            if isinstance(node, list):
                if not key.isdecimal():
                    raise ValueError("List indices must be non-negative integers.")
                key = int(key)
                node[key]  # validate before assignment
            elif not isinstance(node, dict) or key not in node:
                raise KeyError(key)
            node[key] = parse_yaml(value)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unknown station field {path!r}; add supported optional fields to YAML first.") from exc
    if args.control_hz is not None:
        config["control_hz"] = args.control_hz
    hz = config.get("control_hz", 30)
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("control_hz must be finite and positive.")
    if args.save_root:
        config["save_root"] = str(args.save_root.expanduser().resolve())
    if args.task is not None:
        if not args.task.strip():
            raise ValueError("Task must not be empty.")
        config["task_name"] = args.task
    if config.get("data_format", "default") != "default":
        raise ValueError("The pinned recorder supports data_format=default; convert afterward.")
    return config


def apply_station_overrides(args: argparse.Namespace, command: list[str]) -> None:
    config = resolve_station(args)
    if config is None:
        return
    import yaml

    rendered = yaml.safe_dump(config, sort_keys=False)
    if args.dry_run:
        print("Effective station configuration:\n" + rendered)
        path = ROOT / "outputs/collection/sessions/DRY-RUN/station.yaml"
    else:
        path = ROOT / "outputs/collection/sessions" / uuid.uuid4().hex / "station.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(rendered)
        print(f"Session station configuration: {path}", flush=True)
    command[command.index("--station") + 1] = str(path)


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    checkout = args.checkout.expanduser().resolve()
    pin = json.loads(LOCK.read_text())
    try:
        if args.command == "setup":
            setup(checkout, pin, dry_run=args.dry_run)
            return
        command = collection_command(args, checkout)
        if not args.dry_run:
            verify_checkout(checkout, pin)
            if not Path(command[0]).is_file():
                raise ValueError("Collection environment is missing; run setup first.")
        if args.command in {"record", "gui"}:
            apply_station_overrides(args, command)
        # Relative save_root and cameras_config paths are rooted in Open Badger.
        run(command, cwd=ROOT, dry_run=args.dry_run)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Collection error: {exc}\n")


if __name__ == "__main__":
    main()
