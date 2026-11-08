# YAM data collection

Open Badger launches the collection and conversion tools from
[Msuri27/yam-abc-reproduce, matt/single-arm-dev](https://github.com/Msuri27/yam-abc-reproduce/tree/matt/single-arm-dev).
The exact commit and its I2RT submodule revision are recorded in
`third_party/yam-abc-reproduce.lock.json`. The branch is provenance, not a floating
runtime dependency. Commands verify both revisions and reject tracked changes.
Keep station configuration outside the checkout.

## Install on the robot host

Prerequisites: Git, uv, PyYAML in the launcher Python environment (already an
Open Badger dependency), and the fork's system dependencies (`build-essential`,
`python3-dev`, and any camera/CAN permissions needed on the station). Use the
fork's `docs/hardware.md` for physical setup. Setup does not configure CAN,
install sudoers rules, or launch the robot.

```bash
python3 scripts/collect.py setup
```

This clones into the ignored `third_party/yam-abc-reproduce/` directory, checks
out the pinned commit, initializes the pinned I2RT submodule over HTTPS, and
runs `uv sync --frozen` with Python 3.12 and camera, GUI, and LeRobot extras.
The fork owns its `.venv` and lockfile; Open Badger's training environment is
not modified. No upstream policy-training dependency group is selected.
An existing checkout at a different revision is never reset automatically.
Use `--checkout /absolute/path` before the subcommand to select another checkout.

Use `--dry-run` before a subcommand to inspect it without installing packages
or opening hardware.

## Hardware collection

`configs/collection/yam_single.yaml` reproduces the fork's single-arm CAN layout
(can0 follower, can1 motorized leader) with no cameras. Copy and edit it for your
station. Configure the actual gripper, channels, and camera serials before use.
The fork may move the gripper during calibration and ease the follower toward
the leader when teleoperation starts.

```bash
python3 scripts/collect.py cameras
python3 scripts/collect.py gui --station configs/collection/yam_single.yaml
```

The GUI binds to `127.0.0.1:8042` by default; it is the fork's existing GUI,
including its review/train/deploy tabs. This integration installs only collection
and LeRobot conversion dependencies. Use the Collect tab for interactive recording.
For a bounded headless recording (CAN must already be configured):

```bash
python3 scripts/collect.py record \
  --station configs/collection/yam_single.yaml \
  --task 'pick up cup' \
  --episode-seconds 30 \
  --control-hz 30 \
  --save-root outputs/collection/episodes
```

### Collection parameters

`record` records one episode of the requested duration. `--seconds` and
`--episode-seconds` are aliases (default: 10 seconds). In the upstream GUI,
episodes are started/stopped manually; there is no automatic duration control.
Use `record` for fixed-duration episodes.

Both `record` and `gui` accept:

| Option | Effect |
| --- | --- |
| `--control-hz 30` | Control/recording loop rate, independent of camera FPS |
| `--save-root PATH` | Raw episode output directory |
| `--task TEXT` | Episode task label; required for `record`, initial GUI label for `gui` |
| `--cameras PATH` | Camera roster YAML, replacing the station's camera list |
| `--set PATH=VALUE` | Repeatable station overrides, with YAML values and numeric list indices |

Examples:

```bash
python3 scripts/collect.py record \
  --station configs/collection/yam_single.yaml \
  --task 'pick up cup' --episode-seconds 30 --control-hz 30 \
  --set robot.robots.0.channel=can0 \
  --set robot.controllers.0.channel=can1 \
  --set robot.robots.0.gripper=linear_4310
```

After creating your camera roster, add `--cameras /path/to/cameras.yaml`.
You can then override fields such as `--set cameras.0.fps=30`,
`--set cameras.0.width=640`, or `--set 'cameras.0.serial="YOUR_SERIAL"'`.
Camera entries can also be provided in station YAML or replaced as a whole with
`--set 'cameras=[...]'`. Quote numeric serials as YAML strings.

All supported hardware fields remain configurable in station YAML. `--set`
requires an existing path to catch typos; add optional upstream fields (such as
`gripper_limits`) to YAML before overriding them. Changing control Hz does not
change the camera FPS automatically. The requested rate is not a guarantee of
achieved capture timing.

Precedence is station YAML, camera roster, `--set` overrides in order, then
explicit `--control-hz`, `--save-root`, and `--task` flags. Overrides are written
to `outputs/collection/sessions/<id>/station.yaml` and passed to the upstream
process. The source station file stays unchanged; GUI station edits during that
session apply to the session copy. Without overrides, the GUI uses the original
station file. `--dry-run` prints the effective YAML and command without writing
a session file or opening hardware.

All launches use the Open Badger root as their working directory. Relative
`save_root` and `cameras_config` paths in station YAML resolve there, not beside
the YAML file. Prefer inline `cameras` or absolute camera-config paths. The fork
prioritizes inline cameras. The supported camera roles are top, left, right, and wrist. Configure real
camera types and unique serials for your station. Do not reuse the fork author's camera serials.

## Export to LeRobot v3

Pass one completed episode or its immediate task directory, not the directory
containing multiple task folders. Use a new output directory for each export:

```bash
python3 scripts/collect.py convert \
  outputs/collection/episodes/pick_up_cup \
  --repo-id local/yam-single \
  --out datasets/yam-single
```

The exporter calls LeRobot's dataset writer; no Hub upload is requested. It
concatenates each arm's measured joints/gripper into `observation.state` and
commanded joints/gripper into `action`. Camera keys include the stream suffix:
`observation.images.top_rgb`, `left_rgb`, `right_rgb`, and `wrist_rgb`.
One six-joint arm plus gripper produces seven state/action dimensions.

Use the exported directory with `scripts/prepare_dataset.py` to compute any
missing quantile statistics, then `scripts/train.py --backend local` with the
same dataset ID, dataset root, and the actual camera keys. Choose a new manifest
directory for the newly collected dataset. Existing SO100 manifests are unrelated.

## Current scope and validation

The launcher is covered by tests for argument handling, revision checks,
non-executing previews, and conversion destination protection. Hardware recording
and a complete LeRobot conversion must be checked on the collection station.

This pin records arms/grippers and cameras. It does **not** capture LinearBot
base/drivetrain or spine encoders/commands. Those require extending the fork's
runtime, raw episode schema, and exporter together; adding unknown YAML keys is
insufficient because upstream ignores them. ROS 2 and TensorRT are separate work.

The pinned exporter aligns equal-length camera streams by index; it only uses
nearest timestamps when camera lengths differ. Arm arrays are also indexed by
row. It does not enforce a maximum timestamp skew or retain the original capture
timestamps as LeRobot feature columns. Retain raw episodes and validate timing
before treating four-camera/mobile-base data as synchronized. Export FPS is the
rounded configured control frequency.
