# hbc_lab

HBC Lab is a lightweight IsaacLab workspace for humanoid whole-body control experiments.

Project references:

- [`IDEA_PLAN.md`](IDEA_PLAN.md): full research direction and paper plan.
- [`docs/GO2_ARX5_TASK_MIGRATION_GUIDE.md`](docs/GO2_ARX5_TASK_MIGRATION_GUIDE.md): reusable PnP, cart, door, drawer, DRC, keyframe, and action-saturation guidance for the Robot Lab Go2+ARX5 implementation.

The first target is a low-level Unitree G1 velocity locomotion policy that can be trained
before the object-centric interaction pipeline is added. This repository reuses the existing
Unitree RSL-RL runner pattern, then vendors the Unitree official G1 29DoF velocity recipe
into the HBC task namespace so reward and curriculum breakpoints stay local.

## Layout

- `source/hbc_lab/hbc_lab/assets/robots`: local Unitree robot and actuator configs.
- `source/hbc_lab/hbc_lab/tasks/locomotion`: migrated Unitree official locomotion task configs and MDP helpers.
- `source/hbc_lab/hbc_lab/tasks/manager_based/humanoid_tracking/mdp`: legacy reference-motion helpers for later interaction work.
- `scripts/rsl_rl/train.py`: HBC local RSL-RL training entry point.
- `scripts/rsl_rl/play.py`: HBC local RSL-RL playback entry point using `play_env_cfg_entry_point`.

## Dependency Assumption

This repository expects a sibling checkout:

```bash
/home/kaijun/wbc/robot_lab
/home/kaijun/wbc/hbc_lab
```

The launcher script adds HBC, robot_lab, and IsaacLab source paths to `PYTHONPATH`.

The migrated Unitree asset config uses these defaults, which can be overridden per shell:

```bash
export UNITREE_MODEL_DIR=/home/kaijun/wbc/unitree_model
export UNITREE_ROS_DIR=/home/kaijun/wbc/unitree_ros
```

## First Training Run

The Flat and Rough HBC task ids currently both point to the same migrated Unitree official
G1 velocity recipe. The two ids are kept so old commands continue to work while we stabilize
the baseline.

Start training with:

```bash
cd /home/kaijun/wbc/hbc_lab
python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-Tracking-Flat-Unitree-G1-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=50000
```

The rough alias uses the same config:

```bash
python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-Tracking-Rough-Unitree-G1-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=50000
```

Play a trained policy by pointing to the produced checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task=HBC-Isaac-Tracking-Flat-Unitree-G1-v0 \
  --num_envs=64 \
  --checkpoint logs/rsl_rl/hbc_g1_tracking_flat/<run>/model_49999.pt
```

## Current Locomotion Recipe

The current task is the Unitree official G1 29DoF velocity recipe migrated into
`hbc_lab.tasks.locomotion.robots.g1.29dof`. Training keeps the official command curriculum:
it starts from a narrow command range and expands toward the limits when the linear velocity
tracking reward passes the official threshold. Play uses the limit command range immediately
and disables play-side curriculum.

The official command limits are:

- `lin_vel_x=(-0.5, 1.0)`
- `lin_vel_y=(-0.3, 0.3)`
- `ang_vel_z=(-0.2, 0.2)`

## Whole-Body Extension

The first extension task keeps the official velocity baseline intact and adds a
separate task id:

```bash
python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-WholeBody-Unitree-G1-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=50000
```

This task adds terrain `height_scan` observations plus low-frequency
`left_wrist_pose` and `right_wrist_pose` commands. The wrist pose tracking
rewards are registered with zero weight at this stage, so the first goal is to
verify the command and observation pipeline before enabling hand tracking through
curriculum.

Object, end-effector, and sparse video-derived tracking commands should be added in a later
task after this locomotion baseline is stable.
