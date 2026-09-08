# Articulated Cumulative Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one-step Door/Drawer motion deltas with persistent state progress and make rotated door-handle keyframes follow the handle pivot.

**Architecture:** Shared pose-motion helpers compute local-axis pose rotation and normalized multi-keyframe progress. Door and Drawer environments retain task-specific pose construction and phase switching while exposing one normalized progress value to their shared manipulation reward.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based environments, pytest.

---

### Task 1: Shared pose-motion math

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/pose_motion.py`
- Test: `tests/test_sparse_pose_motion.py`

- [x] Add failing tests for rotating both position and orientation about a pivot.
- [x] Add failing tests for continuous two-keyframe progress and persistent state reward.
- [x] Implement `compose_local_axis_pose` and `compute_cumulative_keyframe_progress`.
- [x] Replace the one-step reward API with `0.3 + normalized_progress`.
- [x] Run `pytest -q tests/test_sparse_pose_motion.py`.

### Task 2: Integrate Door and Drawer

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/drawer_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/rewards.py`
- Test: `tests/test_g1_dex1_door_open.py`
- Test: `tests/test_g1_dex1_drawer.py`

- [x] Replace previous-error buffers with per-phase initial-error buffers.
- [x] Compute normalized cumulative progress before phase advancement so transitions remain continuous.
- [x] Rotate Door keyframe positions around the handle joint pivot together with orientation.
- [x] Pass normalized progress to the manipulation reward and log it explicitly.
- [x] Run focused Door/Drawer tests.

### Task 3: Runtime verification and resume

**Files:**
- No repository files.

- [x] Create one-environment Door and Drawer environments and verify finite progress/reward tensors.
- [x] Sync only the changed files to JC/JD without overwriting unrelated server work.
- [x] Stop the old tmux sessions after capturing their latest checkpoints.
- [x] Resume Door on JC and Drawer on JD from those checkpoints for 50,000 total iterations.
- [x] Verify both tmux sessions advance and load the intended checkpoint.
