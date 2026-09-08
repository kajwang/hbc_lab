# GraspRef Pose Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare improved GraspGen 6-DoF guidance against an equal-dimension object-center control in matched from-scratch experiments.

**Architecture:** A pure Torch helper combines fine position guidance with broad-gated linear orientation guidance. Physical grasp drives the mass curriculum, while strict pose-gated grasp remains in DRC. A second task uses the same pose observation dimensions but replaces GraspGen references with object-center position and identity orientation.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based RL, pytest, Hydra, RSL-RL, tmux.

---

### Task 1: Specify pose-guidance behavior

**Files:**
- Modify: `tests/test_g1_dex1_reward_logic.py`
- Modify: `tests/test_g1_dex1_grasp_references.py`

- [x] Add failing tests for broad orientation gradients, physical mass progress, and equal observation dimensions.
- [x] Run the focused tests and confirm they fail on the old implementation.

### Task 2: Implement the minimal reward change

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/multishape_bps_env_cfg.py`

- [x] Use linear normalized angular error behind a broad position gate.
- [x] Keep the strict close-orientation gate unchanged.
- [x] Add physical-grasp curriculum progress and separated orientation diagnostics.
- [x] Add the equal-dimension object-center control task.
- [ ] Run focused tests and confirm they pass.

### Task 3: Verify regressions

**Files:**
- Test: `tests/test_g1_dex1_reward_logic.py`
- Test: `tests/test_g1_dex1_grasp_references.py`
- Test: `tests/test_g1_dex1_hier_drc_static.py`
- Test: `tests/test_g1_dex1_cart_push.py`

- [ ] Run the selected suite and confirm ordinary PnP/Cart behavior remains intact.
- [ ] Inspect the diff for unrelated changes.

### Task 4: Deploy the controlled experiment

**Files:**
- Sync the modified source and tests to `/home/kaijun/hbc_lab` on JC and `/home/zju/code/hbc_lab` on JD.

- [ ] Stop the current GraspRef sessions on both hosts.
- [ ] Start the GraspGen run on JC with no resume/checkpoint arguments.
- [ ] Start the object-center control run on JD with no resume/checkpoint arguments.
- [ ] Confirm both runs report iteration 0/1, actor input dimensions match, BPS is enabled, and no checkpoint is loaded.
