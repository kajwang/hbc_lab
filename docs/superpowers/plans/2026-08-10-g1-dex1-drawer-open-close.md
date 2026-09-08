# G1 Dex1 Drawer Open-Close Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a right-hand G1 Dex1 task that opens a randomly selected Sektion drawer by 0.30 m and then closes it through the shared sparse pose-keyframe interface.

**Architecture:** Reuse the shared contact-label and pose-motion primitives from the latest door task. The drawer environment owns only asset-specific frame selection, keyframe generation, progress buffers, and success logic; observations and rewards preserve the generic sparse-pose contract.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based environments, RSL-RL PPO, pytest.

---

### Task 1: Synchronize The Shared Sparse-Pose Baseline

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/contact_labels.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/pose_motion.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/rewards.py`
- Test: `tests/test_g1_dex1_contact_label_pose.py`
- Test: `tests/test_sparse_pose_motion.py`
- Test: `tests/test_g1_dex1_door_open.py`

- [ ] Synchronize the latest JC generic door/contact-label implementation into the local baseline.
- [ ] Run the three focused tests and confirm they pass unchanged.

### Task 2: Specify Drawer Behavior With Failing Tests

**Files:**
- Create: `tests/test_g1_dex1_drawer.py`

- [ ] Add tests for the Sektion asset, two handle frames, two drawer joints, right-hand mask, random drawer selection, two ordered keyframes, generic motion reward, task registration, and 16-environment debug play configuration.
- [ ] Run `pytest -q tests/test_g1_dex1_drawer.py` and verify failure is caused by the missing task.

### Task 3: Add Cabinet Asset And Scene

**Files:**
- Modify: `source/hbc_lab/hbc_lab/assets/articulated_objects.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/scenes.py`

- [ ] Add the Sektion articulation with both drawer actuators and both handle frames.
- [ ] Add right/left Dex1 contact sensors filtered to both handles.
- [ ] Run the focused drawer test and verify the asset/scene assertions pass.

### Task 4: Implement Drawer Reset And Sparse Motion

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/drawer_env.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/drawer_env_cfg.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/events.py`

- [ ] Reset the cabinet and both drawer joints, then sample top/bottom drawer at 50% probability.
- [ ] Update the selected live handle contact target.
- [ ] Generate the 0.30 m open keyframe and closed-pose keyframe from the selected handle pose.
- [ ] Advance keyframes using masked pose error and stable-step gating.
- [ ] Compute DRC grasp state with the fixed right hand and selected-handle distance.
- [ ] Mark success only after the close keyframe and stable grasp.

### Task 5: Add Generic Observation, Reward, And Registration

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/observations.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/rewards.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/flat_env_cfg.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/agents/rsl_rl_ppo_cfg.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/config/agents/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/__init__.py`

- [ ] Add the generic motion observation with ten-frame actor history.
- [ ] Add shared DRC approach/couple and generic sparse-pose manipulation reward.
- [ ] Register train and play task IDs and PPO configuration.
- [ ] Run the focused drawer and shared door tests until green.

### Task 6: Add Local Visual Launch And Runtime Validation

**Files:**
- Modify: `.vscode/launch.json`

- [ ] Add a 16-environment non-headless train configuration with `env.enable_debug_visualization=True` and the latest low-level policy path.
- [ ] Run compile/static tests.
- [ ] Run a one-environment headless environment smoke test on JD and inspect selected joint/frame movement and finite rollout values.
- [ ] Fix the outward-axis sign only if the measured handle displacement disagrees with the 0.30 m open target.

### Task 7: Synchronize And Stop At Review Gate

**Files:**
- Synchronize: local `/home/kaijun/wbc/hbc_lab`
- Synchronize: JD `/home/zju/code/hbc_lab`

- [ ] Synchronize source, tests, docs, and launch configuration without deleting logs or checkpoints.
- [ ] Verify local and JD tracked/untracked source manifests match.
- [ ] Present implementation details, task dimensions, reward/progress logic, smoke-test evidence, and the exact local launch entry for user review.
- [ ] Do not start the JD 4096-environment tmux training session before explicit user approval.
