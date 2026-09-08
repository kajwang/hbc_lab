# G1 Dex1 Multi-Shape BPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated 4096-environment PnP experiment that varies object shape and supplies a current-frame directional BPS observation without changing the existing randomized PnP task.

**Architecture:** Spawn exactly one of eight training USD assets per environment with a deterministic multi-asset spawner. Precompute a shared 64-point directional BPS descriptor for every asset, transform the selected descriptor from object-local coordinates into the G1 pelvis/root frame at runtime, and keep it outside the ten-frame dynamic observation history. Preserve the existing PnP rewards and mass curriculum while disabling size, height, and extra mass-range randomization.

**Tech Stack:** Isaac Lab manager-based environments, PyTorch, USD/PhysX, RSL-RL PPO, pytest static and numerical tests.

---

### Task 1: Directional BPS data contract

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_shape_bps.py`
- Create: `tests/test_g1_dex1_multishape_bps.py`

- [ ] Write failing tests for deterministic basis generation, geometry-center offsets, descriptor dimensions, and rigid-frame transforms.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py` and confirm the missing module failure.
- [ ] Implement the asset metadata and tensor-only BPS transform helpers.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Multi-asset scene and reset

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/multishape_bps_env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/scenes.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/events.py`
- Modify: `tests/test_g1_dex1_multishape_bps.py`

- [ ] Add failing static tests requiring one multi-asset rigid object per environment and fixed platform height/asset scale.
- [ ] Implement deterministic round-robin multi-asset spawning and shape-aware reset placement.
- [ ] Preserve XY/yaw reset randomization and the existing DRC mass curriculum only.
- [ ] Run the focused tests.

### Task 3: Current-frame BPS policy observation

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/multishape_bps_env_cfg.py`
- Modify: `tests/test_g1_dex1_multishape_bps.py`

- [ ] Add failing tests that dynamic terms have history length 10 while the 192-dimensional BPS term has no history.
- [ ] Add the pelvis/root-frame directional BPS observation and per-term observation history configuration.
- [ ] Keep the critic shape input current-frame and preserve existing privileged terms.
- [ ] Run the focused and existing PnP static tests.

### Task 4: Registration, launch, and metrics

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/__init__.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/agents/rsl_rl_ppo_cfg.py`
- Modify: `.vscode/launch.json`
- Modify: `tests/test_g1_dex1_multishape_bps.py`

- [ ] Add failing registration/config tests for train, no-BPS ablation, and play task IDs.
- [ ] Register the tasks and a dedicated RSL-RL experiment name.
- [ ] Add 4096-env train and small visual play launch entries.
- [ ] Add logging for per-shape success and BPS/gripper alignment without changing rewards.
- [ ] Run the full relevant test subset and Python compilation.

### Task 5: Simulation and JC deployment

**Files:**
- No additional production files expected.

- [ ] Run a 16-environment headless reset/step smoke test locally.
- [ ] Run a short 4096-environment throughput smoke test and check GPU memory and collection speed.
- [ ] Sync only the verified repository state to `jc_5090` without stopping the JD run.
- [ ] Start the BPS experiment in a named tmux session and report its command, log directory, and initial throughput.
