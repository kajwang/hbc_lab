# Grasp Candidate Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-asset, one-environment-per-candidate policy visualization for GraspRef and object-center control checkpoints.

**Architecture:** A thin CLI wrapper delegates inference to the existing Play runner. A post-Hydra config hook selects one asset and sizes the scene, while the environment fixes candidate IDs and separates diagnostic references from policy target references.

**Tech Stack:** Python, PyTorch, NumPy, Isaac Lab config classes and visualization markers, RSL-RL, pytest.

---

### Task 1: Candidate Metadata and Assignment

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/grasp_references.py`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Write failing tests for selecting valid candidate IDs by shape and assigning them deterministically by environment ID.
- [ ] Run the focused tests and confirm missing helper failures.
- [ ] Implement validated metadata lookup and deterministic assignment helpers.
- [ ] Re-run the focused tests.

### Task 2: Play-only Scene Configuration

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/multishape_bps_env_cfg.py`
- Modify: `scripts/rsl_rl/play.py`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Write failing integration assertions for the post-override hook, selected asset spawner, candidate-sized scene, and fixed object pose.
- [ ] Run the tests and confirm failure on missing configuration behavior.
- [ ] Add candidate evaluation fields and `apply_grasp_candidate_evaluation()`.
- [ ] Call the hook after Hydra overrides and before `gym.make()`.
- [ ] Re-run the tests.

### Task 3: Separate Diagnostic and Policy References

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Write failing tests requiring fixed candidate assignment and a selected-reference pose helper.
- [ ] Add `grasp_reference_controls_target` and keep it false for object-center control.
- [ ] Use candidate references for markers in both modes, but only for observations/rewards in GraspRef mode.
- [ ] Re-run the tests.

### Task 4: Thin CLI and Launch Entries

**Files:**
- Create: `scripts/rsl_rl/play_grasp_candidates.py`
- Modify: `.vscode/launch.json`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Write failing tests for mode-to-task translation and forced diagnostic overrides.
- [ ] Implement argument translation and `os.execv()` delegation to `play.py`.
- [ ] Add GraspRef and object-center launch entries with editable asset IDs/checkpoints.
- [ ] Re-run the focused tests.

### Task 5: Verification

**Files:**
- Test: `tests/test_g1_dex1_grasp_references.py`
- Test: `tests/test_g1_dex1_multishape_bps.py`

- [ ] Run focused unit/static tests.
- [ ] Run `py_compile` and `git diff --check`.
- [ ] Start each Play mode with a matching checkpoint and confirm candidate mapping, marker visibility, and network dimensions.
