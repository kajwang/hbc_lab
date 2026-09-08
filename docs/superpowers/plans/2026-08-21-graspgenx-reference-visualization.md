# GraspGenX Reference Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, filter, store, and visualize GraspGenX grasp references for all G1 Dex1 multishape assets, then route a selected reference through the existing contact-label target pose.

**Architecture:** GraspGenX remains an offline dependency on a 5090 server. The hbc repository owns a small adapter that converts raw model output into a validated fixed-size NPZ library, a dependency-free runtime loader, and an IsaacLab visualization task. Runtime HRC code consumes only the NPZ and updates the selected pose from the live object transform.

**Tech Stack:** Python 3.10, PyTorch, NumPy, GraspGenX, IsaacLab, USD, pytest.

---

### Task 1: Candidate Library Tensor API

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/grasp_references.py`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Add failing tensor-only tests for transform validation, quaternion conversion, approach-mode classification, SE(3) NMS, balanced top/side/oblique selection, NPZ round-trip, and live object-to-world pose composition.
- [ ] Run `pytest -q tests/test_g1_dex1_grasp_references.py` and confirm the new module is missing.
- [ ] Implement the minimal tensor API with no IsaacLab or GraspGenX import.
- [ ] Run the focused test and confirm it passes.

### Task 2: GraspGenX Offline Adapter

**Files:**
- Create: `scripts/tools/generate_graspgenx_references.py`
- Create: `scripts/tools/export_multishape_mesh_manifest.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_shape_bps.py`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Add a manifest test requiring all `ALL_SHAPE_NAMES` exactly once with USD path, nominal scale, and stable orientation.
- [ ] Implement manifest export from `OBJECT_SHAPE_SPECS`.
- [ ] Implement a GraspGenX adapter that saves raw transforms/scores per shape and runs the shared deterministic filter into `multishape_grasp_references_k16.npz`.
- [ ] Add resumability: completed raw files are skipped and filtering can be rerun independently.
- [ ] Run the tensor/static tests without requiring GraspGenX.

### Task 3: Candidate Visualization Task

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_grasp_reference_vis/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_grasp_reference_vis/env.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_grasp_reference_vis/env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/__init__.py`
- Modify: `.vscode/launch.json`
- Test: `tests/test_g1_dex1_grasp_reference_vis.py`

- [ ] Add static tests for task registration, 27-shape deterministic assignment, candidate library path, marker-only visualization, and launch arguments.
- [ ] Implement a play-only environment that loads all assets and renders candidate axes colored by approach mode.
- [ ] Add `g1_dex1_grasp_reference_visual_play` with 27 environments and no checkpoint.
- [ ] Run static tests and a 1-environment headless smoke test.

### Task 4: Replace Contact Target With Selected Grasp Pose

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/events.py`
- Test: `tests/test_g1_dex1_grasp_references.py`

- [ ] Add tests that selected candidate position/orientation replace `ContactLabel` target pose while `object_target_pos_w` remains unchanged.
- [ ] Load the reference library and select a valid candidate at reset.
- [ ] Recompute the active target from the live object pose every control step.
- [ ] Extend the existing task target-pose observation with target rotation-6D; do not create a second grasp-reference term.
- [ ] Keep inactive-hand target entries masked.
- [ ] Run the focused and existing contact-label/multishape tests.

### Task 5: Generate All Assets On JC 5090

**Files:**
- External checkout: `/home/kaijun/GraspGenX`
- Raw outputs: `/home/kaijun/graspgenx_outputs/g1_dex1_multishape/`
- Runtime output: `source/hbc_lab/hbc_lab/assets/models/objects/grasp_references/multishape_grasp_references_k16.npz`

- [ ] Clone GraspGenX and install its uv inference environment.
- [x] Verify the released `unitree_g1` descriptor. It is Dex3, so use GraspGenX's cross-embodiment
  sweep-volume API with dimensions measured from the repository's Dex1 USD.
- [ ] Run all 27 nominal-scale assets, resuming around any failed asset.
- [ ] Filter into the fixed-size runtime NPZ and print accepted/raw counts by shape and mode.
- [ ] Sync only the generated NPZ and hbc source changes back to local and JD without replacing unrelated dirty files.

### Task 6: Final Verification

**Files:**
- Test: `tests/test_g1_dex1_grasp_references.py`
- Test: `tests/test_g1_dex1_grasp_reference_vis.py`
- Test: `tests/test_g1_dex1_multishape_bps.py`
- Test: `tests/test_g1_dex1_contact_label_pose.py`

- [ ] Run all focused pure-Python tests.
- [ ] Run the local 27-environment visual task and check object/grasp frame alignment.
- [ ] Record assets with zero or suspicious candidates for threshold adjustment.
- [ ] Provide the exact local launch entry and generation summary for user review.
