# Grasp Reference Scene Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate Dex1 grasp references with support-scene collision filtering and prevent runtime selection of grasps approached from the object's far side.

**Architecture:** GraspGenX continues to generate poses from the segmented object point cloud. A finite support-plane scene point cloud is used only for full-gripper collision filtering over the final-to-pregrasp sweep, while a robot-relative hard accessibility gate runs after live object-pose composition. The compact library grows from 16 to 32 candidates without changing actor observations.

**Tech Stack:** Python, PyTorch, NumPy, trimesh, GraspGenX, pytest, Isaac Lab, tmux.

---

### Task 1: Lock Runtime Accessibility Semantics

**Files:**
- Modify: `tests/test_g1_dex1_grasp_references.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/grasp_references.py`

- [ ] Add a failing test with front and back pregrasp poses and assert the back candidate has exactly zero weight.
- [ ] Add a failing test proving a top-down candidate is not rejected by the horizontal approach-direction gate.
- [ ] Run `pytest -q tests/test_g1_dex1_grasp_references.py -k accessible` and verify the new far-side assertion fails.
- [ ] Implement a hard near-half-space gate from the live robot/object/pregrasp geometry while retaining active-hand preference as a soft weight.
- [ ] Run the focused tests and verify they pass.

### Task 2: Add Support Scene Point-Cloud Collision Filtering

**Files:**
- Modify: `tests/test_g1_dex1_grasp_references.py`
- Modify: `scripts/tools/generate_graspgenx_references.py`

- [ ] Add failing tests requiring a finite support-plane point cloud and swept gripper-to-scene collision filtering.
- [ ] Add a numeric unit test where a below-table sweep is rejected and an above-table sweep is retained.
- [ ] Run the generator-focused tests and verify they fail for the missing scene filter.
- [ ] Sample the platform top as a finite scene cloud in the stable object frame and collision-check the complete Dex1 proxy at five poses over the 0.12 m pregrasp sweep.
- [ ] Keep target-object points separate from scene points so GraspGenX generation still receives only the segmented object.
- [ ] Run the generator tests and verify they pass.

### Task 3: Expand, Regenerate, and Inspect the Library

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/grasp_references.py`
- Modify: `scripts/tools/generate_graspgenx_references.py`
- Modify: `artifacts/graspgenx_g1_dex1/README.md`
- Replace: `source/hbc_lab/hbc_lab/assets/models/objects/grasp_references/multishape_grasp_references_k32.npz`

- [ ] Change the fixed candidate count and default library name from K16 to K32.
- [ ] Run GraspGenX for all enabled assets using the current stable semantic poses and regenerate raw candidates where required.
- [ ] Apply scene filtering, SE(3) NMS, mode balancing, and save the K32 runtime library.
- [ ] Generate contact sheets and inspect wine plus every upright asset for table penetration and far-side-only candidate failure.
- [ ] Update the artifact README with generation parameters and candidate counts.

### Task 4: Verify and Restart the Fair Ablation

**Files:**
- Modify only if paths changed: `.vscode/launch.json`

- [ ] Run `pytest -q tests/test_g1_dex1_grasp_references.py tests/test_g1_dex1_multishape_bps.py tests/test_g1_dex1_hier_drc_static.py`.
- [ ] Sync the exact verified grasp-reference files and library to `jc_5090` and `jd_5090` without overwriting unrelated remote changes.
- [ ] Stop only the prior grasp-reference BPS/no-BPS tmux training sessions.
- [ ] Resume each arm from its corresponding pre-mass-fix checkpoint with identical code, seed policy, environment count, and K32 references; differ only in `object_shape_bps_enabled`.
- [ ] Verify each tmux process is alive and its log reports the expected task, checkpoint, observation dimension, and BPS toggle.
