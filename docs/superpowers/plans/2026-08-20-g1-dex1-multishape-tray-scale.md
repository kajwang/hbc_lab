# G1 Dex1 Multi-Shape Tray And Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the requested graspable assets, spawn-time 0.8-1.2 uniform size variation, shallow-tray supports, compact diagnostics, and all-asset visual evaluation to the existing multi-shape PnP task.

**Architecture:** Extend the existing deterministic `MultiAssetSpawnerCfg` assignment to shape/scale variants while keeping one object actor per environment. Keep canonical BPS metadata per shape and apply the selected uniform scale consistently to descriptors, bounds, and center offsets at runtime. Implement the tray as one compound kinematic USD per support so the number of physics actors does not multiply.

**Tech Stack:** Python, PyTorch, Isaac Lab/Isaac Sim USD assets, pytest, directional BPS.

---

### Task 1: Lock The Experiment Contract In Tests

**Files:**
- Modify: `tests/test_g1_dex1_multishape_bps.py`

- [ ] Add tests for the 24 training shapes, three held-out shapes, YAML-derived base scales, five training scale factors, and 27-environment play assignment.
- [ ] Add numeric tests for scaled directional BPS reconstruction and shape/scale variant decoding.
- [ ] Add static checks for the compound shallow-tray asset and compact log key set.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py` and verify the new assertions fail because the feature is absent.

### Task 2: Vendor The Requested Assets And Extend Shape Metadata

**Files:**
- Copy: requested object directories from `robot_lab/.../assets/models/objects/`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_shape_bps.py`
- Modify: `scripts/tools/generate_multishape_bps.py`

- [ ] Copy complete USD directories including referenced textures and configuration layers.
- [ ] Change shape specifications to support three-axis base scales and add all requested objects.
- [ ] Add helpers that map a spawn variant to shape and size indices and scale directional descriptors correctly.
- [ ] Update BPS generation to apply vector base scales.
- [ ] Run focused helper tests until green.

### Task 3: Spawn Balanced Shape/Scale Variants

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/scenes.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/multishape_bps_env_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/events.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py`

- [ ] Build the training spawner from every training shape crossed with five scale factors; build play from standard-size shapes only.
- [ ] Decode the authored variant metadata after scene creation and store per-environment shape and scale indices.
- [ ] Apply size factors to reset bounds, geometry centers, object frame locations, fall thresholds, and BPS observations.
- [ ] Run multi-shape tests until green.

### Task 4: Replace Flat Supports With Compound Shallow Trays

**Files:**
- Create: `source/hbc_lab/hbc_lab/assets/models/platform/shallow_tray.usda`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/scenes.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`

- [ ] Author one rigid-body USD containing the pedestal collision and four shallow wall colliders.
- [ ] Use it for multi-shape initial and target supports without changing the 0.5 m work height.
- [ ] Render base and walls through pose-driven debug markers so Fabric cannot leave visual geometry at the origin.
- [ ] Run tray static tests until green.

### Task 5: Compact Diagnostics And Add Per-Shape Grasp Rate

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py`

- [ ] Replace per-shape continuous diagnostic groups with one thresholded stable-grasp occupancy metric.
- [ ] Remove per-link/contact/gate/gripper-buffer and redundant curriculum logs.
- [ ] Keep the minimum global DRC, motion, safety, mass, and success metrics needed to diagnose training.
- [ ] Run focused reward and multi-shape tests until green.

### Task 6: Regenerate Data And Verify Runtime Entry Points

**Files:**
- Regenerate: `source/hbc_lab/hbc_lab/assets/models/objects/bps/multishape_directional_bps64.npz`
- Modify: `.vscode/launch.json`

- [ ] Regenerate BPS metadata headlessly and inspect printed physical dimensions for implausible assets.
- [ ] Update both visual play entries to 27 environments while retaining support-platform visualization only.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py tests/test_g1_dex1_reward_logic.py`.
- [ ] Run Python compilation/static checks and, if available, a small headless Isaac Lab environment smoke test.
