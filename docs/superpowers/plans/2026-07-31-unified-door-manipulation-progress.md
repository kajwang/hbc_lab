# Unified Door Manipulation Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove door-specific policy inputs and reward shaping so the door task uses the same live-object-pose-to-target-pose manipulation interface as PnP.

**Architecture:** The door environment will continue to expose the live handle frame as the object/contact target and a fixed opened-door handle frame as the object goal. Generic normalized spatial progress drives manipulation reward, while articulation angles remain internal to latch physics, success checks, and logging.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based environments, pytest.

---

### Task 1: Lock the Unified Observation and Reward Contract

**Files:**
- Modify: `tests/test_g1_dex1_door_open.py`
- Modify: `tests/test_g1_dex1_articulated_progress.py`

- [ ] **Step 1: Replace the door-state observation test**

Assert that `G1Dex1DoorOpenObservationsCfg` directly reuses
`G1Dex1HierDrcObservationsCfg`, and that no `door_articulation_state` or
`door_state` term remains.

- [ ] **Step 2: Replace the handle-gated reward tests**

Test a pure tensor helper with initial distances `[2, 2, 0]` and current
distances `[1, 3, 0]`, expecting normalized spatial progress `[0.5, 0, 0]`.
Assert statically that `door_manip_reward` uses spatial progress and
`env.c_couple`, not `env.door_manipulation_progress.reward`.

- [ ] **Step 3: Add the actuator parity assertion**

Assert that the door handle actuator uses `stiffness=1.0`.

- [ ] **Step 4: Run the focused tests and confirm RED**

Run:

```bash
pytest -q tests/test_g1_dex1_door_open.py tests/test_g1_dex1_articulated_progress.py
```

Expected: failures for the still-present door state, handle-gated reward, and
handle stiffness `10.0`.

### Task 2: Implement Generic Spatial Manipulation Progress

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/progress.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env.py`

- [ ] **Step 1: Replace handle-gated progress with spatial progress**

Implement:

```python
def compute_spatial_progress(
    initial_distance: torch.Tensor,
    current_distance: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    progress = torch.clamp(initial_distance - current_distance, min=0.0)
    valid = initial_distance > eps
    return torch.where(valid, progress / (initial_distance + eps), torch.zeros_like(progress))
```

- [ ] **Step 2: Restore the generic manipulation reward**

Compute initial handle-to-goal distance from `object_initial_pos_w` and
`object_target_pos_w`, then return:

```python
0.7 * spatial_progress + 0.3 * env.c_couple
```

- [ ] **Step 3: Make `d_goal` spatial**

In `_compute_progress`, set `d_goal` to the Euclidean distance between the live
handle frame and `object_target_pos_w`. Keep handle and hinge angle updates for
latch simulation, success checks, and logs.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run:

```bash
pytest -q tests/test_g1_dex1_door_open.py tests/test_g1_dex1_articulated_progress.py
```

Expected: all focused tests pass.

### Task 3: Remove Door-Specific Network Inputs

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env_cfg.py`

- [ ] **Step 1: Reuse the shared observation configuration**

Remove `door_articulation_state` and the policy/critic `door_state` terms.
Configure the door task with `G1Dex1HierDrcObservationsCfg`.

- [ ] **Step 2: Run observation contract tests**

Run:

```bash
pytest -q tests/test_g1_dex1_door_open.py
```

Expected: all door contract tests pass.

### Task 4: Restore Door Handle Dynamics and Verify

**Files:**
- Modify: `source/hbc_lab/hbc_lab/assets/articulated_objects.py`
- Test: `tests/test_g1_dex1_articulated_assets.py`

- [ ] **Step 1: Restore original handle stiffness**

Set the `joint_2` implicit actuator stiffness from `10.0` to `1.0`, retaining
the existing damping.

- [ ] **Step 2: Run the relevant local test suite**

Run:

```bash
pytest -q \
  tests/test_g1_dex1_door_open.py \
  tests/test_g1_dex1_articulated_progress.py \
  tests/test_g1_dex1_articulated_assets.py \
  tests/test_g1_dex1_articulated_registration.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Sync exact changes to `jc_5090`**

Copy only the modified source and test files into `/home/kaijun/hbc_lab` on
`jc_5090`, preserving the remote worktree's unrelated changes.

- [ ] **Step 4: Run the same tests on `jc_5090`**

Run the selected pytest command in `/home/kaijun/hbc_lab`.

- [ ] **Step 5: Run an Isaac Lab smoke test on `jc_5090`**

Launch the door task with a small environment count and verify reset plus
several finite environment steps. Do not claim runtime success without this
fresh remote evidence.
