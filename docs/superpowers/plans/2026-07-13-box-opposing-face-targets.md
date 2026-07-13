# Box Opposing-Face Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guide the two G1 Dex1 hand centers to deterministically assigned centers of the box's opposing longest-axis faces.

**Architecture:** Add a pure-Torch geometry module that computes the current `+X/-X` face centers and selects the minimum-distance hand assignment once per episode. The BoxCarry environment owns the fixed assignment and live world-frame targets; a BoxCarry-specific observation term exposes the assigned targets in the robot root frame. Existing contact support, DRC equations, reward weights, 19D action interface, and low-level policy remain unchanged.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based environments, Isaac Lab `VisualizationMarkers`, pytest.

---

## File Structure

- Create `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/face_targets.py`: simulator-independent quaternion geometry and deterministic hand assignment.
- Create `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/observations.py`: append assigned face targets to the existing BoxCarry task observation.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/__init__.py`: export the new observation module with the task MDP namespace.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env.py`: own assignment buffers, update targets, use face errors for DRC, visualize targets, and log errors.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env_cfg.py`: install the BoxCarry-specific observation function for actor and critic.
- Modify `tests/test_g1_dex1_box_carry.py`: cover geometry, assignment, target-following behavior, observation wiring, and progress integration.

### Task 1: Pure Opposing-Face Geometry

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/face_targets.py`
- Modify: `tests/test_g1_dex1_box_carry.py`

- [ ] **Step 1: Add a module loader and failing geometry tests**

Add `_load_face_targets_module()` beside the existing contact-progress loader, then add tests equivalent to:

```python
def test_long_axis_face_centers_follow_translation_and_yaw():
    module = _load_face_targets_module()
    object_pos = torch.tensor([[1.0, 2.0, 0.1], [0.0, 0.0, 0.1]])
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    yaw_90 = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    positive, negative = module.compute_long_axis_face_centers(
        object_pos, torch.stack((identity, yaw_90)), half_extent=0.15
    )
    assert torch.allclose(positive[0], torch.tensor([1.15, 2.0, 0.1]), atol=1.0e-6)
    assert torch.allclose(negative[0], torch.tensor([0.85, 2.0, 0.1]), atol=1.0e-6)
    assert torch.allclose(positive[1], torch.tensor([0.0, 0.15, 0.1]), atol=1.0e-6)
    assert torch.allclose(negative[1], torch.tensor([0.0, -0.15, 0.1]), atol=1.0e-6)


def test_face_assignment_uses_shorter_pairing_and_remains_explicit():
    module = _load_face_targets_module()
    positive = torch.tensor([[0.0, 0.2, 0.1], [0.0, 0.2, 0.1]])
    negative = torch.tensor([[0.0, -0.2, 0.1], [0.0, -0.2, 0.1]])
    left_hand = torch.tensor([[0.0, 0.4, 0.1], [0.0, -0.4, 0.1]])
    right_hand = -left_hand
    left_positive = module.choose_left_positive_assignment(
        positive, negative, left_hand, right_hand
    )
    assert torch.equal(left_positive, torch.tensor([True, False]))

    left_target, right_target = module.select_assigned_face_targets(
        positive, negative, left_positive
    )
    assert torch.allclose(left_target, torch.stack((positive[0], negative[1])))
    assert torch.allclose(right_target, torch.stack((negative[0], positive[1])))
```

- [ ] **Step 2: Run the tests and verify the missing module fails**

Run:

```bash
pytest -q tests/test_g1_dex1_box_carry.py -k "face_centers or face_assignment"
```

Expected: failure because `mdp/face_targets.py` does not exist.

- [ ] **Step 3: Implement the minimal pure-Torch geometry module**

Implement these functions without importing Isaac Lab:

```python
def _quat_rotate(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vec = quat_wxyz[..., 1:]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


def compute_long_axis_face_centers(
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    half_extent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_offset = torch.zeros_like(object_pos_w)
    local_offset[..., 0] = half_extent
    world_offset = _quat_rotate(object_quat_w, local_offset)
    return object_pos_w + world_offset, object_pos_w - world_offset


def choose_left_positive_assignment(
    positive_w: torch.Tensor,
    negative_w: torch.Tensor,
    left_hand_w: torch.Tensor,
    right_hand_w: torch.Tensor,
) -> torch.Tensor:
    positive_left_cost = torch.norm(left_hand_w - positive_w, dim=-1) + torch.norm(
        right_hand_w - negative_w, dim=-1
    )
    negative_left_cost = torch.norm(left_hand_w - negative_w, dim=-1) + torch.norm(
        right_hand_w - positive_w, dim=-1
    )
    return positive_left_cost <= negative_left_cost


def select_assigned_face_targets(
    positive_w: torch.Tensor,
    negative_w: torch.Tensor,
    left_positive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = left_positive.unsqueeze(-1)
    return torch.where(mask, positive_w, negative_w), torch.where(mask, negative_w, positive_w)
```

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_g1_dex1_box_carry.py -k "face_centers or face_assignment"`

Expected: both tests pass.

- [ ] **Step 5: Commit the geometry unit**

```bash
git add tests/test_g1_dex1_box_carry.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/face_targets.py
git commit -m "feat: add box opposing-face geometry"
```

### Task 2: Episode-Fixed Face Targets And DRC Distances

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env.py`
- Modify: `tests/test_g1_dex1_box_carry.py`

- [ ] **Step 1: Add failing source-level integration assertions**

Add a test that verifies the BoxCarry environment:

```python
def test_box_env_uses_episode_fixed_long_axis_face_targets_for_progress():
    source = _read(CONFIG_ROOT / "box_env.py")
    assert "self.left_face_uses_positive" in source
    assert "self.face_assignment_pending" in source
    assert "compute_long_axis_face_centers" in source
    assert "choose_left_positive_assignment" in source
    assert "select_assigned_face_targets" in source
    assert "left_face_target_pos_w" in source
    assert "right_face_target_pos_w" in source
    assert "self.face_assignment_pending[env_ids] = True" in source
```

- [ ] **Step 2: Run the failing integration test**

Run:

```bash
pytest -q tests/test_g1_dex1_box_carry.py::test_box_env_uses_episode_fixed_long_axis_face_targets_for_progress
```

Expected: failure because no face-target state exists in `box_env.py`.

- [ ] **Step 3: Add buffers and live target updates**

Before `super().__init__`, allocate:

```python
self.left_face_target_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
self.right_face_target_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
self.left_face_uses_positive = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
self.face_assignment_pending = torch.ones(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
```

Add `_update_face_targets()` that computes the current `+X/-X` centers with `half_extent=0.5 * BOX_CUBE_SIZE[0]`, chooses an assignment only for pending environments using current hand-center positions, clears their pending flags, and selects live targets for every environment using the stored assignment.

- [ ] **Step 4: Route progress through assigned face targets**

At the start of `_compute_progress()`, call `_update_face_targets()`. Replace center distances with:

```python
left_distance = torch.norm(hand_center_pos_w[:, 0, :] - self.left_face_target_pos_w, dim=-1)
right_distance = torch.norm(hand_center_pos_w[:, 1, :] - self.right_face_target_pos_w, dim=-1)
```

Keep `compute_bimanual_support_progress`, contact EMA values, DRC equations, mass curriculum, and all reward weights unchanged.

- [ ] **Step 5: Reset assignment state without selecting from stale kinematics**

In `_reset_hier_buffers(env_ids)`, clear both target buffers and set:

```python
self.face_assignment_pending[env_ids] = True
```

The first post-forward observation selects the assignment from current reset kinematics. Later target updates only transform the stored local face assignment with the moving box pose.

- [ ] **Step 6: Run BoxCarry tests**

Run: `pytest -q tests/test_g1_dex1_box_carry.py`

Expected: all BoxCarry tests pass.

- [ ] **Step 7: Commit environment integration**

```bash
git add tests/test_g1_dex1_box_carry.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env.py
git commit -m "feat: guide box carry toward opposing faces"
```

### Task 3: Expose Assigned Face Targets To Actor And Critic

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/__init__.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env_cfg.py`
- Modify: `tests/test_g1_dex1_box_carry.py`

- [ ] **Step 1: Add failing observation wiring test**

Add assertions that `observations.py` calls the existing `object_goal_hand_obs`, calls `env._update_face_targets()`, converts both targets with `quat_apply_inverse`, and appends them in left-then-right order. Also assert that `box_env_cfg.py` installs this function for both `self.observations.policy.task.func` and `self.observations.critic.task.func`.

- [ ] **Step 2: Run the failing observation test**

Run: `pytest -q tests/test_g1_dex1_box_carry.py -k observation`

Expected: failure because the BoxCarry observation module and wiring do not exist.

- [ ] **Step 3: Implement the BoxCarry task observation**

Create:

```python
def box_object_goal_hand_obs(env) -> torch.Tensor:
    base_task_obs = object_goal_hand_obs(env)
    env._update_face_targets()
    robot: Articulation = env.scene["robot"]
    left_target_b = quat_apply_inverse(
        robot.data.root_quat_w,
        env.left_face_target_pos_w - robot.data.root_pos_w,
    )
    right_target_b = quat_apply_inverse(
        robot.data.root_quat_w,
        env.right_face_target_pos_w - robot.data.root_pos_w,
    )
    return torch.cat((base_task_obs, left_target_b, right_target_b), dim=-1)
```

This preserves the existing 14 task scalars and appends six observable target scalars, for a 20-scalar task term.

- [ ] **Step 4: Wire actor and critic to the new function**

In `G1Dex1BoxCarryEnvCfg.__post_init__`, after `super().__post_init__()`:

```python
self.observations.policy.task.func = box_object_goal_hand_obs
self.observations.critic.task.func = box_object_goal_hand_obs
```

Export `observations` from the BoxCarry `mdp/__init__.py` guarded import block.

- [ ] **Step 5: Run observation and complete BoxCarry tests**

Run:

```bash
pytest -q tests/test_g1_dex1_box_carry.py
```

Expected: all tests pass and existing 19D action assertions remain unchanged.

- [ ] **Step 6: Commit observation integration**

```bash
git add tests/test_g1_dex1_box_carry.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/observations.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/__init__.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env_cfg.py
git commit -m "feat: observe assigned box face targets"
```

### Task 4: Target Visualization, Diagnostics, And Smoke Verification

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env.py`
- Modify: `tests/test_g1_dex1_box_carry.py`

- [ ] **Step 1: Add failing visualization and logging assertions**

Assert that `box_env.py` defines separate left/right `VisualizationMarkersCfg` values, visualizes `left_face_target_pos_w` and `right_face_target_pos_w`, and logs:

```text
BoxCarry/left_face_target_error
BoxCarry/right_face_target_error
BoxCarry/left_positive_assignment_ratio
```

- [ ] **Step 2: Run the failing diagnostics test**

Run: `pytest -q tests/test_g1_dex1_box_carry.py -k "visualization or diagnostics"`

Expected: failure because the markers and new log keys do not exist.

- [ ] **Step 3: Implement face-target markers and logs**

Create blue and orange sphere marker configurations with distinct prim paths and `0.035 m` radii. Override `_update_target_pose_visualization()` to call `super()`, refresh face targets, lazily create both marker objects, and visualize their current positions whenever `target_pose_debug_vis` is enabled.

Replace the ambiguous `left_center_distance/right_center_distance` logs with the explicit face-target error names, and log the fraction of environments assigned positive local X to the left hand.

- [ ] **Step 4: Run all focused static/unit tests**

Run:

```bash
pytest -q tests/test_g1_dex1_box_carry.py tests/test_g1_dex1_contact_labels.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Run one simulator smoke iteration**

Run:

```bash
python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-G1-Dex1-BoxCarry-HierDrc-v0 \
  --headless \
  --num_envs=2 \
  --max_iterations=1 \
  env.low_level_policy_path=logs/exported_policies/g1_spherical_posture_actor.pt
```

Expected: the environment resets, the actor reports an input width increased by six, one rollout/learning iteration completes, and no observation shape or target-buffer error occurs.

- [ ] **Step 6: Inspect the worktree and commit diagnostics**

```bash
git diff --check
git status --short
git add tests/test_g1_dex1_box_carry.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/config/box_env.py
git commit -m "feat: visualize box face targets"
```

