# Contact-Conditioned Motion Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace frame-dependent incremental high-level commands with physical-rate commands and add shared contact-conditioned motion regularization that keeps the G1 composed while approaching an interaction target without restricting valid near-contact motion.

**Architecture:** The 19-dimensional high-level policy remains the only trainable controller and the stick-figure hand-center HBC remains frozen. Pure PyTorch helpers compute physical command rates, continuous reachability weights, and normalized motion losses; the IsaacLab environment supplies deployable robot state and live `ContactLabel` geometry. PnP is the first training target, while shared observations and reward hooks are wired consistently across the existing Dex1 tasks.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab manager-based RL, RSL-RL PPO, pytest, tmux.

---

## File Map

- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/high_level_actions.py`: physical-rate decoding, command-rate dataclasses, and hand-center naming.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/commands.py`: hand-center command naming and defaults.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/low_level_observations.py`: consume renamed command fields without changing the frozen HBC observation order.
- Create `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/motion_regularization.py`: pure PyTorch gates, normalized losses, and term aggregation.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`: command-rate history, upper-body IDs, reset/update logic, and diagnostics.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py`: shared deployable execution-state observation.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py`: gather environment state and apply the shared regularizer.
- Modify the Box, Cart, Door, and Drawer observation/reward configs only where necessary to preserve the shared interface.
- Modify `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env_cfg.py`: regularization tolerances and physical limits.
- Create `tests/test_g1_dex1_motion_regularization.py`: pure numerical behavior.
- Create `tests/test_g1_dex1_physical_rate_actions.py`: physical-rate decoder behavior.
- Update `tests/test_g1_dex1_hier_drc_static.py` and `tests/test_g1_dex1_wrist_local_command.py`: renamed interface and shared wiring.

### Task 1: Physical-Rate High-Level Decoder

**Files:**
- Create: `tests/test_g1_dex1_physical_rate_actions.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/high_level_actions.py`

- [ ] **Step 1: Write failing tests for physical-rate integration**

Add tests that create a zero command state, apply the same action for one `0.1 s` step and two `0.05 s` steps, and assert equal final position/posture/orientation. Test exact one-step limits and confirm position remains softly unconstrained.

```python
def make_command_state(left_position=(0.0, 0.0, 0.0)):
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return HighLevelCommandState(
        base_velocity=torch.zeros(1, 3),
        posture_command=torch.tensor([[0.70, 0.0]]),
        left_hand_center_pose_a=torch.cat((torch.tensor([left_position]), quat), dim=-1),
        right_hand_center_pose_a=torch.cat((torch.zeros(1, 3), quat), dim=-1),
        left_grip=torch.zeros(1, 1),
        right_grip=torch.zeros(1, 1),
    )


def test_hand_translation_is_invariant_to_high_level_frequency():
    limits = HighLevelActionLimits()
    action = torch.zeros(1, 19)
    action[:, 5] = 1.0
    initial = make_command_state()

    one, _ = decode_high_level_action(action, initial, limits, dt=0.1)
    half, _ = decode_high_level_action(action, initial, limits, dt=0.05)
    two, _ = decode_high_level_action(action, half, limits, dt=0.05)

    assert torch.allclose(one.left_hand_center_pose_a[:, :3], two.left_hand_center_pose_a[:, :3], atol=1.0e-6)
    assert torch.allclose(one.left_hand_center_pose_a[:, 0], torch.tensor([0.03]), atol=1.0e-6)


def test_decoded_rate_reports_physical_units():
    limits = HighLevelActionLimits()
    action = torch.zeros(1, 19)
    action[:, 3] = 1.0
    action[:, 5] = 1.0
    action[:, 8] = 1.0

    _, rate = decode_high_level_action(action, make_command_state(), limits, dt=0.1)

    assert torch.allclose(rate.posture_velocity[:, 0], torch.tensor([0.12]))
    assert torch.allclose(rate.left_hand_twist[:, 0], torch.tensor([0.30]))
    assert torch.allclose(rate.left_hand_twist[:, 3], torch.tensor([1.00]))


def test_physical_rate_decoder_does_not_hard_clamp_workspace():
    previous = make_command_state(left_position=(0.80, 0.20, -0.10))
    action = torch.zeros(1, 19)
    action[:, 5] = 1.0
    decoded, _ = decode_high_level_action(action, previous, HighLevelActionLimits(), dt=0.1)
    assert decoded.left_hand_center_pose_a[0, 0] > 0.80
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
pytest -q tests/test_g1_dex1_physical_rate_actions.py
```

Expected: failures because `dt`, `HighLevelCommandRateState`, and hand-center field names do not exist.

- [ ] **Step 3: Implement physical-rate decoding**

Replace per-step scales with physical limits and return both command and rate state:

```python
@dataclass(frozen=True)
class HighLevelActionLimits:
    max_lin_vel: float = 0.8
    max_ang_vel: float = 0.5
    max_lin_acc: float = 1.5
    max_ang_acc: float = 1.5
    hand_linear_speed: float = 0.30
    hand_angular_speed: float = 1.00
    root_height_speed: float = 0.12
    torso_pitch_speed: float = 0.40
    root_height_range: tuple[float, float] = (0.42, 0.80)
    torso_pitch_range: tuple[float, float] = (0.0, 0.85)
    wrist_radius_range: tuple[float, float] = (0.20, 0.58)


@dataclass
class HighLevelCommandRateState:
    base_acceleration: torch.Tensor
    posture_velocity: torch.Tensor
    left_hand_twist: torch.Tensor
    right_hand_twist: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            (self.base_acceleration, self.posture_velocity, self.left_hand_twist, self.right_hand_twist), dim=-1
        )
```

Decode base velocity with acceleration limits multiplied by `dt`, posture with `[0.12, 0.40] * dt`, and each hand with `0.30 * dt` translation and `1.00 * dt` local rotvec. Compute reported rates from the actual post-clamp command difference so a posture command at its limit reports zero velocity.

- [ ] **Step 4: Run the focused tests**

Run `pytest -q tests/test_g1_dex1_physical_rate_actions.py`.

Expected: all tests pass.

- [ ] **Step 5: Commit the decoder**

```bash
git add tests/test_g1_dex1_physical_rate_actions.py source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/high_level_actions.py
git commit -m "feat: decode high-level actions as physical rates"
```

### Task 2: Rename Wrist Command State To Hand-Center State

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/commands.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/low_level_observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`
- Modify: `tests/test_g1_dex1_wrist_local_command.py`
- Modify: `tests/test_g1_dex1_hier_drc_static.py`

- [ ] **Step 1: Update static tests to require semantic hand-center names**

Require these identifiers and reject the old Dex1 state identifiers:

```python
assert "left_hand_center_pose_a" in action_source
assert "right_hand_center_pose_a" in action_source
assert "default_left_hand_center_pose_a" in command_source
assert "default_right_hand_center_pose_a" in command_source
assert "left_wrist_pose_b" not in action_source
assert "right_wrist_pose_b" not in action_source
```

- [ ] **Step 2: Run the static tests and verify they fail**

Run:

```bash
pytest -q tests/test_g1_dex1_wrist_local_command.py tests/test_g1_dex1_hier_drc_static.py
```

Expected: failures on the old field names.

- [ ] **Step 3: Rename the Dex1 hierarchy consistently**

Rename command-state fields, command-term buffers, setters, defaults, visualizers, reset logic, low-level observation reads, and reward reads. Keep the frozen low-level observation ordering byte-for-byte unchanged. Use `_a` only for the posture-conditioned shoulder-anchor frame and `_w`/`_b` for world/root frames.

```python
@dataclass
class HighLevelCommandState:
    base_velocity: torch.Tensor
    posture_command: torch.Tensor
    left_hand_center_pose_a: torch.Tensor
    right_hand_center_pose_a: torch.Tensor
    left_grip: torch.Tensor
    right_grip: torch.Tensor
```

- [ ] **Step 4: Run rename-focused and frozen-interface tests**

Run:

```bash
pytest -q tests/test_g1_dex1_wrist_local_command.py tests/test_g1_dex1_hand_center_low_level_static.py tests/test_g1_dex1_hier_drc_static.py
```

Expected: all pass; low-level term ordering assertions remain unchanged.

- [ ] **Step 5: Commit the semantic rename**

```bash
git add source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc tests/test_g1_dex1_wrist_local_command.py tests/test_g1_dex1_hier_drc_static.py
git commit -m "refactor: name Dex1 commands by tracked hand center"
```

### Task 3: Pure Contact-Conditioned Motion Math

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/motion_regularization.py`
- Create: `tests/test_g1_dex1_motion_regularization.py`

- [ ] **Step 1: Write failing gate and loss tests**

Cover monotonic release, inactive-hand invariance, bimanual independence, walking-weight floor, dead-zone behavior, and bounded quality scaling:

```python
def test_inactive_hand_stays_fully_constrained():
    mask = torch.tensor([[0.0, 1.0]])
    gate = torch.tensor([[0.9, 0.9]])
    weight = comfort_weights(mask, gate)
    assert torch.allclose(weight, torch.tensor([[1.0, 0.1]]))


def test_release_gate_increases_as_target_enters_reach():
    distance = torch.tensor([[0.75, 0.60, 0.45]])
    gate = reachability_gate(distance, release_radius=0.60, release_width=0.06)
    assert gate[0, 0] < gate[0, 1] < gate[0, 2]


def test_walking_weight_never_disappears_near_target():
    mask = torch.tensor([[1.0, 0.0]])
    gate = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(walking_posture_weight(mask, gate, near_floor=0.2), torch.tensor([0.2]))


def test_normalized_deadzone_is_zero_inside_tolerance():
    value = normalized_deadzone_square(torch.tensor([0.04, 0.15]), deadzone=0.05, scale=0.10)
    assert torch.allclose(value, torch.tensor([0.0, 1.0]))


def test_quality_reward_is_capped_at_twenty_percent_of_stage_scale():
    reward = scaled_quality_reward(torch.tensor([10.0]), torch.tensor([5.0]), coefficient=0.10, loss_cap=2.0)
    assert torch.allclose(reward, torch.tensor([-2.0]))
```

- [ ] **Step 2: Verify the new tests fail**

Run `pytest -q tests/test_g1_dex1_motion_regularization.py`.

Expected: import failure because the module is absent.

- [ ] **Step 3: Implement the pure functions**

Implement functions with no IsaacLab dependency:

```python
def reachability_gate(distance: torch.Tensor, release_radius: float, release_width: float) -> torch.Tensor:
    return torch.sigmoid((release_radius - distance) / release_width)


def comfort_weights(effector_mask: torch.Tensor, reach_gate: torch.Tensor) -> torch.Tensor:
    return (1.0 - effector_mask) + effector_mask * (1.0 - reach_gate)


def walking_posture_weight(
    effector_mask: torch.Tensor, reach_gate: torch.Tensor, near_floor: float
) -> torch.Tensor:
    body_gate = torch.amax(effector_mask * reach_gate, dim=-1)
    return near_floor + (1.0 - near_floor) * (1.0 - body_gate)


def normalized_deadzone_square(value: torch.Tensor, deadzone: float, scale: float) -> torch.Tensor:
    return torch.square(torch.relu(value - deadzone) / scale)


def scaled_quality_reward(
    stage_scale: torch.Tensor,
    quality_loss: torch.Tensor,
    coefficient: float = 0.10,
    loss_cap: float = 2.0,
) -> torch.Tensor:
    return -coefficient * stage_scale.detach() * torch.clamp(quality_loss, 0.0, loss_cap)
```

Add helpers for radial workspace violation, normalized joint-limit proximity, quaternion angular distance, and mask-normalized bilateral means. Validate positive widths/scales with `ValueError`.

- [ ] **Step 4: Run the pure tests**

Run `pytest -q tests/test_g1_dex1_motion_regularization.py`.

Expected: all tests pass without launching Isaac Sim.

- [ ] **Step 5: Commit the math module**

```bash
git add source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/motion_regularization.py tests/test_g1_dex1_motion_regularization.py
git commit -m "feat: add contact-conditioned motion regularization math"
```

### Task 4: Environment Command-Rate State And Deployable Observations

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/observations.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/observations.py`
- Modify: `tests/test_g1_dex1_hier_drc_static.py`
- Modify: `tests/test_g1_dex1_door_open.py`
- Modify: `tests/test_g1_dex1_drawer.py`

- [ ] **Step 1: Add failing static assertions for buffers and actor terms**

Require the environment to retain current/previous command rates, acceleration, and jerk, and require all policy groups to include the shared execution observation:

```python
assert "self.command_rate" in env_source
assert "self.previous_command_rate" in env_source
assert "self.command_acceleration" in env_source
assert "self.previous_command_acceleration" in env_source
assert "self.command_jerk" in env_source
assert "execution = ObsTerm(func=execution_state_obs" in observation_source
```

- [ ] **Step 2: Run static tests and verify failure**

Run:

```bash
pytest -q tests/test_g1_dex1_hier_drc_static.py tests/test_g1_dex1_door_open.py tests/test_g1_dex1_drawer.py
```

Expected: failures because execution state is not wired.

- [ ] **Step 3: Store physical command derivatives in the environment**

Before decoding, shift previous buffers. Decode with `dt=self.step_dt`, then compute:

```python
self.previous_command_rate.copy_(self.command_rate)
self.previous_command_acceleration.copy_(self.command_acceleration)
self.command_state, rate_state = decode_high_level_action(
    self.last_high_level_action,
    self.command_state,
    self.action_limits,
    dt=self.step_dt,
)
self.command_rate.copy_(rate_state.as_tensor())
self.command_acceleration.copy_((self.command_rate - self.previous_command_rate) / self.step_dt)
self.command_jerk.copy_((self.command_acceleration - self.previous_command_acceleration) / self.step_dt)
```

Reset all derivative buffers to zero for reset environments. Resolve and cache upper-body joint IDs plus left/right shoulder and torso body IDs after simulator initialization.

- [ ] **Step 4: Add `execution_state_obs`**

Return only deployable values in this order:

```text
base linear velocity b                        3
upper-body joint position relative default  17
left/right current hand-center pose b        18
left/right pose error                        12
posture tracking error                        2
decoded command rate                         17
total                                        69
```

Represent each current orientation with 6D rotation and each orientation error with a 3D rotation vector. Add uniform sensor noise to base velocity and joint positions; leave deterministic command-derived terms uncorrupted. Add the same ObsTerm to PnP, Door, and Drawer policy/critic groups; Box and Cart inherit the PnP group.

- [ ] **Step 5: Run observation wiring tests**

Run the static test command from Step 2.

Expected: all pass.

- [ ] **Step 6: Commit execution state**

```bash
git add \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/observations.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/observations.py \
  tests/test_g1_dex1_hier_drc_static.py \
  tests/test_g1_dex1_door_open.py \
  tests/test_g1_dex1_drawer.py
git commit -m "feat: expose deployable high-level execution state"
```

### Task 5: Shared Environment Motion Loss Extraction

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/motion_regularization.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env_cfg.py`
- Modify: `tests/test_g1_dex1_motion_regularization.py`
- Modify: `tests/test_g1_dex1_reward_logic.py`

- [ ] **Step 1: Add failing reward-structure tests**

Require a single shared reward term, soft workspace behavior, and no phase/task-name gate:

```python
assert "def contact_conditioned_motion_reward" in reward_source
assert "motion_quality = RewTerm(" in reward_source
assert "env.contact_label.effector_mask" in reward_source
assert "env.W_app" in reward_source
assert "env.W_couple" in reward_source
assert "env.W_manip" in reward_source
assert "motion_keyframe_index" not in motion_source
assert "contact_mode" not in motion_source
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/test_g1_dex1_motion_regularization.py tests/test_g1_dex1_reward_logic.py
```

Expected: missing reward and configuration failures.

- [ ] **Step 3: Add explicit regularization configuration**

Add fields to `G1Dex1HierDrcEnvCfg`:

```python
motion_release_radius: float = 0.60
motion_release_width: float = 0.06
motion_near_posture_floor: float = 0.20
motion_workspace_radius_range: tuple[float, float] = (0.20, 0.58)
motion_position_deadzone: float = 0.05
motion_position_scale: float = 0.10
motion_orientation_deadzone: float = 0.20
motion_orientation_scale: float = 0.50
motion_root_height_deadzone: float = 0.03
motion_torso_pitch_deadzone: float = 0.08
motion_quality_coefficient: float = 0.10
motion_quality_loss_cap: float = 2.0
```

- [ ] **Step 4: Gather the shared motion terms in the reward**

Compute both shoulder-to-target distances from live body positions and `contact_label.target_region`, then calculate reach gates and comfort weights. Build normalized losses for:

```text
0.20 hand position tracking
0.10 hand orientation tracking
0.10 posture tracking
0.10 radial workspace violation
0.10 root/pelvis tilt safety
0.05 upper-body joint-limit proximity
0.10 hand command comfort
0.05 arm joint comfort
0.10 distance-conditioned walking posture
0.04 command speed
0.02 arm joint speed
0.025 command acceleration
0.015 command jerk
```

Use comfortable hand commands from the command config defaults, and comfortable waist/arm joint poses from `robot.data.default_joint_pos`. Weight command/arm comfort and arm speed per hand with `w_comfort`; weight walking height/pitch and waist/torso terms with `w_walk`. Keep tracking, root tilt, joint limits, acceleration, and jerk always active.

Compute:

```python
stage_scale = (
    env.W_app * approach_scale
    + env.W_couple * couple_scale
    + env.W_manip * manip_scale
)
reward = scaled_quality_reward(
    stage_scale,
    quality_loss,
    coefficient=env.cfg.motion_quality_coefficient,
    loss_cap=env.cfg.motion_quality_loss_cap,
)
```

Register it with `weight=1.0` because the function already returns a negative reward.

- [ ] **Step 5: Log every raw contribution**

Write TensorBoard keys under `Motion/` for reach gates, comfort weights, walking weight, all thirteen normalized losses, total quality loss, scaled reward, command rate/acceleration/jerk norms, and active/inactive hand summaries. Logs must be detached means.

- [ ] **Step 6: Run motion and reward tests**

Run the Step 2 command.

Expected: all pass and the existing PnP task reward weights remain unchanged.

- [ ] **Step 7: Commit shared motion rewards**

```bash
git add source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc tests/test_g1_dex1_motion_regularization.py tests/test_g1_dex1_reward_logic.py
git commit -m "feat: regularize motion from contact reachability"
```

### Task 6: Wire The Shared Reward Across Dex1 Tasks

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/rewards.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/rewards.py`
- Modify: `tests/test_g1_dex1_box_carry.py`
- Modify: `tests/test_g1_dex1_cart_push.py`
- Modify: `tests/test_g1_dex1_door_open.py`
- Modify: `tests/test_g1_dex1_drawer.py`

- [ ] **Step 1: Add failing assertions for common reward wiring**

Each reward config must import `contact_conditioned_motion_reward` and register exactly one `motion_quality` term. Box and Cart use manipulation scale `200.0`; Door and Drawer use `100.0`.

- [ ] **Step 2: Run task static tests and verify failure**

Run:

```bash
pytest -q tests/test_g1_dex1_box_carry.py tests/test_g1_dex1_cart_push.py tests/test_g1_dex1_door_open.py tests/test_g1_dex1_drawer.py
```

- [ ] **Step 3: Register the common reward in every task**

Use this structure, changing only `manip_scale` to match each task's existing DRC function:

```python
motion_quality = RewTerm(
    func=contact_conditioned_motion_reward,
    weight=1.0,
    params={"approach_scale": 2.0, "couple_scale": 20.0, "manip_scale": 200.0},
)
```

Do not add object-type, phase, keyframe, grasp-mode, or task-name conditions.

- [ ] **Step 4: Run all four task tests**

Expected: all pass.

- [ ] **Step 5: Commit shared task wiring**

```bash
git add \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_box_carry_hier_drc/mdp/rewards.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/rewards.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/rewards.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_drawer_hier_drc/mdp/rewards.py \
  tests/test_g1_dex1_box_carry.py \
  tests/test_g1_dex1_cart_push.py \
  tests/test_g1_dex1_door_open.py \
  tests/test_g1_dex1_drawer.py
git commit -m "feat: share motion quality reward across interaction tasks"
```

### Task 7: Full Static And Local IsaacLab Verification

**Files:**
- Modify only if verification reveals a defect in files already listed above.

- [ ] **Step 1: Run formatting and static tests**

Run:

```bash
python -m compileall -q source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc
pytest -q tests/test_g1_dex1_physical_rate_actions.py tests/test_g1_dex1_motion_regularization.py tests/test_g1_dex1_hier_drc_static.py tests/test_g1_dex1_reward_logic.py tests/test_g1_dex1_hand_center_low_level_static.py tests/test_g1_dex1_box_carry.py tests/test_g1_dex1_cart_push.py tests/test_g1_dex1_door_open.py tests/test_g1_dex1_drawer.py
git diff --check
```

Expected: compile succeeds, all tests pass, and `git diff --check` prints nothing.

- [ ] **Step 2: Run a 16-environment headless smoke training**

Run with the local IsaacLab Python environment:

```bash
python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-G1-Dex1-HierDrc-v0 \
  --num_envs=16 \
  --headless \
  --max_iterations=2 \
  --seed=42 \
  --run_name=pnp_contact_motion_smoke \
  env.low_level_policy_path=logs/exported_policies/g1_dex1_handcenter_stickfigure_m49999.pt
```

Expected: two PPO iterations complete; observations, rewards, policy std, and command derivative logs remain finite.

- [ ] **Step 3: Inspect smoke metrics**

Confirm the console or event log contains finite values for:

```text
Motion/left_reach_gate_mean
Motion/right_reach_gate_mean
Motion/inactive_comfort_weight_mean
Motion/walking_weight_mean
Motion/quality_loss_mean
Motion/scaled_reward_mean
Motion/command_rate_norm
Motion/command_acceleration_norm
Motion/command_jerk_norm
```

- [ ] **Step 4: Commit any verification fixes**

If no fix is required, do not create an empty commit. Otherwise commit only the corrected files with `fix: stabilize contact-conditioned motion rollout`.

### Task 8: Synchronize JD And Start The PnP Experiment

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify the drawer process remains stopped**

Run:

```bash
ssh jd_5090 'pgrep -af "Drawer-HierDrc|g1_dex1_drawer" || true'
```

Expected: no drawer training process.

- [ ] **Step 2: Synchronize the reviewed local commit to JD**

Push the local implementation branch, then in `/home/zju/code/hbc_lab` fetch and fast-forward to the exact implementation commit. Refuse to overwrite uncommitted JD changes; inspect and reconcile them first.

- [ ] **Step 3: Run focused tests on JD**

Run in `/home/zju/code/hbc_lab` using `/home/zju/miniconda3/envs/robot_lab/bin/python`:

```bash
/home/zju/miniconda3/envs/robot_lab/bin/python -m pytest -q \
  tests/test_g1_dex1_physical_rate_actions.py \
  tests/test_g1_dex1_motion_regularization.py \
  tests/test_g1_dex1_hier_drc_static.py \
  tests/test_g1_dex1_reward_logic.py
```

Expected: all pass.

- [ ] **Step 4: Verify the frozen HBC checkpoint exists**

Run:

```bash
test -f logs/exported_policies/g1_dex1_handcenter_stickfigure_m49999.pt
```

Expected: exit code zero.

- [ ] **Step 5: Start a fresh 50k PnP run in tmux**

Create session `hbc_pnp_contact_motion_v1_50k` in `/home/zju/code/hbc_lab`:

```bash
tmux new-session -d -s hbc_pnp_contact_motion_v1_50k -c /home/zju/code/hbc_lab \
  /home/zju/miniconda3/envs/robot_lab/bin/python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-G1-Dex1-HierDrc-v0 \
  --num_envs=4096 \
  --headless \
  --max_iterations=50000 \
  --seed=42 \
  --run_name=pnp_contact_conditioned_motion_v1_stickfigure_50k \
  env.low_level_policy_path=logs/exported_policies/g1_dex1_handcenter_stickfigure_m49999.pt
```

- [ ] **Step 6: Verify the first iterations**

Poll the tmux pane until at least iteration 2. Confirm collection and learning complete, all `Motion/` metrics are finite, action std remains nonnegative, and no joint-velocity/nonfinite termination spike appears. Report the tmux session, log directory, implementation commit, and first metrics to the user.

---

## Plan Self-Review

- Spec coverage: physical-rate commands, semantic hand-center naming, deployable observation, continuous target-conditioned arm/body constraints, always-on safety, soft workspace behavior, DRC-relative scale, cross-task wiring, PnP validation, and JD deployment are each assigned to a task.
- Placeholder scan: no TBD, TODO, deferred implementation, or unspecified test step remains.
- Type consistency: `HighLevelCommandState`, `HighLevelCommandRateState`, `left_hand_center_pose_a`, `right_hand_center_pose_a`, `command_rate`, `command_acceleration`, and `command_jerk` use the same names throughout the plan.
