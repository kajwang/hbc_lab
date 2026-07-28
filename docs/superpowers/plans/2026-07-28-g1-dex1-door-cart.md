# G1 Dex1 Door and Cart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trainable G1 Dex1 hierarchical door-opening and bimanual cart-pushing tasks using locally vendored articulated assets.

**Architecture:** Create two thin task subclasses over the existing G1 Dex1 hierarchical environment. Each task owns its articulated state, contact targets, progress, rewards, reset logic, and observations while reusing the current 19-dimensional high-level action and frozen low-level hand-center tracking interface.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab manager-based RL, USD, Gymnasium, RSL-RL, pytest.

---

### Task 1: Vendor and Configure Articulated Assets

**Files:**
- Create: `source/hbc_lab/hbc_lab/assets/articulated_objects.py`
- Create: `source/hbc_lab/hbc_lab/assets/models/articulated/door/door_0_bot.usd`
- Create: `source/hbc_lab/hbc_lab/assets/models/articulated/cart/instance_turn_R_10kg.usd`
- Modify: `pyproject.toml`
- Test: `tests/test_g1_dex1_articulated_assets.py`

- [ ] **Step 1: Write failing asset configuration tests**

Create tests that assert:

```python
assert DOOR_USD.exists()
assert CART_USD.exists()
assert "joint_1" in source
assert "joint_2" in source
assert "RL_turn_joint" in source
assert "assets/models/articulated/**/*.usd" in pyproject_source
```

- [ ] **Step 2: Run the tests and confirm the missing module/assets fail**

Run:

```bash
pytest -q tests/test_g1_dex1_articulated_assets.py
```

Expected: failures naming the missing articulated asset files and configuration.

- [ ] **Step 3: Copy only the two self-contained USD files**

Copy:

```text
robot_lab/.../models/doors/door_0_bot.usd
    -> hbc_lab/assets/models/articulated/door/door_0_bot.usd
robot_lab/.../models/carts/cart_2/instance_turn_R_10kg.usd
    -> hbc_lab/assets/models/articulated/cart/instance_turn_R_10kg.usd
```

- [ ] **Step 4: Define the IsaacLab asset configurations**

Implement:

```python
DOOR_CFG: ArticulationCfg
DOOR_FRAME_CFG: FrameTransformerCfg
CART_CFG: ArticulationCfg
CART_FRAME_CFG: FrameTransformerCfg
```

Preserve the source USD scale, initial rotations, passive actuator parameters,
door hinge/handle actuator parameters, and frame offsets.

- [ ] **Step 5: Include articulated USD files as package data**

Add an explicit setuptools package-data glob for:

```text
assets/models/articulated/**/*.usd
```

- [ ] **Step 6: Run the asset tests**

Run:

```bash
pytest -q tests/test_g1_dex1_articulated_assets.py
```

Expected: all asset tests pass.

### Task 2: Implement Pure Door and Cart Progress Math

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/progress.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/progress.py`
- Test: `tests/test_g1_dex1_articulated_progress.py`

- [ ] **Step 1: Write failing tests for door progress**

Cover:

```python
progress = compute_door_manipulation_progress(
    handle_angle=torch.tensor([0.0, 0.25, 0.5]),
    hinge_angle=torch.tensor([0.0, math.radians(30), math.radians(60)]),
    latch_threshold=0.5,
    hinge_target=math.radians(60),
)
```

Assert normalized handle/hinge progress, a closed hinge gate at zero handle
progress, an open gate at the latch threshold, and the specified reward formula.

- [ ] **Step 2: Write failing tests for bimanual cart progress**

Cover:

```python
couple = bimanual_grasp_confidence(
    torch.tensor([1.0, 0.25, 0.0]),
    torch.tensor([1.0, 1.0, 1.0]),
)
```

Assert results `[1.0, 0.5, 0.0]`. Also test local handle target transforms
under identity and 90-degree yaw, and local forward/lateral goal transforms.

- [ ] **Step 3: Run the tests and confirm missing functions fail**

Run:

```bash
pytest -q tests/test_g1_dex1_articulated_progress.py
```

- [ ] **Step 4: Implement minimal pure-PyTorch progress functions**

Door module exports:

```python
compute_door_manipulation_progress(...)
```

Cart module exports:

```python
bimanual_grasp_confidence(...)
compute_handle_targets(...)
compute_cart_goal(...)
compute_transport_progress(...)
```

- [ ] **Step 5: Run progress tests**

Run:

```bash
pytest -q tests/test_g1_dex1_articulated_progress.py
```

Expected: all progress tests pass.

### Task 3: Build the Door Scene and Environment

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/scenes.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/events.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/observations.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/mdp/rewards.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/door_env_cfg.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/flat_env_cfg.py`
- Test: `tests/test_g1_dex1_door_open.py`

- [ ] **Step 1: Write failing door task tests**

Assert the source contract:

```python
assert "fixed_effector_mask = (0.0, 1.0)" in cfg_source
assert "ContactMode.GRASP" in cfg_source
assert "right_hand_Link1_3" in scene_source
assert "right_hand_Link2_3" in scene_source
assert "door/link_2" in scene_source
assert "_simulate_door_latch" in env_source
assert "handle_angle" in observation_source
assert "hinge_angle" in observation_source
```

- [ ] **Step 2: Run door tests and confirm missing task files fail**

Run:

```bash
pytest -q tests/test_g1_dex1_door_open.py
```

- [ ] **Step 3: Implement the scene and reset events**

Replace the PnP rigid object and platforms with `DOOR_CFG` and
`DOOR_FRAME_CFG`. Add right-pad filtered contact sensors and left inactive-hand
diagnostic sensors. Reset both door joints to zero and place the door nominally
2.0 m ahead.

- [ ] **Step 4: Implement door observations**

Extend the common policy and critic groups with:

```python
door_articulation_state(env) -> Tensor[N, 2]
```

Normalize handle and hinge angles by their task thresholds.

- [ ] **Step 5: Implement door progress, latch effort, rewards, and success**

The environment must:

```python
self.contact_label.set_effector_mask(env_ids, [[0.0, 1.0]])
self.contact_label.set_target_region(env_ids, right_handle_targets)
self._simulate_door_latch()
self.c_couple = update_ema(self.c_couple, right_grasp, alpha=0.2)
```

Use the approved manipulation reward and require 60 degrees plus sustained
right-hand grasp for success.

- [ ] **Step 6: Run door tests**

Run:

```bash
pytest -q tests/test_g1_dex1_door_open.py tests/test_g1_dex1_articulated_progress.py
```

Expected: all door tests pass.

### Task 4: Build the Cart Scene and Environment

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/scenes.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/events.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/mdp/rewards.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/config/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/config/cart_env.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/config/cart_env_cfg.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/config/flat_env_cfg.py`
- Test: `tests/test_g1_dex1_cart_push.py`

- [ ] **Step 1: Write failing cart task tests**

Assert:

```python
assert "fixed_effector_mask = (1.0, 1.0)" in cfg_source
assert "ContactMode.GRASP" in cfg_source
assert "Link1_3" in scene_source
assert "Link2_3" in scene_source
assert "cart/handle" in scene_source
assert "bimanual_grasp_confidence" in env_source
assert "0.3 + progress.transport_progress" in reward_source
assert "object_mass_curriculum_enabled: bool = False" in cfg_source
```

- [ ] **Step 2: Run cart tests and confirm missing task files fail**

Run:

```bash
pytest -q tests/test_g1_dex1_cart_push.py
```

- [ ] **Step 3: Implement cart scene and reset events**

Replace the PnP rigid object and platforms with `CART_CFG` and
`CART_FRAME_CFG`. Add four per-pad handle contact sensors. Reset all cart joints,
sample the cart pose, and transform the approved local goal displacement into
world coordinates.

- [ ] **Step 4: Implement live bimanual target regions and contact progress**

Compute live handle target points at local lateral offsets `+0.18` and `-0.18`
m. Compute each hand's Dex1 grasp independently and combine them using the
geometric mean.

- [ ] **Step 5: Implement DRC, transport reward, diagnostics, and success**

Use mean hand distance for approach, bimanual grasp EMA for `W_manip`, planar
handle-midpoint transport progress, and a 0.25 m success radius sustained for 25
steps.

- [ ] **Step 6: Run cart tests**

Run:

```bash
pytest -q tests/test_g1_dex1_cart_push.py tests/test_g1_dex1_articulated_progress.py
```

Expected: all cart tests pass.

### Task 5: Add PPO Configs, Registration, and Launch Entries

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/agents/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_door_open_hier_drc/config/agents/rsl_rl_ppo_cfg.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/config/agents/__init__.py`
- Create: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_cart_push_hier_drc/config/agents/rsl_rl_ppo_cfg.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/__init__.py`
- Modify: `.vscode/launch.json`
- Test: `tests/test_g1_dex1_articulated_registration.py`

- [ ] **Step 1: Write failing registration tests**

Check Train and Play IDs, experiment names, task imports, and four launch entry
names.

- [ ] **Step 2: Run registration tests and confirm failure**

Run:

```bash
pytest -q tests/test_g1_dex1_articulated_registration.py
```

- [ ] **Step 3: Add registrations and PPO configurations**

Register:

```text
HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-v0
HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-Play-v0
HBC-Isaac-G1-Dex1-CartPush-HierDrc-v0
HBC-Isaac-G1-Dex1-CartPush-HierDrc-Play-v0
```

Use task-specific experiment names and the existing G1 Dex1 PPO hyperparameters.

- [ ] **Step 4: Add launch configurations**

Add train/play configurations for each task with the existing exported
hand-center low-level policy path.

- [ ] **Step 5: Run registration tests**

Run:

```bash
pytest -q tests/test_g1_dex1_articulated_registration.py
```

Expected: all registration tests pass.

### Task 6: Regression and IsaacLab Smoke Verification

**Files:**
- Modify only if smoke verification finds a concrete defect.

- [ ] **Step 1: Run focused and existing G1 Dex1 tests**

Run:

```bash
pytest -q \
  tests/test_g1_dex1_articulated_assets.py \
  tests/test_g1_dex1_articulated_progress.py \
  tests/test_g1_dex1_door_open.py \
  tests/test_g1_dex1_cart_push.py \
  tests/test_g1_dex1_articulated_registration.py \
  tests/test_g1_dex1_contact_labels.py \
  tests/test_g1_dex1_hier_drc_static.py \
  tests/test_g1_dex1_reward_logic.py
```

- [ ] **Step 2: Run the full pure-Python test suite**

Run:

```bash
pytest -q
```

Expected: no regressions.

- [ ] **Step 3: Validate Python syntax**

Run:

```bash
python -m compileall -q source/hbc_lab/hbc_lab tests
```

- [ ] **Step 4: Run small IsaacLab smoke launches**

Launch each task with a small environment count and missing low-level policy
allowed:

```bash
python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-v0 \
  --num_envs=2 --max_iterations=1 \
  env.allow_missing_low_level_policy=True

python scripts/rsl_rl/train.py \
  --task=HBC-Isaac-G1-Dex1-CartPush-HierDrc-v0 \
  --num_envs=2 --max_iterations=1 \
  env.allow_missing_low_level_policy=True
```

Expected: both tasks resolve assets, articulation joints, frames, observations,
and 19-dimensional actions, then complete one iteration without exceptions.

- [ ] **Step 5: Inspect the final diff**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Confirm only the approved asset, task, registration, launch, test, and
documentation files changed.
