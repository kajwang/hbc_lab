# Dex1 Wrist-Local Hand-Center Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate Dex1 low-level hand-center commands from posture-aware hand-base positions and physical wrist-joint rotations, then use the identical anchor transform during hierarchical deployment.

**Architecture:** Add a focused tensor-transform module containing the posture anchor, wrist-chain quaternion, and hand-base-to-hand-center operations. Extend the existing spherical command term with an opt-in wrist-chain mode so other G1 tasks retain their current behavior, configure only the Dex1 task to use it, and replace the duplicated HRC anchor math with the shared helper.

**Tech Stack:** Python, PyTorch tensor APIs, IsaacLab quaternion utilities, pytest static contract tests.

---

### Task 1: Lock the shared transform contract with focused tests

**Files:**
- Create: `tests/test_g1_dex1_wrist_local_command.py`
- Test: `tests/test_g1_dex1_wrist_local_command.py`

- [ ] **Step 1: Write failing contract tests**

Create tests that assert:

```python
def test_shared_pose_transform_module_defines_posture_anchor_and_wrist_chain():
    source = _read(TRANSFORM_PATH)
    assert "def posture_anchor_pose_w(" in source
    assert "def compose_wrist_chain_quat(" in source
    assert "def hand_base_to_hand_center_pose(" in source
    assert "quat_mul(quat_mul(quat_mul(roll_quat, pitch_quat), yaw_quat), fixed_quat)" in source


def test_dex1_command_uses_hand_base_sampling_and_physical_wrist_limits():
    source = _read(DEX1_CONFIG_PATH)
    assert "WRIST_ROLL_LIMIT = math.radians(100.0)" in source
    assert "WRIST_PITCH_YAW_LIMIT = math.radians(80.0)" in source
    assert source.count("l=(0.20, 0.38)") == 2
    assert source.count("l=(0.12, 0.58)") == 2
    assert source.count('orientation_mode="wrist_chain"') == 2
    assert source.count('anchor_height_command_name="posture_command"') == 2
    assert source.count('anchor_pitch_command_name="posture_command"') == 2


def test_hier_builder_reuses_shared_posture_anchor():
    source = _read(HIER_OBS_PATH)
    assert "from hbc_lab.tasks.locomotion.mdp.pose_transforms import posture_anchor_pose_w" in source
    assert "return posture_anchor_pose_w(" in source
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest -q tests/test_g1_dex1_wrist_local_command.py
```

Expected: failures because the shared module and new configuration are not implemented.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_g1_dex1_wrist_local_command.py
git commit -m "test: define Dex1 wrist-local command contract"
```

### Task 2: Implement shared posture and wrist pose transforms

**Files:**
- Create: `source/hbc_lab/hbc_lab/tasks/locomotion/mdp/pose_transforms.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/locomotion/mdp/__init__.py`
- Test: `tests/test_g1_dex1_wrist_local_command.py`

- [ ] **Step 1: Implement the posture-aware anchor**

Add a pure tensor function with this interface:

```python
def posture_anchor_pose_w(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    shoulder_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    posture_command: torch.Tensor,
    anchor_height_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
```

It must construct `root_yaw * pitch_command`, set commanded root height relative to each environment origin, preserve the shoulder lateral offset in the root-yaw frame, and rotate the configured vertical root-to-shoulder offset with commanded pitch.

- [ ] **Step 2: Implement physical wrist-chain composition**

Add:

```python
def compose_wrist_chain_quat(wrist_angles: torch.Tensor, fixed_quat: torch.Tensor) -> torch.Tensor:
    roll_quat = quat_from_euler_xyz(wrist_angles[:, 0], zeros, zeros)
    pitch_quat = quat_from_euler_xyz(zeros, wrist_angles[:, 1], zeros)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, wrist_angles[:, 2])
    return quat_mul(quat_mul(quat_mul(roll_quat, pitch_quat), yaw_quat), fixed_quat)
```

The order matches the asset chain `wrist_roll(X) -> wrist_pitch(Y) -> wrist_yaw(Z) -> fixed hand palm`.

- [ ] **Step 3: Implement hand-base to hand-center conversion**

Add:

```python
def hand_base_to_hand_center_pose(
    hand_base_pos: torch.Tensor,
    hand_base_quat: torch.Tensor,
    hand_center_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hand_base_pos + quat_apply(hand_base_quat, hand_center_offset), hand_base_quat
```

Export the helpers through the locomotion `mdp` package.

- [ ] **Step 4: Run the focused tests**

```bash
pytest -q tests/test_g1_dex1_wrist_local_command.py
```

Expected: transform source-contract assertions pass; command-configuration assertions remain red until Task 3.

- [ ] **Step 5: Commit the shared transform**

```bash
git add source/hbc_lab/hbc_lab/tasks/locomotion/mdp/pose_transforms.py source/hbc_lab/hbc_lab/tasks/locomotion/mdp/__init__.py
git commit -m "feat: add shared Dex1 pose transforms"
```

### Task 3: Generate Dex1 hand-center commands from hand-base and wrist samples

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/locomotion/mdp/commands/spherical_pose_command.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof/whole_body_spherical_posture_dex1_hand_center_env_cfg.py`
- Modify: `tests/test_g1_dex1_hand_center_low_level_static.py`
- Test: `tests/test_g1_dex1_wrist_local_command.py`

- [ ] **Step 1: Add opt-in wrist-chain command configuration**

Extend `SphericalLevelPoseCommandCfg` with per-command fixed-palm quaternion and hand-center offset fields. In `orientation_mode="wrist_chain"`, sample the existing range fields as physical `(wrist_roll, wrist_pitch, wrist_yaw)`, compose the chain quaternion, retain the spherical Cartesian point as the hand-base origin, and convert it to the final 7D hand-center command.

- [ ] **Step 2: Preserve existing command behavior for other tasks**

Keep `local_delta` and `azimuth` branches unchanged. Only the wrist-chain branch uses the new offset and fixed transform, so the non-Dex1 locomotion tasks do not change command semantics.

- [ ] **Step 3: Configure posture-aware Dex1 commands**

For both hands set:

```python
anchor_height_command_name="posture_command"
anchor_height_command_index=0
anchor_height_offset=0.43
anchor_pitch_command_name="posture_command"
anchor_pitch_command_index=1
orientation_mode="wrist_chain"
hand_center_offset=(0.0, 0.09734, 0.0142)
```

Use the asset fixed-palm quaternions independently:

```python
left_fixed_palm_quat=(0.70710677, 0.0, 0.0, -0.70710677)
right_fixed_palm_quat=(0.7073882, 0.0, 0.0, -0.7068252)
```

Set initial radii to `[0.20, 0.38]` m, limit radii to `[0.12, 0.58]` m, roll limit to `radians(100)`, and pitch/yaw limits to `radians(80)`.

- [ ] **Step 4: Keep the orientation curriculum and play override coherent**

Retain the existing orientation curriculum mechanism but update names/comments so `roll`, `ee_pitch`, and `yaw` now explicitly mean wrist-joint samples. Keep play as a small manual command block rather than automatically forcing full joint limits.

- [ ] **Step 5: Run low-level command tests**

```bash
pytest -q tests/test_g1_dex1_wrist_local_command.py tests/test_g1_dex1_hand_center_low_level_static.py tests/test_g1_spherical_whole_body_static.py
```

Expected: all selected tests pass and generic spherical command tests still confirm backward-compatible branches.

- [ ] **Step 6: Commit the sampler change**

```bash
git add source/hbc_lab/hbc_lab/tasks/locomotion/mdp/commands/spherical_pose_command.py source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof/whole_body_spherical_posture_dex1_hand_center_env_cfg.py tests/test_g1_dex1_hand_center_low_level_static.py
git commit -m "feat: sample Dex1 commands through wrist kinematics"
```

### Task 4: Align hierarchical deployment with the low-level anchor

**Files:**
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/low_level_observations.py`
- Modify: `tests/test_g1_dex1_hier_drc_static.py`
- Test: `tests/test_g1_dex1_wrist_local_command.py`

- [ ] **Step 1: Replace duplicated HRC anchor math**

Import `posture_anchor_pose_w` and call it from `_anchor_pose_w` using the robot root pose, selected shoulder position, environment origins, high-level posture command, and `0.43` m offset. Do not change observation order, term dimensions, or 7D hand-center command representation.

- [ ] **Step 2: Update the static HRC contract test**

Assert that the shared helper is imported and used, while the hand-center FrameTransformer remains the source of current EE pose.

- [ ] **Step 3: Run focused and regression tests**

```bash
pytest -q tests/test_g1_dex1_wrist_local_command.py tests/test_g1_dex1_hier_drc_static.py tests/test_g1_dex1_hand_center_low_level_static.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Compile changed Python modules**

```bash
python -m compileall -q \
  source/hbc_lab/hbc_lab/tasks/locomotion/mdp/pose_transforms.py \
  source/hbc_lab/hbc_lab/tasks/locomotion/mdp/commands/spherical_pose_command.py \
  source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof/whole_body_spherical_posture_dex1_hand_center_env_cfg.py \
  source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/low_level_observations.py
```

Expected: exit status 0.

- [ ] **Step 5: Commit HRC alignment**

```bash
git add source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/low_level_observations.py tests/test_g1_dex1_hier_drc_static.py tests/test_g1_dex1_wrist_local_command.py
git commit -m "fix: align HRC hand commands with low-level frames"
```

### Task 5: Final verification and training handoff

**Files:**
- Verify only; no unrelated source edits.

- [ ] **Step 1: Run the complete static test suite**

```bash
pytest -q
```

Expected: all repository tests pass.

- [ ] **Step 2: Inspect the final diff and worktree state**

```bash
git diff 5be37b9..HEAD --check
git status --short
```

Expected: no whitespace errors; the pre-existing cart configuration modification remains uncommitted and untouched.

- [ ] **Step 3: Report simulator validation boundary**

Document that the new command distribution requires a fresh low-level run and export before HRC training. If an Isaac Sim launch is not available in the current shell, explicitly report that the simulator smoke test remains for the user’s training host rather than claiming it ran.
