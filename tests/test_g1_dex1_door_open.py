from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
TASK_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex1_door_open_hier_drc"
MDP_ROOT = TASK_ROOT / "mdp"
CONFIG_ROOT = TASK_ROOT / "config"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing door task file: {path}"
    return path.read_text()


def test_door_uses_shared_contact_conditioned_motion_quality():
    source = _read(MDP_ROOT / "rewards.py")

    assert "contact_conditioned_motion_reward" in source
    assert source.count("motion_quality = RewTerm(") == 1
    assert '"manip_scale": 100.0' in source


def test_door_scene_replaces_pnp_object_with_articulation_and_handle_contacts():
    source = _read(MDP_ROOT / "scenes.py")

    assert "DOOR_CFG" in source
    assert "DOOR_FRAME_CFG" in source
    assert "object_init_platform = None" in source
    assert "object_target_platform = None" in source
    assert "{ENV_REGEX_NS}/door/link_2" in source
    assert "right_hand_Link1_3" in source
    assert "right_hand_Link2_3" in source
    assert "left_hand_Link1_3" in source
    assert "left_hand_Link2_3" in source


def test_door_contact_label_fixes_right_hand_grasp():
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "fixed_effector_mask = (0.0, 1.0)" in cfg_source
    assert "ContactMode.GRASP" in cfg_source
    assert "self.contact_label.set_target_region_pose" in env_source
    assert "self.cfg.commands.high_level.fixed_effector_mask" in env_source
    assert "self.active_hand[env_ids] = torch.argmax(mask, dim=-1)" in env_source
    assert "self.active_hand[env_ids] = RIGHT_HAND" not in env_source
    assert "self.inactive_hand_contact" in env_source
    assert 'self._step_link_contact["right_Link1_2"]' in env_source
    assert "torch.where(self.active_hand == 0, right_contact, left_contact)" in env_source


def test_door_observations_reuse_shared_contact_label_interface():
    source = _read(MDP_ROOT / "observations.py")

    assert "interaction_motion_obs" in source
    assert "G1Dex1DoorOpenObservationsCfg" in source
    assert "gather_keyframe(env.motion_target_pos_w" in source
    assert "gather_keyframe(env.motion_target_quat_w" in source
    assert "motion_keyframe_index.to" not in source
    assert "motion_guide_valid.to" not in source
    assert "door_articulation_state" not in source
    assert "handle_angle" not in source


def test_door_actor_uses_shared_deployable_execution_state():
    source = _read(MDP_ROOT / "observations.py")

    assert "execution_state_obs" in source
    assert "execution = ObsTerm(func=execution_state_obs" in source


def test_door_env_applies_latch_and_uses_grasp_for_drc():
    source = _read(CONFIG_ROOT / "door_env.py")

    assert "def _simulate_door_latch" in source
    assert "door_initial_pose_pending" in source
    assert "door_target_update_delay" in source
    assert "door_latch_stiffness" in source
    assert "door_latch_damping" in source
    assert "self._simulate_door_latch()" in source
    assert "compute_active_hand_grasp_progress" in source
    assert "self.c_couple = update_ema(self.c_couple, progress.grasp, alpha=0.2)" in source
    assert "compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)" in source
    assert "def _update_object_mass_curriculum" in source


def test_door_reward_uses_generic_sparse_pose_progress():
    reward_source = _read(MDP_ROOT / "rewards.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "generic_pose_manip_reward" in reward_source
    assert "env.motion_progress" in reward_source
    assert "approach_root_object_facing_reward" in reward_source
    assert "env.W_app * root_object_facing_reward(env)" in reward_source
    assert "unlock_progress" not in reward_source
    assert "open_progress" not in reward_source
    assert "handle_angle" not in reward_source
    assert "hinge_angle" not in reward_source
    assert "+ 0.3" not in reward_source
    assert "inactive_hand_contact_penalty" in reward_source
    assert "object_fall" not in reward_source
    assert "hier_drc_reward" in reward_source
    assert "_update_motion_guide" in env_source
    assert "compute_cumulative_keyframe_progress" in env_source
    assert "motion_phase_initial_error" in env_source
    assert "motion_previous_error" not in env_source
    assert "self.d_goal = torch.norm(self._door_handle_pos_w() - self.object_target_pos_w, dim=-1)" in env_source


def test_door_unlock_keyframe_rotates_position_about_handle_pivot():
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "compose_local_axis_pose" in env_source
    assert "def _door_handle_pivot_pos_w" in env_source
    assert "self.motion_target_pos_w[env_ids, 0] = unlock_pos_w" in env_source
    assert "self.motion_target_pos_w[env_ids, 1] = open_pos_w" in env_source


def test_door_reset_and_success_use_articulation_state():
    events_source = _read(MDP_ROOT / "events.py")
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "def set_default_root_state_from_current_pose" in events_source
    assert "asset.data.root_link_pose_w[env_ids].clone()" in events_source
    assert "set_default_object_root_state = EventTerm" in events_source
    assert "func=mdp.reset_root_state_uniform" in events_source
    assert "reset_object_joints = EventTerm" in events_source
    assert "func=mdp.reset_joints_by_offset" in events_source
    assert "reset_door_articulation" not in events_source
    assert '"x": (2.0, 2.0)' in events_source
    assert "door_hinge_target: float = math.radians(60.0)" in cfg_source
    assert "motion_unlock_angle: float = math.pi / 2.0" in cfg_source
    assert "door_latch_handle_threshold: float = math.radians(82.0)" in cfg_source
    assert "self.hinge_angle >= self.cfg.door_hinge_target" in env_source
    assert "self.c_couple > self.cfg.success_couple_threshold" in env_source


def test_door_task_preserves_high_level_and_low_level_interfaces():
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "class G1Dex1DoorOpenEnv(G1Dex1HierDrcEnv)" in env_source
    assert "class G1Dex1DoorOpenEnvCfg(G1Dex1HierDrcEnvCfg)" in cfg_source
    assert "action_dim" not in cfg_source
    assert "low_level_policy" not in env_source
    assert "object_mass_curriculum_enabled: bool = False" in cfg_source


def test_door_visualizes_current_handle_and_both_keyframes_with_shared_world_frames():
    base_env_source = _read(
        HBC_ROOT / "tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py"
    )
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")

    assert '"current"' in base_env_source
    assert '"keyframe_0"' in base_env_source
    assert '"keyframe_1"' in base_env_source
    assert "flatten_motion_frame_markers" in base_env_source
    assert "self.motion_current_pos_w" in base_env_source
    assert "self.motion_target_pos_w" in base_env_source
    assert "self.motion_frame_visualizer.visualize" in base_env_source
    assert "self.motion_keyframe_debug_vis = enabled" in cfg_source
    assert "self.scene.object_frame.debug_vis = False" in cfg_source
