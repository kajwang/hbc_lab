from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex3_hier_drc"
MDP_ROOT = PKG_ROOT / "mdp"
CONFIG_ROOT = PKG_ROOT / "config"


def _read(path: Path) -> str:
    return path.read_text()


def test_g1_dex3_hier_drc_registers_custom_high_level_env():
    init_source = _read(PKG_ROOT / "__init__.py")
    tasks_source = _read(REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/__init__.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert 'id="HBC-Isaac-G1-Dex3-HierDrc-v0"' in init_source
    assert 'id="HBC-Isaac-G1-Dex3-HierDrc-Play-v0"' in init_source
    assert "config.g1_dex3_env:G1Dex3HierDrcEnv" in init_source
    assert "config.flat_env_cfg:G1Dex3HierDrcFlatEnvCfg" in init_source
    assert "play_env_cfg_entry_point" in init_source
    assert "config.flat_env_cfg:G1Dex3HierDrcFlatPlayEnvCfg" in init_source
    assert "config.agents.rsl_rl_ppo_cfg:G1Dex3HierDrcPPORunnerCfg" in init_source
    assert 'hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc' in tasks_source
    assert "FixedCmd" not in init_source
    assert "g1_dex3_hier_drc_fixed_cmd_train" not in launch_source


def test_g1_dex3_hier_env_calls_exported_low_level_policy_without_training_low_level():
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")
    cfg_source = _read(CONFIG_ROOT / "g1_dex3_env_cfg.py")
    low_level_source = _read(MDP_ROOT / "low_level_policy.py")
    obs_source = _read(MDP_ROOT / "low_level_observations.py")

    assert "class G1Dex3HierDrcEnv(ManagerBasedRLEnv)" in env_source
    assert "LowLevelPolicyWrapper(cfg.low_level_policy_path" in env_source
    assert "self.low_level_policy(low_obs)" in env_source
    assert "robot.set_joint_position_target(joint_target, joint_ids=joint_ids)" in env_source
    assert "robot.find_joints(self.body_joint_names, preserve_order=False)" in env_source
    assert "Dex3GripperController" in env_source
    assert "self.gripper_controller.apply" in env_source
    assert "low_level_policy_path: str = \"\"" in cfg_source
    assert "allow_missing_low_level_policy: bool = False" in cfg_source
    assert "A frozen low-level policy is required" in env_source
    assert "_fixed_high_level_command_state" not in env_source
    assert "use_fixed_high_level_command" not in cfg_source
    assert "train_low_level" not in cfg_source
    assert "torch.jit.load" in low_level_source
    assert "class G1SphericalPostureLowLevelObsBuilder" in obs_source
    assert "robot.find_joints(self.joint_names, preserve_order=False)" in obs_source
    assert "history_length" in obs_source
    assert "height_scan" in obs_source
    assert "height_scan.clip(-1.0, 5.0)" in obs_source
    assert "left_current = left_current.clip(-2.0, 2.0)" in obs_source
    assert "right_current = right_current.clip(-2.0, 2.0)" in obs_source
    assert "left_error = left_error.clip(-1.0, 1.0)" in obs_source
    assert "right_error = right_error.clip(-1.0, 1.0)" in obs_source
    assert "posture_error = self._posture_error(command_state, torso_body_id).clip(-1.0, 1.0)" in obs_source
    assert "_term_history" in obs_source
    assert "_push_term_history" in obs_source
    assert "_flatten_term_major_history" in obs_source
    assert "torch.cat(term_history_values, dim=-1)" in obs_source
    assert "self._history_valid[env_ids] = False" in obs_source
    assert "self.history.reshape" not in obs_source


def test_hier_low_level_wrist_targets_use_spherical_shoulder_anchor_frame():
    obs_source = _read(MDP_ROOT / "low_level_observations.py")
    command_source = _read(MDP_ROOT / "commands.py")

    assert "left_shoulder_pitch_link" in obs_source
    assert "right_shoulder_pitch_link" in obs_source
    assert "_anchor_pose_w" in obs_source
    assert "_target_pos_w(self, pose_b: torch.Tensor, side: str" in obs_source
    assert "root_cmd_pos_w[:, 2]" in obs_source
    assert "anchor_offset_b[:, 2] = self.anchor_height_offset" in obs_source
    assert "math_utils.quat_apply_inverse(root_yaw_quat" in obs_source
    assert "math_utils.quat_apply(anchor_quat_w" in obs_source

    assert "_target_pose_w(self, pose_b: torch.Tensor, side: str" in command_source
    assert "left_shoulder_pitch_link" in command_source
    assert "right_shoulder_pitch_link" in command_source
    assert "self.posture_command" in command_source
    assert "default_root_height: float = 0.8" in command_source
    assert "0.955177693375944" in command_source
    assert "0.296033062472777" in command_source


def test_high_level_action_and_command_are_dual_hand_but_contact_label_is_hand_level_only():
    action_source = _read(MDP_ROOT / "high_level_actions.py")
    command_source = _read(MDP_ROOT / "commands.py")
    progress_source = _read(MDP_ROOT / "contact_progress.py")

    assert "left_wrist_pose_b" in action_source
    assert "right_wrist_pose_b" in action_source
    assert "left_grip" in action_source
    assert "right_grip" in action_source
    assert "posture_command" in action_source
    assert "action_dim: int = 19" in _read(CONFIG_ROOT / "g1_dex3_env_cfg.py")

    assert "class G1Dex3HierCommand" in command_source
    assert "active_hand" in command_source
    assert "sample_active_hands" in command_source

    assert "ContactLabel" not in progress_source
    assert "thumb_index" not in progress_source
    assert "thumb_middle" not in progress_source
    assert "compute_hand_contact_confidence" in progress_source
    assert "compute_active_hand_grasp_progress" in progress_source
    assert "select_active_hand_value" in progress_source


def test_hand_contact_sensors_are_split_then_aggregated_to_hand_level():
    scene_source = _read(MDP_ROOT / "scenes.py")
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")
    cfg_source = _read(CONFIG_ROOT / "g1_dex3_env_cfg.py")

    assert "LEFT_HAND_CONTACT_SENSOR_NAMES" in scene_source
    assert "RIGHT_HAND_CONTACT_SENSOR_NAMES" in scene_source
    assert "LEFT_HAND_THUMB_CONTACT_SENSOR_NAMES" in scene_source
    assert "LEFT_HAND_INDEX_CONTACT_SENSOR_NAMES" in scene_source
    assert "LEFT_HAND_MIDDLE_CONTACT_SENSOR_NAMES" in scene_source
    assert "RIGHT_HAND_THUMB_CONTACT_SENSOR_NAMES" in scene_source
    assert "RIGHT_HAND_INDEX_CONTACT_SENSOR_NAMES" in scene_source
    assert "RIGHT_HAND_MIDDLE_CONTACT_SENSOR_NAMES" in scene_source
    assert "left_hand_object_contact =" not in scene_source
    assert "right_hand_object_contact =" not in scene_source
    assert "compute_dex3_hand_contact_components" in env_source
    assert "thumb_names = LEFT_HAND_THUMB_CONTACT_SENSOR_NAMES" in env_source
    assert "index_names = RIGHT_HAND_INDEX_CONTACT_SENSOR_NAMES" in env_source
    assert "self._sum_sensor_group_force(thumb_names)" in env_source
    assert "self._sum_sensor_group_force(index_names)" in env_source
    assert "self.cfg.scene.left_hand_object_contact" not in cfg_source
    assert "ContactLabel" not in env_source


def test_observations_are_safe_during_manager_initialization():
    obs_source = _read(MDP_ROOT / "observations.py")

    assert "def hand_center_positions_w" in obs_source
    assert "env.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w" in obs_source
    assert "hasattr(env, \"command_state\")" in obs_source
    assert "torch.zeros(env.num_envs, 2" in obs_source


def test_hier_task_disables_terrain_curriculum_for_flat_interaction_debugging():
    curriculum_source = _read(MDP_ROOT / "curriculums.py")
    cfg_source = _read(CONFIG_ROOT / "g1_dex3_env_cfg.py")
    scene_source = _read(MDP_ROOT / "scenes.py")
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")

    assert "class G1Dex3HierDrcCurriculumCfg" in curriculum_source
    assert "CurrTerm" not in curriculum_source
    assert "terrain_levels" not in curriculum_source
    assert "get_command(\"high_level\")" not in curriculum_source
    assert "terrain_levels_vel" not in curriculum_source
    assert "get_command(\"base_velocity\")" not in curriculum_source
    assert "self.scene.height_scanner.update_period = self.low_level_decimation * self.sim.dt" in cfg_source
    assert 'terrain_type="plane"' in scene_source
    assert "terrain_generator=None" in scene_source
    assert "terrain_generator.num_rows" not in flat_cfg_source


def test_hier_reset_keeps_dex3_hand_joints_stable():
    events_source = _read(MDP_ROOT / "events.py")

    assert "def reset_robot_body_and_hand_joints" in events_source
    assert "asset.find_joints(joint_names, preserve_order=False)" in events_source
    assert "G1_29DOF_BODY_JOINT_NAMES" in events_source
    assert "G1_DEX3_LEFT_HAND_JOINT_NAMES" in events_source
    assert "G1_DEX3_RIGHT_HAND_JOINT_NAMES" in events_source
    assert "joint_vel[:, len(body_joint_ids) :] = 0.0" in events_source
    assert "asset.set_joint_position_target(hand_default_pos" in events_source
    assert "func=reset_robot_body_and_hand_joints" in events_source
    assert "func=mdp.reset_joints_by_scale" not in events_source


def test_low_level_last_action_obs_uses_raw_actor_output_like_training_env():
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")
    cfg_source = _read(CONFIG_ROOT / "g1_dex3_env_cfg.py")

    assert "low_level_action_clip: float | None = None" in cfg_source
    assert "raw_low_action = self.low_level_policy(low_obs)" in env_source
    assert "self._last_low_level_action = raw_low_action.detach()" in env_source
    assert "self._last_low_level_action = applied_low_action.detach()" not in env_source
    assert "if self.cfg.low_level_action_clip is not None:" in env_source


def test_hier_train_visualizes_all_high_level_commands_by_default():
    command_source = _read(MDP_ROOT / "commands.py")
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")

    assert "self.commands.high_level.debug_vis = True" in flat_cfg_source
    assert "self.target_pose_debug_vis = True" in flat_cfg_source
    assert "OBJECT_INITIAL_MARKER_CFG" in env_source
    assert "object_initial_pose_visualizer" in env_source
    assert "self.object_initial_pose_visualizer.visualize(self.object_initial_pos_w)" in env_source
    assert "posture_command_visualizer" in command_source
    assert "grip_visualizer" in command_source
    assert "active_hand_visualizer" in command_source
    assert "posture_command_visualizer_cfg" in command_source
    assert "grip_visualizer_cfg" in command_source
    assert "active_hand_visualizer_cfg" in command_source
    assert "_resolve_posture_arrow" in command_source
    assert "_resolve_grip_marker_scale" in command_source


def test_hier_flat_debug_task_uses_single_right_hand_and_same_side_object_sampling():
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")

    assert "self.commands.high_level.left_hand_probability = 0.0" in flat_cfg_source
    assert 'self.events.reset_object.params["pose_range"]["y"] = (-0.35, -0.05)' in flat_cfg_source


def test_hier_logs_posture_and_workspace_clamp_diagnostics():
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")

    assert "def _log_high_level_diagnostics" in env_source
    assert "HL/root_height_cmd_mean" in env_source
    assert "HL/torso_pitch_cmd_mean" in env_source
    assert "HL/active_wrist_cmd_x_mean" in env_source
    assert "HL/active_wrist_cmd_y_mean" in env_source
    assert "HL/active_wrist_cmd_z_mean" in env_source
    assert "HL/active_wrist_cmd_z_min_ratio" in env_source
    assert "HL/active_wrist_cmd_x_max_ratio" in env_source
    assert "HL/active_grip_mean" in env_source
    assert "HL/inactive_grip_mean" in env_source
    assert "HL/active_grip_closed_ratio" in env_source
    assert "HL/active_wrist_target_object_dist" in env_source
    assert "HL/active_wrist_tracking_error" in env_source
    assert "HL/active_hand_center_to_wrist_target_dist" in env_source
    assert "self._active_target_tracking_diagnostics()" in env_source
    assert "self._log_high_level_diagnostics()" in env_source


def test_hier_reset_restores_delta_command_state_to_command_defaults():
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")

    assert "def _reset_command_state" in env_source
    assert "self.command_state.base_velocity[env_ids] = self.high_level_command.base_velocity[env_ids]" in env_source
    assert "self.command_state.posture_command[env_ids] = self.high_level_command.posture_command[env_ids]" in env_source
    assert "self.command_state.left_wrist_pose_b[env_ids] = self.high_level_command.left_wrist_pose_b[env_ids]" in env_source
    assert "self.command_state.right_wrist_pose_b[env_ids] = self.high_level_command.right_wrist_pose_b[env_ids]" in env_source
    assert "self.command_state.left_grip[env_ids] = self.high_level_command.left_grip[env_ids]" in env_source
    assert "self.command_state.right_grip[env_ids] = self.high_level_command.right_grip[env_ids]" in env_source
    assert "self._reset_command_state(env_ids)" in env_source


def test_hier_task_uses_visualized_hand_center_frames_for_object_distance():
    scene_source = _read(MDP_ROOT / "scenes.py")
    obs_source = _read(MDP_ROOT / "observations.py")
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")
    progress_source = env_source.split("def _compute_progress", 1)[1].split("def _check_success", 1)[0]

    assert "FrameTransformerCfg" in scene_source
    assert "OffsetCfg" in scene_source
    assert "hand_center_frame = FrameTransformerCfg" in scene_source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/torso_link"' in scene_source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link"' in scene_source
    assert 'name="left_hand_center"' in scene_source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link"' in scene_source
    assert 'name="right_hand_center"' in scene_source
    assert "OffsetCfg(pos=(0.07, 0.0, 0.0))" in scene_source
    assert "debug_vis=True" in scene_source

    assert "from .scenes import HAND_CENTER_FRAME_NAME" in obs_source
    assert "HAND_CENTER_FRAME_NAME" in env_source
    assert "hand_center_pos_w = env.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w" in obs_source
    assert "hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w" in progress_source
    assert "return hand_center_pos_w[:, 0, :], hand_center_pos_w[:, 1, :]" in obs_source
    assert "left_pos_w = hand_center_pos_w[:, 0, :]" in progress_source
    assert "left_wrist_body_id" not in progress_source
    assert "right_wrist_body_id" not in progress_source


def test_hier_play_does_not_hard_code_closed_grippers_by_default():
    cfg_source = _read(CONFIG_ROOT / "g1_dex3_env_cfg.py")
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")

    assert "debug_fixed_gripper: bool = False" in cfg_source
    assert "debug_fixed_left_grip: float = 1.0" in cfg_source
    assert "debug_fixed_right_grip: float = 1.0" in cfg_source
    assert "class G1Dex3HierDrcFlatEnvCfg" in flat_cfg_source
    assert "self.debug_fixed_gripper = True" not in flat_cfg_source
    assert "def _apply_debug_gripper_override" in env_source
    assert "self.command_state.left_grip[:] = self.cfg.debug_fixed_left_grip" in env_source
    assert "self.command_state.right_grip[:] = self.cfg.debug_fixed_right_grip" in env_source
    assert "self._apply_debug_gripper_override()" in env_source


def test_hier_couple_reward_does_not_penalize_active_grip_before_contact():
    reward_source = _read(MDP_ROOT / "rewards.py")

    assert "early_close_penalty" not in reward_source
    assert "active_grip_window" in reward_source
    assert "env.active_grip * active_grip_window" in reward_source


def test_hier_object_starts_on_half_meter_platform():
    objects_source = _read(REPO_ROOT / "source/hbc_lab/hbc_lab/assets/objects.py")
    scene_source = _read(MDP_ROOT / "scenes.py")
    events_source = _read(MDP_ROOT / "events.py")
    env_source = _read(CONFIG_ROOT / "g1_dex3_env.py")

    assert "OBJECT_PLATFORM_HEIGHT = 0.5" in objects_source
    assert "OBJECT_PLATFORM_SIZE = (0.24, 0.30, OBJECT_PLATFORM_HEIGHT)" in objects_source
    assert "SMALL_CUBE_SIZE = 0.056" in objects_source
    assert "SMALL_CUBE_HALF_HEIGHT = 0.5 * SMALL_CUBE_SIZE" in objects_source
    assert "OBJECT_ON_PLATFORM_Z = OBJECT_PLATFORM_HEIGHT + SMALL_CUBE_HALF_HEIGHT" in objects_source
    assert "OBJECT_INIT_PLATFORM_CFG" in objects_source
    assert "OBJECT_TARGET_PLATFORM_CFG" in objects_source
    assert "kinematic_enabled=True" not in objects_source
    assert "disable_gravity=True" in objects_source
    assert "mass=1.0e6" in objects_source
    assert "size=(SMALL_CUBE_SIZE, SMALL_CUBE_SIZE, SMALL_CUBE_SIZE)" in objects_source
    assert "init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)" in objects_source
    assert "object_init_platform: RigidObjectCfg = OBJECT_INIT_PLATFORM_CFG" in scene_source
    assert "object_target_platform: RigidObjectCfg = OBJECT_TARGET_PLATFORM_CFG" in scene_source
    assert "def reset_object_and_support_platforms" in events_source
    assert "self.object_initial_pos_w[env_ids] = object_pos_w" not in env_source
    assert "self.object_target_pos_w[env_ids] = object_pos_w + target_offset" not in env_source
    assert "func=reset_object_and_support_platforms" in events_source
    assert "object_goal_radius_range" in events_source
    assert "env.object_initial_pos_w[env_ids] = object_pos_w" in events_source
    assert "env.object_target_pos_w[env_ids] = target_pos_w" in events_source
    assert "env.scene[init_platform_cfg.name].write_root_state_to_sim" in events_source
    assert "env.scene[target_platform_cfg.name].write_root_state_to_sim" in events_source
    assert "obj.write_root_state_to_sim" in events_source
    assert "def _reset_idx(self, env_ids)" in env_source
    assert "super()._reset_idx(env_ids)" in env_source
