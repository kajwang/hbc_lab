from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
ASSET_ROOT = HBC_ROOT / "assets/robots"
TASK_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex1_hier_drc"
MDP_ROOT = TASK_ROOT / "mdp"
CONFIG_ROOT = TASK_ROOT / "config"
TRAIN_PATH = REPO_ROOT / "scripts/rsl_rl/train.py"


def _read(path: Path) -> str:
    return path.read_text()


def test_vendored_unitree_assets_are_included_as_package_data():
    pyproject_source = _read(REPO_ROOT / "pyproject.toml")

    assert "[tool.setuptools.package-data]" in pyproject_source
    assert "assets/models/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1/*.usd" in pyproject_source
    assert "assets/models/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1/configuration/*.usd" in pyproject_source


def test_g1_dex1_asset_cfg_uses_official_unitree_gripper_asset_and_body_joint_order():
    unitree_source = _read(ASSET_ROOT / "unitree.py")
    vendored_asset = (
        HBC_ROOT
        / "assets/models/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex1"
        / "g1_29dof_with_dex1_rev_1_0.usd"
    )

    assert "UNITREE_SIM_ISAACLAB_ASSETS_DIR" in unitree_source
    assert "Path(__file__).resolve()" in unitree_source
    assert "models/unitree_sim_isaaclab/assets" in unitree_source
    assert "/home/kaijun/wbc/unitree_sim_isaaclab/assets" not in unitree_source
    assert "G1_DEX1_LEFT_GRIPPER_JOINT_NAMES" in unitree_source
    assert "G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES" in unitree_source
    assert "UNITREE_G1_29DOF_DEX1_CFG" in unitree_source
    assert "robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd" in unitree_source
    assert vendored_asset.exists()
    assert '"left_hand_Joint1_1"' in unitree_source
    assert '"left_hand_Joint2_1"' in unitree_source
    assert '"right_hand_Joint1_1"' in unitree_source
    assert '"right_hand_Joint2_1"' in unitree_source
    assert "**UNITREE_G1_29DOF_CFG.actuators" in unitree_source
    assert "gripper" in unitree_source
    assert "joint_sdk_names=[\n        *G1_29DOF_BODY_JOINT_NAMES," in unitree_source


def test_g1_dex1_gripper_uses_limit_endpoints_for_open_and_close_direction():
    unitree_source = _read(ASSET_ROOT / "unitree.py")
    gripper_source = _read(MDP_ROOT / "gripper.py")

    assert '".*_hand_Joint[12]_1": 0.047' not in unitree_source
    assert '".*_hand_Joint[12]_1": -0.02' in unitree_source
    assert "DEX1_OPEN_POSITION = -0.02" in gripper_source
    assert "DEX1_CLOSE_POSITION = 0.024" in gripper_source
    assert "enabled_self_collisions=False" in unitree_source
    assert "stiffness=800.0" in unitree_source
    assert "friction=0.0" in unitree_source


def test_g1_dex1_task_is_registered_separately_from_dex3():
    task_init = _read(TASK_ROOT / "__init__.py")
    tasks_init = _read(HBC_ROOT / "tasks/__init__.py")

    assert "HBC-Isaac-G1-Dex1-HierDrc-v0" in task_init
    assert "HBC-Isaac-G1-Dex1-HierDrc-Play-v0" in task_init
    assert "g1_dex1_hier_drc" in tasks_init
    assert "g1_dex3_hier_drc" in tasks_init


def test_g1_dex1_samples_both_active_hands_and_exposes_contact_label_mask():
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")
    command_source = _read(MDP_ROOT / "commands.py")
    observation_source = _read(MDP_ROOT / "observations.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")

    assert "self.commands.high_level.left_hand_probability = 0.5" in flat_cfg_source
    assert "self.commands.high_level.left_hand_probability = 0.0" not in flat_cfg_source
    assert "self.contact_label.set_single_active_hand" in command_source
    assert "env.contact_label.effector_mask" in observation_source
    assert "env.contact_label.target_region" in observation_source
    assert "target_region_b = target_region_b * effector_mask.unsqueeze(-1)" in observation_source
    assert "ContactLabel.from_active_hand" in env_source
    assert "ContactMode.GRASP" in env_source
    assert "contact_mode" not in observation_source.split("def object_goal_hand_obs", maxsplit=1)[1].split(
        "def grip_obs", maxsplit=1
    )[0]


def test_g1_dex1_actor_uses_ten_step_observation_history_only():
    observation_source = _read(MDP_ROOT / "observations.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")
    policy_source = observation_source.split("    class PolicyCfg(ObsGroup):", maxsplit=1)[1].split(
        "    class CriticCfg(ObsGroup):", maxsplit=1
    )[0]
    critic_source = observation_source.split("    class CriticCfg(ObsGroup):", maxsplit=1)[1]

    assert "self.history_length = 10" in policy_source
    assert "self.flatten_history_dim = True" in policy_source
    assert "self.history_length = 10" not in critic_source
    assert "self.obs_buf = self.observation_manager.compute(update_history=True)" in env_source


def test_g1_dex1_hier_env_reuses_current_low_level_policy_interface():
    cfg_source = _read(CONFIG_ROOT / "g1_dex1_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")

    assert "action_dim: int = 19" in cfg_source
    assert "low_level_policy_path: str = \"\"" in cfg_source
    assert "LowLevelPolicyWrapper" in env_source
    assert "self.low_level_obs_builder.build(" in env_source
    assert "Dex1GripperController" in env_source
    assert "self.gripper_controller.apply(self.command_state.left_grip, self.command_state.right_grip)" in env_source
    assert "HL/left_gripper_target_mean" in env_source
    assert "HL/right_gripper_joint_pos_mean" in env_source
    assert "HL/right_gripper_target_buffer_mean" in env_source
    assert "HL/action_saturation_ratio" in env_source
    assert "HL/action_abs_mean" in env_source


def test_g1_dex1_uses_trainable_checkpoint_policy_std():
    train_source = _read(TRAIN_PATH)
    agent_source = _read(CONFIG_ROOT / "agents/rsl_rl_ppo_cfg.py")

    assert "def configure_runner_policy_noise" not in train_source
    assert "policy_noise_std_override" not in train_source
    assert "policy_noise_std_override" not in agent_source
    assert "freeze_policy_noise_std" not in agent_source
    assert "init_noise_std=0.6" in agent_source
    assert "entropy_coef=0.005" in agent_source


def test_g1_dex1_goal_is_sampled_behind_init_relative_to_robot():
    events_source = _read(MDP_ROOT / "events.py")

    assert 'robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]' in events_source
    assert "away_xy = object_root_pos_w[:, :2] - robot_root_pos_w[:, :2]" in events_source
    assert "away_heading = torch.atan2(away_xy[:, 1:2], away_xy[:, 0:1])" in events_source
    assert "heading_jitter = torch.empty_like(radius).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)" in events_source
    assert "heading = away_heading + heading_jitter" in events_source


def test_g1_dex1_couple_reward_uses_grasp_window_and_inner_pad_contact():
    progress_source = _read(MDP_ROOT / "contact_progress.py")
    reward_source = _read(MDP_ROOT / "rewards.py")
    scenes_source = _read(MDP_ROOT / "scenes.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")

    assert "def compute_gripper_contact_components" in progress_source
    assert "contact = torch.minimum(left_contact, right_contact)" in progress_source
    assert "pinch = contact * pinch_score" in progress_source
    assert "grasp = contact * grip * close_allowed_gate" in progress_source
    assert "def active_inner_pad_contact" in reward_source
    assert "INNER_PAD_CONTACT_GATE_SCALE = 0.05" in reward_source
    assert "left_link1_3 = env._step_link_contact[\"left_Link1_3\"]" in reward_source
    assert "left_link2_3 = env._step_link_contact[\"left_Link2_3\"]" in reward_source
    assert "grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.1)" in reward_source
    assert "pad_gate = torch.clamp(inner_pad_contact / INNER_PAD_CONTACT_GATE_SCALE, min=0.0, max=1.0)" in reward_source
    assert "gated_gripper_close1 = gripper_close * pad_gate" in reward_source
    assert "gated_gripper_close2 = gripper_close * grasp_window" in reward_source
    assert "0.4 * near" in reward_source
    assert "+ 0.1 * pad_gate" in reward_source
    assert "+ 0.1 * env.c_pinch" in reward_source
    assert "+ 0.2 * env.c_grasp" in reward_source
    assert "def root_object_facing_reward" in reward_source
    assert "DRC/root_object_facing_mean" in reward_source
    assert "root_object_facing = RewTerm(func=root_object_facing_reward, weight=2.0)" in reward_source
    assert "Couple/grasp_window_mean" in env_source
    assert "Couple/pad_gate_mean" in env_source
    assert "ApproachOnly/" not in env_source
    assert "# self.c_couple = update_ema(self.c_couple, progress.pinch, alpha=0.2)" in env_source
    assert "self.c_couple = update_ema(self.c_couple, progress.grasp, alpha=0.2)" in env_source
    assert "left_gripper_finger_contact" in scenes_source
    assert "right_gripper_finger_contact" in scenes_source
    assert "left_hand_base_link" in scenes_source
    assert "right_hand_base_link" in scenes_source
    assert "left_hand_Link1_3" in scenes_source
    assert "left_hand_Link2_3" in scenes_source
    assert "right_hand_Link1_3" in scenes_source
    assert "right_hand_Link2_3" in scenes_source
    assert "0.09734" in scenes_source
    assert ".*hand.*" in reward_source


def test_g1_dex1_records_all_gripper_link_contact_diagnostics():
    scenes_source = _read(MDP_ROOT / "scenes.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")

    assert "DEX1_LINK_CONTACT_SENSOR_NAMES" in scenes_source
    assert "left_hand_Link1_2" in scenes_source
    assert "left_hand_Link1_3" in scenes_source
    assert "left_hand_Link2_2" in scenes_source
    assert "left_hand_Link2_3" in scenes_source
    assert "right_hand_Link1_2" in scenes_source
    assert "right_hand_Link1_3" in scenes_source
    assert "right_hand_Link2_2" in scenes_source
    assert "right_hand_Link2_3" in scenes_source
    assert "ContactLink/{key}_mean" in env_source
    assert "ContactLink/active_{link_name}_force" in env_source


def test_g1_dex1_logs_success_before_successful_envs_are_reset():
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")
    step_source = env_source.split("    def step(", maxsplit=1)[1]

    assert "step_success_count = self.task_succeeded.sum().float().detach()" in step_source
    assert 'self.extras["log"]["Task/success_count"] = step_success_count' in step_source


def test_g1_dex1_logs_active_link_contact_with_pre_reset_active_hand():
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")
    step_source = env_source.split("    def step(", maxsplit=1)[1]

    assert "step_active_hand = self.active_hand.clone()" in step_source
    assert '(step_active_hand == 0).float().mean()' in step_source
    assert "self._log_link_contact_diagnostics(step_active_hand)" in step_source


def test_g1_dex1_high_level_wrist_commands_keep_workspace_as_diagnostics_only():
    action_source = _read(MDP_ROOT / "high_level_actions.py")

    assert "workspace_min" in action_source
    assert "workspace_max" in action_source
    assert "_clamp_position_to_workspace" not in action_source
    assert "torch.maximum(torch.minimum(position, upper), lower)" not in action_source


def test_g1_dex1_hier_low_level_obs_matches_hand_center_low_level_policy_interface():
    obs_source = _read(MDP_ROOT / "low_level_observations.py")

    assert "from .scenes import HAND_CENTER_FRAME_NAME" in obs_source
    assert "self.left_wrist_body_id" not in obs_source
    assert "self.right_wrist_body_id" not in obs_source
    assert 'find_bodies("left_wrist_yaw_link")' not in obs_source
    assert 'find_bodies("right_wrist_yaw_link")' not in obs_source
    assert "frame_sensor = self.env.scene[HAND_CENTER_FRAME_NAME]" in obs_source
    assert "frame_sensor.data.target_pos_w[:, frame_index]" in obs_source
    assert "frame_sensor.data.target_quat_w[:, frame_index]" in obs_source
    assert "def _target_pose_w" in obs_source
    assert "math_utils.compute_pose_error" in obs_source
    assert "math_utils.quat_apply_inverse(robot.data.root_quat_w, rot_error_w)" in obs_source
    assert "left_orientation_error" in obs_source
    assert "right_orientation_error" in obs_source


def test_g1_dex1_hier_low_level_obs_adds_orientation_error_terms_to_history():
    obs_source = _read(MDP_ROOT / "low_level_observations.py")

    single_frame_terms = obs_source.split("    def _single_frame_terms(", maxsplit=1)[1]
    return_block = single_frame_terms.split("        return (", maxsplit=1)[1].split("        )", maxsplit=1)[0]
    assert "left_current" in return_block
    assert "right_current" in return_block
    assert "left_error" in return_block
    assert "right_error" in return_block
    assert "left_orientation_error" in return_block
    assert "right_orientation_error" in return_block
    assert return_block.index("left_orientation_error") > return_block.index("right_error")
    assert return_block.index("command_state.posture_command") > return_block.index("right_error")
    assert return_block.index("posture_error") > return_block.index("command_state.posture_command")
    assert return_block.index("left_orientation_error") > return_block.index("posture_error")
    assert return_block.index("right_orientation_error") > return_block.index("left_orientation_error")


def test_g1_dex1_hier_action_orientation_uses_wrist_local_delta_and_matching_default():
    action_source = _read(MDP_ROOT / "high_level_actions.py")
    command_source = _read(MDP_ROOT / "commands.py")

    assert "pose[:, 3:] = _normalize_quat(_quat_mul(pose[:, 3:], delta_quat))" in action_source
    assert "pose[:, 3:] = _normalize_quat(_quat_mul(delta_quat, pose[:, 3:]))" not in action_source
    assert command_source.count("0.7071067811865476") >= 2
    assert command_source.count("-0.7071067811865475") >= 2
    assert "0.955177693375944" not in command_source
    assert "0.296033062472777" not in command_source


def test_g1_dex1_hier_tracking_penalty_and_diagnostics_use_hand_center_not_wrist_link():
    reward_source = _read(MDP_ROOT / "rewards.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")

    assert "def both_hand_center_tracking_error_penalty" in reward_source
    assert "both_wrist_tracking_error_penalty" not in reward_source
    assert "HAND_CENTER_FRAME_NAME" in reward_source
    assert "hand_center_pos_w = env.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w" in reward_source
    assert "left_wrist_body_id" not in reward_source
    assert "right_wrist_body_id" not in reward_source
    assert "left_wrist_body_id" not in env_source
    assert "right_wrist_body_id" not in env_source
    assert "HL/active_hand_center_tracking_error" in env_source
    assert "HL/active_wrist_tracking_error" not in env_source
    assert "HL/active_hand_center_to_wrist_target_dist" not in env_source


def test_g1_dex1_launch_entries_exist():
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert '"name": "g1_dex1_hier_drc_train"' in launch_source
    assert '"--task=HBC-Isaac-G1-Dex1-HierDrc-v0"' in launch_source
    assert '"name": "g1_dex1_hier_drc_play"' in launch_source
    assert '"--task=HBC-Isaac-G1-Dex1-HierDrc-Play-v0"' in launch_source
