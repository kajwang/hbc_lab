from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof"
MDP_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp"


def _read(path: Path) -> str:
    return path.read_text()


def test_dex1_hand_center_low_level_task_is_registered_and_launchable():
    init_source = _read(G1_ROOT / "__init__.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert 'id="HBC-Isaac-WholeBody-SphericalPosture-Dex1HandCenter-Unitree-G1-v0"' in init_source
    assert (
        "whole_body_spherical_posture_dex1_hand_center_env_cfg:SphericalPostureDex1HandCenterEnvCfg"
        in init_source
    )
    assert (
        "whole_body_spherical_posture_dex1_hand_center_env_cfg:SphericalPostureDex1HandCenterPlayEnvCfg"
        in init_source
    )
    assert '"name": "g1_whole_body_spherical_posture_dex1_train"' in launch_source
    assert '"name": "g1_whole_body_spherical_posture_dex1_play"' in launch_source
    assert "--task=HBC-Isaac-WholeBody-SphericalPosture-Dex1HandCenter-Unitree-G1-v0" in launch_source


def test_dex1_hand_center_task_uses_dex1_body_policy_dims_and_hand_center_frames():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")

    assert "UNITREE_G1_29DOF_DEX1_CFG" in source
    assert "G1_29DOF_BODY_JOINT_NAMES" in source
    assert "G1_DEX1_LEFT_GRIPPER_JOINT_NAMES" in source
    assert "G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES" in source
    assert "class Dex1HandCenterActionsCfg" in source
    assert "joint_names=G1_29DOF_BODY_JOINT_NAMES" in source
    assert 'HAND_CENTER_FRAME_NAME = "hand_center_frame"' in source
    assert "HAND_CENTER_OFFSET = OffsetCfg(pos=(0.0, 0.09734, 0.0142))" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/left_hand_base_link"' in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/right_hand_base_link"' in source
    assert 'name="left_hand_center"' in source
    assert 'name="right_hand_center"' in source
    assert "reset_robot_body_and_gripper_joints" in source
    assert "joint_vel[:, len(body_joint_ids) :] = 0.0" in source


def test_dex1_hand_center_task_tracks_frame_transformer_pose_with_physical_wrist_ranges():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")
    reward_source = _read(MDP_ROOT / "rewards.py")
    observation_source = _read(MDP_ROOT / "observations.py")
    command_source = _read(MDP_ROOT / "commands/spherical_pose_command.py")

    assert 'body_name="left_hand_base_link"' in source
    assert 'body_name="right_hand_base_link"' in source
    assert "tracked_frame_sensor_name=HAND_CENTER_FRAME_NAME" in source
    assert "tracked_frame_index=0" in source
    assert "tracked_frame_index=1" in source
    assert source.count("roll=(-0.50, 0.50)") == 2
    assert "WRIST_ROLL_LIMIT = math.radians(100.0)" in source
    assert source.count("roll=(-WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT)") == 2
    assert source.count("ee_pitch=(-0.12, 0.12)") == 2
    assert "WRIST_PITCH_YAW_LIMIT = math.radians(80.0)" in source
    assert source.count("ee_pitch=(-WRIST_PITCH_YAW_LIMIT, WRIST_PITCH_YAW_LIMIT)") == 2
    assert source.count("yaw=(-0.12, 0.12)") == 2
    assert source.count("yaw=(-WRIST_PITCH_YAW_LIMIT, WRIST_PITCH_YAW_LIMIT)") == 2
    assert "func=mdp.frame_transformer_pose_command_position_error_w_in_root_frame" in source
    assert "func=mdp.frame_pose_command_position_error_w_exp" in source
    assert "func=mdp.frame_pose_command_position_error_w_tanh" in source
    assert "func=mdp.frame_pose_command_orientation_error_w_tanh" in source
    assert '"frame_sensor_name": HAND_CENTER_FRAME_NAME' in source
    assert "def frame_transformer_pose_in_root_frame" in observation_source
    assert "def frame_transformer_pose_command_position_error_w_in_root_frame" in observation_source
    assert "def frame_transformer_pose_command_orientation_error_w_in_root_frame" in observation_source
    assert "def frame_pose_command_position_error_w_exp" in reward_source
    assert "def frame_pose_command_orientation_error_w_tanh" in reward_source
    assert 'orientation_mode: str = "azimuth"' in command_source
    assert "tracked_frame_sensor_name: str | None = None" in command_source
    assert "target_pos_w[:, self.cfg.tracked_frame_index]" in command_source
    assert source.count("weight=1.0") >= 4


def test_dex1_hand_center_curricula_use_actual_duration_and_direct_errors():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")
    curriculum_source = _read(MDP_ROOT / "curriculums.py")
    spherical_command_source = _read(MDP_ROOT / "commands/spherical_pose_command.py")
    posture_command_source = _read(MDP_ROOT / "commands/posture_command.py")

    assert "def _episode_duration_s(" in curriculum_source
    assert "env.episode_length_buf[env_ids]" in curriculum_source
    assert "min_episode_fraction" in curriculum_source
    assert "position_error_sum" in spherical_command_source
    assert "orientation_error_sum" in spherical_command_source
    assert "root_height_error_sum" in posture_command_source
    assert "torso_pitch_error_sum" in posture_command_source
    assert '"error_sum_name": "orientation_error_sum"' in source
    assert '"success_threshold": 0.30' in source
    assert source.count('"min_episode_fraction": 0.5') == 3
    assert '"reward_term_names"' not in source.split("orientation_cmd_levels = CurrTerm(", 1)[1].split(
        "posture_cmd_levels = CurrTerm(", 1
    )[0]


def test_dex1_hand_center_logs_orientation_axes_and_wrist_limit_usage():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")
    command_source = _read(MDP_ROOT / "commands/spherical_pose_command.py")

    for metric_name in (
        "orientation_roll_error",
        "orientation_pitch_error",
        "orientation_yaw_error",
        "wrist_roll_limit_ratio",
        "wrist_pitch_limit_ratio",
        "wrist_yaw_limit_ratio",
    ):
        assert f'self.metrics["{metric_name}"]' in command_source
    assert 'wrist_joint_names=(' in source
    assert '"left_wrist_roll_joint"' in source
    assert '"right_wrist_roll_joint"' in source


def test_dex1_hand_center_current_pose_observations_use_hand_center_frame():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")

    assert "left_wrist_pose_current = ObsTerm" in source
    assert "right_wrist_pose_current = ObsTerm" in source
    assert source.count("func=mdp.frame_transformer_pose_in_root_frame") == 4
    assert "left_wrist_orientation_error = ObsTerm" in source
    assert "right_wrist_orientation_error = ObsTerm" in source
    assert source.count("func=mdp.frame_transformer_pose_command_orientation_error_w_in_root_frame") == 4
    assert source.count('"frame_sensor_name": HAND_CENTER_FRAME_NAME') >= 16
    assert source.count('"frame_index": 0') >= 8
    assert source.count('"frame_index": 1') >= 8


def test_dex1_hand_center_orientation_target_uses_physical_wrist_chain_sampling():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")
    command_source = _read(MDP_ROOT / "commands/spherical_pose_command.py")

    assert source.count('orientation_mode="wrist_chain"') == 2
    assert 'if self.cfg.orientation_mode == "wrist_chain":' in command_source
    assert "compose_wrist_chain_quat(euler_angles, self.fixed_palm_quat)" in command_source
    assert "hand_base_to_hand_center_pose(" in command_source
    assert source.count("fixed_palm_quat=") == 2
    assert source.count("hand_center_offset=(0.0, 0.09734, 0.0142)") == 2


def test_dex1_hand_center_task_disables_terrain_level_curriculum_but_keeps_other_curricula():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")
    curriculum_source = _read(MDP_ROOT / "curriculums.py")

    train_source = source.split("class SphericalPostureDex1HandCenterEnvCfg")[1].split(
        "class SphericalPostureDex1HandCenterPlayEnvCfg"
    )[0]
    assert "terrain_levels = None" in source
    assert "self.curriculum.terrain_levels = None" in source
    assert "self.curriculum.lin_vel_cmd_levels = None" not in train_source
    assert "wrist_pose_cmd_levels = CurrTerm" in source
    assert "orientation_cmd_levels = CurrTerm" in source
    assert "func=mdp.spherical_pose_orientation_cmd_levels" in source
    assert '"error_sum_name": "orientation_error_sum"' in source
    assert '"roll_delta": 0.35' in source
    assert '"ee_pitch_delta": 0.04' in source
    assert '"yaw_delta": 0.04' in source
    assert "posture_cmd_levels = CurrTerm" in source
    assert "self.curriculum.lin_vel_cmd_levels = None" in source
    assert "self.curriculum.wrist_pose_cmd_levels = None" in source
    assert "self.curriculum.orientation_cmd_levels = None" in source
    assert "self.curriculum.posture_cmd_levels = None" in source
    assert "def spherical_pose_orientation_cmd_levels" in curriculum_source
    assert "ranges.roll = _expand_uniform_range" in curriculum_source
    assert "ranges.ee_pitch = _expand_uniform_range" in curriculum_source
    assert "ranges.yaw = _expand_uniform_range" in curriculum_source
    assert 'for range_name in ("roll", "ee_pitch", "yaw")' in curriculum_source
