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
    assert '"name": "g1_whole_body_spherical_posture_dex1_hand_center_train"' in launch_source
    assert '"name": "g1_whole_body_spherical_posture_dex1_hand_center_play"' in launch_source
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


def test_dex1_hand_center_task_tracks_frame_transformer_pose_with_conservative_orientation_ranges():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")
    reward_source = _read(MDP_ROOT / "rewards.py")
    observation_source = _read(MDP_ROOT / "observations.py")
    command_source = _read(MDP_ROOT / "commands/spherical_pose_command.py")

    assert 'body_name="left_hand_base_link"' in source
    assert 'body_name="right_hand_base_link"' in source
    assert "tracked_frame_sensor_name=HAND_CENTER_FRAME_NAME" in source
    assert "tracked_frame_index=0" in source
    assert "tracked_frame_index=1" in source
    assert "roll=(-0.08, 0.08)" in source
    assert "ee_pitch=(-0.08, 0.08)" in source
    assert "yaw=(-0.08, 0.08)" in source
    assert "roll=(-0.20, 0.20)" in source
    assert "ee_pitch=(-0.20, 0.20)" in source
    assert "yaw=(-0.20, 0.20)" in source
    assert "func=mdp.frame_transformer_pose_command_position_error_w_in_root_frame" in source
    assert "func=mdp.frame_pose_command_position_error_w_exp" in source
    assert "func=mdp.frame_pose_command_position_error_w_tanh" in source
    assert "func=mdp.frame_pose_command_orientation_error_w_tanh" in source
    assert '"frame_sensor_name": HAND_CENTER_FRAME_NAME' in source
    assert "def frame_transformer_pose_in_root_frame" in observation_source
    assert "def frame_transformer_pose_command_position_error_w_in_root_frame" in observation_source
    assert "def frame_pose_command_position_error_w_exp" in reward_source
    assert "def frame_pose_command_orientation_error_w_tanh" in reward_source
    assert "tracked_frame_sensor_name: str | None = None" in command_source
    assert "target_pos_w[:, self.cfg.tracked_frame_index]" in command_source


def test_dex1_hand_center_task_disables_terrain_level_curriculum_but_keeps_other_curricula():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex1_hand_center_env_cfg.py")

    train_source = source.split("class SphericalPostureDex1HandCenterEnvCfg")[1].split(
        "class SphericalPostureDex1HandCenterPlayEnvCfg"
    )[0]
    assert "terrain_levels = None" in source
    assert "self.curriculum.terrain_levels = None" in source
    assert "self.curriculum.lin_vel_cmd_levels = None" not in train_source
    assert "wrist_pose_cmd_levels = CurrTerm" in source
    assert "posture_cmd_levels = CurrTerm" in source
    assert "self.curriculum.lin_vel_cmd_levels = None" in source
    assert "self.curriculum.wrist_pose_cmd_levels = None" in source
    assert "self.curriculum.posture_cmd_levels = None" in source
