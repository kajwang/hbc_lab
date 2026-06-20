from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof"


def _read(path: Path) -> str:
    return path.read_text()


def test_dex3_fixed_hands_low_level_play_task_is_registered_and_launchable():
    init_source = _read(G1_ROOT / "__init__.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert 'id="HBC-Isaac-WholeBody-SphericalPosture-Dex3FixedHands-Unitree-G1-v0"' in init_source
    assert "whole_body_spherical_posture_dex3_fixed_env_cfg:SphericalPostureDex3FixedHandsEnvCfg" in init_source
    assert "whole_body_spherical_posture_dex3_fixed_env_cfg:SphericalPostureDex3FixedHandsPlayEnvCfg" in init_source
    assert '"name": "g1_whole_body_spherical_posture_dex3_fixed_play"' in launch_source
    assert "--task=HBC-Isaac-WholeBody-SphericalPosture-Dex3FixedHands-Unitree-G1-v0" in launch_source


def test_dex3_fixed_hands_task_preserves_low_level_body_policy_dimensions():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex3_fixed_env_cfg.py")

    assert "UNITREE_G1_29DOF_DEX3_CFG" in source
    assert "G1_29DOF_BODY_JOINT_NAMES" in source
    assert "G1_DEX3_LEFT_HAND_JOINT_NAMES" in source
    assert "G1_DEX3_RIGHT_HAND_JOINT_NAMES" in source
    assert "class Dex3FixedHandsActionsCfg" in source
    assert "joint_names=G1_29DOF_BODY_JOINT_NAMES" in source
    assert "class Dex3FixedHandsObservationsCfg" in source
    assert 'params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_BODY_JOINT_NAMES)}' in source
    assert "SphericalPostureWholeBodyEnvCfg" in source
    assert "class SphericalPostureDex3FixedHandsPlayEnvCfg" in source
    assert "self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges" in source
    assert "self.commands.posture_command.ranges = self.commands.posture_command.limit_ranges" in source
    assert "self.curriculum.wrist_pose_cmd_levels = None" in source
    assert "self.curriculum.posture_cmd_levels = None" in source


def test_dex3_fixed_hands_task_closes_hands_by_default_and_resets_them_to_default():
    source = _read(G1_ROOT / "whole_body_spherical_posture_dex3_fixed_env_cfg.py")

    assert "DEX3_CLOSED_HAND_JOINT_POS" in source
    assert "pos=(0.0, 0.0, 0.8)" in source
    assert '"left_hip_pitch_joint": -0.1' in source
    assert '".*_knee_joint": 0.3' in source
    assert '".*_elbow_joint": 0.97' in source
    assert '"left_hand_thumb_1_joint": 1.0' in source
    assert '"left_hand_index_1_joint": -1.74' in source
    assert '"right_hand_thumb_1_joint": -1.0' in source
    assert '"right_hand_index_1_joint": 1.74' in source
    assert "reset_robot_body_and_hand_joints" in source
    assert "joint_ids=body_joint_ids + hand_joint_ids" in source
    assert "joint_vel[:, len(body_joint_ids) :] = 0.0" in source
