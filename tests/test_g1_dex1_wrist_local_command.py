from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORM_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp/pose_transforms.py"
COMMAND_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp/commands/spherical_pose_command.py"
DEX1_CONFIG_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof/whole_body_spherical_posture_dex1_hand_center_env_cfg.py"
)
HIER_OBS_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/low_level_observations.py"
)
HIER_COMMAND_PATH = (
    REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/commands.py"
)


def _read(path: Path) -> str:
    return path.read_text()


def test_shared_pose_transform_module_defines_posture_anchor_and_wrist_chain():
    source = _read(TRANSFORM_PATH)

    assert "def posture_anchor_pose_w(" in source
    assert "def compose_wrist_chain_quat(" in source
    assert "def hand_base_to_hand_center_pose(" in source
    assert "quat_mul(quat_mul(quat_mul(roll_quat, pitch_quat), yaw_quat), fixed_quat)" in source
    assert "hand_base_pos + quat_apply(hand_base_quat, hand_center_offset)" in source


def test_dex1_command_uses_stick_figure_sampling_and_physical_wrist_limits():
    source = _read(DEX1_CONFIG_PATH)
    command_source = _read(COMMAND_PATH)

    assert "class Dex1HandCenterCommandsCfg(velocity_env_cfg.CommandsCfg):" in source
    assert source.index("posture_command = mdp.UniformLevelPostureCommandCfg(") < source.index(
        "left_wrist_pose = mdp.SphericalLevelPoseCommandCfg("
    )
    assert "WRIST_ROLL_LIMIT = math.radians(100.0)" in source
    assert "WRIST_PITCH_YAW_LIMIT = math.radians(80.0)" in source
    assert source.count('sampling_mode="stick_figure"') == 2
    assert source.count('anchor_body_name="left_shoulder_roll_link"') == 1
    assert source.count('anchor_body_name="right_shoulder_roll_link"') == 1
    assert source.count("use_full_anchor_offset=True") == 2
    assert source.count("upper_arm_length=UPPER_ARM_LENGTH") == 2
    assert source.count("forearm_length=FOREARM_LENGTH") == 2
    assert source.count("upper_arm_roll=(-0.5 * math.pi, 0.5 * math.pi)") == 4
    assert source.count("elbow_flexion=(0.45, 1.05)") == 2
    assert source.count("elbow_flexion=(0.15, 1.85)") == 2
    assert source.count('orientation_mode="wrist_chain"') == 2
    assert source.count('anchor_height_command_name="posture_command"') == 2
    assert source.count('anchor_pitch_command_name="posture_command"') == 2
    assert "anchor_height_offset=0.43" not in source
    assert source.count("hand_center_offset=(0.0, 0.09734, 0.0142)") == 2
    assert "compose_wrist_chain_quat" in command_source
    assert "stick_figure_hand_center_pose" in command_source
    assert "self.anchor_offset_b" in command_source
    stick_figure_branch = command_source.split('if self.cfg.sampling_mode == "stick_figure":', 1)[1].split(
        "self.spherical_command[env_ids, 0]", 1
    )[0]
    assert "self.robot.data.body_quat_w[env_ids, self.wrist_parent_body_idx]" not in stick_figure_branch
    assert "return" in stick_figure_branch


def test_hier_builder_reuses_shared_posture_anchor_without_changing_command_shape():
    source = _read(HIER_OBS_PATH)
    command_source = _read(HIER_COMMAND_PATH)

    assert "from hbc_lab.tasks.locomotion.mdp.pose_transforms import posture_anchor_pose_w" in source
    assert "return posture_anchor_pose_w(" in source
    assert "from hbc_lab.tasks.locomotion.mdp.pose_transforms import posture_anchor_pose_w" in command_source
    assert "return posture_anchor_pose_w(" in command_source
    assert "command_state.left_wrist_pose_b" in source
    assert "command_state.right_wrist_pose_b" in source
    assert "pose_b[:, :3]" in source
    assert "pose_b[:, 3:]" in source
