from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
ASSET_ROOT = HBC_ROOT / "assets/robots"
SKILL_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex3_hier_drc"


def _read(path: Path) -> str:
    return path.read_text()


def test_g1_dex3_asset_cfg_uses_oasis_hand_asset_and_keeps_body_policy_compatibility():
    unitree_source = _read(ASSET_ROOT / "unitree.py")

    assert "UNITREE_G1_29DOF_DEX3_CFG" in unitree_source
    assert "UNITREE_G1_29DOF_DEX3_CFG = UNITREE_G1_29DOF_CFG.replace(" in unitree_source
    assert "**UNITREE_G1_29DOF_CFG.actuators" in unitree_source
    assert "**UNITREE_G1_29DOF_MIMIC_CFG.actuators" not in unitree_source
    assert "pos=(0.0, 0.0, 0.8)" in unitree_source
    assert '"left_hip_pitch_joint": -0.1' in unitree_source
    assert '".*_knee_joint": 0.3' in unitree_source
    assert "g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd" in unitree_source
    assert '"hand": ImplicitActuatorCfg' in unitree_source
    assert "'.*_hand_thumb_0_joint'" in unitree_source
    assert "effort_limit_sim=10.0" in unitree_source
    assert "stiffness=20.0" in unitree_source
    assert "damping=1.0" in unitree_source
    assert '"left_hand_thumb_0_joint"' in unitree_source
    assert '"right_hand_middle_1_joint"' in unitree_source


def test_g1_dex3_high_level_gripper_and_contact_progress_modules_are_local():
    gripper_source = _read(SKILL_ROOT / "mdp/gripper.py")
    progress_source = _read(SKILL_ROOT / "mdp/contact_progress.py")

    assert "LEFT_DEX3_OPEN_POSE" in gripper_source
    assert "LEFT_DEX3_CLOSE_POSE" in gripper_source
    assert "RIGHT_DEX3_CLOSE_POSE" in gripper_source
    assert "def interpolate_dex3_hand_pose" in gripper_source
    assert "class Dex3GripperController" in gripper_source
    assert "robot.set_joint_position_target" in gripper_source
    assert "grip = 1.0 means fully closed" in gripper_source

    assert "def sample_active_hands" in progress_source
    assert "def select_active_hand_value" in progress_source
    assert "def compute_hand_contact_confidence" in progress_source
    assert "def compute_dex3_hand_contact_components" in progress_source
    assert "def compute_active_hand_grasp_progress" in progress_source
    assert "opposition" in progress_source
    assert "finger_count" in progress_source
    assert "active_hand" in progress_source
    assert "ContactLabel" not in progress_source
    assert "thumb_index" not in progress_source
    assert "robot_lab" not in gripper_source
    assert "robot_lab" not in progress_source
