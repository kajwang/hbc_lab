from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
ASSET_ROOT = HBC_ROOT / "assets/robots"
TASK_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex1_hier_drc"
MDP_ROOT = TASK_ROOT / "mdp"
CONFIG_ROOT = TASK_ROOT / "config"


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


def test_g1_dex1_task_is_registered_separately_from_dex3():
    task_init = _read(TASK_ROOT / "__init__.py")
    tasks_init = _read(HBC_ROOT / "tasks/__init__.py")

    assert "HBC-Isaac-G1-Dex1-HierDrc-v0" in task_init
    assert "HBC-Isaac-G1-Dex1-HierDrc-Play-v0" in task_init
    assert "g1_dex1_hier_drc" in tasks_init
    assert "g1_dex3_hier_drc" in tasks_init


def test_g1_dex1_hier_env_reuses_current_low_level_policy_interface():
    cfg_source = _read(CONFIG_ROOT / "g1_dex1_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "g1_dex1_env.py")

    assert "action_dim: int = 19" in cfg_source
    assert "low_level_policy_path: str = \"\"" in cfg_source
    assert "LowLevelPolicyWrapper" in env_source
    assert "self.low_level_obs_builder.build(" in env_source
    assert "Dex1GripperController" in env_source
    assert "self.gripper_controller.apply(self.command_state.left_grip, self.command_state.right_grip)" in env_source


def test_g1_dex1_uses_go2_style_two_finger_contact_progress_and_rewards():
    progress_source = _read(MDP_ROOT / "contact_progress.py")
    reward_source = _read(MDP_ROOT / "rewards.py")
    scenes_source = _read(MDP_ROOT / "scenes.py")

    assert "def compute_gripper_contact_components" in progress_source
    assert "contact = torch.minimum(left_contact, right_contact)" in progress_source
    assert "pinch = contact * pinch_score" in progress_source
    assert "grasp = contact * grip * close_allowed_gate" in progress_source
    assert "0.35 * env.c_contact" in reward_source
    assert "0.20 * env.c_pinch" in reward_source
    assert "early_close_penalty" in reward_source
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


def test_g1_dex1_launch_entries_exist():
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert '"name": "g1_dex1_hier_drc_train"' in launch_source
    assert '"--task=HBC-Isaac-G1-Dex1-HierDrc-v0"' in launch_source
    assert '"name": "g1_dex1_hier_drc_play"' in launch_source
    assert '"--task=HBC-Isaac-G1-Dex1-HierDrc-Play-v0"' in launch_source
