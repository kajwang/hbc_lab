import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
TASK_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex1_box_carry_hier_drc"
MDP_ROOT = TASK_ROOT / "mdp"
CONFIG_ROOT = TASK_ROOT / "config"


def _read(path: Path) -> str:
    return path.read_text()


def _load_contact_progress_module():
    module_path = MDP_ROOT / "contact_progress.py"
    assert module_path.exists(), f"Missing BoxCarry contact progress module: {module_path}"
    spec = importlib.util.spec_from_file_location("g1_dex1_box_contact_progress_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bimanual_support_progress_rewards_each_required_region_and_requires_both_hands():
    module = _load_contact_progress_module()
    progress = module.compute_bimanual_support_progress(
        left_distance=torch.tensor([0.2, 0.1]),
        right_distance=torch.tensor([0.4, 0.3]),
        left_region_contacts=torch.tensor([[0.8, 0.1, 0.0], [0.9, 0.0, 0.0]]),
        right_region_contacts=torch.tensor([[0.0, 0.6, 0.1], [0.0, 0.0, 0.0]]),
    )

    assert torch.allclose(progress.distance, torch.tensor([0.4, 0.3]))
    assert torch.allclose(progress.left_support, torch.tensor([0.3, 0.3]))
    assert torch.allclose(progress.right_support, torch.tensor([0.7 / 3.0, 0.0]))
    assert torch.allclose(progress.bimanual_support, torch.tensor([0.7 / 3.0, 0.0]))


def test_box_carry_task_uses_ground_cube_far_goal_and_no_platforms():
    assets_source = _read(HBC_ROOT / "assets/objects.py")
    scenes_source = _read(MDP_ROOT / "scenes.py")
    events_source = _read(MDP_ROOT / "events.py")

    assert "BOX_CUBE_SIZE = (0.30, 0.25, 0.20)" in assets_source
    assert "BOX_CUBE_CENTER_Z = 0.5 * BOX_CUBE_SIZE[2]" in assets_source
    assert "BOX_CUBE_OBJECT_CFG" in assets_source
    assert "size=BOX_CUBE_SIZE" in assets_source
    assert "mass=10.0" in assets_source
    assert "object: RigidObjectCfg = BOX_CUBE_OBJECT_CFG" in scenes_source
    assert "object_init_platform = None" in scenes_source
    assert "object_target_platform = None" in scenes_source
    assert "left_palm_contact" in scenes_source
    assert "right_palm_contact" in scenes_source
    support_keys = scenes_source.split("BOX_SUPPORT_CONTACT_KEYS = (", maxsplit=1)[1].split(")", maxsplit=1)[0]
    assert '"palm"' in support_keys
    assert '"Link1_2"' in support_keys
    assert '"Link2_2"' in support_keys
    assert "Link1_3" not in support_keys
    assert "Link2_3" not in support_keys
    assert '"x": (1.2, 1.8)' in events_source
    assert '"object_goal_radius_range": (1.5, 2.5)' in events_source
    assert "away_heading" in events_source
    assert "env.object_initial_pos_w[env_ids] = object_pos_w" in events_source
    assert "env.object_target_pos_w[env_ids] = target_pos_w" in events_source


def test_box_carry_uses_bimanual_contact_label_and_removes_gripper_close_shaping():
    contact_labels_source = _read(
        HBC_ROOT / "tasks/manager_based/skill/g1_dex1_hier_drc/mdp/contact_labels.py"
    )
    cfg_source = _read(CONFIG_ROOT / "box_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "box_env.py")
    rewards_source = _read(MDP_ROOT / "rewards.py")

    assert "BIMANUAL_BOX_SUPPORT" in contact_labels_source
    assert "fixed_effector_mask = (1.0, 1.0)" in cfg_source
    assert "ContactMode.BIMANUAL_BOX_SUPPORT" in cfg_source
    assert "compute_bimanual_support_progress" in env_source
    assert "self.c_couple = update_ema(self.c_couple, progress.bimanual_support" in env_source
    assert "gripper_close" not in rewards_source
    assert "gated_gripper_close" not in rewards_source
    assert "early_close" not in rewards_source
    assert "env.bimanual_support_contact" in rewards_source
    assert "return 0.7 * progress + 0.3" in rewards_source
    assert "0.3 * env.c_couple" not in rewards_source


def test_box_carry_has_independent_registration_agent_and_launch_entries():
    task_init = _read(TASK_ROOT / "__init__.py")
    tasks_init = _read(HBC_ROOT / "tasks/__init__.py")
    agent_source = _read(CONFIG_ROOT / "agents/rsl_rl_ppo_cfg.py")
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert "HBC-Isaac-G1-Dex1-BoxCarry-HierDrc-v0" in task_init
    assert "HBC-Isaac-G1-Dex1-BoxCarry-HierDrc-Play-v0" in task_init
    assert "g1_dex1_box_carry_hier_drc" in tasks_init
    assert 'experiment_name = "g1_dex1_box_carry_hier_drc"' in agent_source
    assert "self.object_mass_start_w = 1.0" in flat_cfg_source
    assert '"name": "g1_dex1_box_carry_hier_drc_train"' in launch_source
    assert '"name": "g1_dex1_box_carry_hier_drc_play"' in launch_source


def test_box_carry_mass_curriculum_ends_at_two_kg():
    cfg_source = _read(CONFIG_ROOT / "box_env_cfg.py")

    assert "object_mass_start_mass: float = 10.0" in cfg_source
    assert "object_mass_ref_mass: float = 5.0" in cfg_source
    assert "object_mass_anchor_mass: float = 2.0" in cfg_source
    assert "object_mass_final_mass: float = 2.0" in cfg_source
