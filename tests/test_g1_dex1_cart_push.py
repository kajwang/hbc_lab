from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = (
    ROOT
    / "source"
    / "hbc_lab"
    / "hbc_lab"
    / "tasks"
    / "manager_based"
    / "skill"
    / "g1_dex1_cart_push_hier_drc"
)


def _source(relative_path: str) -> str:
    path = TASK / relative_path
    assert path.exists(), f"Missing cart task file: {path}"
    return path.read_text()


def test_cart_scene_uses_articulated_cart_and_inner_pad_contacts():
    source = _source("mdp/scenes.py")
    assert "CART_CFG" in source
    assert "CART_FRAME_CFG" in source
    assert "cart/handle" in source
    assert "Link1_3" in source
    assert "Link2_3" in source
    assert "object_init_platform = None" in source
    assert "object_target_platform = None" in source


def test_cart_reset_samples_local_goal_and_resets_all_joints():
    source = _source("mdp/events.py")
    assert "compute_cart_goal" in source
    for joint_name in (
        "RL_joint",
        "RR_joint",
        "FL_joint",
        "FR_joint",
        "RL_turn_joint",
        "RR_turn_joint",
    ):
        assert joint_name in source
    assert "cart_goal_displacement_x" in source
    assert "cart_goal_displacement_y" in source


def test_cart_is_bimanual_grasp_without_mass_curriculum():
    source = _source("config/cart_env_cfg.py")
    assert "fixed_effector_mask = (1.0, 1.0)" in source
    assert "ContactMode.GRASP" in source
    assert "object_mass_curriculum_enabled: bool = False" in source
    assert "cart_goal_displacement_x: tuple[float, float] = (2.0, 4.0)" in source
    assert "cart_goal_displacement_y: tuple[float, float] = (-0.5, 0.5)" in source


def test_cart_progress_requires_both_grippers():
    source = _source("config/cart_env.py")
    assert "compute_handle_targets" in source
    assert "bimanual_grasp_confidence" in source
    assert "left_grasp" in source
    assert "right_grasp" in source
    assert "self.c_couple" in source
    assert "self.W_manip" in source


def test_cart_manipulation_reward_is_dense_transport_progress():
    source = _source("mdp/rewards.py")
    assert "0.3 + progress.transport_progress" in source
    assert "object_fall" not in source


def test_cart_keeps_low_level_interface_inherited():
    source = _source("config/cart_env.py")
    assert "class G1Dex1CartPushEnv(G1Dex1HierDrcEnv)" in source
    assert "low_level_policy" not in source
