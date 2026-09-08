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


def test_cart_push_uses_shared_contact_conditioned_motion_quality():
    source = _source("mdp/rewards.py")

    assert "contact_conditioned_motion_reward" in source
    assert source.count("motion_quality = RewTerm(") == 1
    assert '"manip_scale": 200.0' in source


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
    assert "compute_planar_heading_quat" in source
    assert "env.cart_goal_forward" in source
    assert "env.cart_goal_lateral" in source
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
    env_source = _source("config/cart_env.py")
    assert "fixed_effector_mask = (1.0, 1.0)" in source
    assert "ContactMode.GRASP" in source
    assert "object_mass_curriculum_enabled: bool = False" in source
    assert 'self.scene["object"].data.default_mass.sum(dim=-1)' in env_source
    assert "cart_handle_target_half_width: float = 0.1" in source
    assert "cart_goal_displacement_x: tuple[float, float] = (2.0, 4.0)" in source
    assert "cart_goal_displacement_y: tuple[float, float] = (-0.5, 0.5)" in source


def test_cart_progress_requires_both_grippers():
    source = _source("config/cart_env.py")
    assert "compute_handle_targets" in source
    assert "bimanual_approach_distance" in source
    assert "bimanual_grasp_confidence" in source
    assert "cart_goal_pending" in source
    assert "cart_goal_update_delay" in source
    assert "left_grasp" in source
    assert "right_grasp" in source
    assert "self.c_couple" in source
    assert "self.W_manip" in source


def test_cart_goal_freezes_only_after_physics_settling_steps():
    env_source = _source("config/cart_env.py")
    cfg_source = _source("config/cart_env_cfg.py")
    update_source = env_source.split("def _update_contact_target_regions", 1)[1].split(
        "def _compute_progress", 1
    )[0]

    assert "cart_goal_settle_steps: int = 2" in cfg_source
    assert "cart_goal_update_delay" not in update_source
    assert "self._update_cart_goal_while_settling(handle_pos_w)" in env_source
    assert "self.cart_goal_update_delay[settling_ids] -= 1" in env_source


def test_cart_manipulation_reward_is_dense_transport_progress():
    source = _source("mdp/rewards.py")
    assert "0.3 + progress.transport_progress" in source
    assert "object_fall" not in source


def test_cart_couple_reward_credits_each_gripper_but_drc_requires_both():
    env_source = _source("config/cart_env.py")
    rewards_source = _source("mdp/rewards.py")

    assert "self.independent_contact" in env_source
    assert "self.independent_grasp" in env_source
    assert "0.2 * env.independent_contact" in rewards_source
    assert "0.2 * env.independent_grasp" in rewards_source
    assert "self.c_couple = update_ema(self.c_couple, self.bimanual_grasp" in env_source


def test_cart_keeps_low_level_interface_inherited():
    source = _source("config/cart_env.py")
    assert "class G1Dex1CartPushEnv(G1Dex1HierDrcEnv)" in source
    assert "low_level_policy" not in source
