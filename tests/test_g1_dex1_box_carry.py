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


def _load_face_targets_module():
    module_path = MDP_ROOT / "face_targets.py"
    assert module_path.exists(), f"Missing BoxCarry face-target module: {module_path}"
    spec = importlib.util.spec_from_file_location("g1_dex1_box_face_targets_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_long_axis_face_centers_follow_object_translation_and_yaw():
    module = _load_face_targets_module()
    object_pos = torch.tensor([[1.0, 2.0, 0.1], [0.0, 0.0, 0.1]])
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    yaw_90 = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])

    positive, negative = module.compute_long_axis_face_centers(
        object_pos,
        torch.stack((identity, yaw_90)),
        half_extent=0.15,
    )

    assert torch.allclose(positive[0], torch.tensor([1.15, 2.0, 0.1]), atol=1.0e-6)
    assert torch.allclose(negative[0], torch.tensor([0.85, 2.0, 0.1]), atol=1.0e-6)
    assert torch.allclose(positive[1], torch.tensor([0.0, 0.15, 0.1]), atol=1.0e-6)
    assert torch.allclose(negative[1], torch.tensor([0.0, -0.15, 0.1]), atol=1.0e-6)


def test_face_assignment_uses_the_shorter_pairing():
    module = _load_face_targets_module()
    positive = torch.tensor([[0.0, 0.2, 0.1], [0.0, 0.2, 0.1]])
    negative = torch.tensor([[0.0, -0.2, 0.1], [0.0, -0.2, 0.1]])
    left_hand = torch.tensor([[0.0, 0.4, 0.1], [0.0, -0.4, 0.1]])
    right_hand = torch.tensor([[0.0, -0.4, 0.1], [0.0, 0.4, 0.1]])

    left_positive = module.choose_left_positive_assignment(
        positive,
        negative,
        left_hand,
        right_hand,
    )
    left_target, right_target = module.select_assigned_face_targets(
        positive,
        negative,
        left_positive,
    )

    assert torch.equal(left_positive, torch.tensor([True, False]))
    assert torch.allclose(left_target, torch.stack((positive[0], negative[1])))
    assert torch.allclose(right_target, torch.stack((negative[0], positive[1])))


def test_manip_reward_requires_lift_before_transport_progress():
    module = _load_contact_progress_module()
    progress = module.compute_lift_gated_transport_progress(
        lift_height=torch.tensor([0.0, 0.20, 0.40]),
        transport_progress=torch.tensor([1.0, 0.8, 0.8]),
        lift_target_height=0.40,
    )

    assert torch.allclose(progress.lift_progress, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(progress.transport_gate, progress.lift_progress)
    assert torch.allclose(progress.reward, torch.tensor([0.3, 0.59, 0.88]), atol=1.0e-6)

    env_source = _read(CONFIG_ROOT / "box_env.py")
    rewards_source = _read(MDP_ROOT / "rewards.py")
    assert "self.d_goal_xy = torch.norm((object_pos_w - self.object_target_pos_w)[:, :2]" in env_source
    assert "env.d_goal_xy" in rewards_source
    assert '"DRC/lift_progress_mean"' in rewards_source
    assert '"DRC/transport_progress_mean"' in rewards_source


def test_bimanual_support_separates_contact_gate_from_support_density():
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
    assert torch.allclose(progress.left_contact_gate, torch.tensor([0.8, 0.9]))
    assert torch.allclose(progress.right_contact_gate, torch.tensor([0.6, 0.0]))
    assert torch.allclose(progress.couple_gate, torch.tensor([0.6, 0.0]))
    assert torch.allclose(progress.support_density, torch.tensor([4.0 / 15.0, 0.15]))


def test_bimanual_support_reward_gates_each_hand_by_its_own_distance():
    module = _load_contact_progress_module()
    assert hasattr(module, "compute_independent_support_reward_terms")

    left_gate, right_gate, gated_support, early_contact = module.compute_independent_support_reward_terms(
        left_distance=torch.tensor([0.0]),
        right_distance=torch.tensor([0.15]),
        left_support=torch.tensor([0.8]),
        right_support=torch.tensor([0.6]),
        distance_scale=0.15,
    )
    expected_right_gate = 1.0 - torch.tanh(torch.tensor([1.0]))

    assert torch.allclose(left_gate, torch.ones(1))
    assert torch.allclose(right_gate, expected_right_gate)
    assert torch.allclose(gated_support, 0.5 * (0.8 + 0.6 * expected_right_gate))
    assert torch.allclose(early_contact, 0.5 * 0.6 * (1.0 - expected_right_gate))


def test_bimanual_couple_progress_requires_both_distance_gates():
    module = _load_contact_progress_module()
    assert hasattr(module, "compute_bimanual_distance_gated_couple")

    valid_couple = module.compute_bimanual_distance_gated_couple(
        left_gate=torch.tensor([0.8, 0.4]),
        right_gate=torch.tensor([0.2, 0.9]),
        raw_couple=torch.tensor([0.7, 0.5]),
    )

    assert torch.allclose(valid_couple, torch.tensor([0.14, 0.20]))


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
    assert '"palm"' not in support_keys
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
    assert "self.bimanual_support_contact = progress.support_density" in env_source
    assert "self.bimanual_contact_gate = progress.couple_gate" in env_source
    assert "self.c_grasp = update_ema(self.c_grasp, progress.support_density" in env_source
    assert "self.c_couple = update_ema(self.c_couple, self.distance_gated_couple" in env_source
    assert "gripper_close" not in rewards_source
    assert "gated_gripper_close" not in rewards_source
    assert "early_close" not in rewards_source
    assert "left_support=progress.left_support" in env_source
    assert "right_support=progress.right_support" in env_source
    assert "compute_lift_gated_transport_progress" in rewards_source
    assert "lift_target_height=0.40" in rewards_source
    assert "0.3 * env.c_couple" not in rewards_source


def test_box_carry_gates_support_per_hand_and_distance_gates_couple_progress():
    env_source = _read(CONFIG_ROOT / "box_env.py")
    rewards_source = _read(MDP_ROOT / "rewards.py")

    assert "compute_independent_support_reward_terms" in env_source
    assert "distance_scale=0.15" in env_source
    assert "0.5 * both_near" in rewards_source
    assert "+ 0.5 * env.gated_support" in rewards_source
    assert "- 0.25 * env.early_contact" in rewards_source
    assert "position_couple" not in rewards_source
    assert "compute_bimanual_distance_gated_couple" in env_source
    assert "self.distance_gated_couple" in env_source
    assert "self.c_couple = update_ema(self.c_couple, self.distance_gated_couple" in env_source


def test_box_carry_has_no_task_specific_object_leg_contact_penalty():
    scenes_source = _read(MDP_ROOT / "scenes.py")
    env_source = _read(CONFIG_ROOT / "box_env.py")
    cfg_source = _read(CONFIG_ROOT / "box_env_cfg.py")
    rewards_source = _read(MDP_ROOT / "rewards.py")

    assert "object_leg_contact" not in scenes_source
    assert "BOX_FORBIDDEN_LEG_BODY_NAMES" not in scenes_source
    assert "OBJECT_LEG_CONTACT_SENSOR_NAME" not in scenes_source
    assert "object_leg_contact" not in env_source
    assert "OBJECT_LEG_CONTACT_SENSOR_NAME" not in env_source
    assert "object_leg_contact_force_threshold" not in cfg_source
    assert "OBJECT_LEG_CONTACT_SENSOR_NAME" not in cfg_source
    assert "object_leg_contact_penalty" not in rewards_source


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


def test_box_env_uses_episode_fixed_face_targets_for_progress():
    env_source = _read(CONFIG_ROOT / "box_env.py")

    assert "self.left_face_uses_positive" in env_source
    assert "self.face_assignment_pending" in env_source
    assert "compute_long_axis_face_centers" in env_source
    assert "choose_left_positive_assignment" in env_source
    assert "select_assigned_face_targets" in env_source
    assert "0.5 * BOX_CUBE_SIZE[0]" in env_source
    assert "hand_center_pos_w[:, 0, :] - self.left_face_target_pos_w" in env_source
    assert "hand_center_pos_w[:, 1, :] - self.right_face_target_pos_w" in env_source
    assert "raw_couple=progress.couple_gate" in env_source
    assert "self.face_assignment_pending[env_ids] = True" in env_source
    assert "compute_bimanual_position_relation" not in env_source


def test_box_carry_observes_assigned_face_targets_in_root_frame():
    observations_path = MDP_ROOT / "observations.py"
    cfg_source = _read(CONFIG_ROOT / "box_env_cfg.py")

    assert observations_path.exists()
    observations_source = _read(observations_path)
    assert "object_goal_hand_obs(env)" in observations_source
    assert "env._update_face_targets()" in observations_source
    assert "env.left_face_target_pos_w - robot.data.root_pos_w" in observations_source
    assert "env.right_face_target_pos_w - robot.data.root_pos_w" in observations_source
    assert "quat_apply_inverse" in observations_source
    assert "self.observations.policy.task.func = box_object_goal_hand_obs" in cfg_source
    assert "self.observations.critic.task.func = box_object_goal_hand_obs" in cfg_source


def test_box_carry_uses_command_state_and_ten_frame_actor_history():
    cfg_source = _read(CONFIG_ROOT / "box_env_cfg.py")
    base_observations_source = _read(
        HBC_ROOT / "tasks/manager_based/skill/g1_dex1_hier_drc/mdp/observations.py"
    )

    assert "self.observations.policy.history_length = 10" in cfg_source
    assert "def high_level_command_state_obs" in base_observations_source
    assert "command_state = ObsTerm(func=high_level_command_state_obs)" in base_observations_source
    assert "matrix_from_quat" in base_observations_source


def test_box_carry_visualizes_and_logs_assigned_face_targets():
    env_source = _read(CONFIG_ROOT / "box_env.py")
    mdp_init_source = _read(MDP_ROOT / "__init__.py")

    assert (MDP_ROOT / "face_targets.py").exists()
    assert "face_targets" in mdp_init_source
    assert "LEFT_FACE_TARGET_MARKER_CFG" in env_source
    assert "RIGHT_FACE_TARGET_MARKER_CFG" in env_source
    assert "self.left_face_target_visualizer.visualize(self.left_face_target_pos_w)" in env_source
    assert "self.right_face_target_visualizer.visualize(self.right_face_target_pos_w)" in env_source
    assert '"BoxCarry/left_face_target_error"' in env_source
    assert '"BoxCarry/right_face_target_error"' in env_source
    assert '"BoxCarry/left_positive_assignment_ratio"' in env_source


def test_box_carry_forwards_pre_reset_active_hand_to_base_contact_logging():
    env_source = _read(CONFIG_ROOT / "box_env.py")

    assert "def _log_link_contact_diagnostics(self, active_hand: torch.Tensor | None = None)" in env_source
    assert "super()._log_link_contact_diagnostics(active_hand)" in env_source
