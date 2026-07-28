from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
TASK_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex1_door_open_hier_drc"
MDP_ROOT = TASK_ROOT / "mdp"
CONFIG_ROOT = TASK_ROOT / "config"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing door task file: {path}"
    return path.read_text()


def test_door_scene_replaces_pnp_object_with_articulation_and_handle_contacts():
    source = _read(MDP_ROOT / "scenes.py")

    assert "DOOR_CFG" in source
    assert "DOOR_FRAME_CFG" in source
    assert "object_init_platform = None" in source
    assert "object_target_platform = None" in source
    assert "{ENV_REGEX_NS}/door/link_2" in source
    assert "right_hand_Link1_3" in source
    assert "right_hand_Link2_3" in source
    assert "left_hand_Link1_3" in source
    assert "left_hand_Link2_3" in source


def test_door_contact_label_fixes_right_hand_grasp():
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "fixed_effector_mask = (0.0, 1.0)" in cfg_source
    assert "ContactMode.GRASP" in cfg_source
    assert "self.contact_label.set_target_region" in env_source
    assert "self.active_hand[env_ids] = RIGHT_HAND" in env_source
    assert "self.inactive_left_contact" in env_source


def test_door_actor_observes_normalized_handle_and_hinge_state():
    source = _read(MDP_ROOT / "observations.py")

    assert "def door_articulation_state" in source
    assert "env.handle_angle / env.cfg.door_latch_handle_threshold" in source
    assert "env.hinge_angle / env.cfg.door_hinge_target" in source
    assert "door_state = ObsTerm(func=door_articulation_state)" in source
    policy_source = source.split("class PolicyCfg", maxsplit=1)[1].split("class CriticCfg", maxsplit=1)[0]
    assert "door_state" in policy_source


def test_door_env_applies_latch_and_uses_grasp_for_drc():
    source = _read(CONFIG_ROOT / "door_env.py")

    assert "def _simulate_door_latch" in source
    assert "door_initial_pose_pending" in source
    assert "door_target_update_delay" in source
    assert "door_latch_stiffness" in source
    assert "door_latch_damping" in source
    assert "self._simulate_door_latch()" in source
    assert "compute_active_hand_grasp_progress" in source
    assert "self.c_couple = update_ema(self.c_couple, progress.grasp, alpha=0.2)" in source
    assert "compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)" in source
    assert "def _update_object_mass_curriculum" in source


def test_door_reward_gates_hinge_progress_by_handle_progress():
    source = _read(MDP_ROOT / "rewards.py")

    assert "env.door_manipulation_progress.reward" in source
    assert "inactive_left_contact_penalty" in source
    assert "object_fall" not in source
    assert "hier_drc_reward" in source


def test_door_reset_and_success_use_articulation_state():
    events_source = _read(MDP_ROOT / "events.py")
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "reset_door_articulation" in events_source
    assert 'door.find_joints(["joint_1", "joint_2"]' in events_source
    assert '"x": (2.0, 2.0)' in events_source
    assert "door_hinge_target: float = math.radians(60.0)" in cfg_source
    assert "door_latch_handle_threshold: float = 0.5" in cfg_source
    assert "self.hinge_angle >= self.cfg.door_hinge_target" in env_source
    assert "self.c_couple > self.cfg.success_couple_threshold" in env_source


def test_door_task_preserves_high_level_and_low_level_interfaces():
    cfg_source = _read(CONFIG_ROOT / "door_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "door_env.py")

    assert "class G1Dex1DoorOpenEnv(G1Dex1HierDrcEnv)" in env_source
    assert "class G1Dex1DoorOpenEnvCfg(G1Dex1HierDrcEnvCfg)" in cfg_source
    assert "action_dim" not in cfg_source
    assert "low_level_policy" not in env_source
    assert "object_mass_curriculum_enabled: bool = False" in cfg_source
