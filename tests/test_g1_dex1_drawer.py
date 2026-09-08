from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
ASSET_SOURCE = HBC_ROOT / "assets/articulated_objects.py"
TASK_ROOT = HBC_ROOT / "tasks/manager_based/skill/g1_dex1_drawer_hier_drc"
CONFIG_ROOT = TASK_ROOT / "config"
MDP_ROOT = TASK_ROOT / "mdp"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing drawer task file: {path}"
    return path.read_text()


def test_drawer_uses_shared_contact_conditioned_motion_quality():
    source = _read(MDP_ROOT / "rewards.py")

    assert "contact_conditioned_motion_reward" in source
    assert source.count("motion_quality = RewTerm(") == 1
    assert '"manip_scale": 100.0' in source


def test_drawer_asset_uses_sektion_cabinet_with_two_drawers_and_handles():
    asset_source = _read(ASSET_SOURCE)
    scene_source = _read(MDP_ROOT / "scenes.py")

    assert "SEKTION_CABINET_CFG" in asset_source
    assert "Sektion_Cabinet/sektion_cabinet_instanceable.usd" in asset_source
    assert '"drawer_top_joint"' in asset_source
    assert '"drawer_bottom_joint"' in asset_source
    assert 'name="drawer_handle_top"' in asset_source
    assert 'name="drawer_handle_bottom"' in asset_source
    assert "drawer_handle_top" in scene_source
    assert "drawer_handle_bottom" in scene_source
    assert "right_hand_Link1_3" in scene_source
    assert "right_hand_Link2_3" in scene_source


def test_drawer_contact_force_selects_only_the_current_environment_handle_filter():
    scene_source = _read(MDP_ROOT / "scenes.py")
    env_source = _read(CONFIG_ROOT / "drawer_env.py")

    top_filter = scene_source.index('"{ENV_REGEX_NS}/Cabinet/drawer_handle_top"')
    bottom_filter = scene_source.index('"{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom"')
    assert top_filter < bottom_filter
    assert "def _sum_sensor_force" in env_source
    assert "select_contact_filter_force" in env_source
    assert "filter_indices=self.selected_drawer" in env_source


def test_drawer_randomizes_selected_level_but_fixes_right_hand_grasp():
    cfg_source = _read(CONFIG_ROOT / "drawer_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "drawer_env.py")

    assert "top_drawer_probability: float = 0.5" in cfg_source
    assert "fixed_effector_mask = (0.0, 1.0)" in cfg_source
    assert "ContactMode.GRASP" in cfg_source
    assert "torch.rand" in env_source
    assert "self.cfg.top_drawer_probability" in env_source
    assert "self.selected_drawer" in env_source
    assert "self.contact_label.set_target_region_pose" in env_source


def test_drawer_actor_uses_shared_deployable_execution_state():
    source = _read(MDP_ROOT / "observations.py")

    assert "execution_state_obs" in source
    assert "execution = ObsTerm(func=execution_state_obs" in source


def test_drawer_motion_is_open_030_then_close_with_translation_only():
    cfg_source = _read(CONFIG_ROOT / "drawer_env_cfg.py")
    env_source = _read(CONFIG_ROOT / "drawer_env.py")

    assert "drawer_open_distance: float = 0.30" in cfg_source
    assert "self.motion_target_pos_w[env_ids, 0]" in env_source
    assert "closed_pos_w + open_delta_w" in env_source
    assert "self.motion_target_pos_w[env_ids, 1] = closed_pos_w" in env_source
    assert "self.motion_rotation_mask[env_ids] = 0.0" in env_source
    assert "self.motion_keyframe_index[advance] += 1" in env_source
    assert "self.motion_sequence_complete" in env_source


def test_drawer_actor_and_reward_keep_generic_sparse_pose_interface():
    obs_source = _read(MDP_ROOT / "observations.py")
    reward_source = _read(MDP_ROOT / "rewards.py")
    env_source = _read(CONFIG_ROOT / "drawer_env.py")

    assert "interaction_motion_obs" in obs_source
    assert "history_length = 10" in obs_source
    assert "gather_keyframe(env.motion_target_pos_w" in obs_source
    assert "gather_keyframe(env.motion_target_quat_w" in obs_source
    assert "motion_keyframe_index.to" not in obs_source
    assert "motion_guide_valid.to" not in obs_source
    assert "generic_pose_manip_reward" in reward_source
    assert "env.motion_progress" in reward_source
    assert "compute_cumulative_keyframe_progress" in env_source
    assert "motion_phase_initial_error" in env_source
    assert "motion_previous_error" not in env_source
    assert "drawer_joint_pos" not in obs_source
    assert "drawer_joint_pos" not in reward_source
    assert "open_drawer_bonus" not in reward_source
    assert "multi_stage_open_drawer" not in reward_source


def test_drawer_registration_and_local_visual_train_launch():
    registration_source = _read(TASK_ROOT / "__init__.py")
    task_registry_source = _read(HBC_ROOT / "tasks/__init__.py")
    flat_cfg_source = _read(CONFIG_ROOT / "flat_env_cfg.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert "HBC-Isaac-G1-Dex1-Drawer-HierDrc-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-Drawer-HierDrc-Play-v0" in registration_source
    assert 'import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_drawer_hier_drc")' in task_registry_source
    assert "self.scene.num_envs = 16" in flat_cfg_source
    assert "self.enable_debug_visualization = True" in flat_cfg_source
    assert '"name": "g1_dex1_drawer_hier_drc_train_visual_16"' in launch_source
    assert '"--num_envs=16"' in launch_source
    assert '"env.enable_debug_visualization=True"' in launch_source


def test_drawer_visualizes_selected_handle_and_both_keyframes_with_shared_world_frames():
    base_env_source = _read(
        HBC_ROOT / "tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py"
    )
    cfg_source = _read(CONFIG_ROOT / "drawer_env_cfg.py")

    assert '"current"' in base_env_source
    assert '"keyframe_0"' in base_env_source
    assert '"keyframe_1"' in base_env_source
    assert "flatten_motion_frame_markers" in base_env_source
    assert "self.motion_current_quat_w" in base_env_source
    assert "self.motion_target_quat_w" in base_env_source
    assert "self.motion_frame_visualizer.visualize" in base_env_source
    assert "self.motion_keyframe_debug_vis = enabled" in cfg_source
    assert "self.scene.object_frame.debug_vis = False" in cfg_source
