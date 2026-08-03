from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof"
MDP_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp"


def _read(path: Path) -> str:
    return path.read_text()


def test_g1_spherical_whole_body_task_is_registered_and_has_launch_entries():
    init_source = _read(G1_ROOT / "__init__.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert 'id="HBC-Isaac-WholeBody-Spherical-Unitree-G1-v0"' in init_source
    assert "whole_body_spherical_env_cfg:SphericalWholeBodyEnvCfg" in init_source
    assert "whole_body_spherical_env_cfg:SphericalWholeBodyPlayEnvCfg" in init_source
    assert '"name": "g1_whole_body_spherical_train"' in launch_source
    assert '"name": "g1_whole_body_spherical_play"' in launch_source
    assert "--task=HBC-Isaac-WholeBody-Spherical-Unitree-G1-v0" in launch_source


def test_spherical_command_generator_is_local_debuggable_and_optionally_fixed_height():
    command_source = _read(MDP_ROOT / "commands/spherical_pose_command.py")
    command_init_source = _read(MDP_ROOT / "commands/__init__.py")

    assert "class SphericalPoseCommand(CommandTerm)" in command_source
    assert "class SphericalLevelPoseCommandCfg(CommandTermCfg)" in command_source
    assert "self.env = env" in command_source
    assert "limit_ranges: Ranges = MISSING" in command_source
    assert "anchor_body_name: str = MISSING" in command_source
    assert "anchor_height_command_name: str | None = None" in command_source
    assert "anchor_height_command_index: int = 0" in command_source
    assert "anchor_height_offset: float = 0.0" in command_source
    assert "anchor_pitch_command_name: str | None = None" in command_source
    assert "anchor_pitch_command_index: int = 1" in command_source
    assert "anchor_pitch_scale: float = 1.0" in command_source
    assert "anchor_pitch_offset: float = 0.0" in command_source
    assert "fixed_anchor_height: float | None = None" in command_source
    assert "def _spherical_to_anchor_xyz" in command_source
    assert "torch.cos(pitch)" in command_source
    assert "torch.sin(azimuth)" in command_source
    assert "if self.cfg.anchor_height_command_name is not None:" in command_source
    assert "anchor_height_command = self.env.command_manager.get_command(self.cfg.anchor_height_command_name)" in command_source
    assert "posture_command = torch.stack(" in command_source
    assert "return posture_anchor_pose_w(" in command_source
    assert "self.env.scene.env_origins" in command_source
    assert "self.cfg.anchor_height_offset" in command_source
    assert "if self.cfg.fixed_anchor_height is not None:" in command_source
    assert "anchor_pos_w[:, 2] = self.cfg.fixed_anchor_height" in command_source
    assert "yaw_quat" in command_source
    assert "posture_anchor_pose_w" in command_source
    assert "quat_mul" in command_source
    assert "if self.cfg.anchor_pitch_command_name is not None:" in command_source
    assert "anchor_pitch_command = self.env.command_manager.get_command(self.cfg.anchor_pitch_command_name)" in command_source
    assert "anchor_pitch_command[:, self.cfg.anchor_pitch_command_index]" in command_source
    assert "self.cfg.anchor_pitch_scale" in command_source
    assert "self.cfg.anchor_pitch_offset" in command_source
    assert "pitch_quat = quat_from_euler_xyz(zeros, anchor_pitch, zeros)" in command_source
    assert "anchor_quat_w = quat_mul(anchor_quat_w, pitch_quat)" in command_source
    assert "self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])" in command_source
    assert "SphericalLevelPoseCommandCfg" in command_init_source


def test_spherical_whole_body_config_uses_shoulder_anchored_octants_and_train_visualization():
    source = _read(G1_ROOT / "whole_body_spherical_env_cfg.py")

    assert "from .whole_body_env_cfg import WholeBodyEnvCfg, WholeBodyPlayEnvCfg" in source
    assert "class SphericalWholeBodyCommandsCfg" in source
    assert "left_wrist_pose = mdp.SphericalLevelPoseCommandCfg" in source
    assert "right_wrist_pose = mdp.SphericalLevelPoseCommandCfg" in source
    assert 'body_name="left_wrist_yaw_link"' in source
    assert 'body_name="right_wrist_yaw_link"' in source
    assert 'anchor_body_name="left_shoulder_pitch_link"' in source
    assert 'anchor_body_name="right_shoulder_pitch_link"' in source
    assert "fixed_anchor_height=1.23" in source
    assert "debug_vis=True" in source
    assert "l=(0.28, 0.45)" in source
    assert "l=(0.20, 0.75)" in source
    assert "pitch=(-0.5 * math.pi, 0.0)" in source
    assert "azimuth=(0.0, 0.5 * math.pi)" in source
    assert "azimuth=(-0.5 * math.pi, 0.0)" in source
    assert "func=mdp.spherical_pose_radius_cmd_levels" in source


def test_spherical_task_uses_command_world_target_for_rewards_and_observations():
    config_source = _read(G1_ROOT / "whole_body_spherical_env_cfg.py")
    reward_source = _read(MDP_ROOT / "rewards.py")
    observation_source = _read(MDP_ROOT / "observations.py")
    curriculum_source = _read(MDP_ROOT / "curriculums.py")

    assert "body_pose_command_position_error_w_exp" in config_source
    assert "body_pose_command_position_error_w_tanh" in config_source
    assert "body_pose_command_position_error_w_l2" in config_source
    assert "body_pose_command_orientation_error_w_tanh" in config_source
    assert "body_pose_command_position_error_w_in_root_frame" in config_source
    assert "def body_pose_command_position_error_w_l2" in reward_source
    assert "command_term.pose_command_w" in reward_source
    assert 'if hasattr(command_term, "_update_pose_command_w"):' in reward_source
    assert "command_term._update_pose_command_w()" in reward_source
    assert "def body_pose_command_position_error_w_in_root_frame" in observation_source
    assert 'if hasattr(command_term, "_update_pose_command_w"):' in observation_source
    assert "command_term._update_pose_command_w()" in observation_source
    assert "subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_w)" in observation_source
    assert "def spherical_pose_radius_cmd_levels" in curriculum_source
    assert "ranges.l = _expand_uniform_range(ranges.l, limit_ranges.l, radius_delta, env.device)" in curriculum_source
    assert 'for range_name in ("l",)' in curriculum_source
