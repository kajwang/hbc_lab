from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof"
MDP_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp"


def _read(path: Path) -> str:
    return path.read_text()


def test_g1_whole_body_task_is_registered_separately_from_velocity_baseline():
    init_source = _read(G1_ROOT / "__init__.py")

    assert 'id="HBC-Isaac-WholeBody-Unitree-G1-v0"' in init_source
    assert "whole_body_env_cfg:WholeBodyEnvCfg" in init_source
    assert "whole_body_env_cfg:WholeBodyPlayEnvCfg" in init_source
    assert "velocity_env_cfg:RobotEnvCfg" in init_source


def test_whole_body_env_extends_velocity_with_wrist_pose_commands_and_height_map():
    source = _read(G1_ROOT / "whole_body_env_cfg.py")

    assert "from . import velocity_env_cfg" in source
    assert "class WholeBodyCommandsCfg(velocity_env_cfg.CommandsCfg)" in source
    assert "left_wrist_pose = mdp.UniformLevelPoseCommandCfg" in source
    assert "right_wrist_pose = mdp.UniformLevelPoseCommandCfg" in source
    assert "limit_ranges=mdp.UniformLevelPoseCommandCfg.Ranges" in source
    assert 'body_name="left_wrist_yaw_link"' in source
    assert 'body_name="right_wrist_yaw_link"' in source
    assert "class WholeBodyObservationsCfg(velocity_env_cfg.ObservationsCfg)" in source
    assert "height_scan = ObsTerm" in source
    assert 'command_name": "left_wrist_pose"' in source
    assert 'command_name": "right_wrist_pose"' in source
    assert "left_wrist_pose_current = ObsTerm" in source
    assert "right_wrist_pose_current = ObsTerm" in source
    assert "left_wrist_pose_error = ObsTerm" in source
    assert "right_wrist_pose_error = ObsTerm" in source
    assert "func=mdp.body_pose_in_root_frame" in source
    assert "func=mdp.body_pose_command_position_error_in_root_frame" in source
    assert "class CriticCfg(velocity_env_cfg.ObservationsCfg.CriticCfg)" in source
    assert "class WholeBodyRewardsCfg(velocity_env_cfg.RewardsCfg)" in source
    assert "track_left_wrist_pose = RewTerm" in source
    assert "track_right_wrist_pose = RewTerm" in source
    assert "track_left_wrist_pose_fine = RewTerm" in source
    assert "track_right_wrist_pose_fine = RewTerm" in source
    assert "track_left_wrist_orientation = RewTerm" in source
    assert "track_right_wrist_orientation = RewTerm" in source
    assert "penalty_left_wrist_pose_error = RewTerm" in source
    assert "penalty_right_wrist_pose_error = RewTerm" in source
    assert "joint_deviation_arms = RewTerm" in source
    assert "weight=-0.005" in source
    assert "weight=1.0" in source
    assert '"std": 0.06' in source
    assert "weight=-0.80" in source
    assert "func=mdp.body_pose_command_position_error_tanh" in source
    assert "func=mdp.body_pose_command_orientation_error_tanh" in source
    assert "func=mdp.body_pose_command_position_error_l2" in source
    assert "class WholeBodyCurriculumCfg(velocity_env_cfg.CurriculumCfg)" in source
    assert "wrist_pose_cmd_levels = CurrTerm" in source
    assert "func=mdp.pose_position_cmd_levels" in source
    assert '"penalty_term_names": ("penalty_left_wrist_pose_error", "penalty_right_wrist_pose_error")' in source
    assert '"success_threshold": 0.08' in source
    assert "class WholeBodyEnvCfg(velocity_env_cfg.RobotEnvCfg)" in source
    assert "class WholeBodyPlayEnvCfg(WholeBodyEnvCfg)" in source
    assert "self.curriculum.wrist_pose_cmd_levels = None" in source


def test_whole_body_play_keeps_wrist_commands_in_training_distribution():
    source = _read(G1_ROOT / "whole_body_env_cfg.py")

    assert "self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges" in source
    assert "self.commands.left_wrist_pose.ranges = self.commands.left_wrist_pose.limit_ranges" not in source
    assert "self.commands.right_wrist_pose.ranges = self.commands.right_wrist_pose.limit_ranges" not in source


def test_wrist_pose_z_ranges_are_relative_to_robot_root_not_world_height():
    source = _read(G1_ROOT / "whole_body_env_cfg.py")

    assert "pos_z=(-0.22, 0.02)" in source
    assert "pos_z=(-0.30, 0.28)" in source
    assert "pos_z=(0.72, 0.92)" not in source
    assert "pos_z=(0.55, 1.05)" not in source


def test_wrist_pose_sampling_starts_from_a_reachable_near_body_workspace():
    source = _read(G1_ROOT / "whole_body_env_cfg.py")

    assert "pos_x=(0.10, 0.24)" in source
    assert "pos_y=(0.18, 0.34)" in source
    assert "pos_y=(-0.34, -0.18)" in source
    assert "pos_z=(-0.22, 0.02)" in source
    assert "yaw=(0.5 * math.pi - 0.15, 0.5 * math.pi + 0.15)" in source
    assert "yaw=(-0.5 * math.pi - 0.15, -0.5 * math.pi + 0.15)" in source
    assert "pos_x=(0.08, 0.38)" in source
    assert "pos_y=(0.08, 0.42)" in source
    assert "pos_y=(-0.42, -0.08)" in source
    assert "pos_z=(-0.30, 0.28)" in source


def test_whole_body_tracking_helpers_are_local_for_breakpoints():
    reward_source = _read(MDP_ROOT / "rewards.py")
    command_source = _read(MDP_ROOT / "commands/velocity_command.py")
    curriculum_source = _read(MDP_ROOT / "curriculums.py")
    observation_source = _read(MDP_ROOT / "observations.py")

    assert "def body_pose_command_position_error_exp" in reward_source
    assert "def body_pose_command_position_error_l2" in reward_source
    assert "def body_pose_command_position_error_tanh" in reward_source
    assert "def body_pose_command_orientation_error_l2" in reward_source
    assert "def body_pose_command_orientation_error_tanh" in reward_source
    assert "env.command_manager.get_command(command_name)" in reward_source
    assert "combine_frame_transforms" in reward_source
    assert "quat_error_magnitude" in reward_source
    assert "quat_mul" in reward_source
    assert "def body_pose_command_position_error_w_l2" in reward_source
    assert "class UniformLevelPoseCommandCfg(UniformPoseCommandCfg)" in command_source
    assert "def pose_cmd_levels" in curriculum_source
    assert "command_names" in curriculum_source
    assert "def pose_position_cmd_levels" in curriculum_source
    assert "penalty_term_names" in curriculum_source
    assert 'for range_name in ("pos_x", "pos_y", "pos_z")' in curriculum_source
    assert 'for range_name in ("roll", "pitch", "yaw")' not in curriculum_source
    assert "def body_pose_in_root_frame" in observation_source
    assert "def body_pose_command_position_error_in_root_frame" in observation_source
    assert "subtract_frame_transforms" in observation_source
