from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/robots/g1/29dof"
MDP_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp"


def _read(path: Path) -> str:
    return path.read_text()


def test_g1_spherical_posture_task_is_registered_and_has_launch_entries():
    init_source = _read(G1_ROOT / "__init__.py")
    launch_source = _read(REPO_ROOT / ".vscode/launch.json")

    assert 'id="HBC-Isaac-WholeBody-SphericalPosture-Unitree-G1-v0"' in init_source
    assert "whole_body_spherical_posture_env_cfg:SphericalPostureWholeBodyEnvCfg" in init_source
    assert "whole_body_spherical_posture_env_cfg:SphericalPostureWholeBodyPlayEnvCfg" in init_source
    assert '"name": "g1_whole_body_spherical_posture_train"' in launch_source
    assert '"name": "g1_whole_body_spherical_posture_play"' in launch_source
    assert "--task=HBC-Isaac-WholeBody-SphericalPosture-Unitree-G1-v0" in launch_source


def test_posture_command_generator_is_local_debuggable_and_two_dimensional():
    command_source = _read(MDP_ROOT / "commands/posture_command.py")
    command_init_source = _read(MDP_ROOT / "commands/__init__.py")

    assert "class UniformPostureCommand(CommandTerm)" in command_source
    assert "class UniformLevelPostureCommandCfg(CommandTermCfg)" in command_source
    assert "self.posture_command = torch.zeros(self.num_envs, 2" in command_source
    assert "root_height" in command_source
    assert "torso_pitch" in command_source
    assert "euler_xyz_from_quat" in command_source
    assert "asset.data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]" in command_source
    assert "self.metrics[\"root_height_error\"]" in command_source
    assert "self.metrics[\"torso_pitch_error\"]" in command_source
    assert "GREEN_ARROW_X_MARKER_CFG" in command_source
    assert "import isaaclab.sim as sim_utils" in command_source
    assert "posture_command_visualizer_cfg" in command_source
    assert 'posture_command_visualizer_cfg.markers["arrow"].scale = (0.9, 0.025, 0.025)' in command_source
    assert 'diffuse_color=(1.0, 0.05, 0.65)' in command_source
    assert "self.posture_command_visualizer.visualize" in command_source
    assert "arrow_pos_w[:, 2] = self.env.scene.env_origins[:, 2] + self.posture_command[:, 0]" in command_source
    assert "quat_from_euler_xyz(zeros, self.posture_command[:, 1], zeros)" in command_source
    assert "UniformLevelPostureCommandCfg" in command_init_source


def test_spherical_posture_config_adds_posture_command_obs_rewards_and_curriculum():
    source = _read(G1_ROOT / "whole_body_spherical_posture_env_cfg.py")

    assert "from .whole_body_spherical_env_cfg import SphericalWholeBodyEnvCfg" in source
    assert "class SphericalPostureWholeBodyCommandsCfg" in source
    assert "posture_command = mdp.UniformLevelPostureCommandCfg" in source
    assert 'body_name="torso_link"' in source
    assert "debug_vis=True" in source
    assert "root_height=(0.76, 0.80)" in source
    assert "root_height=(0.55, 0.80)" in source
    assert "torso_pitch=(0.0, 0.12)" in source
    assert "torso_pitch=(0.0, 0.45)" in source
    assert "posture_command = ObsTerm" in source
    assert 'params={"command_name": "posture_command"}' in source
    assert "posture_command_error = ObsTerm" in source
    assert "func=mdp.posture_command_error" in source
    assert "track_root_height = RewTerm" in source
    assert "func=mdp.root_height_command_error_l2" in source
    assert "track_torso_pitch = RewTerm" in source
    assert "func=mdp.torso_pitch_command_error_l2" in source
    assert "base_height = None" in source
    assert "flat_orientation_l2 = None" in source
    assert "joint_deviation_waists = RewTerm" in source
    assert 'joint_names=["waist_yaw_joint", "waist_roll_joint"]' in source
    assert 'joint_names=["waist.*"]' not in source
    assert "weight=-0.05" in source
    assert "posture_cmd_levels = CurrTerm" in source
    assert "func=mdp.posture_cmd_levels" in source


def test_posture_tracking_helpers_are_local_and_curriculum_expands_only_posture_ranges():
    reward_source = _read(MDP_ROOT / "rewards.py")
    observation_source = _read(MDP_ROOT / "observations.py")
    curriculum_source = _read(MDP_ROOT / "curriculums.py")

    assert "def current_posture" in observation_source
    assert "def posture_command_error" in observation_source
    assert "def root_height_command_error_l2" in reward_source
    assert "def torso_pitch_command_error_l2" in reward_source
    assert "def posture_cmd_levels" in curriculum_source
    assert "ranges.root_height = _expand_lower_bound" in curriculum_source
    assert "ranges.torso_pitch = _expand_upper_bound" in curriculum_source
    assert 'for range_name in ("root_height", "torso_pitch")' in curriculum_source
