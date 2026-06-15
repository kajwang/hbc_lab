from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HBC_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab"
LOCOMOTION_ROOT = HBC_ROOT / "tasks/locomotion"
G1_ROOT = LOCOMOTION_ROOT / "robots/g1/29dof"
ASSET_ROOT = HBC_ROOT / "assets/robots"


def _read(path: Path) -> str:
    return path.read_text()


def test_official_unitree_assets_are_local_and_configurable():
    unitree_source = _read(ASSET_ROOT / "unitree.py")
    actuator_source = _read(ASSET_ROOT / "unitree_actuators.py")

    assert 'UNITREE_MODEL_DIR = os.environ.get("UNITREE_MODEL_DIR", "/home/kaijun/wbc/unitree_model")' in unitree_source
    assert 'UNITREE_ROS_DIR = os.environ.get("UNITREE_ROS_DIR", "/home/kaijun/wbc/unitree_ros")' in unitree_source
    assert "from hbc_lab.assets.robots import unitree_actuators" in unitree_source
    assert "UnitreeUsdFileCfg" in unitree_source
    assert "UNITREE_G1_29DOF_CFG" in unitree_source
    assert '"N7520-14.3": ImplicitActuatorCfg' in unitree_source
    assert '"N7520-22.5": ImplicitActuatorCfg' in unitree_source
    assert '"N5020-16": ImplicitActuatorCfg' in unitree_source
    assert "class UnitreeActuator(" in actuator_source
    assert "UnitreeActuatorCfg_N7520_14p3" in actuator_source
    assert "unitree_rl_lab" not in unitree_source


def test_g1_velocity_env_is_official_recipe_in_hbc_namespace():
    env_source = _read(G1_ROOT / "velocity_env_cfg.py")

    assert "from hbc_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG" in env_source
    assert "from hbc_lab.tasks.locomotion import mdp" in env_source
    assert "ManagerBasedRLEnvCfg" in env_source
    assert "UniformLevelVelocityCommandCfg" in env_source
    assert "lin_vel_x=(-0.1, 0.1)" in env_source
    assert "lin_vel_y=(-0.1, 0.1)" in env_source
    assert "ang_vel_z=(-0.1, 0.1)" in env_source
    assert "lin_vel_x=(-0.5, 1.0)" in env_source
    assert "lin_vel_y=(-0.3, 0.3)" in env_source
    assert "ang_vel_z=(-0.2, 0.2)" in env_source
    assert "track_lin_vel_xy = RewTerm" in env_source
    assert "gait = RewTerm" in env_source
    assert "feet_clearance = RewTerm" in env_source
    assert "bad_orientation = DoneTerm" in env_source
    assert "terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)" in env_source
    assert "lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)" in env_source
    assert "self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges" in env_source
    assert "self.curriculum.terrain_levels = None" in env_source
    assert "self.curriculum.lin_vel_cmd_levels = None" in env_source
    assert "self.scene.terrain.terrain_generator.curriculum = False" in env_source
    assert "unitree_rl_lab" not in env_source
    assert "robot_lab.tasks.manager_based.locomotion.velocity.mdp" not in env_source


def test_g1_registration_keeps_hbc_task_ids_and_official_agent_entrypoint():
    init_source = _read(G1_ROOT / "__init__.py")

    assert 'id="HBC-Isaac-Tracking-Rough-Unitree-G1-v0"' in init_source
    assert 'id="HBC-Isaac-Tracking-Flat-Unitree-G1-v0"' in init_source
    assert "isaaclab.envs:ManagerBasedRLEnv" in init_source
    assert "velocity_env_cfg:RobotEnvCfg" in init_source
    assert "velocity_env_cfg:RobotPlayEnvCfg" in init_source
    assert "hbc_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg" in init_source
    assert "unitree_rl_lab" not in init_source


def test_locomotion_mdp_is_local_for_debuggable_breakpoints():
    mdp_init_source = _read(LOCOMOTION_ROOT / "mdp/__init__.py")
    reward_source = _read(LOCOMOTION_ROOT / "mdp/rewards.py")
    command_source = _read(LOCOMOTION_ROOT / "mdp/commands/velocity_command.py")
    curriculum_source = _read(LOCOMOTION_ROOT / "mdp/curriculums.py")

    assert "from .commands import *" in mdp_init_source
    assert "from .curriculums import *" in mdp_init_source
    assert "from .rewards import *" in mdp_init_source
    assert "def track_lin_vel_xy_yaw_frame_exp" in reward_source
    assert "def track_ang_vel_z_exp" in reward_source
    assert "def feet_gait" in reward_source
    assert "class UniformLevelVelocityCommandCfg" in command_source
    assert "limit_ranges" in command_source
    assert "def lin_vel_cmd_levels" in curriculum_source
    assert "reward > reward_term.weight * 0.8" in curriculum_source


def test_task_registration_imports_official_locomotion_package_only():
    task_init_source = _read(HBC_ROOT / "tasks/__init__.py")
    legacy_g1_root = HBC_ROOT / "tasks/manager_based/humanoid_tracking/g1"
    legacy_g1_init_source = _read(legacy_g1_root / "__init__.py")

    assert 'hbc_lab.tasks.locomotion.robots.g1.29dof' in task_init_source
    assert "robot_lab" not in task_init_source
    assert "gym.register" not in legacy_g1_init_source
    assert "HBC-Isaac-Tracking-Rough-Unitree-G1-v0" not in legacy_g1_init_source
    assert not (legacy_g1_root / "rough_env_cfg.py").exists()
    assert not (legacy_g1_root / "flat_env_cfg.py").exists()
    assert not (legacy_g1_root / "agents/rsl_rl_ppo_cfg.py").exists()
