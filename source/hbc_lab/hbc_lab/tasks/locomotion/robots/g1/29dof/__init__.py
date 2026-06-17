import gymnasium as gym

gym.register(
    id="HBC-Isaac-Tracking-Rough-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"hbc_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-Tracking-Flat-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"hbc_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-WholeBody-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:WholeBodyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:WholeBodyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"hbc_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-WholeBody-Spherical-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_spherical_env_cfg:SphericalWholeBodyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.whole_body_spherical_env_cfg:SphericalWholeBodyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"hbc_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-WholeBody-SphericalPosture-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_spherical_posture_env_cfg:SphericalPostureWholeBodyEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.whole_body_spherical_posture_env_cfg:SphericalPostureWholeBodyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"hbc_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
