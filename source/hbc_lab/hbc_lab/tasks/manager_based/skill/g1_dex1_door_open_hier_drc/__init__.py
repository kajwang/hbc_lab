"""Gym registration for the G1 Dex1 hierarchical door-opening task."""

import gymnasium as gym


gym.register(
    id="HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-v0",
    entry_point=f"{__name__}.config.door_env:G1Dex1DoorOpenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DoorOpenFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DoorOpenFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1DoorOpenPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-Play-v0",
    entry_point=f"{__name__}.config.door_env:G1Dex1DoorOpenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DoorOpenFlatPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DoorOpenFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1DoorOpenPPORunnerCfg"
        ),
    },
)
