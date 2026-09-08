"""Gym registration for the G1 Dex1 hierarchical drawer task."""

import gymnasium as gym


gym.register(
    id="HBC-Isaac-G1-Dex1-Drawer-HierDrc-v0",
    entry_point=f"{__name__}.config.drawer_env:G1Dex1DrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DrawerFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DrawerFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1DrawerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-Drawer-HierDrc-Play-v0",
    entry_point=f"{__name__}.config.drawer_env:G1Dex1DrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DrawerFlatPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1DrawerFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1DrawerPPORunnerCfg"
        ),
    },
)
