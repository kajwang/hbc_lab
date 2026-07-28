"""Gym registration for the G1 Dex1 hierarchical bimanual cart-pushing task."""

import gymnasium as gym


gym.register(
    id="HBC-Isaac-G1-Dex1-CartPush-HierDrc-v0",
    entry_point=f"{__name__}.config.cart_env:G1Dex1CartPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1CartPushFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1CartPushFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1CartPushPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-CartPush-HierDrc-Play-v0",
    entry_point=f"{__name__}.config.cart_env:G1Dex1CartPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1CartPushFlatPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1CartPushFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1CartPushPPORunnerCfg"
        ),
    },
)
