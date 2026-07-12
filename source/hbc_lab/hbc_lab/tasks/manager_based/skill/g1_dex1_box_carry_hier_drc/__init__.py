"""Gym registration for the G1 Dex1 bimanual BoxCarry task."""

import gymnasium as gym


gym.register(
    id="HBC-Isaac-G1-Dex1-BoxCarry-HierDrc-v0",
    entry_point=f"{__name__}.config.box_env:G1Dex1BoxCarryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1BoxCarryFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1BoxCarryFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1BoxCarryPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-BoxCarry-HierDrc-Play-v0",
    entry_point=f"{__name__}.config.box_env:G1Dex1BoxCarryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1BoxCarryFlatPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1BoxCarryFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.config.agents.rsl_rl_ppo_cfg:G1Dex1BoxCarryPPORunnerCfg",
    },
)
