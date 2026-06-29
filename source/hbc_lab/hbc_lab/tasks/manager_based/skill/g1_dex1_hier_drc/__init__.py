"""Gym registration for G1 Dex1 hierarchical interaction tasks."""

import gymnasium as gym


gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1HierDrcFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1HierDrcFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-Play-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1HierDrcFlatPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex1HierDrcFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcPPORunnerCfg",
    },
)
