"""Gym registration for G1 Dex3 hierarchical interaction tasks."""

import gymnasium as gym


gym.register(
    id="HBC-Isaac-G1-Dex3-HierDrc-v0",
    entry_point=f"{__name__}.config.g1_dex3_env:G1Dex3HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex3HierDrcFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex3HierDrcFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex3HierDrcPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex3-HierDrc-Play-v0",
    entry_point=f"{__name__}.config.g1_dex3_env:G1Dex3HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex3HierDrcFlatPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.flat_env_cfg:G1Dex3HierDrcFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex3HierDrcPPORunnerCfg",
    },
)
