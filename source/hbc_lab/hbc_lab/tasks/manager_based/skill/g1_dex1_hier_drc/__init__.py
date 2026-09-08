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

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-Randomized-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.randomized_env_cfg:G1Dex1HierDrcRandomizedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.randomized_env_cfg:G1Dex1HierDrcRandomizedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcRandomizedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-Randomized-Play-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.randomized_env_cfg:G1Dex1HierDrcRandomizedPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.randomized_env_cfg:G1Dex1HierDrcRandomizedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcRandomizedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcMultiShapePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-Play-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapePlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcMultiShapePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeGraspRefEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcMultiShapePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-Play-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcMultiShapePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeObjectCenterControlEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeObjectCenterControlPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcMultiShapePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-Play-v0",
    entry_point=f"{__name__}.config.g1_dex1_env:G1Dex1HierDrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeObjectCenterControlPlayEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.config.multishape_bps_env_cfg:G1Dex1HierDrcMultiShapeObjectCenterControlPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg:G1Dex1HierDrcMultiShapePPORunnerCfg",
    },
)
