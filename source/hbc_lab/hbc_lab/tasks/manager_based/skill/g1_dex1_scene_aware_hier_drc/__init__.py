"""Gym registration for the first environment-aware whole-body interaction benchmark."""

import gymnasium as gym


_ENV_ENTRY_POINT = f"{__name__}.config.scene_aware_env:G1Dex1SceneAwareEnv"
_MULTI_GEOMETRY_ENV_ENTRY_POINT = f"{__name__}.config.multi_geometry_env:G1Dex1MultiGeometryEnv"
_CFG_MODULE = f"{__name__}.config.env_cfg"
_AGENT_MODULE = f"{__name__}.config.agents.rsl_rl_ppo_cfg"


gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-HierDrc-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwarePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwarePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-HierDrc-Play-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwarePlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwarePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwarePPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-NoPerception-HierDrc-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareNoPerceptionEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareNoPerceptionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwareNoPerceptionPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-NoPerception-HierDrc-Play-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareNoPerceptionPlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareNoPerceptionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwareNoPerceptionPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-Squashed-HierDrc-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwarePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwareSquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-Squashed-HierDrc-Play-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwarePlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwarePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwareSquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-Geometry-Squashed-HierDrc-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareGeometryEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareGeometryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwareGeometrySquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-SceneAware-Geometry-Squashed-HierDrc-Play-v0",
    entry_point=_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareGeometryPlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1SceneAwareGeometryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1SceneAwareGeometrySquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-MultiGeometry-Squashed-HierDrc-v0",
    entry_point=_MULTI_GEOMETRY_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1MultiGeometryEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1MultiGeometryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1MultiGeometrySquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-MultiGeometry-Squashed-HierDrc-Play-v0",
    entry_point=_MULTI_GEOMETRY_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1MultiGeometryPlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1MultiGeometryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1MultiGeometrySquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-MultiGeometry-OOD-Squashed-HierDrc-Play-v0",
    entry_point=_MULTI_GEOMETRY_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1MultiGeometryOodPlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1MultiGeometryOodPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1MultiGeometrySquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-ReachOver-Squashed-HierDrc-v0",
    entry_point=_MULTI_GEOMETRY_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1ReachOverEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1ReachOverPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1ReachOverSquashedPPORunnerCfg",
    },
)

gym.register(
    id="HBC-Isaac-G1-Dex1-ReachOver-Squashed-HierDrc-Play-v0",
    entry_point=_MULTI_GEOMETRY_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1ReachOverPlayEnvCfg",
        "play_env_cfg_entry_point": f"{_CFG_MODULE}:G1Dex1ReachOverPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:G1Dex1ReachOverSquashedPPORunnerCfg",
    },
)
