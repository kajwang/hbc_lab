from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.agents.rsl_rl_ppo_cfg import (
    G1Dex1HierDrcPPORunnerCfg,
)


@configclass
class G1Dex1SceneAwarePPORunnerCfg(G1Dex1HierDrcPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "g1_dex1_scene_aware_hier_drc"
    load_optimizer: bool = False


@configclass
class G1Dex1SceneAwareNoPerceptionPPORunnerCfg(G1Dex1HierDrcPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "g1_dex1_scene_aware_no_perception_hier_drc"
    load_optimizer: bool = False


@configclass
class G1Dex1SceneAwareSquashedPPORunnerCfg(G1Dex1SceneAwarePPORunnerCfg):
    """Action-only experiment with a properly bounded Gaussian policy."""

    experiment_name = "g1_dex1_scene_aware_squashed_hier_drc"
    policy = RslRlPpoActorCriticCfg(
        class_name="SquashedGaussianActorCritic",
        init_noise_std=0.30,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        self.algorithm.class_name = "SquashedGaussianPPO"
        # Keep PPO updates inside a latent-policy trust region. Without adaptive
        # KL control, finite updates can drive the pre-tanh actor into an
        # unrecoverable saturated regime even though environment actions remain bounded.
        self.algorithm.schedule = "adaptive"
        self.algorithm.learning_rate = 5.0e-4


@configclass
class G1Dex1SceneAwareGeometrySquashedPPORunnerCfg(G1Dex1SceneAwareSquashedPPORunnerCfg):
    experiment_name = "g1_dex1_scene_aware_geometry_squashed_hier_drc"


@configclass
class G1Dex1MultiGeometrySquashedPPORunnerCfg(G1Dex1SceneAwareSquashedPPORunnerCfg):
    experiment_name = "g1_dex1_multi_geometry_squashed_hier_drc"


@configclass
class G1Dex1ReachOverSquashedPPORunnerCfg(G1Dex1SceneAwareSquashedPPORunnerCfg):
    experiment_name = "g1_dex1_reach_over_squashed_hier_drc"
