from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1Dex1HierDrcPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 100
    experiment_name = "g1_dex1_hier_drc"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1Dex1HierDrcRandomizedPPORunnerCfg(G1Dex1HierDrcPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "g1_dex1_hier_drc_randomized"


@configclass
class G1Dex1HierDrcMultiShapePPORunnerCfg(G1Dex1HierDrcPPORunnerCfg):
    max_iterations = 50000
    experiment_name = "g1_dex1_hier_drc_multishape"
    load_optimizer: bool = False
