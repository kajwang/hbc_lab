from __future__ import annotations

from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env_cfg import (
    G1Dex1HierDrcEnvCfg,
)

from ..mdp.events import G1Dex1CartPushEventCfg
from ..mdp.rewards import G1Dex1CartPushRewardsCfg
from ..mdp.scenes import CART_PAD_CONTACT_SENSOR_NAMES, G1Dex1CartPushSceneCfg


@configclass
class G1Dex1CartPushEnvCfg(G1Dex1HierDrcEnvCfg):
    scene: G1Dex1CartPushSceneCfg = G1Dex1CartPushSceneCfg(num_envs=4096, env_spacing=12.0)
    events: G1Dex1CartPushEventCfg = G1Dex1CartPushEventCfg()
    rewards: G1Dex1CartPushRewardsCfg = G1Dex1CartPushRewardsCfg()

    object_mass_curriculum_enabled: bool = False
    cart_handle_target_half_width: float = 0.1
    cart_goal_settle_steps: int = 2
    cart_goal_displacement_x: tuple[float, float] = (2.0, 4.0)
    cart_goal_displacement_y: tuple[float, float] = (-0.5, 0.5)
    success_distance: float = 0.25
    success_couple_threshold: float = 0.45
    success_steps: int = 25

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 40.0
        self.commands.high_level.fixed_effector_mask = (1.0, 1.0)
        self.commands.high_level.contact_mode = int(ContactMode.GRASP)
        self.events.reset_object.params["cart_goal_displacement_x"] = self.cart_goal_displacement_x
        self.events.reset_object.params["cart_goal_displacement_y"] = self.cart_goal_displacement_y
        for sensor_name in CART_PAD_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
