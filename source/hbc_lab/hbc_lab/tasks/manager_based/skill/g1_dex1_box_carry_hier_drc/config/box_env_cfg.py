from __future__ import annotations

from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env_cfg import G1Dex1HierDrcEnvCfg
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.contact_labels import ContactMode

from ..mdp.events import G1Dex1BoxCarryEventCfg
from ..mdp.rewards import G1Dex1BoxCarryRewardsCfg
from ..mdp.scenes import (
    BOX_PALM_CONTACT_SENSOR_NAMES,
    G1Dex1BoxCarrySceneCfg,
)


@configclass
class G1Dex1BoxCarryEnvCfg(G1Dex1HierDrcEnvCfg):
    scene: G1Dex1BoxCarrySceneCfg = G1Dex1BoxCarrySceneCfg(num_envs=4096, env_spacing=10.0)
    events: G1Dex1BoxCarryEventCfg = G1Dex1BoxCarryEventCfg()
    rewards: G1Dex1BoxCarryRewardsCfg = G1Dex1BoxCarryRewardsCfg()

    object_goal_radius_range: tuple[float, float] = (1.5, 2.5)
    object_mass_start_mass: float = 10.0
    object_mass_ref_mass: float = 5.0
    object_mass_anchor_mass: float = 2.0
    object_mass_final_mass: float = 2.0
    object_mass_min: float = 0.5
    object_mass_max: float = 15.0
    success_distance: float = 0.20

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.commands.high_level.fixed_effector_mask = (1.0, 1.0)
        self.commands.high_level.contact_mode = int(ContactMode.BIMANUAL_BOX_SUPPORT)
        for sensor_name in BOX_PALM_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
