from __future__ import annotations

from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env_cfg import G1Dex1HierDrcEnvCfg

from ..mdp.events import G1Dex1DrawerEventCfg
from ..mdp.observations import G1Dex1DrawerObservationsCfg
from ..mdp.rewards import G1Dex1DrawerRewardsCfg
from ..mdp.scenes import DRAWER_PAD_CONTACT_SENSOR_NAMES, G1Dex1DrawerSceneCfg


@configclass
class G1Dex1DrawerEnvCfg(G1Dex1HierDrcEnvCfg):
    scene: G1Dex1DrawerSceneCfg = G1Dex1DrawerSceneCfg(num_envs=4096, env_spacing=6.0)
    events: G1Dex1DrawerEventCfg = G1Dex1DrawerEventCfg()
    observations: G1Dex1DrawerObservationsCfg = G1Dex1DrawerObservationsCfg()
    rewards: G1Dex1DrawerRewardsCfg = G1Dex1DrawerRewardsCfg()

    object_mass_curriculum_enabled: bool = False
    top_drawer_probability: float = 0.5
    drawer_open_distance: float = 0.30
    drawer_open_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    motion_position_scale: float = 0.30
    motion_rotation_scale: float = 0.5
    motion_position_tolerance: float = 0.035
    motion_rotation_tolerance: float = 0.15
    motion_keyframe_stable_steps: int = 5
    success_couple_threshold: float = 0.45
    success_steps: int = 25

    def apply_debug_visualization(self) -> None:
        super().apply_debug_visualization()
        enabled = self.enable_debug_visualization
        self.motion_keyframe_debug_vis = enabled
        self.scene.object_frame.debug_vis = False

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.commands.high_level.fixed_effector_mask = (0.0, 1.0)
        self.commands.high_level.contact_mode = int(ContactMode.GRASP)
        for sensor_name in DRAWER_PAD_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
