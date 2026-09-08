from __future__ import annotations

import math

from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env_cfg import (
    G1Dex1HierDrcEnvCfg,
)

from ..mdp.events import G1Dex1DoorOpenEventCfg
from ..mdp.observations import G1Dex1DoorOpenObservationsCfg
from ..mdp.rewards import G1Dex1DoorOpenRewardsCfg
from ..mdp.scenes import DOOR_PAD_CONTACT_SENSOR_NAMES, G1Dex1DoorOpenSceneCfg


@configclass
class G1Dex1DoorOpenEnvCfg(G1Dex1HierDrcEnvCfg):
    scene: G1Dex1DoorOpenSceneCfg = G1Dex1DoorOpenSceneCfg(num_envs=4096, env_spacing=8.0)
    events: G1Dex1DoorOpenEventCfg = G1Dex1DoorOpenEventCfg()
    observations: G1Dex1DoorOpenObservationsCfg = G1Dex1DoorOpenObservationsCfg()
    rewards: G1Dex1DoorOpenRewardsCfg = G1Dex1DoorOpenRewardsCfg()

    object_mass_curriculum_enabled: bool = False
    door_latch_handle_threshold: float = math.radians(82.0)
    door_latch_hinge_release_threshold: float = 0.2
    door_latch_stiffness: float = 5000.0
    door_latch_damping: float = 10.0
    door_hinge_target: float = math.radians(60.0)
    motion_unlock_angle: float = math.pi / 2.0
    motion_unlock_axis_local: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    motion_position_scale: float = 0.25
    motion_rotation_scale: float = 0.5
    motion_position_tolerance: float = 0.06
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
        for sensor_name in DOOR_PAD_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
