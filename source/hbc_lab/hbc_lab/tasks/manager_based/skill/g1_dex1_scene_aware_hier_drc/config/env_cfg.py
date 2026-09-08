from __future__ import annotations

from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env_cfg import G1Dex1HierDrcEnvCfg

from ..mdp.events import G1Dex1SceneAwareEventCfg
from ..mdp.multi_geometry import REACH_OVER_FAMILY
from ..mdp.multi_geometry_events import G1Dex1MultiGeometryEventCfg
from ..mdp.multi_geometry_scenes import G1Dex1MultiGeometrySceneCfg
from ..mdp.observations import G1Dex1SceneAwareObservationsCfg
from ..mdp.rewards import G1Dex1SceneAwareGeometryRewardsCfg, G1Dex1SceneAwareRewardsCfg
from ..mdp.scenes import G1Dex1SceneAwareSceneCfg


@configclass
class G1Dex1SceneAwareEnvCfg(G1Dex1HierDrcEnvCfg):
    scene: G1Dex1SceneAwareSceneCfg = G1Dex1SceneAwareSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: G1Dex1SceneAwareObservationsCfg = G1Dex1SceneAwareObservationsCfg()
    rewards: G1Dex1SceneAwareRewardsCfg = G1Dex1SceneAwareRewardsCfg()
    events: G1Dex1SceneAwareEventCfg = G1Dex1SceneAwareEventCfg()

    environment_perception_enabled: bool = True
    environment_scan_max_distance: float = 3.0
    environment_voxel_half_life_s: float = 2.0
    environment_voxel_min_occupancy: float = 0.02
    environment_scan_debug_vis: bool = False
    environment_voxel_debug_vis: bool = False
    environment_voxel_debug_threshold: float = 0.08
    environment_voxel_debug_max_points_per_env: int = 768
    environment_voxel_debug_max_envs: int = 16

    def apply_debug_visualization(self) -> None:
        super().apply_debug_visualization()
        self.environment_scan_debug_vis = (
            self.environment_scan_debug_vis or self.enable_debug_visualization
        )
        self.environment_voxel_debug_vis = (
            self.environment_voxel_debug_vis or self.enable_debug_visualization
        )

    def __post_init__(self):
        super().__post_init__()
        self.commands.high_level.left_hand_probability = 0.5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex1SceneAwarePlayEnvCfg(G1Dex1SceneAwareEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.object_mass_start_w = 1.0
        self.enable_debug_visualization = True
        self.environment_scan_debug_vis = True
        self.environment_voxel_debug_vis = True


@configclass
class G1Dex1SceneAwareNoPerceptionEnvCfg(G1Dex1SceneAwareEnvCfg):
    environment_perception_enabled: bool = False


@configclass
class G1Dex1SceneAwareNoPerceptionPlayEnvCfg(G1Dex1SceneAwarePlayEnvCfg):
    environment_perception_enabled: bool = False


@configclass
class G1Dex1SceneAwareGeometryEnvCfg(G1Dex1SceneAwareEnvCfg):
    rewards: G1Dex1SceneAwareGeometryRewardsCfg = G1Dex1SceneAwareGeometryRewardsCfg()


@configclass
class G1Dex1SceneAwareGeometryPlayEnvCfg(G1Dex1SceneAwareGeometryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.object_mass_start_w = 1.0
        self.enable_debug_visualization = True
        self.environment_scan_debug_vis = True
        self.environment_voxel_debug_vis = True


@configclass
class G1Dex1MultiGeometryEnvCfg(G1Dex1SceneAwareGeometryEnvCfg):
    scene: G1Dex1MultiGeometrySceneCfg = G1Dex1MultiGeometrySceneCfg(
        num_envs=4096,
        env_spacing=10.0,
    )
    events: G1Dex1MultiGeometryEventCfg = G1Dex1MultiGeometryEventCfg()

    geometry_curriculum_enabled: bool = True
    geometry_preview_sweep: bool = False
    geometry_ood_mode: bool = False
    train_family_id: int = -1
    # Play-only exact scene selector. Negative ids retain the balanced training layout.
    play_active_id: int = -1
    play_env_id: int = -1
    play_geo_level: float = 1.0
    geometry_curriculum_start_level: float = 0.0
    geometry_safe_reach_distance: float = 0.32
    geometry_safe_clearance: float = 0.04
    geometry_safe_reach_steps: int = 10
    geometry_curriculum_safe_reach_ema_alpha: float = 0.01
    geometry_curriculum_advance_threshold: float = 0.45
    geometry_curriculum_update_interval: int = 500
    geometry_curriculum_level_step: float = 0.10

    object_mass_family_start_level: float = 0.0


@configclass
class G1Dex1MultiGeometryPlayEnvCfg(G1Dex1MultiGeometryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.geometry_curriculum_enabled = False
        self.geometry_preview_sweep = False
        self.play_active_id = 1
        self.play_env_id = 0
        self.play_geo_level = 1.0
        self.object_mass_family_start_level = 1.0
        self.object_mass_start_w = 1.0
        self.enable_debug_visualization = True
        self.environment_scan_debug_vis = True
        self.environment_voxel_debug_vis = True


@configclass
class G1Dex1MultiGeometryOodPlayEnvCfg(G1Dex1MultiGeometryPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.geometry_ood_mode = True
        self.geometry_curriculum_start_level = 1.0


@configclass
class G1Dex1ReachOverEnvCfg(G1Dex1MultiGeometryEnvCfg):
    train_family_id: int = REACH_OVER_FAMILY


@configclass
class G1Dex1ReachOverPlayEnvCfg(G1Dex1MultiGeometryPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.play_env_id = REACH_OVER_FAMILY
        self.geometry_preview_sweep = True
