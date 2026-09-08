from __future__ import annotations

from isaaclab.utils import configclass

from ..mdp.events import reset_randomized_object_and_support_platforms, set_rigid_object_collection_material
from ..mdp.scenes import (
    DEX1_LINK_CONTACT_SENSOR_NAMES,
    G1Dex1HierDrcRandomizedSceneCfg,
    RANDOMIZED_OBJECT_CONTACT_FILTER,
)
from .flat_env_cfg import G1Dex1HierDrcFlatEnvCfg


@configclass
class G1Dex1HierDrcRandomizedEnvCfg(G1Dex1HierDrcFlatEnvCfg):
    scene: G1Dex1HierDrcRandomizedSceneCfg = G1Dex1HierDrcRandomizedSceneCfg(
        num_envs=4096,
        env_spacing=10.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.domain_randomization_curriculum_enabled = True
        self.domain_randomization_start_level = 0.0
        self.object_mass_start_w = 0.0
        self.events.reset_object.func = reset_randomized_object_and_support_platforms
        self.events.object_physics_material.func = set_rigid_object_collection_material
        self.events.object_physics_material.params = {
            "static_friction": 1.5,
            "dynamic_friction": 1.2,
            "restitution": 0.0,
        }
        for _, _, sensor_name in DEX1_LINK_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).filter_prim_paths_expr = RANDOMIZED_OBJECT_CONTACT_FILTER


@configclass
class G1Dex1HierDrcRandomizedPlayEnvCfg(G1Dex1HierDrcRandomizedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.domain_randomization_curriculum_enabled = False
        self.domain_randomization_start_level = 1.0
        self.object_mass_start_w = self.object_mass_anchor_w
        self.enable_debug_visualization = True
        self.platform_pose_debug_vis = True
