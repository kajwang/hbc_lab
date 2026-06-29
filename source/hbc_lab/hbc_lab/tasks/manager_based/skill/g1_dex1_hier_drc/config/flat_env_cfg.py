from __future__ import annotations

from isaaclab.utils import configclass

from .g1_dex1_env_cfg import G1Dex1HierDrcEnvCfg


@configclass
class G1Dex1HierDrcFlatEnvCfg(G1Dex1HierDrcEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.high_level.debug_vis = True
        self.commands.high_level.left_hand_probability = 0.0
        self.events.reset_object.params["pose_range"]["y"] = (-0.35, -0.05)
        self.target_pose_debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex1HierDrcFlatPlayEnvCfg(G1Dex1HierDrcFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.high_level.debug_vis = True
        self.target_pose_debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
