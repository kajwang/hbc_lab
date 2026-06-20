from __future__ import annotations

from isaaclab.utils import configclass

from .g1_dex3_env_cfg import G1Dex3HierDrcEnvCfg


@configclass
class G1Dex3HierDrcFlatEnvCfg(G1Dex3HierDrcEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.high_level.debug_vis = True
        self.target_pose_debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex3HierDrcFlatPlayEnvCfg(G1Dex3HierDrcFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.high_level.debug_vis = True
        self.target_pose_debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
