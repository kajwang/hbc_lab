from __future__ import annotations

from isaaclab.utils import configclass

from .drawer_env_cfg import G1Dex1DrawerEnvCfg


@configclass
class G1Dex1DrawerFlatEnvCfg(G1Dex1DrawerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex1DrawerFlatPlayEnvCfg(G1Dex1DrawerFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.enable_debug_visualization = True
