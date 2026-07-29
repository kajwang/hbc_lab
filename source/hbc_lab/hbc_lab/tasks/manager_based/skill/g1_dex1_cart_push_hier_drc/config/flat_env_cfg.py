from __future__ import annotations

from isaaclab.utils import configclass

from .cart_env_cfg import G1Dex1CartPushEnvCfg


@configclass
class G1Dex1CartPushFlatEnvCfg(G1Dex1CartPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex1CartPushFlatPlayEnvCfg(G1Dex1CartPushFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.enable_debug_visualization = True
