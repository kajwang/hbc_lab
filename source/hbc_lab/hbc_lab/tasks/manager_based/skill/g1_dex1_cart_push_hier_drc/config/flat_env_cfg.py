from __future__ import annotations

from isaaclab.utils import configclass

from .cart_env_cfg import G1Dex1CartPushEnvCfg


@configclass
class G1Dex1CartPushFlatEnvCfg(G1Dex1CartPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.high_level.debug_vis = True
        self.target_pose_debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex1CartPushFlatPlayEnvCfg(G1Dex1CartPushFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
