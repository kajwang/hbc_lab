from __future__ import annotations

from isaaclab.utils import configclass

from .box_env_cfg import G1Dex1BoxCarryEnvCfg


@configclass
class G1Dex1BoxCarryFlatEnvCfg(G1Dex1BoxCarryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1Dex1BoxCarryFlatPlayEnvCfg(G1Dex1BoxCarryFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.object_mass_start_w = 1.0
        self.enable_debug_visualization = True
