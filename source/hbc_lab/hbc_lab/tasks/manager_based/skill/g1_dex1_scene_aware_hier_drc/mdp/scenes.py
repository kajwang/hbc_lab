from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import G1Dex1HierDrcSceneCfg

TABLE_TOP_SIZE = (0.90, 1.10, 0.08)
TABLE_NOMINAL_CLEARANCE = 0.72
TABLE_LEG_SIZE = (0.07, 0.07, 0.72)
TABLE_LEG_CENTERS = (
    (-0.415, 0.515, 0.36),
    (-0.415, -0.515, 0.36),
    (0.415, 0.515, 0.36),
    (0.415, -0.515, 0.36),
)
# Four interlaced scan phases resolve thin interaction geometry without paying
# for the full dense lattice every simulation step. Each phase evaluates about
# 300 rays; their union over 0.4 s is a 25 x 49 angular lattice.
ENVIRONMENT_LIDAR_VERTICAL_CHANNELS = 13
ENVIRONMENT_LIDAR_HORIZONTAL_SAMPLES = 25
ENVIRONMENT_LIDAR_INTERLACE_PHASES = 4
ENVIRONMENT_LIDAR_EFFECTIVE_VERTICAL_CHANNELS = 25
ENVIRONMENT_LIDAR_EFFECTIVE_HORIZONTAL_SAMPLES = 49
ENVIRONMENT_LIDAR_RAY_COUNT = (
    ENVIRONMENT_LIDAR_VERTICAL_CHANNELS * ENVIRONMENT_LIDAR_HORIZONTAL_SAMPLES
)
ENVIRONMENT_LIDAR_VERTICAL_FOV = (-60.0, 60.0)
ENVIRONMENT_LIDAR_HORIZONTAL_FOV = (-75.0, 75.0)
ENVIRONMENT_LIDAR_OFFSET = (0.12, 0.0, 0.10)
# Root-yaw-frame rolling occupancy volume. Tensor storage is (z, y, x), while
# these public values use Cartesian (x, y, z) order. The 16/16/10 cm cells are
# a bounded-cost intermediate step between the old 25/25/20 cm MLP input and
# GALLANT's 5 cm CNN volume.
ENVIRONMENT_VOXEL_SHAPE_XYZ = (20, 20, 24)
ENVIRONMENT_VOXEL_MIN_XYZ = (-0.8, -1.6, -0.8)
ENVIRONMENT_VOXEL_MAX_XYZ = (2.4, 1.6, 1.6)
ENVIRONMENT_VOXEL_COUNT = (
    ENVIRONMENT_VOXEL_SHAPE_XYZ[0]
    * ENVIRONMENT_VOXEL_SHAPE_XYZ[1]
    * ENVIRONMENT_VOXEL_SHAPE_XYZ[2]
)
TABLE_USD_PATH = Path(__file__).resolve().parents[5] / "assets" / "models" / "platform" / "scene_table.usda"


def _scene_table_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/scene_table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 6.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TABLE_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0e6),
        ),
    )


@configclass
class G1Dex1SceneAwareSceneCfg(G1Dex1HierDrcSceneCfg):
    """PnP scene paired between open ground and a low table overhang."""

    object_init_platform = None
    object_target_platform = None
    scene_table = _scene_table_cfg()
