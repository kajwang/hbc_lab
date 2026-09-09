from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .scenes import G1Dex1SceneAwareSceneCfg


GAP_WALL_SIZE = (1.10, 0.10, 1.10)
PILLAR_SIZE = (0.18, 0.18, 0.85)
OOD_ARCH_POST_SIZE = (0.16, 0.16, 1.22)
OOD_ARCH_TOP_SIZE = (0.16, 1.30, 0.16)
OOD_ARCH_OPENING_WIDTH = 0.86
OOD_ARCH_CLEARANCE = OOD_ARCH_POST_SIZE[2]
OPEN_PLATFORM_SIZE = (0.60, 0.60, 0.08)
REACH_OVER_BARRIER_THICKNESS = 0.08
REACH_OVER_BARRIER_WIDTH = 1.50
# Every wall is above the apple.  Increasing level raises the rim and moves it
# farther from the object, requiring an over-then-down hand trajectory.
REACH_OVER_BARRIER_HEIGHTS = (0.48, 0.60, 0.72)
REACH_OVER_SUPPORT_SIZE = (0.32, 0.32, 0.04)
REACH_OVER_SUPPORT_SURFACE_HEIGHT = 0.28
CABINET_USD_PATH = Path(__file__).resolve().parents[5] / "assets" / "models" / "platform" / "three_level_cabinet.usda"
CABINET_BOARD_SIZE = (0.50, 1.20, 0.06)
CABINET_SIDE_SIZE = (0.50, 0.06, 1.50)
CABINET_BACK_SIZE = (0.06, 1.20, 1.50)
CABINET_LOCAL_CENTERS = (
    (0.0, 0.0, 0.03),
    (0.0, 0.0, 0.51),
    (0.0, 0.0, 0.99),
    (0.0, 0.0, 1.47),
    (0.0, 0.57, 0.75),
    (0.0, -0.57, 0.75),
    (0.22, 0.0, 0.75),
)
CABINET_BOX_SIZES = (
    CABINET_BOARD_SIZE,
    CABINET_BOARD_SIZE,
    CABINET_BOARD_SIZE,
    CABINET_BOARD_SIZE,
    CABINET_SIDE_SIZE,
    CABINET_SIDE_SIZE,
    CABINET_BACK_SIZE,
)
CABINET_SHELF_SURFACE_HEIGHT = {
    "bottom": 0.06,
    "middle": 0.54,
    "top": 1.02,
}


def _obstacle_cfg(
    prim_name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    parking_height: float,
):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, parking_height)),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0e6),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _cabinet_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/three_level_cabinet",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 9.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CABINET_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0e6),
        ),
    )


@configclass
class G1Dex1MultiGeometrySceneCfg(G1Dex1SceneAwareSceneCfg):
    three_level_cabinet = _cabinet_cfg()
    front_pillar = _obstacle_cfg("front_pillar", PILLAR_SIZE, (0.72, 0.25, 0.16), 12.0)
    gap_wall_left = _obstacle_cfg("gap_wall_left", GAP_WALL_SIZE, (0.18, 0.38, 0.72), 14.0)
    gap_wall_right = _obstacle_cfg("gap_wall_right", GAP_WALL_SIZE, (0.18, 0.38, 0.72), 16.0)
    ood_arch_left_post = _obstacle_cfg(
        "ood_arch_left_post", OOD_ARCH_POST_SIZE, (0.68, 0.20, 0.58), 18.0
    )
    ood_arch_right_post = _obstacle_cfg(
        "ood_arch_right_post", OOD_ARCH_POST_SIZE, (0.68, 0.20, 0.58), 20.0
    )
    ood_arch_top = _obstacle_cfg(
        "ood_arch_top", OOD_ARCH_TOP_SIZE, (0.82, 0.30, 0.66), 22.0
    )
    open_platform = _obstacle_cfg(
        "open_platform",
        OPEN_PLATFORM_SIZE,
        (0.42, 0.45, 0.48),
        24.0,
    )
    reach_over_barrier_low = _obstacle_cfg(
        "reach_over_barrier_low",
        (REACH_OVER_BARRIER_THICKNESS, REACH_OVER_BARRIER_WIDTH, REACH_OVER_BARRIER_HEIGHTS[0]),
        (0.18, 0.55, 0.62),
        26.0,
    )
    reach_over_barrier_mid = _obstacle_cfg(
        "reach_over_barrier_mid",
        (REACH_OVER_BARRIER_THICKNESS, REACH_OVER_BARRIER_WIDTH, REACH_OVER_BARRIER_HEIGHTS[1]),
        (0.18, 0.55, 0.62),
        28.0,
    )
    reach_over_barrier_high = _obstacle_cfg(
        "reach_over_barrier_high",
        (REACH_OVER_BARRIER_THICKNESS, REACH_OVER_BARRIER_WIDTH, REACH_OVER_BARRIER_HEIGHTS[2]),
        (0.18, 0.55, 0.62),
        30.0,
    )
    reach_over_support = _obstacle_cfg(
        "reach_over_support",
        REACH_OVER_SUPPORT_SIZE,
        (0.46, 0.48, 0.50),
        32.0,
    )
