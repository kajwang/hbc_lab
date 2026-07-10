from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


OBJECTS_MODEL_DIR = Path(__file__).resolve().parent / "models" / "objects"
OBJECT_PLATFORM_HEIGHT = 0.5
OBJECT_PLATFORM_SIZE = (0.24, 0.30, OBJECT_PLATFORM_HEIGHT)
OBJECT_PLATFORM_CENTER_Z = OBJECT_PLATFORM_HEIGHT * 0.5
APPLE_SCALE = 0.007
APPLE_OBJECT_FRAME_OFFSET_Z = 4.5 * APPLE_SCALE
OBJECT_ROOT_ON_PLATFORM_Z = OBJECT_PLATFORM_HEIGHT


def _make_object_platform_cfg(prim_name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_PLATFORM_SIZE,
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.26, 0.28)),
        ),
    )


OBJECT_INIT_PLATFORM_CFG = _make_object_platform_cfg("object_init_platform")
OBJECT_TARGET_PLATFORM_CFG = _make_object_platform_cfg("object_target_platform")

APPLE_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(OBJECTS_MODEL_DIR / "fruit" / "apple" / "apple_rigid.usd"),
        scale=(APPLE_SCALE, APPLE_SCALE, APPLE_SCALE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)

SMALL_CUBE_OBJECT_CFG = APPLE_OBJECT_CFG
