from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


OBJECT_PLATFORM_HEIGHT = 0.5
OBJECT_PLATFORM_SIZE = (0.35, 0.45, OBJECT_PLATFORM_HEIGHT)
OBJECT_PLATFORM_CENTER_Z = OBJECT_PLATFORM_HEIGHT * 0.5
SMALL_CUBE_HALF_HEIGHT = 0.04
OBJECT_ON_PLATFORM_Z = 0.54


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
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
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

SMALL_CUBE_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.CuboidCfg(
        size=(0.08, 0.08, 0.08),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
)
