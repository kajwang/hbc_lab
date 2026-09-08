from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg, patterns
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from hbc_lab.assets.objects import (
    APPLE_OBJECT_CFG,
    APPLE_OBJECT_FRAME_OFFSET_Z,
    APPLE_SCALE,
    OBJECT_INIT_PLATFORM_CFG,
    OBJECT_PLATFORM_HEIGHT,
    OBJECT_PLATFORM_SIZE,
    OBJECT_TARGET_PLATFORM_CFG,
)
from hbc_lab.assets.robots.unitree import UNITREE_G1_29DOF_DEX1_CFG

from .object_shape_bps import (
    ALL_SHAPE_NAMES,
    GRASP_REF_ALL_SHAPE_NAMES,
    GRASP_REF_TRAIN_SHAPE_NAMES,
    MULTISHAPE_SIZE_FACTORS,
    OBJECT_SHAPE_SPECS,
    OBJECT_MODEL_ROOT,
    SHAPE_INDEX_MASS_BASE,
    SHAPE_INDEX_MASS_STRIDE,
    TRAIN_SHAPE_NAMES,
)


LEFT_GRIPPER_CONTACT_SENSOR_NAMES = (
    "left_gripper_finger_contact",
    "left_gripper_opposing_finger_contact",
)
RIGHT_GRIPPER_CONTACT_SENSOR_NAMES = (
    "right_gripper_finger_contact",
    "right_gripper_opposing_finger_contact",
)
DEX1_LINK_CONTACT_SENSOR_NAMES = (
    ("left", "Link1_2", "left_gripper_link1_2_contact"),
    ("left", "Link1_3", "left_gripper_finger_contact"),
    ("left", "Link2_2", "left_gripper_link2_2_contact"),
    ("left", "Link2_3", "left_gripper_opposing_finger_contact"),
    ("right", "Link1_2", "right_gripper_link1_2_contact"),
    ("right", "Link1_3", "right_gripper_finger_contact"),
    ("right", "Link2_2", "right_gripper_link2_2_contact"),
    ("right", "Link2_3", "right_gripper_opposing_finger_contact"),
)
OBJECT_CONTACT_FILTER = ["{ENV_REGEX_NS}/object"]
PNP_OBJECT_SCALE_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
RANDOMIZED_OBJECT_PLATFORM_HEIGHT = 0.7
RANDOMIZED_OBJECT_CONTACT_FILTER = [
    f"{{ENV_REGEX_NS}}/object_size_{index}" for index in range(len(PNP_OBJECT_SCALE_FACTORS))
]
HAND_CENTER_FRAME_NAME = "hand_center_frame"
HAND_CENTER_OFFSET = OffsetCfg(pos=(0.0, 0.09734, 0.0142))
SHALLOW_TRAY_USD_PATH = OBJECT_MODEL_ROOT.parent / "platform" / "shallow_tray.usda"
SHALLOW_TRAY_SIZE = (0.34, 0.40, OBJECT_PLATFORM_HEIGHT)
SHALLOW_TRAY_WALL_HEIGHT = 0.05
SHALLOW_TRAY_WALL_THICKNESS = 0.025

HAND_CENTER_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex1HierDrc/hand_center_frame")
HAND_CENTER_FRAME_MARKER_CFG.markers["frame"].scale = (0.07, 0.07, 0.07)
OBJECT_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex1HierDrc/object_frame")
OBJECT_FRAME_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)


@configclass
class G1Dex1HierDrcSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.46, 0.48)),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_G1_29DOF_DEX1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg = APPLE_OBJECT_CFG
    object_init_platform: RigidObjectCfg = OBJECT_INIT_PLATFORM_CFG
    object_target_platform: RigidObjectCfg = OBJECT_TARGET_PLATFORM_CFG
    object_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/object",
        debug_vis=True,
        visualizer_cfg=OBJECT_FRAME_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/object",
                name="object",
                offset=OffsetCfg(pos=(0.0, 0.0, APPLE_OBJECT_FRAME_OFFSET_Z)),
            ),
        ],
    )
    hand_center_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        debug_vis=True,
        visualizer_cfg=HAND_CENTER_FRAME_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_hand_base_link",
                name="left_hand_center",
                offset=HAND_CENTER_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_base_link",
                name="right_hand_center",
                offset=HAND_CENTER_OFFSET,
            ),
        ],
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    left_gripper_link1_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_Link1_2",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_gripper_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_Link1_3",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_gripper_link2_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_Link2_2",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_gripper_opposing_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_Link2_3",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_gripper_link1_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_Link1_2",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_gripper_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_Link1_3",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_gripper_link2_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_Link2_2",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_gripper_opposing_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_Link2_3",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


def _randomized_apple_pool() -> dict[str, RigidObjectCfg]:
    objects = {}
    for index, scale_factor in enumerate(PNP_OBJECT_SCALE_FACTORS):
        scale = APPLE_SCALE * scale_factor
        objects[f"object_size_{index}"] = APPLE_OBJECT_CFG.replace(
            prim_path=f"{{ENV_REGEX_NS}}/object_size_{index}",
            spawn=APPLE_OBJECT_CFG.spawn.replace(scale=(scale, scale, scale)),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, -5.0 - index),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
    return objects


def _randomized_platform_cfg(platform_cfg: RigidObjectCfg) -> RigidObjectCfg:
    return platform_cfg.replace(
        spawn=platform_cfg.spawn.replace(
            size=(OBJECT_PLATFORM_SIZE[0], OBJECT_PLATFORM_SIZE[1], RANDOMIZED_OBJECT_PLATFORM_HEIGHT),
            rigid_props=platform_cfg.spawn.rigid_props.replace(kinematic_enabled=True),
            visible=False,
        )
    )


@configclass
class G1Dex1HierDrcRandomizedSceneCfg(G1Dex1HierDrcSceneCfg):
    """PnP scene with a small, pre-scaled apple pool for safe runtime size curricula."""

    object: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=_randomized_apple_pool())
    object_init_platform: RigidObjectCfg = _randomized_platform_cfg(OBJECT_INIT_PLATFORM_CFG)
    object_target_platform: RigidObjectCfg = _randomized_platform_cfg(OBJECT_TARGET_PLATFORM_CFG)
    object_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        debug_vis=True,
        visualizer_cfg=OBJECT_FRAME_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/object_size_{index}",
                name=f"object_size_{index}",
                offset=OffsetCfg(pos=(0.0, 0.0, APPLE_OBJECT_FRAME_OFFSET_Z * scale_factor)),
            )
            for index, scale_factor in enumerate(PNP_OBJECT_SCALE_FACTORS)
        ],
    )


def _multishape_asset_cfgs(
    shape_names: tuple[str, ...],
    size_factors: tuple[float, ...],
) -> list[sim_utils.UsdFileCfg]:
    assets = []
    for shape_name in shape_names:
        spec = OBJECT_SHAPE_SPECS[shape_name]
        for size_factor in size_factors:
            variant_index = len(assets)
            assets.append(
                sim_utils.UsdFileCfg(
                    usd_path=str(spec.usd_path),
                    scale=tuple(component * size_factor for component in spec.scale),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=1.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    # Spawn-time mass carries deterministic variant metadata. The mass
                    # curriculum overwrites the simulated value after environment setup.
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=SHAPE_INDEX_MASS_BASE + SHAPE_INDEX_MASS_STRIDE * variant_index,
                    ),
                    activate_contact_sensors=True,
                )
            )
    return assets


def _multishape_tray_cfg(platform_cfg: RigidObjectCfg) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=platform_cfg.prim_path,
        init_state=platform_cfg.init_state,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SHALLOW_TRAY_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visible=False,
        ),
    )


def _multishape_flat_platform_cfg(platform_cfg: RigidObjectCfg) -> RigidObjectCfg:
    """Create a visible flat support with the tray footprint and no side walls."""
    initial_x = 2.0 if "object_init_platform" in platform_cfg.prim_path else -2.0
    return RigidObjectCfg(
        prim_path=platform_cfg.prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(initial_x, 0.0, 0.5 * SHALLOW_TRAY_SIZE[2]),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=SHALLOW_TRAY_SIZE,
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


def _multishape_object_cfg(
    shape_names: tuple[str, ...],
    size_factors: tuple[float, ...],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=_multishape_asset_cfgs(shape_names, size_factors),
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class G1Dex1HierDrcMultiShapeBpsSceneCfg(G1Dex1HierDrcSceneCfg):
    """One deterministically balanced training shape per environment."""

    object: RigidObjectCfg = _multishape_object_cfg(TRAIN_SHAPE_NAMES, MULTISHAPE_SIZE_FACTORS)
    object_init_platform: RigidObjectCfg = _multishape_tray_cfg(OBJECT_INIT_PLATFORM_CFG)
    object_target_platform: RigidObjectCfg = _multishape_tray_cfg(OBJECT_TARGET_PLATFORM_CFG)
    object_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/object",
        debug_vis=False,
        visualizer_cfg=OBJECT_FRAME_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/object",
                name="object_root",
            ),
        ],
    )


@configclass
class G1Dex1HierDrcMultiShapeBpsEvalSceneCfg(G1Dex1HierDrcMultiShapeBpsSceneCfg):
    """Training and held-out shapes for visual evaluation."""

    object: RigidObjectCfg = _multishape_object_cfg(ALL_SHAPE_NAMES, (1.0,))


@configclass
class G1Dex1HierDrcMultiShapeGraspRefSceneCfg(G1Dex1HierDrcMultiShapeBpsSceneCfg):
    """Grasp-reference training scene with unobstructed flat supports."""

    object: RigidObjectCfg = _multishape_object_cfg(GRASP_REF_TRAIN_SHAPE_NAMES, (1.0,))
    object_init_platform: RigidObjectCfg = _multishape_flat_platform_cfg(OBJECT_INIT_PLATFORM_CFG)
    object_target_platform: RigidObjectCfg = _multishape_flat_platform_cfg(OBJECT_TARGET_PLATFORM_CFG)


@configclass
class G1Dex1HierDrcMultiShapeGraspRefEvalSceneCfg(G1Dex1HierDrcMultiShapeGraspRefSceneCfg):
    """Training and held-out shapes on unobstructed flat supports."""

    object: RigidObjectCfg = _multishape_object_cfg(GRASP_REF_ALL_SHAPE_NAMES, (1.0,))
