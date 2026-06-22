from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg, patterns
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from hbc_lab.assets.objects import OBJECT_INIT_PLATFORM_CFG, OBJECT_TARGET_PLATFORM_CFG, SMALL_CUBE_OBJECT_CFG
from hbc_lab.assets.robots.unitree import UNITREE_G1_29DOF_DEX3_CFG


LEFT_HAND_PALM_CONTACT_SENSOR_NAMES = (
    "left_hand_palm_contact",
)
LEFT_HAND_THUMB_CONTACT_SENSOR_NAMES = (
    "left_hand_thumb_0_contact",
    "left_hand_thumb_1_contact",
    "left_hand_thumb_2_contact",
)
LEFT_HAND_INDEX_CONTACT_SENSOR_NAMES = (
    "left_hand_index_0_contact",
    "left_hand_index_1_contact",
)
LEFT_HAND_MIDDLE_CONTACT_SENSOR_NAMES = (
    "left_hand_middle_0_contact",
    "left_hand_middle_1_contact",
)
LEFT_HAND_CONTACT_SENSOR_NAMES = (
    LEFT_HAND_PALM_CONTACT_SENSOR_NAMES
    + LEFT_HAND_THUMB_CONTACT_SENSOR_NAMES
    + LEFT_HAND_INDEX_CONTACT_SENSOR_NAMES
    + LEFT_HAND_MIDDLE_CONTACT_SENSOR_NAMES
)

RIGHT_HAND_PALM_CONTACT_SENSOR_NAMES = (
    "right_hand_palm_contact",
)
RIGHT_HAND_THUMB_CONTACT_SENSOR_NAMES = (
    "right_hand_thumb_0_contact",
    "right_hand_thumb_1_contact",
    "right_hand_thumb_2_contact",
)
RIGHT_HAND_INDEX_CONTACT_SENSOR_NAMES = (
    "right_hand_index_0_contact",
    "right_hand_index_1_contact",
)
RIGHT_HAND_MIDDLE_CONTACT_SENSOR_NAMES = (
    "right_hand_middle_0_contact",
    "right_hand_middle_1_contact",
)
RIGHT_HAND_CONTACT_SENSOR_NAMES = (
    RIGHT_HAND_PALM_CONTACT_SENSOR_NAMES
    + RIGHT_HAND_THUMB_CONTACT_SENSOR_NAMES
    + RIGHT_HAND_INDEX_CONTACT_SENSOR_NAMES
    + RIGHT_HAND_MIDDLE_CONTACT_SENSOR_NAMES
)
OBJECT_CONTACT_FILTER = ["{ENV_REGEX_NS}/object"]
HAND_CENTER_FRAME_NAME = "hand_center_frame"

HAND_CENTER_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex3HierDrc/hand_center_frame")
HAND_CENTER_FRAME_MARKER_CFG.markers["frame"].scale = (0.07, 0.07, 0.07)


@configclass
class G1Dex3HierDrcSceneCfg(InteractiveSceneCfg):
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

    robot: ArticulationCfg = UNITREE_G1_29DOF_DEX3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg = SMALL_CUBE_OBJECT_CFG
    object_init_platform: RigidObjectCfg = OBJECT_INIT_PLATFORM_CFG
    object_target_platform: RigidObjectCfg = OBJECT_TARGET_PLATFORM_CFG
    hand_center_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        debug_vis=True,
        visualizer_cfg=HAND_CENTER_FRAME_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link",
                name="left_hand_center",
                offset=OffsetCfg(pos=(0.07, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link",
                name="right_hand_center",
                offset=OffsetCfg(pos=(0.07, 0.0, 0.0)),
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
    left_hand_palm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_index_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_index_0_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_index_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_index_1_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_middle_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_middle_0_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_middle_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_middle_1_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_thumb_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_thumb_0_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_thumb_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_thumb_1_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    left_hand_thumb_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_thumb_2_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_palm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_index_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_0_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_index_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_middle_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_0_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_middle_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_thumb_0_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_0_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_thumb_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_1_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_hand_thumb_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
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
