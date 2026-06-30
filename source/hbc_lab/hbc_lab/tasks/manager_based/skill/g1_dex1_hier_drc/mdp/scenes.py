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

from hbc_lab.assets.objects import (
    APPLE_OBJECT_CFG,
    APPLE_OBJECT_FRAME_OFFSET_Z,
    OBJECT_INIT_PLATFORM_CFG,
    OBJECT_TARGET_PLATFORM_CFG,
)
from hbc_lab.assets.robots.unitree import UNITREE_G1_29DOF_DEX1_CFG


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
HAND_CENTER_FRAME_NAME = "hand_center_frame"
HAND_CENTER_OFFSET = OffsetCfg(pos=(0.0, 0.09734, 0.0142))

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
