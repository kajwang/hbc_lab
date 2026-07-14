from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from hbc_lab.assets.objects import BOX_CUBE_OBJECT_CFG
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import (
    G1Dex1HierDrcSceneCfg,
    HAND_CENTER_FRAME_NAME,
    OBJECT_CONTACT_FILTER,
)


BOX_PALM_CONTACT_SENSOR_NAMES = ("left_palm_contact", "right_palm_contact")
OBJECT_LEG_CONTACT_SENSOR_NAME = "object_leg_contact"
BOX_SUPPORT_CONTACT_KEYS = (
    "Link1_2",
    "Link2_2",
)
BOX_FORBIDDEN_LEG_BODY_NAMES = (
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
)

BOX_OBJECT_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(
    prim_path="/Visuals/G1Dex1BoxCarry/object_frame"
)
BOX_OBJECT_FRAME_MARKER_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class G1Dex1BoxCarrySceneCfg(G1Dex1HierDrcSceneCfg):
    object: RigidObjectCfg = BOX_CUBE_OBJECT_CFG
    object_init_platform = None
    object_target_platform = None
    object_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/object",
        debug_vis=True,
        visualizer_cfg=BOX_OBJECT_FRAME_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/object",
                name="object",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )
    left_palm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_base_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    right_palm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_base_link",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=OBJECT_CONTACT_FILTER,
    )
    object_leg_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/object",
        history_length=0,
        track_air_time=False,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in BOX_FORBIDDEN_LEG_BODY_NAMES
        ],
    )
