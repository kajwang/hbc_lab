from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass

from hbc_lab.assets.articulated_objects import CART_CFG, CART_FRAME_CFG
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import G1Dex1HierDrcSceneCfg


CART_HANDLE_CONTACT_FILTER = ["{ENV_REGEX_NS}/cart/handle"]
CART_PAD_CONTACT_SENSOR_NAMES = (
    "left_gripper_finger_contact",
    "left_gripper_opposing_finger_contact",
    "right_gripper_finger_contact",
    "right_gripper_opposing_finger_contact",
)


def _cart_contact_sensor(body_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=CART_HANDLE_CONTACT_FILTER,
    )


@configclass
class G1Dex1CartPushSceneCfg(G1Dex1HierDrcSceneCfg):
    object: ArticulationCfg = CART_CFG
    object_init_platform = None
    object_target_platform = None
    object_frame: FrameTransformerCfg = CART_FRAME_CFG

    left_gripper_link1_2_contact = _cart_contact_sensor("left_hand_Link1_2")
    left_gripper_finger_contact = _cart_contact_sensor("left_hand_Link1_3")
    left_gripper_link2_2_contact = _cart_contact_sensor("left_hand_Link2_2")
    left_gripper_opposing_finger_contact = _cart_contact_sensor("left_hand_Link2_3")
    right_gripper_link1_2_contact = _cart_contact_sensor("right_hand_Link1_2")
    right_gripper_finger_contact = _cart_contact_sensor("right_hand_Link1_3")
    right_gripper_link2_2_contact = _cart_contact_sensor("right_hand_Link2_2")
    right_gripper_opposing_finger_contact = _cart_contact_sensor("right_hand_Link2_3")
