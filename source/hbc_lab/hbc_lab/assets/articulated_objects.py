from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


ARTICULATED_MODEL_DIR = Path(__file__).resolve().parent / "models" / "articulated"

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


DOOR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/door",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ARTICULATED_MODEL_DIR / "door" / "door_0_bot.usd"),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
        },
    ),
    actuators={
        "hinge": ImplicitActuatorCfg(
            joint_names_expr=["joint_1"],
            stiffness=0.0,
            damping=5.0,
            friction=0.2,
        ),
        "handle": ImplicitActuatorCfg(
            joint_names_expr=["joint_2"],
            stiffness=1.0,
            damping=1.0,
        ),
    },
)

DOOR_FRAME_CFG = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/door/link_1",
    debug_vis=True,
    visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/DoorFrameTransformer"),
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/door/link_2",
            name="door_handle",
            offset=OffsetCfg(
                pos=(-0.04, 0.0, 0.04),
                rot=(0.707, 0.0, 0.707, 0.0),
            ),
        ),
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/door/link_0",
            name="door_handle_goal",
            offset=OffsetCfg(
                pos=(-0.31822, -0.26341, -0.5),
                rot=(0.0, 0.0, 1.0, 0.0),
            ),
        ),
    ],
)


CART_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/cart",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ARTICULATED_MODEL_DIR / "cart" / "instance_turn_R_10kg.usd"),
        scale=(0.8, 0.8, 0.8),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.5, 0.5, -0.5, -0.5),
        joint_pos={
            "RL_joint": 0.0,
            "RR_joint": 0.0,
            "FL_joint": 0.0,
            "FR_joint": 0.0,
            "RL_turn_joint": 0.0,
            "RR_turn_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["RL_joint", "RR_joint", "FL_joint", "FR_joint"],
            effort_limit=0.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.1,
            friction=0.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["RL_turn_joint", "RR_turn_joint"],
            effort_limit=0.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=1.0,
            friction=0.2,
        ),
    },
)

CART_FRAME_CFG = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/cart/handle",
    debug_vis=True,
    visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CartFrameTransformer"),
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/cart/handle",
            name="cart_handle",
            offset=OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.707, 0.0, 0.707, 0.0),
            ),
        ),
    ],
)


SEKTION_CABINET_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Cabinet",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "door_left_joint": 0.0,
            "door_right_joint": 0.0,
            "drawer_bottom_joint": 0.0,
            "drawer_top_joint": 0.0,
        },
    ),
    actuators={
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
            effort_limit_sim=87.0,
            velocity_limit_sim=100.0,
            stiffness=1.0,
            damping=0.1,
            friction=0.1,
        ),
        "doors": ImplicitActuatorCfg(
            joint_names_expr=["door_left_joint", "door_right_joint"],
            effort_limit_sim=87.0,
            velocity_limit_sim=100.0,
            stiffness=10.0,
            damping=2.5,
        ),
    },
)

SEKTION_CABINET_FRAME_CFG = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
    debug_vis=True,
    visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/SektionCabinetFrameTransformer"),
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
            name="drawer_handle_top",
            offset=OffsetCfg(
                pos=(0.305, 0.0, 0.01),
                rot=(0.5, 0.5, -0.5, -0.5),
            ),
        ),
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom",
            name="drawer_handle_bottom",
            offset=OffsetCfg(
                pos=(0.305, 0.0, 0.01),
                rot=(0.5, 0.5, -0.5, -0.5),
            ),
        ),
    ],
)
