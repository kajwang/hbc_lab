from __future__ import annotations

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.assets.robots.unitree import (
    G1_29DOF_BODY_JOINT_NAMES,
    G1_DEX3_LEFT_HAND_JOINT_NAMES,
    G1_DEX3_RIGHT_HAND_JOINT_NAMES,
    UNITREE_G1_29DOF_DEX3_CFG,
)
from hbc_lab.tasks.locomotion import mdp

from . import velocity_env_cfg
from .whole_body_spherical_posture_env_cfg import (
    SphericalPostureWholeBodyEnvCfg,
    SphericalPostureWholeBodyObservationsCfg,
)


DEX3_CLOSED_HAND_JOINT_POS = {
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": 1.0,
    "left_hand_thumb_2_joint": 1.74,
    "left_hand_middle_0_joint": -1.57,
    "left_hand_middle_1_joint": -1.74,
    "left_hand_index_0_joint": -1.57,
    "left_hand_index_1_joint": -1.74,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -1.0,
    "right_hand_thumb_2_joint": -1.74,
    "right_hand_middle_0_joint": 1.57,
    "right_hand_middle_1_joint": 1.74,
    "right_hand_index_0_joint": 1.57,
    "right_hand_index_1_joint": 1.74,
}

DEX3_FIXED_HAND_ROBOT_CFG = UNITREE_G1_29DOF_DEX3_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
            **DEX3_CLOSED_HAND_JOINT_POS,
        },
        joint_vel={".*": 0.0},
    )
)


@configclass
class Dex3FixedHandsSceneCfg(velocity_env_cfg.RobotSceneCfg):
    """Original low-level scene with only the robot USD swapped to Dex3 hands."""

    robot: ArticulationCfg = DEX3_FIXED_HAND_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Dex3FixedHandsActionsCfg(velocity_env_cfg.ActionsCfg):
    """Keep the trained low-level action dimension: 29 body joints only."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_BODY_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class Dex3FixedHandsObservationsCfg(SphericalPostureWholeBodyObservationsCfg):
    """Keep joint observations aligned with the original 29-DoF body policy."""

    @configclass
    class PolicyCfg(SphericalPostureWholeBodyObservationsCfg.PolicyCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_BODY_JOINT_NAMES)},
            noise=velocity_env_cfg.Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_BODY_JOINT_NAMES)},
            scale=0.05,
            noise=velocity_env_cfg.Unoise(n_min=-1.5, n_max=1.5),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(SphericalPostureWholeBodyObservationsCfg.CriticCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_BODY_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_BODY_JOINT_NAMES)},
            scale=0.05,
        )

    critic: CriticCfg = CriticCfg()


def _find_joint_ids(asset: Articulation, joint_names: list[str]) -> list[int]:
    joint_ids, resolved_joint_names = asset.find_joints(joint_names, preserve_order=False)
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(f"Expected {len(joint_names)} joints, got {len(joint_ids)}: {resolved_joint_names}")
    return list(joint_ids)


def reset_robot_body_and_hand_joints(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset body joints like the original task and keep Dex3 hands at their closed default."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_joint_ids = _find_joint_ids(asset, G1_29DOF_BODY_JOINT_NAMES)
    hand_joint_ids = _find_joint_ids(asset, G1_DEX3_LEFT_HAND_JOINT_NAMES + G1_DEX3_RIGHT_HAND_JOINT_NAMES)

    joint_ids = body_joint_ids + hand_joint_ids
    joint_pos = asset.data.default_joint_pos[env_ids[:, None], joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids[:, None], joint_ids].clone()

    body_shape = joint_pos[:, : len(body_joint_ids)].shape
    joint_pos[:, : len(body_joint_ids)] *= math_utils.sample_uniform(
        *position_range,
        body_shape,
        joint_pos.device,
    )
    joint_vel[:, : len(body_joint_ids)] *= math_utils.sample_uniform(
        *velocity_range,
        body_shape,
        joint_vel.device,
    )
    joint_vel[:, len(body_joint_ids) :] = 0.0

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids[:, None], joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=body_joint_ids + hand_joint_ids,
        env_ids=env_ids,
    )
    hand_default_pos = asset.data.default_joint_pos[env_ids[:, None], hand_joint_ids]
    asset.set_joint_position_target(hand_default_pos, joint_ids=hand_joint_ids, env_ids=env_ids)


@configclass
class Dex3FixedHandsEventCfg(velocity_env_cfg.EventCfg):
    reset_robot_joints = EventTerm(
        func=reset_robot_body_and_hand_joints,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class SphericalPostureDex3FixedHandsEnvCfg(SphericalPostureWholeBodyEnvCfg):
    """Low-level policy compatibility check: original task, Dex3 USD, fixed closed hands."""

    scene: Dex3FixedHandsSceneCfg = Dex3FixedHandsSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Dex3FixedHandsActionsCfg = Dex3FixedHandsActionsCfg()
    observations: Dex3FixedHandsObservationsCfg = Dex3FixedHandsObservationsCfg()
    events: Dex3FixedHandsEventCfg = Dex3FixedHandsEventCfg()


@configclass
class SphericalPostureDex3FixedHandsPlayEnvCfg(SphericalPostureDex3FixedHandsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.posture_command.ranges = self.commands.posture_command.limit_ranges
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.wrist_pose_cmd_levels = None
        self.curriculum.posture_cmd_levels = None
        self.commands.left_wrist_pose.debug_vis = True
        self.commands.right_wrist_pose.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
