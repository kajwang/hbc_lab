from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.assets.robots.unitree import (
    G1_29DOF_BODY_JOINT_NAMES,
    G1_DEX1_LEFT_GRIPPER_JOINT_NAMES,
    G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES,
    UNITREE_G1_29DOF_DEX1_CFG,
)
from hbc_lab.tasks.locomotion import mdp

from . import velocity_env_cfg
from .whole_body_spherical_posture_env_cfg import (
    SphericalPostureWholeBodyEnvCfg,
    SphericalPostureWholeBodyObservationsCfg,
    SphericalPostureWholeBodyRewardsCfg,
)


HAND_CENTER_FRAME_NAME = "hand_center_frame"
HAND_CENTER_OFFSET = OffsetCfg(pos=(0.0, 0.09734, 0.0142))
HAND_CENTER_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(
    prim_path="/Visuals/G1Dex1LowLevel/hand_center_frame"
)
HAND_CENTER_FRAME_MARKER_CFG.markers["frame"].scale = (0.07, 0.07, 0.07)
WRIST_ROLL_LIMIT = math.radians(100.0)
WRIST_PITCH_YAW_LIMIT = math.radians(80.0)


@configclass
class Dex1HandCenterSceneCfg(velocity_env_cfg.RobotSceneCfg):
    """Original low-level scene with the robot USD swapped to Dex1 grippers."""

    robot: ArticulationCfg = UNITREE_G1_29DOF_DEX1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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


@configclass
class Dex1HandCenterActionsCfg(velocity_env_cfg.ActionsCfg):
    """Keep the low-level action dimension at the original 29 body joints."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_BODY_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class Dex1HandCenterCommandsCfg(velocity_env_cfg.CommandsCfg):
    """Sample hand bases spherically, then map physical wrist angles to hand centers."""

    # Wrist commands consume this term during resampling, so it must be declared first.
    posture_command = mdp.UniformLevelPostureCommandCfg(
        asset_name="robot",
        body_name="torso_link",
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformLevelPostureCommandCfg.Ranges(
            root_height=(0.70, 0.80),
            torso_pitch=(0.0, 0.25),
        ),
        limit_ranges=mdp.UniformLevelPostureCommandCfg.Ranges(
            root_height=(0.42, 0.80),
            torso_pitch=(0.0, 0.85),
        ),
    )

    left_wrist_pose = mdp.SphericalLevelPoseCommandCfg(
        asset_name="robot",
        body_name="left_hand_base_link",
        anchor_body_name="left_shoulder_pitch_link",
        tracked_frame_sensor_name=HAND_CENTER_FRAME_NAME,
        tracked_frame_index=0,
        anchor_height_command_name="posture_command",
        anchor_height_command_index=0,
        anchor_height_offset=0.43,
        anchor_pitch_command_name="posture_command",
        anchor_pitch_command_index=1,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        make_quat_unique=True,
        orientation_mode="wrist_chain",
        wrist_parent_body_name="left_elbow_link",
        wrist_joint_names=(
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ),
        fixed_palm_quat=(0.70710677, 0.0, 0.0, -0.70710677),
        hand_center_offset=(0.0, 0.09734, 0.0142),
        ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.20, 0.38),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(0.0, 0.5 * math.pi),
            roll=(-0.50, 0.50),
            ee_pitch=(-0.12, 0.12),
            yaw=(-0.12, 0.12),
        ),
        limit_ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.12, 0.58),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(0.0, 0.5 * math.pi),
            roll=(-WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT),
            ee_pitch=(-WRIST_PITCH_YAW_LIMIT, WRIST_PITCH_YAW_LIMIT),
            yaw=(-WRIST_PITCH_YAW_LIMIT, WRIST_PITCH_YAW_LIMIT),
        ),
    )

    right_wrist_pose = mdp.SphericalLevelPoseCommandCfg(
        asset_name="robot",
        body_name="right_hand_base_link",
        anchor_body_name="right_shoulder_pitch_link",
        tracked_frame_sensor_name=HAND_CENTER_FRAME_NAME,
        tracked_frame_index=1,
        anchor_height_command_name="posture_command",
        anchor_height_command_index=0,
        anchor_height_offset=0.43,
        anchor_pitch_command_name="posture_command",
        anchor_pitch_command_index=1,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        make_quat_unique=True,
        orientation_mode="wrist_chain",
        wrist_parent_body_name="right_elbow_link",
        wrist_joint_names=(
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ),
        fixed_palm_quat=(0.7073882, 0.0, 0.0, -0.7068252),
        hand_center_offset=(0.0, 0.09734, 0.0142),
        ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.20, 0.38),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(-0.5 * math.pi, 0.0),
            roll=(-0.50, 0.50),
            ee_pitch=(-0.12, 0.12),
            yaw=(-0.12, 0.12),
        ),
        limit_ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.12, 0.58),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(-0.5 * math.pi, 0.0),
            roll=(-WRIST_ROLL_LIMIT, WRIST_ROLL_LIMIT),
            ee_pitch=(-WRIST_PITCH_YAW_LIMIT, WRIST_PITCH_YAW_LIMIT),
            yaw=(-WRIST_PITCH_YAW_LIMIT, WRIST_PITCH_YAW_LIMIT),
        ),
    )


@configclass
class Dex1HandCenterObservationsCfg(SphericalPostureWholeBodyObservationsCfg):
    """Track the gripper hand-center frames while keeping 29-DoF body observations."""

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
        left_wrist_pose_current = ObsTerm(
            func=mdp.frame_transformer_pose_in_root_frame,
            params={
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 0,
            },
            clip=(-2.0, 2.0),
        )
        right_wrist_pose_current = ObsTerm(
            func=mdp.frame_transformer_pose_in_root_frame,
            params={
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 1,
            },
            clip=(-2.0, 2.0),
        )
        left_wrist_pose_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 0,
            },
            clip=(-1.0, 1.0),
        )
        right_wrist_pose_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 1,
            },
            clip=(-1.0, 1.0),
        )
        left_wrist_orientation_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_orientation_error_w_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 0,
            },
            clip=(-math.pi, math.pi),
        )
        right_wrist_orientation_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_orientation_error_w_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 1,
            },
            clip=(-math.pi, math.pi),
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
        left_wrist_pose_current = ObsTerm(
            func=mdp.frame_transformer_pose_in_root_frame,
            params={
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 0,
            },
            clip=(-2.0, 2.0),
        )
        right_wrist_pose_current = ObsTerm(
            func=mdp.frame_transformer_pose_in_root_frame,
            params={
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 1,
            },
            clip=(-2.0, 2.0),
        )
        left_wrist_pose_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 0,
            },
            clip=(-1.0, 1.0),
        )
        right_wrist_pose_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 1,
            },
            clip=(-1.0, 1.0),
        )
        left_wrist_orientation_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_orientation_error_w_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 0,
            },
            clip=(-math.pi, math.pi),
        )
        right_wrist_orientation_error = ObsTerm(
            func=mdp.frame_transformer_pose_command_orientation_error_w_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "frame_sensor_name": HAND_CENTER_FRAME_NAME,
                "frame_index": 1,
            },
            clip=(-math.pi, math.pi),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class Dex1HandCenterRewardsCfg(SphericalPostureWholeBodyRewardsCfg):
    """Replace wrist-link tracking rewards with Dex1 hand-center frame tracking."""

    track_left_wrist_pose = RewTerm(
        func=mdp.frame_pose_command_position_error_w_exp,
        weight=0.10,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.25,
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 0,
        },
    )
    track_right_wrist_pose = RewTerm(
        func=mdp.frame_pose_command_position_error_w_exp,
        weight=0.10,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.25,
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 1,
        },
    )
    track_left_wrist_pose_fine = RewTerm(
        func=mdp.frame_pose_command_position_error_w_tanh,
        weight=1.0,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.06,
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 0,
        },
    )
    track_right_wrist_pose_fine = RewTerm(
        func=mdp.frame_pose_command_position_error_w_tanh,
        weight=1.0,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.06,
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 1,
        },
    )
    track_left_wrist_orientation = RewTerm(
        func=mdp.frame_pose_command_orientation_error_w_tanh,
        weight=1.0,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.75,
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 0,
        },
    )
    track_right_wrist_orientation = RewTerm(
        func=mdp.frame_pose_command_orientation_error_w_tanh,
        weight=1.0,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.75,
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 1,
        },
    )
    penalty_left_wrist_pose_error = RewTerm(
        func=mdp.frame_pose_command_position_error_w_l2,
        weight=-0.80,
        params={
            "command_name": "left_wrist_pose",
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 0,
        },
    )
    penalty_right_wrist_pose_error = RewTerm(
        func=mdp.frame_pose_command_position_error_w_l2,
        weight=-0.80,
        params={
            "command_name": "right_wrist_pose",
            "frame_sensor_name": HAND_CENTER_FRAME_NAME,
            "frame_index": 1,
        },
    )


@configclass
class Dex1HandCenterCurriculumCfg(velocity_env_cfg.CurriculumCfg):
    """Keep velocity, hand-center radius, and posture curricula, but no terrain-level curriculum."""

    terrain_levels = None
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    wrist_pose_cmd_levels = CurrTerm(
        func=mdp.spherical_pose_radius_cmd_levels,
        params={
            "command_names": ("left_wrist_pose", "right_wrist_pose"),
            "error_sum_name": "position_error_sum",
            "success_threshold": 0.08,
            "radius_delta": 0.03,
            "min_episode_fraction": 0.8,
        },
    )
    orientation_cmd_levels = CurrTerm(
        func=mdp.spherical_pose_orientation_cmd_levels,
        params={
            "command_names": ("left_wrist_pose", "right_wrist_pose"),
            "error_sum_name": "orientation_error_sum",
            "success_threshold": 0.30,
            "roll_delta": 0.35,
            "ee_pitch_delta": 0.04,
            "yaw_delta": 0.04,
            "min_episode_fraction": 0.8,
        },
    )
    posture_cmd_levels = CurrTerm(
        func=mdp.posture_cmd_levels,
        params={
            "command_name": "posture_command",
            "error_sum_names": ("root_height_error_sum", "torso_pitch_error_sum"),
            "success_threshold": 0.05,
            "root_height_delta": 0.03,
            "torso_pitch_delta": 0.05,
            "min_episode_fraction": 0.8,
        },
    )


def _find_joint_ids(asset: Articulation, joint_names: list[str]) -> list[int]:
    joint_ids, resolved_joint_names = asset.find_joints(joint_names, preserve_order=False)
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(f"Expected {len(joint_names)} joints, got {len(joint_ids)}: {resolved_joint_names}")
    return list(joint_ids)


def reset_robot_body_and_gripper_joints(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset body joints like the original task and hold Dex1 grippers at their default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_joint_ids = _find_joint_ids(asset, G1_29DOF_BODY_JOINT_NAMES)
    gripper_joint_ids = _find_joint_ids(
        asset,
        G1_DEX1_LEFT_GRIPPER_JOINT_NAMES + G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES,
    )

    joint_ids = body_joint_ids + gripper_joint_ids
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
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
    gripper_default_pos = asset.data.default_joint_pos[env_ids[:, None], gripper_joint_ids]
    asset.set_joint_position_target(gripper_default_pos, joint_ids=gripper_joint_ids, env_ids=env_ids)


@configclass
class Dex1HandCenterEventCfg(velocity_env_cfg.EventCfg):
    reset_robot_joints = EventTerm(
        func=reset_robot_body_and_gripper_joints,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class SphericalPostureDex1HandCenterEnvCfg(SphericalPostureWholeBodyEnvCfg):
    """Dex1 low-level whole-body tracking task using gripper hand centers as end effectors."""

    scene: Dex1HandCenterSceneCfg = Dex1HandCenterSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: Dex1HandCenterActionsCfg = Dex1HandCenterActionsCfg()
    commands: Dex1HandCenterCommandsCfg = Dex1HandCenterCommandsCfg()
    observations: Dex1HandCenterObservationsCfg = Dex1HandCenterObservationsCfg()
    rewards: Dex1HandCenterRewardsCfg = Dex1HandCenterRewardsCfg()
    curriculum: Dex1HandCenterCurriculumCfg = Dex1HandCenterCurriculumCfg()
    events: Dex1HandCenterEventCfg = Dex1HandCenterEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class SphericalPostureDex1HandCenterPlayEnvCfg(SphericalPostureDex1HandCenterEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.left_wrist_pose.ranges = self.commands.left_wrist_pose.limit_ranges
        self.commands.right_wrist_pose.ranges = self.commands.right_wrist_pose.limit_ranges
        self.commands.posture_command.ranges = self.commands.posture_command.limit_ranges
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.wrist_pose_cmd_levels = None
        self.curriculum.orientation_cmd_levels = None
        self.curriculum.posture_cmd_levels = None
        self.commands.left_wrist_pose.debug_vis = True
        self.commands.right_wrist_pose.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        for cmd in (self.commands.left_wrist_pose, self.commands.right_wrist_pose):
            cmd.ranges.roll = (-0.85, 0.85)
            cmd.ranges.ee_pitch = (-0.16, 0.16)
            cmd.ranges.yaw = (-0.16, 0.16)
