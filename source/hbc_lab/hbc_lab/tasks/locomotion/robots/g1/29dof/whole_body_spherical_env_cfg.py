import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp

from . import whole_body_env_cfg
from .whole_body_env_cfg import WholeBodyEnvCfg, WholeBodyPlayEnvCfg


@configclass
class SphericalWholeBodyCommandsCfg(whole_body_env_cfg.WholeBodyCommandsCfg):
    """Shoulder-anchored spherical wrist commands for wider arm workspace training."""

    left_wrist_pose = mdp.SphericalLevelPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",
        anchor_body_name="left_shoulder_pitch_link",
        fixed_anchor_height=1.23,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        make_quat_unique=True,
        ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.28, 0.45),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(0.0, 0.5 * math.pi),
            roll=(-0.10, 0.10),
            ee_pitch=(-0.10, 0.10),
            yaw=(-0.10, 0.10),
        ),
        limit_ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.20, 0.75),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(0.0, 0.5 * math.pi),
            roll=(-0.30, 0.30),
            ee_pitch=(-0.30, 0.30),
            yaw=(-0.30, 0.30),
        ),
    )

    right_wrist_pose = mdp.SphericalLevelPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        anchor_body_name="right_shoulder_pitch_link",
        fixed_anchor_height=1.23,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        make_quat_unique=True,
        ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.28, 0.45),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(-0.5 * math.pi, 0.0),
            roll=(-0.10, 0.10),
            ee_pitch=(-0.10, 0.10),
            yaw=(-0.10, 0.10),
        ),
        limit_ranges=mdp.SphericalLevelPoseCommandCfg.Ranges(
            l=(0.20, 0.75),
            pitch=(-0.5 * math.pi, 0.0),
            azimuth=(-0.5 * math.pi, 0.0),
            roll=(-0.30, 0.30),
            ee_pitch=(-0.30, 0.30),
            yaw=(-0.30, 0.30),
        ),
    )


@configclass
class SphericalWholeBodyObservationsCfg(whole_body_env_cfg.WholeBodyObservationsCfg):
    """Whole-body observations with fixed-height command target errors."""

    @configclass
    class PolicyCfg(whole_body_env_cfg.WholeBodyObservationsCfg.PolicyCfg):
        left_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )
        right_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(whole_body_env_cfg.WholeBodyObservationsCfg.CriticCfg):
        left_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )
        right_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_w_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class SphericalWholeBodyRewardsCfg(whole_body_env_cfg.WholeBodyRewardsCfg):
    """Wrist tracking rewards against fixed-height spherical command targets."""

    track_left_wrist_pose = RewTerm(
        func=mdp.body_pose_command_position_error_w_exp,
        weight=0.10,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    track_right_wrist_pose = RewTerm(
        func=mdp.body_pose_command_position_error_w_exp,
        weight=0.10,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )
    track_left_wrist_pose_fine = RewTerm(
        func=mdp.body_pose_command_position_error_w_tanh,
        weight=1.0,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    track_right_wrist_pose_fine = RewTerm(
        func=mdp.body_pose_command_position_error_w_tanh,
        weight=1.0,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )
    track_left_wrist_orientation = RewTerm(
        func=mdp.body_pose_command_orientation_error_w_tanh,
        weight=0.05,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    track_right_wrist_orientation = RewTerm(
        func=mdp.body_pose_command_orientation_error_w_tanh,
        weight=0.05,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )
    penalty_left_wrist_pose_error = RewTerm(
        func=mdp.body_pose_command_position_error_w_l2,
        weight=-0.80,
        params={
            "command_name": "left_wrist_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    penalty_right_wrist_pose_error = RewTerm(
        func=mdp.body_pose_command_position_error_w_l2,
        weight=-0.80,
        params={
            "command_name": "right_wrist_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )


@configclass
class SphericalWholeBodyCurriculumCfg(whole_body_env_cfg.WholeBodyCurriculumCfg):
    """Curriculum for expanding the spherical wrist-command radius."""

    wrist_pose_cmd_levels = CurrTerm(
        func=mdp.spherical_pose_radius_cmd_levels,
        params={
            "command_names": ("left_wrist_pose", "right_wrist_pose"),
            "error_sum_name": "position_error_sum",
            "success_threshold": 0.08,
            "radius_delta": 0.03,
            "min_episode_fraction": 0.5,
        },
    )


@configclass
class SphericalWholeBodyEnvCfg(WholeBodyEnvCfg):
    """G1 whole-body wrist tracking with shoulder-centered spherical commands."""

    observations: SphericalWholeBodyObservationsCfg = SphericalWholeBodyObservationsCfg()
    commands: SphericalWholeBodyCommandsCfg = SphericalWholeBodyCommandsCfg()
    rewards: SphericalWholeBodyRewardsCfg = SphericalWholeBodyRewardsCfg()
    curriculum: SphericalWholeBodyCurriculumCfg = SphericalWholeBodyCurriculumCfg()


@configclass
class SphericalWholeBodyPlayEnvCfg(SphericalWholeBodyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.wrist_pose_cmd_levels = None
        self.commands.left_wrist_pose.debug_vis = True
        self.commands.right_wrist_pose.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
