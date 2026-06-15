import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp

from . import velocity_env_cfg


@configclass
class WholeBodyCommandsCfg(velocity_env_cfg.CommandsCfg):
    """Low-frequency commands for the first whole-body extension."""

    left_wrist_pose = mdp.UniformLevelPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",
        resampling_time_range=(2.0, 4.0),
        debug_vis=False,
        make_quat_unique=True,
        ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.24),
            pos_y=(0.18, 0.34),
            pos_z=(-0.22, 0.02),
            roll=(-0.10, 0.10),
            pitch=(-0.10, 0.10),
            yaw=(0.5 * math.pi - 0.15, 0.5 * math.pi + 0.15),
        ),
        limit_ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.08, 0.38),
            pos_y=(0.08, 0.42),
            pos_z=(-0.30, 0.28),
            roll=(-0.30, 0.30),
            pitch=(-0.30, 0.30),
            yaw=(0.5 * math.pi - 0.50, 0.5 * math.pi + 0.50),
        ),
    )

    right_wrist_pose = mdp.UniformLevelPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        resampling_time_range=(2.0, 4.0),
        debug_vis=False,
        make_quat_unique=True,
        ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.24),
            pos_y=(-0.34, -0.18),
            pos_z=(-0.22, 0.02),
            roll=(-0.10, 0.10),
            pitch=(-0.10, 0.10),
            yaw=(-0.5 * math.pi - 0.15, -0.5 * math.pi + 0.15),
        ),
        limit_ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.08, 0.38),
            pos_y=(-0.42, -0.08),
            pos_z=(-0.30, 0.28),
            roll=(-0.30, 0.30),
            pitch=(-0.30, 0.30),
            yaw=(-0.5 * math.pi - 0.50, -0.5 * math.pi + 0.50),
        ),
    )


@configclass
class WholeBodyObservationsCfg(velocity_env_cfg.ObservationsCfg):
    """Policy observations for terrain-aware whole-body command tracking."""

    @configclass
    class PolicyCfg(velocity_env_cfg.ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        left_wrist_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_wrist_pose"},
        )
        right_wrist_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_wrist_pose"},
        )
        left_wrist_pose_current = ObsTerm(
            func=mdp.body_pose_in_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link")},
            clip=(-2.0, 2.0),
        )
        right_wrist_pose_current = ObsTerm(
            func=mdp.body_pose_in_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link")},
            clip=(-2.0, 2.0),
        )
        left_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )
        right_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(velocity_env_cfg.ObservationsCfg.CriticCfg):
        left_wrist_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_wrist_pose"},
        )
        right_wrist_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_wrist_pose"},
        )
        left_wrist_pose_current = ObsTerm(
            func=mdp.body_pose_in_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link")},
            clip=(-2.0, 2.0),
        )
        right_wrist_pose_current = ObsTerm(
            func=mdp.body_pose_in_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link")},
            clip=(-2.0, 2.0),
        )
        left_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_in_root_frame,
            params={
                "command_name": "left_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )
        right_wrist_pose_error = ObsTerm(
            func=mdp.body_pose_command_position_error_in_root_frame,
            params={
                "command_name": "right_wrist_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            },
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class WholeBodyRewardsCfg(velocity_env_cfg.RewardsCfg):
    """Whole-body wrist tracking rewards layered on top of locomotion."""

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    track_left_wrist_pose = RewTerm(
        func=mdp.body_pose_command_position_error_exp,
        weight=0.10,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    track_right_wrist_pose = RewTerm(
        func=mdp.body_pose_command_position_error_exp,
        weight=0.10,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )
    track_left_wrist_pose_fine = RewTerm(
        func=mdp.body_pose_command_position_error_tanh,
        weight=1.0,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    track_right_wrist_pose_fine = RewTerm(
        func=mdp.body_pose_command_position_error_tanh,
        weight=1.0,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )
    track_left_wrist_orientation = RewTerm(
        func=mdp.body_pose_command_orientation_error_tanh,
        weight=0.05,
        params={
            "command_name": "left_wrist_pose",
            "std": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    track_right_wrist_orientation = RewTerm(
        func=mdp.body_pose_command_orientation_error_tanh,
        weight=0.05,
        params={
            "command_name": "right_wrist_pose",
            "std": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )
    penalty_left_wrist_pose_error = RewTerm(
        func=mdp.body_pose_command_position_error_l2,
        weight=-0.80,
        params={
            "command_name": "left_wrist_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
        },
    )
    penalty_right_wrist_pose_error = RewTerm(
        func=mdp.body_pose_command_position_error_l2,
        weight=-0.80,
        params={
            "command_name": "right_wrist_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
        },
    )


@configclass
class WholeBodyCurriculumCfg(velocity_env_cfg.CurriculumCfg):
    """Curriculum for expanding the wrist command workspace."""

    wrist_pose_cmd_levels = CurrTerm(
        func=mdp.pose_position_cmd_levels,
        params={
            "command_names": ("left_wrist_pose", "right_wrist_pose"),
            "penalty_term_names": ("penalty_left_wrist_pose_error", "penalty_right_wrist_pose_error"),
            "success_threshold": 0.08,
            "position_delta": 0.03,
        },
    )


@configclass
class WholeBodyEnvCfg(velocity_env_cfg.RobotEnvCfg):
    """G1 locomotion with terrain and wrist pose command interfaces."""

    observations: WholeBodyObservationsCfg = WholeBodyObservationsCfg()
    commands: WholeBodyCommandsCfg = WholeBodyCommandsCfg()
    rewards: WholeBodyRewardsCfg = WholeBodyRewardsCfg()
    curriculum: WholeBodyCurriculumCfg = WholeBodyCurriculumCfg()


@configclass
class WholeBodyPlayEnvCfg(WholeBodyEnvCfg):
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
