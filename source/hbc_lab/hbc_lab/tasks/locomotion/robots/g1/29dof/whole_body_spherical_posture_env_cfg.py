from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp

from . import whole_body_spherical_env_cfg
from .whole_body_spherical_env_cfg import SphericalWholeBodyEnvCfg


@configclass
class SphericalPostureWholeBodyCommandsCfg(whole_body_spherical_env_cfg.SphericalWholeBodyCommandsCfg):
    """Spherical wrist commands plus low-dimensional whole-body posture commands."""

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


@configclass
class SphericalPostureWholeBodyObservationsCfg(whole_body_spherical_env_cfg.SphericalWholeBodyObservationsCfg):
    """Add posture command and posture tracking error to actor and critic observations."""

    @configclass
    class PolicyCfg(whole_body_spherical_env_cfg.SphericalWholeBodyObservationsCfg.PolicyCfg):
        posture_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "posture_command"},
        )
        posture_command_error = ObsTerm(
            func=mdp.posture_command_error,
            params={
                "command_name": "posture_command",
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(whole_body_spherical_env_cfg.SphericalWholeBodyObservationsCfg.CriticCfg):
        posture_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "posture_command"},
        )
        posture_command_error = ObsTerm(
            func=mdp.posture_command_error,
            params={
                "command_name": "posture_command",
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class SphericalPostureWholeBodyRewardsCfg(whole_body_spherical_env_cfg.SphericalWholeBodyRewardsCfg):
    """Replace fixed upright/base-height rewards with commanded posture tracking."""

    base_height = None
    flat_orientation_l2 = None

    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint"])},
    )
    track_root_height = RewTerm(
        func=mdp.root_height_command_error_l2,
        weight=-20.0,
        params={
            "command_name": "posture_command",
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    track_torso_pitch = RewTerm(
        func=mdp.torso_pitch_command_error_l2,
        weight=-8.0,
        params={
            "command_name": "posture_command",
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )


@configclass
class SphericalPostureWholeBodyCurriculumCfg(whole_body_spherical_env_cfg.SphericalWholeBodyCurriculumCfg):
    """Curriculum for wrist workspace and posture command range expansion."""

    posture_cmd_levels = CurrTerm(
        func=mdp.posture_cmd_levels,
        params={
            "command_name": "posture_command",
            "error_sum_names": ("root_height_error_sum", "torso_pitch_error_sum"),
            "success_threshold": 0.05,
            "root_height_delta": 0.03,
            "torso_pitch_delta": 0.05,
            "min_episode_fraction": 0.5,
        },
    )


@configclass
class SphericalPostureWholeBodyEnvCfg(SphericalWholeBodyEnvCfg):
    """G1 spherical wrist tracking with commanded crouch and torso pitch."""

    observations: SphericalPostureWholeBodyObservationsCfg = SphericalPostureWholeBodyObservationsCfg()
    commands: SphericalPostureWholeBodyCommandsCfg = SphericalPostureWholeBodyCommandsCfg()
    rewards: SphericalPostureWholeBodyRewardsCfg = SphericalPostureWholeBodyRewardsCfg()
    curriculum: SphericalPostureWholeBodyCurriculumCfg = SphericalPostureWholeBodyCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        for wrist_pose_command in (self.commands.left_wrist_pose, self.commands.right_wrist_pose):
            wrist_pose_command.anchor_height_command_name = "posture_command"
            wrist_pose_command.anchor_height_command_index = 0
            wrist_pose_command.anchor_height_offset = 0.43
            wrist_pose_command.anchor_pitch_command_name = "posture_command"
            wrist_pose_command.anchor_pitch_command_index = 1
            wrist_pose_command.anchor_pitch_scale = 1.0


@configclass
class SphericalPostureWholeBodyPlayEnvCfg(SphericalPostureWholeBodyEnvCfg):
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
