from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hbc_lab.tasks.locomotion import mdp

from .scenes import HAND_CENTER_FRAME_NAME
from .object_shape_bps import (
    apply_directional_bps_ablation,
    directional_bps_in_grasp_frame,
    directional_bps_in_robot_root_frame,
    scale_directional_bps_offsets,
)


def hand_center_positions_w(env) -> tuple[torch.Tensor, torch.Tensor]:
    hand_center_pos_w = env.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
    return hand_center_pos_w[:, 0, :], hand_center_pos_w[:, 1, :]


def object_goal_hand_obs(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    env._update_contact_target_regions()
    goal_pos_w = env.object_target_pos_w
    left_pos_w, right_pos_w = hand_center_positions_w(env)
    goal_pos_b = quat_apply_inverse(robot.data.root_quat_w, goal_pos_w - robot.data.root_pos_w)
    left_pos_b = quat_apply_inverse(robot.data.root_quat_w, left_pos_w - robot.data.root_pos_w)
    right_pos_b = quat_apply_inverse(robot.data.root_quat_w, right_pos_w - robot.data.root_pos_w)
    target_region_w = env.contact_label.target_region
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, target_region_w.shape[1], -1)
    target_region_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        (target_region_w - robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, -1, 3)
    effector_mask = env.contact_label.effector_mask.to(dtype=goal_pos_b.dtype)
    target_region_b = target_region_b * effector_mask.unsqueeze(-1)
    return torch.cat((goal_pos_b, left_pos_b, right_pos_b, target_region_b.flatten(1), effector_mask), dim=-1)


def object_goal_hand_pose_obs(env) -> torch.Tensor:
    """Task state with the selected 6-DoF contact target in the robot root frame."""
    robot: Articulation = env.scene["robot"]
    env._update_contact_target_regions()
    goal_pos_b = quat_apply_inverse(
        robot.data.root_quat_w,
        env.object_target_pos_w - robot.data.root_pos_w,
    )
    left_pos_w, right_pos_w = hand_center_positions_w(env)
    left_pos_b = quat_apply_inverse(robot.data.root_quat_w, left_pos_w - robot.data.root_pos_w)
    right_pos_b = quat_apply_inverse(robot.data.root_quat_w, right_pos_w - robot.data.root_pos_w)

    target_region_w = env.contact_label.target_region
    target_orientation_w = env.contact_label.target_orientation
    root_pos_w = robot.data.root_pos_w.unsqueeze(1).expand(-1, target_region_w.shape[1], -1)
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, target_region_w.shape[1], -1)
    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w.reshape(-1, 3),
        root_quat_w.reshape(-1, 4),
        target_region_w.reshape(-1, 3),
        target_orientation_w.reshape(-1, 4),
    )
    target_pos_b = target_pos_b.reshape(env.num_envs, 2, 3)
    target_rot_6d_b = (
        matrix_from_quat(target_quat_b)
        .reshape(env.num_envs, 2, 3, 3)[..., :, :2]
        .transpose(-1, -2)
        .reshape(env.num_envs, 2, 6)
    )
    effector_mask = env.contact_label.effector_mask.to(dtype=goal_pos_b.dtype)
    target_pose_b = torch.cat((target_pos_b, target_rot_6d_b), dim=-1) * effector_mask.unsqueeze(-1)
    return torch.cat(
        (goal_pos_b, left_pos_b, right_pos_b, target_pose_b.flatten(1), effector_mask),
        dim=-1,
    )


def grip_obs(env) -> torch.Tensor:
    if not hasattr(env, "command_state"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return torch.cat((env.command_state.left_grip, env.command_state.right_grip), dim=-1)


def high_level_command_state_obs(env) -> torch.Tensor:
    if not hasattr(env, "command_state"):
        return torch.zeros(env.num_envs, 23, device=env.device)

    command = env.command_state
    left_rot_6d = (
        matrix_from_quat(command.left_hand_center_pose_a[:, 3:])[:, :, :2].transpose(1, 2).reshape(-1, 6)
    )
    right_rot_6d = (
        matrix_from_quat(command.right_hand_center_pose_a[:, 3:])[:, :, :2].transpose(1, 2).reshape(-1, 6)
    )
    return torch.cat(
        (
            command.base_velocity,
            command.posture_command,
            command.left_hand_center_pose_a[:, :3],
            left_rot_6d,
            command.right_hand_center_pose_a[:, :3],
            right_rot_6d,
        ),
        dim=-1,
    )


def execution_state_obs(
    env,
    sensor_noise: bool = False,
    base_velocity_noise: float = 0.10,
    joint_position_noise: float = 0.01,
) -> torch.Tensor:
    if not all(
        hasattr(env, name)
        for name in ("command_state", "command_rate", "upper_body_joint_ids", "torso_body_id")
    ):
        return torch.zeros(env.num_envs, 69, device=env.device)

    robot: Articulation = env.scene["robot"]
    base_linear_velocity = robot.data.root_lin_vel_b
    upper_body_joint_position = (
        robot.data.joint_pos[:, env.upper_body_joint_ids]
        - robot.data.default_joint_pos[:, env.upper_body_joint_ids]
    )
    if sensor_noise:
        base_linear_velocity = base_linear_velocity + torch.empty_like(base_linear_velocity).uniform_(
            -base_velocity_noise, base_velocity_noise
        )
        upper_body_joint_position = upper_body_joint_position + torch.empty_like(
            upper_body_joint_position
        ).uniform_(-joint_position_noise, joint_position_noise)

    frame_sensor = env.scene[HAND_CENTER_FRAME_NAME]
    current_pose_b: list[torch.Tensor] = []
    pose_error_b: list[torch.Tensor] = []
    command_poses = (
        env.command_state.left_hand_center_pose_a,
        env.command_state.right_hand_center_pose_a,
    )
    for frame_index, (side, command_pose) in enumerate(zip(("left", "right"), command_poses)):
        current_pos_w = frame_sensor.data.target_pos_w[:, frame_index]
        current_quat_w = frame_sensor.data.target_quat_w[:, frame_index]
        current_pos_b, current_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            current_pos_w,
            current_quat_w,
        )
        current_rot_6d = matrix_from_quat(current_quat_b)[:, :, :2].transpose(1, 2).reshape(-1, 6)
        current_pose_b.append(torch.cat((current_pos_b, current_rot_6d), dim=-1))

        target_pos_w, target_quat_w = env.low_level_obs_builder._target_pose_w(
            command_pose,
            side,
            env.command_state.posture_command,
        )
        position_error_b = math_utils.quat_apply_inverse(
            robot.data.root_quat_w,
            target_pos_w - current_pos_w,
        )
        _, orientation_error_w = math_utils.compute_pose_error(
            current_pos_w,
            current_quat_w,
            target_pos_w,
            target_quat_w,
        )
        orientation_error_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, orientation_error_w)
        pose_error_b.append(torch.cat((position_error_b, orientation_error_b), dim=-1))

    root_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    _, torso_pitch, _ = math_utils.euler_xyz_from_quat(robot.data.body_quat_w[:, env.torso_body_id])
    current_posture = torch.stack((root_height, torso_pitch), dim=-1)
    posture_error = env.command_state.posture_command - current_posture

    return torch.cat(
        (
            base_linear_velocity,
            upper_body_joint_position,
            current_pose_b[0],
            current_pose_b[1],
            pose_error_b[0],
            pose_error_b[1],
            posture_error,
            env.command_rate,
        ),
        dim=-1,
    )


def last_high_level_action(env) -> torch.Tensor:
    return env.last_high_level_action


def privileged_drc_obs(env) -> torch.Tensor:
    return torch.stack(
        (
            env.d_active_hand,
            env.c_contact,
            env.c_couple,
            env.d_goal,
            env.W_app,
            env.W_couple,
            env.W_manip,
        ),
        dim=-1,
    )


def directional_object_bps_obs(env) -> torch.Tensor:
    """Current object geometry and orientation, expressed in the robot root frame."""
    if env.object_shape_surface_offsets_o is None:
        raise RuntimeError("Directional BPS observation requires object shape metadata")
    robot: Articulation = env.scene["robot"]
    offsets_o = env.object_shape_surface_offsets_o[env.active_object_index]
    offsets_o = scale_directional_bps_offsets(
        offsets_o,
        env.object_shape_bps_basis_o,
        env.object_size_scale,
    )
    offsets_b = directional_bps_in_robot_root_frame(
        offsets_o,
        env._object_root_quat_w(),
        robot.data.root_quat_w,
    )
    offsets_b = apply_directional_bps_ablation(
        offsets_b,
        enabled=env.cfg.object_shape_bps_enabled,
    )
    return offsets_b.flatten(1)


def grasp_frame_object_bps_obs(env) -> torch.Tensor:
    """Object surface points expressed in the selected grasp-reference frame."""
    if env.grasp_reference_positions_o is None:
        return directional_object_bps_obs(env)
    shape_index = env.active_object_index
    candidate_index = env.selected_grasp_reference_index
    descriptor_g = directional_bps_in_grasp_frame(
        env.object_shape_surface_offsets_o[shape_index],
        env.object_shape_bps_basis_o,
        env.object_geometry_center_offsets_o[shape_index],
        env.grasp_reference_positions_o[shape_index, candidate_index],
        env.grasp_reference_quaternions_o[shape_index, candidate_index],
        env.object_size_scale,
    )
    descriptor_g = apply_directional_bps_ablation(
        descriptor_g,
        enabled=env.cfg.object_shape_bps_enabled,
    )
    return descriptor_g.flatten(1)


@configclass
class G1Dex1HierDrcObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        task = ObsTerm(func=object_goal_hand_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        grip = ObsTerm(func=grip_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        command_state = ObsTerm(func=high_level_command_state_obs)
        execution = ObsTerm(func=execution_state_obs, params={"sensor_noise": True})
        last_high_action = ObsTerm(func=last_high_level_action)

        def __post_init__(self):
            self.history_length = 10
            self.flatten_history_dim = True
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        task = ObsTerm(func=object_goal_hand_obs)
        grip = ObsTerm(func=grip_obs)
        command_state = ObsTerm(func=high_level_command_state_obs)
        execution = ObsTerm(func=execution_state_obs)
        last_high_action = ObsTerm(func=last_high_level_action)
        drc = ObsTerm(func=privileged_drc_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


def _enable_dynamic_term_history(group: ObsGroup) -> None:
    for term in (
        group.base_ang_vel,
        group.projected_gravity,
        group.task,
        group.grip,
        group.command_state,
        group.execution,
        group.last_high_action,
    ):
        term.history_length = 10
        term.flatten_history_dim = True


@configclass
class G1Dex1HierDrcMultiShapeObservationsCfg:
    @configclass
    class PolicyCfg(G1Dex1HierDrcObservationsCfg.PolicyCfg):
        object_shape = ObsTerm(func=directional_object_bps_obs, history_length=0)

        def __post_init__(self):
            super().__post_init__()
            self.history_length = None
            _enable_dynamic_term_history(self)

    @configclass
    class CriticCfg(G1Dex1HierDrcObservationsCfg.CriticCfg):
        object_shape = ObsTerm(func=directional_object_bps_obs, history_length=0)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1Dex1HierDrcMultiShapeGraspRefObservationsCfg:
    @configclass
    class PolicyCfg(G1Dex1HierDrcMultiShapeObservationsCfg.PolicyCfg):
        task = ObsTerm(func=object_goal_hand_pose_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_shape = ObsTerm(func=grasp_frame_object_bps_obs, history_length=0)

    @configclass
    class CriticCfg(G1Dex1HierDrcMultiShapeObservationsCfg.CriticCfg):
        task = ObsTerm(func=object_goal_hand_pose_obs)
        object_shape = ObsTerm(func=grasp_frame_object_bps_obs, history_length=0)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
