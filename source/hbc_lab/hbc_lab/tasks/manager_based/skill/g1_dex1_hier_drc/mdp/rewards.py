from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp

from .motion_regularization import (
    comfort_weights,
    mask_normalized_mean,
    normalized_deadzone_square,
    normalized_joint_limit_proximity,
    quaternion_angular_distance,
    radial_workspace_violation,
    reachability_gate,
    scaled_quality_reward,
    stable_grasp_released_pose_guidance,
    walking_posture_weight,
)
from .scenes import HAND_CENTER_FRAME_NAME


INNER_PAD_CONTACT_GATE_SCALE = 0.05


def approach_reward(env) -> torch.Tensor:
    return torch.exp(-env.d_active_hand / 0.5)


def active_inner_pad_contact(env) -> torch.Tensor:
    if not hasattr(env, "_step_link_contact"):
        return torch.zeros_like(env.d_active_hand)

    left_active = env.active_hand == 0
    left_link1_3 = env._step_link_contact["left_Link1_3"]
    left_link2_3 = env._step_link_contact["left_Link2_3"]
    right_link1_3 = env._step_link_contact["right_Link1_3"]
    right_link2_3 = env._step_link_contact["right_Link2_3"]
    active_link1_3 = torch.where(left_active, left_link1_3, right_link1_3)
    active_link2_3 = torch.where(left_active, left_link2_3, right_link2_3)
    return torch.maximum(active_link1_3, active_link2_3)


def couple_reward(env) -> torch.Tensor:
    near = 1.0 - torch.tanh(env.d_active_hand / 0.35)
    grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.1)
    inner_pad_contact = active_inner_pad_contact(env)
    pad_gate = torch.clamp(inner_pad_contact / INNER_PAD_CONTACT_GATE_SCALE, min=0.0, max=1.0)
    gripper_close = env.active_grip
    pad_gated_gripper_close = gripper_close * pad_gate
    pad_early_close_penalty = gripper_close * (1.0 - pad_gate)
    close_ready_gate = torch.maximum(grasp_window, pad_gate)
    gated_gripper_close = gripper_close * close_ready_gate
    early_close_penalty = gripper_close * (1.0 - close_ready_gate)
    env._couple_grasp_window = grasp_window.detach()
    env._couple_pad_gate = pad_gate.detach()
    env._couple_pad_gated_gripper_close = pad_gated_gripper_close.detach()
    env._couple_pad_early_close_penalty = pad_early_close_penalty.detach()
    env._couple_gated_gripper_close = gated_gripper_close.detach()
    env._couple_early_close_penalty = early_close_penalty.detach()
    return (
        0.4 * near
        + 0.1 * pad_gate
        + 0.1 * env.c_pinch
        + 0.2 * env.c_grasp
        + 0.2 * gated_gripper_close
        - 0.2 * early_close_penalty
    )


def grasp_pose_guidance_reward(env) -> torch.Tensor:
    if not env.cfg.grasp_pose_guidance_enabled:
        return torch.zeros_like(env.d_active_hand)

    position_guidance, orientation_guidance, position_score, orientation_score, release_gate = (
        stable_grasp_released_pose_guidance(
            env.d_active_hand,
            env.active_orientation_error,
            torch.clamp(env.c_physical_grasp.detach(), min=0.0, max=1.0),
            position_scale=env.cfg.grasp_pose_position_scale,
            release_threshold=env.cfg.grasp_pose_release_threshold,
            release_width=env.cfg.grasp_pose_release_width,
        )
    )
    reward = (
        env.cfg.grasp_pose_position_reward_weight * position_guidance
        + env.cfg.grasp_pose_orientation_reward_weight * orientation_guidance
    )
    env.extras["log"]["GraspPose/position_score_mean"] = position_score.detach().mean()
    env.extras["log"]["GraspPose/orientation_score_mean"] = orientation_score.detach().mean()
    env.extras["log"]["GraspPose/stable_release_gate_mean"] = release_gate.detach().mean()
    env.extras["log"]["GraspPose/guidance_reward_mean"] = reward.detach().mean()
    return reward


def manip_reward(env) -> torch.Tensor:
    initial = torch.norm(env.object_initial_pos_w - env.object_target_pos_w, dim=-1)
    progress = torch.clamp(initial - env.d_goal, min=0.0) / (initial + 1e-5)
    return 0.7 * progress + 0.3


def hier_drc_reward(env, approach_scale: float = 2.0, couple_scale: float = 20.0, manip_scale: float = 200.0) -> torch.Tensor:
    r_app = approach_reward(env)
    r_couple = couple_reward(env)
    r_manip = manip_reward(env)
    r_pose_guidance = grasp_pose_guidance_reward(env)
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    env.extras["log"]["GraspPose/couple_weighted_guidance_mean"] = (
        env.W_couple * r_pose_guidance
    ).detach().mean()
    return (
        env.W_app * approach_scale * r_app
        + env.W_couple * couple_scale * r_couple
        + env.W_manip * manip_scale * r_manip
        + env.W_couple * r_pose_guidance
    )


def root_object_facing_reward(env, eps: float = 1.0e-6) -> torch.Tensor:
    robot = env.scene["robot"]
    object_pos_w = env._object_frame_pos_w()
    root_to_object_xy = object_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    root_to_object_dir = root_to_object_xy / torch.clamp(torch.norm(root_to_object_xy, dim=-1, keepdim=True), min=eps)

    forward_b = torch.zeros_like(robot.data.root_pos_w)
    forward_b[:, 0] = 1.0
    forward_w = math_utils.quat_apply(math_utils.yaw_quat(robot.data.root_quat_w), forward_b)[:, :2]
    forward_dir = forward_w / torch.clamp(torch.norm(forward_w, dim=-1, keepdim=True), min=eps)

    facing_cos = torch.sum(forward_dir * root_to_object_dir, dim=-1)
    reward = torch.square(torch.clamp(facing_cos, min=0.0, max=1.0))
    env.extras["log"]["DRC/root_object_facing_mean"] = reward.mean()
    return reward


def _capped(loss: torch.Tensor) -> torch.Tensor:
    return torch.clamp(loss, min=0.0, max=4.0)


def _motion_rate_scale(env, dtype: torch.dtype) -> torch.Tensor:
    cache_name = "_motion_rate_scale"
    cached = getattr(env, cache_name, None)
    if cached is None or cached.dtype != dtype:
        limits = env.action_limits
        cached = torch.tensor(
            (
                limits.max_lin_acc,
                limits.max_lin_acc,
                limits.max_ang_acc,
                limits.root_height_speed,
                limits.torso_pitch_speed,
                *([limits.hand_linear_speed] * 3),
                *([limits.hand_angular_speed] * 3),
                *([limits.hand_linear_speed] * 3),
                *([limits.hand_angular_speed] * 3),
            ),
            device=env.device,
            dtype=dtype,
        )
        setattr(env, cache_name, cached)
    return cached


def _hand_tracking_losses(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    frame_sensor = env.scene[HAND_CENTER_FRAME_NAME]
    position_losses = []
    orientation_losses = []
    command_poses = (
        env.command_state.left_hand_center_pose_a,
        env.command_state.right_hand_center_pose_a,
    )
    for frame_index, (side, command_pose) in enumerate(zip(("left", "right"), command_poses)):
        target_pos_w, target_quat_w = env.low_level_obs_builder._target_pose_w(
            command_pose,
            side,
            env.command_state.posture_command,
        )
        current_pos_w = frame_sensor.data.target_pos_w[:, frame_index]
        current_quat_w = frame_sensor.data.target_quat_w[:, frame_index]
        position_error = torch.linalg.norm(target_pos_w - current_pos_w, dim=-1)
        orientation_error = quaternion_angular_distance(target_quat_w, current_quat_w)
        position_losses.append(
            normalized_deadzone_square(
                position_error,
                env.cfg.motion_position_deadzone,
                env.cfg.motion_position_scale,
            )
        )
        orientation_losses.append(
            normalized_deadzone_square(
                orientation_error,
                env.cfg.motion_orientation_deadzone,
                env.cfg.motion_orientation_scale,
            )
        )
    return torch.stack(position_losses, dim=-1), torch.stack(orientation_losses, dim=-1)


def _hand_command_comfort_loss(env, comfort: torch.Tensor) -> torch.Tensor:
    command_cfg = env.cfg.commands.high_level
    command_poses = torch.stack(
        (
            env.command_state.left_hand_center_pose_a,
            env.command_state.right_hand_center_pose_a,
        ),
        dim=1,
    )
    default_poses = torch.tensor(
        (
            command_cfg.default_left_hand_center_pose_a,
            command_cfg.default_right_hand_center_pose_a,
        ),
        device=env.device,
        dtype=command_poses.dtype,
    ).unsqueeze(0)
    position_loss = torch.square(
        torch.linalg.norm(command_poses[..., :3] - default_poses[..., :3], dim=-1)
        / env.cfg.motion_command_position_scale
    )
    orientation_loss = torch.square(
        quaternion_angular_distance(command_poses[..., 3:], default_poses[..., 3:])
        / env.cfg.motion_command_orientation_scale
    )
    return mask_normalized_mean(_capped(0.5 * (position_loss + orientation_loss)), comfort)


def _arm_comfort_and_speed_losses(env, comfort: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    upper_ids = env.upper_body_joint_ids
    joint_position_relative = robot.data.joint_pos[:, upper_ids] - robot.data.default_joint_pos[:, upper_ids]
    joint_velocity = robot.data.joint_vel[:, upper_ids]
    arm_position = torch.stack(
        (
            torch.mean(torch.square(joint_position_relative[:, 3:10] / env.cfg.motion_arm_position_scale), dim=-1),
            torch.mean(torch.square(joint_position_relative[:, 10:17] / env.cfg.motion_arm_position_scale), dim=-1),
        ),
        dim=-1,
    )
    arm_speed = torch.stack(
        (
            torch.mean(torch.square(joint_velocity[:, 3:10] / env.cfg.motion_arm_velocity_scale), dim=-1),
            torch.mean(torch.square(joint_velocity[:, 10:17] / env.cfg.motion_arm_velocity_scale), dim=-1),
        ),
        dim=-1,
    )
    return (
        mask_normalized_mean(_capped(arm_position), comfort),
        mask_normalized_mean(_capped(arm_speed), comfort),
    )


def _walking_posture_loss(env, walking_weight: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    root_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    default_height = env.cfg.commands.high_level.default_root_height
    torso_quat_w = robot.data.body_quat_w[:, env.torso_body_id]
    _, torso_pitch_w, _ = math_utils.euler_xyz_from_quat(torso_quat_w)
    torso_quat_b = math_utils.quat_mul(math_utils.quat_inv(robot.data.root_quat_w), torso_quat_w)
    torso_roll_b, _, torso_yaw_b = math_utils.euler_xyz_from_quat(torso_quat_b)
    upper_position_relative = (
        robot.data.joint_pos[:, env.upper_body_joint_ids]
        - robot.data.default_joint_pos[:, env.upper_body_joint_ids]
    )
    waist_yaw_roll = upper_position_relative[:, :2]
    terms = torch.stack(
        (
            normalized_deadzone_square(root_height - default_height, 0.03, 0.12),
            normalized_deadzone_square(torso_pitch_w, 0.08, 0.35),
            normalized_deadzone_square(torso_roll_b, 0.08, 0.35),
            normalized_deadzone_square(torso_yaw_b, 0.08, 0.35),
            normalized_deadzone_square(waist_yaw_roll[:, 0], 0.10, 0.40),
            normalized_deadzone_square(waist_yaw_roll[:, 1], 0.10, 0.40),
        ),
        dim=-1,
    )
    return walking_weight * _capped(torch.mean(terms, dim=-1))


def contact_conditioned_motion_reward(
    env,
    approach_scale: float = 2.0,
    couple_scale: float = 20.0,
    manip_scale: float = 200.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    effector_mask = env.contact_label.effector_mask.to(dtype=robot.data.root_pos_w.dtype)
    shoulder_pos_w = torch.stack(
        (
            robot.data.body_pos_w[:, env.left_shoulder_body_id],
            robot.data.body_pos_w[:, env.right_shoulder_body_id],
        ),
        dim=1,
    )
    shoulder_target_distance = torch.linalg.norm(env.contact_label.target_region - shoulder_pos_w, dim=-1)
    reach_gate = reachability_gate(
        shoulder_target_distance,
        release_radius=env.cfg.motion_release_radius,
        release_width=env.cfg.motion_release_width,
    )
    comfort = comfort_weights(effector_mask, reach_gate)
    walking_weight = walking_posture_weight(
        effector_mask,
        reach_gate,
        near_floor=env.cfg.motion_near_posture_floor,
    )

    hand_position, hand_orientation = _hand_tracking_losses(env)
    hand_position_loss = _capped(torch.mean(hand_position, dim=-1))
    hand_orientation_loss = _capped(torch.mean(hand_orientation, dim=-1))

    root_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    _, torso_pitch, _ = math_utils.euler_xyz_from_quat(robot.data.body_quat_w[:, env.torso_body_id])
    posture_error = torch.stack(
        (
            normalized_deadzone_square(
                env.command_state.posture_command[:, 0] - root_height,
                env.cfg.motion_root_height_deadzone,
                env.cfg.motion_root_height_scale,
            ),
            normalized_deadzone_square(
                env.command_state.posture_command[:, 1] - torso_pitch,
                env.cfg.motion_torso_pitch_deadzone,
                env.cfg.motion_torso_pitch_scale,
            ),
        ),
        dim=-1,
    )
    posture_tracking_loss = _capped(torch.mean(posture_error, dim=-1))

    hand_commands_a = torch.stack(
        (
            env.command_state.left_hand_center_pose_a[:, :3],
            env.command_state.right_hand_center_pose_a[:, :3],
        ),
        dim=1,
    )
    workspace_loss = _capped(
        torch.mean(
            radial_workspace_violation(
                hand_commands_a,
                radius_range=env.cfg.motion_workspace_radius_range,
                scale=env.cfg.motion_workspace_scale,
            ),
            dim=-1,
        )
    )

    gravity_b = robot.data.projected_gravity_b
    root_tilt = torch.acos(
        torch.clamp(-gravity_b[:, 2] / torch.clamp(torch.linalg.norm(gravity_b, dim=-1), min=1.0e-6), -1.0, 1.0)
    )
    root_tilt_loss = _capped(
        normalized_deadzone_square(
            root_tilt,
            env.cfg.motion_root_tilt_deadzone,
            env.cfg.motion_root_tilt_scale,
        )
    )

    upper_joint_position = robot.data.joint_pos[:, env.upper_body_joint_ids]
    upper_joint_limits = robot.data.soft_joint_pos_limits[:, env.upper_body_joint_ids]
    joint_limit_loss = _capped(
        torch.mean(
            normalized_joint_limit_proximity(
                upper_joint_position,
                upper_joint_limits,
                margin_fraction=env.cfg.motion_joint_limit_margin,
            ),
            dim=-1,
        )
    )

    hand_command_comfort_loss = _hand_command_comfort_loss(env, comfort)
    arm_joint_comfort_loss, arm_joint_speed_loss = _arm_comfort_and_speed_losses(env, comfort)
    walking_loss = _walking_posture_loss(env, walking_weight)

    hand_twists = torch.stack((env.command_rate[:, 5:11], env.command_rate[:, 11:17]), dim=1)
    command_speed_per_hand = 0.5 * (
        torch.square(torch.linalg.norm(hand_twists[..., :3], dim=-1) / env.action_limits.hand_linear_speed)
        + torch.square(torch.linalg.norm(hand_twists[..., 3:], dim=-1) / env.action_limits.hand_angular_speed)
    )
    command_speed_loss = mask_normalized_mean(_capped(command_speed_per_hand), comfort)

    rate_scale = _motion_rate_scale(env, env.command_rate.dtype)
    smoothing_time = env.cfg.motion_command_smoothing_time
    command_acceleration_loss = _capped(
        torch.mean(torch.square(env.command_acceleration * smoothing_time / rate_scale), dim=-1)
    )
    command_jerk_loss = _capped(
        torch.mean(torch.square(env.command_jerk * smoothing_time * smoothing_time / rate_scale), dim=-1)
    )

    losses = {
        "hand_position_tracking": hand_position_loss,
        "hand_orientation_tracking": hand_orientation_loss,
        "posture_tracking": posture_tracking_loss,
        "radial_workspace": workspace_loss,
        "root_tilt": root_tilt_loss,
        "upper_body_joint_limit": joint_limit_loss,
        "hand_command_comfort": hand_command_comfort_loss,
        "arm_joint_comfort": arm_joint_comfort_loss,
        "walking_posture": walking_loss,
        "command_speed": command_speed_loss,
        "arm_joint_speed": arm_joint_speed_loss,
        "command_acceleration": command_acceleration_loss,
        "command_jerk": command_jerk_loss,
    }
    weights = {
        "hand_position_tracking": 0.20,
        "hand_orientation_tracking": 0.10,
        "posture_tracking": 0.10,
        "radial_workspace": 0.10,
        "root_tilt": 0.10,
        "upper_body_joint_limit": 0.05,
        "hand_command_comfort": 0.10,
        "arm_joint_comfort": 0.05,
        "walking_posture": 0.10,
        "command_speed": 0.04,
        "arm_joint_speed": 0.02,
        "command_acceleration": 0.025,
        "command_jerk": 0.015,
    }
    quality_loss = sum(weights[name] * loss for name, loss in losses.items())
    stage_scale = env.W_app * approach_scale + env.W_couple * couple_scale + env.W_manip * manip_scale
    reward = scaled_quality_reward(
        stage_scale,
        quality_loss,
        coefficient=env.cfg.motion_quality_coefficient,
        loss_cap=env.cfg.motion_quality_loss_cap,
    )

    for name in (
        "hand_position_tracking",
        "hand_orientation_tracking",
        "posture_tracking",
        "root_tilt",
        "walking_posture",
    ):
        loss = losses[name]
        env.extras["log"][f"Motion/{name}_loss"] = loss.detach().mean()
    env.extras["log"]["Motion/quality_loss_mean"] = quality_loss.detach().mean()
    env.extras["log"]["Motion/scaled_reward_mean"] = reward.detach().mean()
    return reward


def task_success_reward(env) -> torch.Tensor:
    return env.task_succeeded.float() * 100.0


def object_fall_penalty(env) -> torch.Tensor:
    return env.object_fallen.float()


@configclass
class G1Dex1HierDrcRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    motion_quality = RewTerm(
        func=contact_conditioned_motion_reward,
        weight=1.0,
        params={"approach_scale": 2.0, "couple_scale": 20.0, "manip_scale": 200.0},
    )
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    object_fall = RewTerm(func=object_fall_penalty, weight=-2.0)
    root_object_facing = RewTerm(func=root_object_facing_reward, weight=2.0)
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*|.*gripper.*|.*finger.*|.*hand.*).*"],
            ),
        },
    )
