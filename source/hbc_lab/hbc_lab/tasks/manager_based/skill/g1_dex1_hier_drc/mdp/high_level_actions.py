from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HighLevelActionLimits:
    max_lin_vel: float = 0.8
    max_ang_vel: float = 0.5
    max_lin_acc: float = 1.5
    max_ang_acc: float = 1.5
    hand_linear_speed: float = 0.30
    hand_angular_speed: float = 1.00
    root_height_speed: float = 0.12
    torso_pitch_speed: float = 0.40
    root_height_range: tuple[float, float] = (0.42, 0.80)
    torso_pitch_range: tuple[float, float] = (0.0, 0.85)
    wrist_radius_range: tuple[float, float] = (0.20, 0.58)


@dataclass
class HighLevelCommandState:
    base_velocity: torch.Tensor
    posture_command: torch.Tensor
    left_hand_center_pose_a: torch.Tensor
    right_hand_center_pose_a: torch.Tensor
    left_grip: torch.Tensor
    right_grip: torch.Tensor


@dataclass
class HighLevelCommandRateState:
    base_acceleration: torch.Tensor
    posture_velocity: torch.Tensor
    left_hand_twist: torch.Tensor
    right_hand_twist: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            (self.base_acceleration, self.posture_velocity, self.left_hand_twist, self.right_hand_twist), dim=-1
        )


def _limit_rate(target: torch.Tensor, previous: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
    delta = torch.clamp(target - previous, min=-max_delta, max=max_delta)
    return previous + delta


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1.0e-6)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    small_angle_scale = 0.5 - angle * angle / 48.0
    scale = torch.where(angle < 1.0e-6, small_angle_scale, torch.sin(half_angle) / angle)
    return _normalize_quat(torch.cat((torch.cos(half_angle), rotvec * scale), dim=-1))


def _decode_hand_center(
    raw_action: torch.Tensor,
    previous_pose: torch.Tensor,
    start: int,
    limits: HighLevelActionLimits,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose = previous_pose.clone()
    linear_velocity = raw_action[:, start : start + 3] * limits.hand_linear_speed
    angular_velocity = raw_action[:, start + 3 : start + 6] * limits.hand_angular_speed
    pose[:, :3] = pose[:, :3] + linear_velocity * dt
    delta_quat = _quat_from_rotvec(angular_velocity * dt)
    pose[:, 3:] = _normalize_quat(_quat_mul(pose[:, 3:], delta_quat))
    grip = (raw_action[:, start + 6 : start + 7] + 1.0) * 0.5
    return pose, grip.clamp(0.0, 1.0), torch.cat((linear_velocity, angular_velocity), dim=-1)


def decode_high_level_action(
    raw_action: torch.Tensor,
    previous: HighLevelCommandState,
    limits: HighLevelActionLimits,
    dt: float,
) -> tuple[HighLevelCommandState, HighLevelCommandRateState]:
    if raw_action.shape[-1] != 19:
        raise ValueError(f"Expected high-level action shape (num_envs, 19), got {tuple(raw_action.shape)}")
    if dt <= 0.0:
        raise ValueError(f"Expected a positive high-level dt, got {dt}")
    raw_action = torch.clamp(raw_action, -1.0, 1.0)

    target_base_velocity = torch.empty_like(previous.base_velocity)
    target_base_velocity[:, 0] = raw_action[:, 0] * limits.max_lin_vel
    target_base_velocity[:, 1] = raw_action[:, 1] * limits.max_lin_vel
    target_base_velocity[:, 2] = raw_action[:, 2] * limits.max_ang_vel
    max_base_delta = torch.tensor(
        (limits.max_lin_acc, limits.max_lin_acc, limits.max_ang_acc),
        device=raw_action.device,
        dtype=raw_action.dtype,
    ) * dt
    base_velocity = _limit_rate(target_base_velocity, previous.base_velocity, max_base_delta)
    base_acceleration = (base_velocity - previous.base_velocity) / dt

    posture_speed = torch.tensor(
        (limits.root_height_speed, limits.torso_pitch_speed),
        device=raw_action.device,
        dtype=raw_action.dtype,
    )
    posture_command = previous.posture_command + raw_action[:, 3:5] * posture_speed * dt
    root_min, root_max = limits.root_height_range
    pitch_min, pitch_max = limits.torso_pitch_range
    posture_command[:, 0] = posture_command[:, 0].clamp(root_min, root_max)
    posture_command[:, 1] = posture_command[:, 1].clamp(pitch_min, pitch_max)
    posture_velocity = (posture_command - previous.posture_command) / dt

    left_hand_center_pose_a, left_grip, left_hand_twist = _decode_hand_center(
        raw_action,
        previous.left_hand_center_pose_a,
        5,
        limits,
        dt,
    )
    right_hand_center_pose_a, right_grip, right_hand_twist = _decode_hand_center(
        raw_action,
        previous.right_hand_center_pose_a,
        12,
        limits,
        dt,
    )

    return (
        HighLevelCommandState(
            base_velocity=base_velocity,
            posture_command=posture_command,
            left_hand_center_pose_a=left_hand_center_pose_a,
            right_hand_center_pose_a=right_hand_center_pose_a,
            left_grip=left_grip,
            right_grip=right_grip,
        ),
        HighLevelCommandRateState(
            base_acceleration=base_acceleration,
            posture_velocity=posture_velocity,
            left_hand_twist=left_hand_twist,
            right_hand_twist=right_hand_twist,
        ),
    )
