from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HighLevelActionLimits:
    max_lin_vel: float = 0.8
    max_ang_vel: float = 0.5
    max_command_rate: float = 0.25
    wrist_delta_scale: float = 0.06
    wrist_rot_delta_scale: float = 0.18
    posture_delta_scale: tuple[float, float] = (0.04, 0.08)
    root_height_range: tuple[float, float] = (0.42, 0.80)
    torso_pitch_range: tuple[float, float] = (0.0, 0.85)
    left_workspace_min: tuple[float, float, float] = (0.05, 0.05, -0.55)
    left_workspace_max: tuple[float, float, float] = (0.85, 0.65, 0.35)
    right_workspace_min: tuple[float, float, float] = (0.05, -0.65, -0.55)
    right_workspace_max: tuple[float, float, float] = (0.85, -0.05, 0.35)


@dataclass
class HighLevelCommandState:
    base_velocity: torch.Tensor
    posture_command: torch.Tensor
    left_wrist_pose_b: torch.Tensor
    right_wrist_pose_b: torch.Tensor
    left_grip: torch.Tensor
    right_grip: torch.Tensor


def _limit_rate(target: torch.Tensor, previous: torch.Tensor, max_delta: float) -> torch.Tensor:
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


def _clamp_pose_pos(pose: torch.Tensor, lower: tuple[float, float, float], upper: tuple[float, float, float]) -> torch.Tensor:
    lower_t = torch.tensor(lower, device=pose.device, dtype=pose.dtype)
    upper_t = torch.tensor(upper, device=pose.device, dtype=pose.dtype)
    pose[:, :3] = torch.max(torch.min(pose[:, :3], upper_t), lower_t)
    return pose


def _decode_wrist(
    raw_action: torch.Tensor,
    previous_pose: torch.Tensor,
    start: int,
    limits: HighLevelActionLimits,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = previous_pose.clone()
    pose[:, :3] = pose[:, :3] + raw_action[:, start : start + 3] * limits.wrist_delta_scale
    delta_quat = _quat_from_rotvec(raw_action[:, start + 3 : start + 6] * limits.wrist_rot_delta_scale)
    pose[:, 3:] = _normalize_quat(_quat_mul(delta_quat, pose[:, 3:]))
    pose = _clamp_pose_pos(pose, lower, upper)
    grip = (raw_action[:, start + 6 : start + 7] + 1.0) * 0.5
    return pose, grip.clamp(0.0, 1.0)


def decode_high_level_action(
    raw_action: torch.Tensor,
    previous: HighLevelCommandState,
    limits: HighLevelActionLimits,
) -> HighLevelCommandState:
    if raw_action.shape[-1] != 19:
        raise ValueError(f"Expected high-level action shape (num_envs, 19), got {tuple(raw_action.shape)}")
    raw_action = torch.clamp(raw_action, -1.0, 1.0)

    target_base_velocity = torch.empty_like(previous.base_velocity)
    target_base_velocity[:, 0] = raw_action[:, 0] * limits.max_lin_vel
    target_base_velocity[:, 1] = raw_action[:, 1] * limits.max_lin_vel
    target_base_velocity[:, 2] = raw_action[:, 2] * limits.max_ang_vel
    base_velocity = _limit_rate(target_base_velocity, previous.base_velocity, limits.max_command_rate)

    posture_command = previous.posture_command.clone()
    posture_delta = torch.tensor(limits.posture_delta_scale, device=raw_action.device, dtype=raw_action.dtype)
    posture_command = posture_command + raw_action[:, 3:5] * posture_delta
    root_min, root_max = limits.root_height_range
    pitch_min, pitch_max = limits.torso_pitch_range
    posture_command[:, 0] = posture_command[:, 0].clamp(root_min, root_max)
    posture_command[:, 1] = posture_command[:, 1].clamp(pitch_min, pitch_max)

    left_wrist_pose_b, left_grip = _decode_wrist(
        raw_action,
        previous.left_wrist_pose_b,
        5,
        limits,
        limits.left_workspace_min,
        limits.left_workspace_max,
    )
    right_wrist_pose_b, right_grip = _decode_wrist(
        raw_action,
        previous.right_wrist_pose_b,
        12,
        limits,
        limits.right_workspace_min,
        limits.right_workspace_max,
    )

    return HighLevelCommandState(
        base_velocity=base_velocity,
        posture_command=posture_command,
        left_wrist_pose_b=left_wrist_pose_b,
        right_wrist_pose_b=right_wrist_pose_b,
        left_grip=left_grip,
        right_grip=right_grip,
    )
