from __future__ import annotations

import torch


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(quaternion, dim=-1)


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
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


def quaternion_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = normalize_quaternion(quaternion)
    pure_vector = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, pure_vector),
        quaternion_conjugate(quaternion),
    )
    return rotated[..., 1:]


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = torch.nn.functional.normalize(axis, dim=-1)
    half_angle = 0.5 * angle
    return torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            axis * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )


def compose_local_axis_rotation(
    frame_quat: torch.Tensor,
    axis_local: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    delta_local = quat_from_axis_angle(axis_local, angle)
    return normalize_quaternion(quaternion_multiply(frame_quat, delta_local))


def compose_local_axis_pose(
    frame_pos: torch.Tensor,
    frame_quat: torch.Tensor,
    pivot_pos: torch.Tensor,
    axis_local: torch.Tensor,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_quat = normalize_quaternion(frame_quat)
    delta_local = quat_from_axis_angle(axis_local, angle)
    delta_world = quaternion_multiply(
        quaternion_multiply(frame_quat, delta_local),
        quaternion_conjugate(frame_quat),
    )
    target_pos = pivot_pos + quaternion_apply(delta_world, frame_pos - pivot_pos)
    target_quat = normalize_quaternion(quaternion_multiply(frame_quat, delta_local))
    return target_pos, target_quat


def quaternion_log_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = normalize_quaternion(current)
    target = normalize_quaternion(target)
    error = quaternion_multiply(target, quaternion_conjugate(current))
    error = torch.where(error[..., :1] < 0.0, -error, error)
    vector = error[..., 1:]
    vector_norm = torch.norm(vector, dim=-1, keepdim=True)
    scalar = torch.clamp(error[..., :1], min=0.0, max=1.0)
    angle = 2.0 * torch.atan2(vector_norm, scalar)
    eps = torch.finfo(error.dtype).eps
    scale = torch.where(vector_norm > eps, angle / torch.clamp(vector_norm, min=eps), 2.0)
    return vector * scale


def compute_masked_pose_error(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    position_mask: torch.Tensor,
    rotation_mask: torch.Tensor,
    position_scale: float,
    rotation_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if position_scale <= 0.0 or rotation_scale <= 0.0:
        raise ValueError("pose error scales must be positive")
    position_delta = (target_pos - current_pos) * position_mask
    rotation_delta = quaternion_log_error(current_quat, target_quat) * rotation_mask
    position_error = torch.norm(position_delta, dim=-1) / position_scale
    orientation_error = torch.norm(rotation_delta, dim=-1) / rotation_scale
    return position_error + orientation_error, position_error, orientation_error


def gather_keyframe(values: torch.Tensor, keyframe_index: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("keyframe values must have shape (num_envs, num_keyframes, ...)")
    keyframe_index = keyframe_index.to(device=values.device, dtype=torch.long)
    if keyframe_index.shape != (values.shape[0],):
        raise ValueError(
            f"keyframe_index must have shape ({values.shape[0]},), got {tuple(keyframe_index.shape)}"
        )
    batch_index = torch.arange(values.shape[0], device=values.device)
    return values[batch_index, keyframe_index]


def flatten_motion_frame_markers(
    current_pos_w: torch.Tensor,
    current_quat_w: torch.Tensor,
    keyframe_pos_w: torch.Tensor,
    keyframe_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten current and keyframe world poses for frame-marker visualization."""
    if current_pos_w.shape != (keyframe_pos_w.shape[0], 3):
        raise ValueError("current_pos_w and keyframe_pos_w must share the environment dimension")
    if current_quat_w.shape != (keyframe_quat_w.shape[0], 4):
        raise ValueError("current_quat_w and keyframe_quat_w must share the environment dimension")
    if keyframe_pos_w.ndim != 3 or keyframe_pos_w.shape[-1] != 3:
        raise ValueError("keyframe_pos_w must have shape (num_envs, num_keyframes, 3)")
    if keyframe_quat_w.shape != (*keyframe_pos_w.shape[:2], 4):
        raise ValueError("keyframe_quat_w must have shape (num_envs, num_keyframes, 4)")

    positions = torch.cat((current_pos_w.unsqueeze(1), keyframe_pos_w), dim=1)
    orientations = torch.cat((current_quat_w.unsqueeze(1), keyframe_quat_w), dim=1)
    marker_indices = torch.arange(
        positions.shape[1],
        device=positions.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(positions.shape[0], -1)
    return positions.flatten(0, 1), orientations.flatten(0, 1), marker_indices.flatten()


def compute_cumulative_keyframe_progress(
    current_error: torch.Tensor,
    phase_initial_error: torch.Tensor,
    keyframe_index: torch.Tensor,
    num_keyframes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_keyframes <= 0:
        raise ValueError("num_keyframes must be positive")
    if current_error.shape != phase_initial_error.shape or current_error.shape != keyframe_index.shape:
        raise ValueError("progress inputs must have matching shapes")
    eps = torch.finfo(current_error.dtype).eps
    phase_fraction = 1.0 - current_error / torch.clamp(phase_initial_error, min=eps)
    phase_fraction = torch.clamp(phase_fraction, min=0.0, max=1.0)
    cumulative = keyframe_index.to(dtype=current_error.dtype) + phase_fraction
    cumulative = torch.clamp(cumulative, min=0.0, max=float(num_keyframes))
    return cumulative, cumulative / float(num_keyframes)


def generic_pose_manip_reward(
    normalized_progress: torch.Tensor,
    hold_bonus: float = 0.3,
    progress_gain: float = 1.0,
) -> torch.Tensor:
    return hold_bonus + progress_gain * torch.clamp(normalized_progress, min=0.0, max=1.0)
