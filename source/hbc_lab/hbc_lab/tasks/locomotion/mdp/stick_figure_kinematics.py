from __future__ import annotations

import torch


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply WXYZ quaternions."""
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


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by WXYZ unit quaternions."""
    quat_vector = quat[..., 1:]
    uv = torch.cross(quat_vector, vector, dim=-1)
    uuv = torch.cross(quat_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _axis_quat(angle: torch.Tensor, axis: int) -> torch.Tensor:
    half_angle = 0.5 * angle
    quat = torch.zeros((*angle.shape, 4), device=angle.device, dtype=angle.dtype)
    quat[..., 0] = torch.cos(half_angle)
    quat[..., axis + 1] = torch.sin(half_angle)
    return quat


def _expand_vector(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(0).expand(batch_size, -1)
    return value


def _quat_from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    quat = torch.stack(
        (
            torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)),
            torch.copysign(torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)), m21 - m12),
            torch.copysign(torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)), m02 - m20),
            torch.copysign(torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)), m10 - m01),
        ),
        dim=-1,
    )
    return torch.nn.functional.normalize(0.5 * quat, dim=-1)


def stick_figure_hand_center_pose(
    upper_pitch: torch.Tensor,
    upper_azimuth: torch.Tensor,
    upper_roll: torch.Tensor,
    elbow_flexion: torch.Tensor,
    wrist_angles: torch.Tensor,
    upper_arm_length: float,
    forearm_length: float,
    fixed_palm_quat: torch.Tensor,
    hand_center_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a coupled hand-center pose from a seven-DoF stick-figure arm."""
    batch_size = upper_pitch.shape[0]
    fixed_palm_quat = _expand_vector(fixed_palm_quat, batch_size)
    hand_center_offset = _expand_vector(hand_center_offset, batch_size)

    cos_pitch = torch.cos(upper_pitch)
    upper_direction = torch.stack(
        (
            cos_pitch * torch.cos(upper_azimuth),
            cos_pitch * torch.sin(upper_azimuth),
            torch.sin(upper_pitch),
        ),
        dim=-1,
    )

    # Project a back-up reference into the plane normal to the upper arm. Negating
    # it gives a natural bend that points down for horizontal arms and forward for lowered arms.
    bend_reference = torch.tensor((-1.0, 0.0, 1.0), device=upper_pitch.device, dtype=upper_pitch.dtype)
    bend_reference = bend_reference.unsqueeze(0).expand(batch_size, -1)
    bend_direction = bend_reference - (
        torch.sum(bend_reference * upper_direction, dim=-1, keepdim=True) * upper_direction
    )
    bend_direction = -torch.nn.functional.normalize(bend_direction, dim=-1)
    bend_direction = (
        torch.cos(upper_roll).unsqueeze(-1) * bend_direction
        + torch.sin(upper_roll).unsqueeze(-1) * torch.cross(upper_direction, bend_direction, dim=-1)
    )
    hinge_axis = torch.nn.functional.normalize(torch.cross(upper_direction, bend_direction, dim=-1), dim=-1)

    elbow_pos = upper_arm_length * upper_direction
    forearm_direction = (
        torch.cos(elbow_flexion).unsqueeze(-1) * upper_direction
        + torch.sin(elbow_flexion).unsqueeze(-1) * bend_direction
    )
    hand_base_pos = elbow_pos + forearm_length * forearm_direction

    forearm_z_axis = torch.nn.functional.normalize(torch.cross(forearm_direction, hinge_axis, dim=-1), dim=-1)
    forearm_matrix = torch.stack((forearm_direction, hinge_axis, forearm_z_axis), dim=-1)
    forearm_quat = _quat_from_rotation_matrix(forearm_matrix)

    wrist_quat = _quat_mul(
        _quat_mul(_axis_quat(wrist_angles[:, 0], axis=0), _axis_quat(wrist_angles[:, 1], axis=1)),
        _axis_quat(wrist_angles[:, 2], axis=2),
    )
    hand_base_quat = _quat_mul(forearm_quat, _quat_mul(wrist_quat, fixed_palm_quat))
    hand_base_quat = torch.nn.functional.normalize(hand_base_quat, dim=-1)
    hand_center_pos = hand_base_pos + _quat_apply(hand_base_quat, hand_center_offset)
    return hand_center_pos, hand_base_quat
