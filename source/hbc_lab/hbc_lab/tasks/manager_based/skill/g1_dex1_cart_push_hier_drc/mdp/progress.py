from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CartManipulationProgress:
    transport_progress: torch.Tensor


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[..., 1:]
    uv = torch.linalg.cross(quat_vector, vector, dim=-1)
    uuv = torch.linalg.cross(quat_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def bimanual_grasp_confidence(left_grasp: torch.Tensor, right_grasp: torch.Tensor) -> torch.Tensor:
    product = torch.clamp(left_grasp * right_grasp, min=0.0, max=1.0)
    return torch.sqrt(product)


def bimanual_approach_distance(
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
) -> torch.Tensor:
    if left_distance.shape != right_distance.shape:
        raise ValueError("left_distance and right_distance must have identical shapes")
    return torch.maximum(left_distance, right_distance)


def compute_handle_targets(
    handle_pos_w: torch.Tensor,
    handle_quat_w: torch.Tensor,
    half_width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The configured handle frame uses x as forward, y along the crossbar, and z as up.
    local_offset = torch.zeros_like(handle_pos_w)
    local_offset[:, 1] = half_width
    offset_w = _quat_apply(handle_quat_w, local_offset)
    return handle_pos_w + offset_w, handle_pos_w - offset_w


def compute_planar_heading_quat(frame_quat_w: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Extract a yaw-only frame from the handle frame's projected forward axis."""
    forward_local = torch.zeros(frame_quat_w.shape[0], 3, device=frame_quat_w.device, dtype=frame_quat_w.dtype)
    forward_local[:, 0] = 1.0
    forward_w = _quat_apply(frame_quat_w, forward_local)
    forward_xy = forward_w[:, :2]
    norm_xy = torch.linalg.norm(forward_xy, dim=-1)
    yaw = torch.where(
        norm_xy > eps,
        torch.atan2(forward_xy[:, 1], forward_xy[:, 0]),
        torch.zeros_like(norm_xy),
    )
    half_yaw = 0.5 * yaw
    heading_quat_w = torch.zeros_like(frame_quat_w)
    heading_quat_w[:, 0] = torch.cos(half_yaw)
    heading_quat_w[:, 3] = torch.sin(half_yaw)
    return heading_quat_w


def compute_cart_goal(
    initial_pos_w: torch.Tensor,
    initial_quat_w: torch.Tensor,
    forward: torch.Tensor,
    lateral: torch.Tensor,
) -> torch.Tensor:
    local_offset = torch.zeros_like(initial_pos_w)
    local_offset[:, 0] = forward
    local_offset[:, 1] = lateral
    return initial_pos_w + _quat_apply(initial_quat_w, local_offset)


def compute_transport_progress(
    initial_distance: torch.Tensor,
    current_distance: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    progress = torch.clamp(initial_distance - current_distance, min=0.0)
    valid_distance = initial_distance > eps
    return torch.where(valid_distance, progress / (initial_distance + eps), torch.zeros_like(progress))
