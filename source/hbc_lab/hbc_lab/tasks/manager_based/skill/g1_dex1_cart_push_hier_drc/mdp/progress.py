from __future__ import annotations

import torch


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[..., 1:]
    uv = torch.linalg.cross(quat_vector, vector, dim=-1)
    uuv = torch.linalg.cross(quat_vector, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def bimanual_grasp_confidence(left_grasp: torch.Tensor, right_grasp: torch.Tensor) -> torch.Tensor:
    product = torch.clamp(left_grasp * right_grasp, min=0.0, max=1.0)
    return torch.sqrt(product)


def compute_handle_targets(
    handle_pos_w: torch.Tensor,
    handle_quat_w: torch.Tensor,
    half_width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_offset = torch.zeros_like(handle_pos_w)
    local_offset[:, 1] = half_width
    offset_w = _quat_apply(handle_quat_w, local_offset)
    return handle_pos_w + offset_w, handle_pos_w - offset_w


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
