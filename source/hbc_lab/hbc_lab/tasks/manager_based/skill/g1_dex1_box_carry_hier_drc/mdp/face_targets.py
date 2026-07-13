from __future__ import annotations

import torch


def _quat_rotate(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vec = quat_wxyz[..., 1:]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


def compute_long_axis_face_centers(
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    half_extent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the centers of the box's local +X and -X faces in world coordinates."""
    local_offset = torch.zeros_like(object_pos_w)
    local_offset[..., 0] = half_extent
    world_offset = _quat_rotate(object_quat_w, local_offset)
    return object_pos_w + world_offset, object_pos_w - world_offset


def choose_left_positive_assignment(
    positive_w: torch.Tensor,
    negative_w: torch.Tensor,
    left_hand_w: torch.Tensor,
    right_hand_w: torch.Tensor,
) -> torch.Tensor:
    """Choose the lower-cost one-to-one pairing between hands and opposing faces."""
    positive_left_cost = torch.norm(left_hand_w - positive_w, dim=-1) + torch.norm(
        right_hand_w - negative_w,
        dim=-1,
    )
    negative_left_cost = torch.norm(left_hand_w - negative_w, dim=-1) + torch.norm(
        right_hand_w - positive_w,
        dim=-1,
    )
    return positive_left_cost <= negative_left_cost


def select_assigned_face_targets(
    positive_w: torch.Tensor,
    negative_w: torch.Tensor,
    left_positive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve live left/right targets from an episode-fixed face assignment."""
    mask = left_positive.unsqueeze(-1)
    left_target = torch.where(mask, positive_w, negative_w)
    right_target = torch.where(mask, negative_w, positive_w)
    return left_target, right_target
