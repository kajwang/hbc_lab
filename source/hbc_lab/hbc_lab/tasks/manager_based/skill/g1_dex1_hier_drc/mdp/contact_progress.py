from __future__ import annotations

from dataclasses import dataclass

import torch

from hbc_lab.tasks.manager_based.skill.contact_labels import LEFT_HAND, RIGHT_HAND


@dataclass
class GripperContactComponents:
    contact: torch.Tensor
    grasp: torch.Tensor
    left_contact: torch.Tensor
    right_contact: torch.Tensor
    left_force: torch.Tensor
    right_force: torch.Tensor
    pinch: torch.Tensor
    pinch_score: torch.Tensor
    cos_sim: torch.Tensor
    close_position_gate: torch.Tensor
    orientation_error: torch.Tensor
    close_orientation_gate: torch.Tensor


@dataclass
class ActiveHandGraspProgress:
    active_hand: torch.Tensor
    contact: torch.Tensor
    grasp: torch.Tensor
    distance: torch.Tensor
    grip: torch.Tensor
    left_contact: torch.Tensor
    right_contact: torch.Tensor
    left_force: torch.Tensor
    right_force: torch.Tensor
    pinch: torch.Tensor
    pinch_score: torch.Tensor
    cos_sim: torch.Tensor
    close_position_gate: torch.Tensor
    orientation_error: torch.Tensor
    close_orientation_gate: torch.Tensor


def sample_active_hands(
    num_envs: int,
    device: torch.device,
    left_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample which hand should interact with the object in each environment."""
    if not 0.0 <= left_probability <= 1.0:
        raise ValueError(f"left_probability must be in [0, 1], got {left_probability}")
    if left_probability == 1.0:
        return torch.full((num_envs,), LEFT_HAND, dtype=torch.long, device=device)
    if left_probability == 0.0:
        return torch.full((num_envs,), RIGHT_HAND, dtype=torch.long, device=device)
    left_mask = torch.rand(num_envs, device=device, generator=generator) < left_probability
    return torch.where(
        left_mask,
        torch.full((num_envs,), LEFT_HAND, dtype=torch.long, device=device),
        torch.full((num_envs,), RIGHT_HAND, dtype=torch.long, device=device),
    )


def select_active_hand_value(left_value: torch.Tensor, right_value: torch.Tensor, active_hand: torch.Tensor) -> torch.Tensor:
    """Select a left/right tensor value by the sampled active hand."""
    if left_value.shape != right_value.shape:
        raise ValueError(f"Left and right values must have identical shapes, got {left_value.shape} and {right_value.shape}")
    active_hand = active_hand.to(device=left_value.device).reshape(-1)
    if active_hand.shape[0] != left_value.shape[0]:
        raise ValueError(f"active_hand batch size {active_hand.shape[0]} does not match value batch {left_value.shape[0]}")
    mask_shape = (active_hand.shape[0],) + (1,) * (left_value.ndim - 1)
    return torch.where((active_hand == LEFT_HAND).reshape(mask_shape), left_value, right_value)


def _grip_scalar(grip: torch.Tensor) -> torch.Tensor:
    if grip.ndim == 2 and grip.shape[-1] == 1:
        grip = grip.squeeze(-1)
    if grip.ndim != 1:
        raise ValueError(f"Expected grip shape (N,) or (N, 1), got {tuple(grip.shape)}")
    return torch.clamp(grip, 0.0, 1.0)


def _contact_confidence(force_w: torch.Tensor, force_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    force = torch.norm(force_w, dim=-1)
    contact = 1.0 - torch.exp(-force / force_threshold)
    return contact.clamp(0.0, 1.0), force


def _quaternion_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[:-1] != vector.shape[:-1] or quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector batch shapes must match")
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def symmetric_parallel_gripper_orientation_error(
    current_quat_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Measure Dex1 orientation error while treating a 180-degree jaw swap as equivalent."""
    if current_quat_w.shape != target_quat_w.shape or current_quat_w.shape[-1] != 4:
        raise ValueError("current and target quaternions must have equal shape ending in 4")
    current_quat_w = torch.nn.functional.normalize(current_quat_w, dim=-1)
    target_quat_w = torch.nn.functional.normalize(target_quat_w, dim=-1)
    local_closing = torch.zeros(*current_quat_w.shape[:-1], 3, device=current_quat_w.device, dtype=current_quat_w.dtype)
    local_approach = torch.zeros_like(local_closing)
    local_closing[..., 0] = 1.0
    local_approach[..., 1] = 1.0
    current_closing = _quaternion_apply_wxyz(current_quat_w, local_closing)
    target_closing = _quaternion_apply_wxyz(target_quat_w, local_closing)
    current_approach = _quaternion_apply_wxyz(current_quat_w, local_approach)
    target_approach = _quaternion_apply_wxyz(target_quat_w, local_approach)
    closing_cosine = torch.abs(torch.sum(current_closing * target_closing, dim=-1)).clamp(-1.0, 1.0)
    approach_cosine = torch.sum(current_approach * target_approach, dim=-1).clamp(-1.0, 1.0)
    return torch.maximum(torch.acos(closing_cosine), torch.acos(approach_cosine))


def orientation_close_gate(
    orientation_error: torch.Tensor,
    *,
    zero_error: float,
    transition_width: float,
) -> torch.Tensor:
    """Return one below the full-score error and zero at or beyond ``zero_error``."""
    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive")
    return torch.clamp((zero_error - orientation_error) / transition_width, min=0.0, max=1.0)


def compute_gripper_contact_components(
    left_finger_force_w: torch.Tensor,
    right_finger_force_w: torch.Tensor,
    grip: torch.Tensor,
    distance: torch.Tensor,
    force_threshold: float,
    close_distance: float = 0.20,
    close_gate_width: float = 0.10,
    soft_pinch_start: float = 0.2,
    orientation_error: torch.Tensor | None = None,
    close_orientation_zero_error: float = 1.0471975512,
    close_orientation_gate_width: float = 0.6981317008,
) -> GripperContactComponents:
    """Compute go2-style two-finger contact and grasp progress for one active Dex1 gripper."""
    grip = _grip_scalar(grip)
    left_contact, left_force = _contact_confidence(left_finger_force_w, force_threshold)
    right_contact, right_force = _contact_confidence(right_finger_force_w, force_threshold)
    contact = torch.minimum(left_contact, right_contact)

    eps = 1.0e-6
    left_dir = left_finger_force_w / (left_force.unsqueeze(-1) + eps)
    right_dir = right_finger_force_w / (right_force.unsqueeze(-1) + eps)
    cos_sim = torch.sum(left_dir * right_dir, dim=-1)
    pinch_score = torch.clamp((-cos_sim - soft_pinch_start) / (1.0 - soft_pinch_start), min=0.0, max=1.0)
    pinch = contact * pinch_score

    close_position_gate = torch.clamp((close_distance - distance) / close_gate_width, min=0.0, max=1.0)
    if orientation_error is None:
        orientation_error = torch.zeros_like(distance)
    close_orientation_gate = orientation_close_gate(
        orientation_error,
        zero_error=close_orientation_zero_error,
        transition_width=close_orientation_gate_width,
    )
    grasp = contact * grip * close_position_gate * close_orientation_gate
    return GripperContactComponents(
        contact=contact,
        grasp=grasp,
        left_contact=left_contact,
        right_contact=right_contact,
        left_force=left_force,
        right_force=right_force,
        pinch=pinch,
        pinch_score=pinch_score,
        cos_sim=cos_sim,
        close_position_gate=close_position_gate,
        orientation_error=orientation_error,
        close_orientation_gate=close_orientation_gate,
    )


def compute_active_hand_grasp_progress(
    left_gripper_left_force_w: torch.Tensor,
    left_gripper_right_force_w: torch.Tensor,
    right_gripper_left_force_w: torch.Tensor,
    right_gripper_right_force_w: torch.Tensor,
    active_hand: torch.Tensor,
    left_grip: torch.Tensor,
    right_grip: torch.Tensor,
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
    force_threshold: float,
    close_distance: float = 0.20,
    close_gate_width: float = 0.10,
    left_orientation_error: torch.Tensor | None = None,
    right_orientation_error: torch.Tensor | None = None,
    close_orientation_zero_error: float = 1.0471975512,
    close_orientation_gate_width: float = 0.6981317008,
) -> ActiveHandGraspProgress:
    """Resolve grasp progress using only the sampled active parallel gripper."""
    distance = select_active_hand_value(left_distance, right_distance, active_hand)
    grip = select_active_hand_value(_grip_scalar(left_grip), _grip_scalar(right_grip), active_hand)
    left_force_w = select_active_hand_value(left_gripper_left_force_w, right_gripper_left_force_w, active_hand)
    right_force_w = select_active_hand_value(left_gripper_right_force_w, right_gripper_right_force_w, active_hand)
    if (left_orientation_error is None) != (right_orientation_error is None):
        raise ValueError("left and right orientation errors must be provided together")
    orientation_error = (
        torch.zeros_like(distance)
        if left_orientation_error is None
        else select_active_hand_value(left_orientation_error, right_orientation_error, active_hand)
    )
    components = compute_gripper_contact_components(
        left_finger_force_w=left_force_w,
        right_finger_force_w=right_force_w,
        grip=grip,
        distance=distance,
        force_threshold=force_threshold,
        close_distance=close_distance,
        close_gate_width=close_gate_width,
        orientation_error=orientation_error,
        close_orientation_zero_error=close_orientation_zero_error,
        close_orientation_gate_width=close_orientation_gate_width,
    )
    return ActiveHandGraspProgress(
        active_hand=active_hand,
        contact=components.contact,
        grasp=components.grasp,
        distance=distance,
        grip=grip,
        left_contact=components.left_contact,
        right_contact=components.right_contact,
        left_force=components.left_force,
        right_force=components.right_force,
        pinch=components.pinch,
        pinch_score=components.pinch_score,
        cos_sim=components.cos_sim,
        close_position_gate=components.close_position_gate,
        orientation_error=components.orientation_error,
        close_orientation_gate=components.close_orientation_gate,
    )
