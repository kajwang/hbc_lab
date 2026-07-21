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


def compute_gripper_contact_components(
    left_finger_force_w: torch.Tensor,
    right_finger_force_w: torch.Tensor,
    grip: torch.Tensor,
    distance: torch.Tensor,
    force_threshold: float,
    close_distance: float = 0.20,
    close_gate_width: float = 0.10,
    soft_pinch_start: float = 0.2,
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

    close_allowed_gate = torch.clamp((close_distance - distance) / close_gate_width, min=0.0, max=1.0)
    grasp = contact * grip * close_allowed_gate
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
) -> ActiveHandGraspProgress:
    """Resolve grasp progress using only the sampled active parallel gripper."""
    distance = select_active_hand_value(left_distance, right_distance, active_hand)
    grip = select_active_hand_value(_grip_scalar(left_grip), _grip_scalar(right_grip), active_hand)
    left_force_w = select_active_hand_value(left_gripper_left_force_w, right_gripper_left_force_w, active_hand)
    right_force_w = select_active_hand_value(left_gripper_right_force_w, right_gripper_right_force_w, active_hand)
    components = compute_gripper_contact_components(
        left_finger_force_w=left_force_w,
        right_finger_force_w=right_force_w,
        grip=grip,
        distance=distance,
        force_threshold=force_threshold,
        close_distance=close_distance,
        close_gate_width=close_gate_width,
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
    )
