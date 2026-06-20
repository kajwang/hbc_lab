from __future__ import annotations

from dataclasses import dataclass

import torch


LEFT_HAND = 0
RIGHT_HAND = 1


@dataclass
class HandContactComponents:
    contact: torch.Tensor
    force: torch.Tensor


@dataclass
class ActiveHandGraspProgress:
    active_hand: torch.Tensor
    contact: torch.Tensor
    grasp: torch.Tensor
    distance: torch.Tensor
    grip: torch.Tensor


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


def compute_hand_contact_confidence(hand_force_w: torch.Tensor, force_threshold: float) -> HandContactComponents:
    """Convert a whole-hand object contact force into a smooth contact confidence."""
    force = torch.norm(hand_force_w, dim=-1)
    contact = 1.0 - torch.exp(-force / force_threshold)
    return HandContactComponents(contact=contact.clamp(0.0, 1.0), force=force)


def _grip_scalar(grip: torch.Tensor) -> torch.Tensor:
    if grip.ndim == 2 and grip.shape[-1] == 1:
        grip = grip.squeeze(-1)
    if grip.ndim != 1:
        raise ValueError(f"Expected grip shape (N,) or (N, 1), got {tuple(grip.shape)}")
    return torch.clamp(grip, 0.0, 1.0)


def compute_active_hand_grasp_progress(
    left_components: HandContactComponents,
    right_components: HandContactComponents,
    active_hand: torch.Tensor,
    left_grip: torch.Tensor,
    right_grip: torch.Tensor,
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
    close_distance: float = 0.20,
    close_gate_width: float = 0.10,
) -> ActiveHandGraspProgress:
    """Resolve progress using only the sampled active hand's object contact."""
    left_grip = _grip_scalar(left_grip)
    right_grip = _grip_scalar(right_grip)
    contact = select_active_hand_value(left_components.contact, right_components.contact, active_hand)
    distance = select_active_hand_value(left_distance, right_distance, active_hand)
    grip = select_active_hand_value(left_grip, right_grip, active_hand)
    close_allowed_gate = torch.clamp((close_distance - distance) / close_gate_width, min=0.0, max=1.0)
    grasp = contact * grip * close_allowed_gate
    return ActiveHandGraspProgress(
        active_hand=active_hand,
        contact=contact,
        grasp=grasp,
        distance=distance,
        grip=grip,
    )
