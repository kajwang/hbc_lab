from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BimanualSupportProgress:
    distance: torch.Tensor
    left_distance: torch.Tensor
    right_distance: torch.Tensor
    left_support: torch.Tensor
    right_support: torch.Tensor
    left_contact_gate: torch.Tensor
    right_contact_gate: torch.Tensor
    couple_gate: torch.Tensor
    support_density: torch.Tensor


def compute_independent_support_reward_terms(
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
    left_support: torch.Tensor,
    right_support: torch.Tensor,
    distance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate each hand's support using its own distance to the assigned box face."""
    if distance_scale <= 0.0:
        raise ValueError(f"distance_scale must be positive, got {distance_scale}")
    if not (left_distance.shape == right_distance.shape == left_support.shape == right_support.shape):
        raise ValueError("distance and support tensors must have identical shapes")

    left_gate = 1.0 - torch.tanh(left_distance / distance_scale)
    right_gate = 1.0 - torch.tanh(right_distance / distance_scale)
    gated_support = 0.5 * (left_gate * left_support + right_gate * right_support)
    early_contact = 0.5 * ((1.0 - left_gate) * left_support + (1.0 - right_gate) * right_support)
    return left_gate, right_gate, gated_support, early_contact


def compute_bimanual_support_progress(
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
    left_region_contacts: torch.Tensor,
    right_region_contacts: torch.Tensor,
) -> BimanualSupportProgress:
    """Compute center approach and two-hand support from semantic contact regions."""
    if left_distance.shape != right_distance.shape:
        raise ValueError("left_distance and right_distance must have identical shapes")
    if left_region_contacts.shape != right_region_contacts.shape:
        raise ValueError("left and right region contacts must have identical shapes")
    if left_region_contacts.ndim != 2 or left_region_contacts.shape[0] != left_distance.shape[0]:
        raise ValueError("region contacts must have shape (num_envs, num_regions)")
    left_support = torch.mean(left_region_contacts, dim=-1)
    right_support = torch.mean(right_region_contacts, dim=-1)
    left_contact_gate = torch.amax(left_region_contacts, dim=-1)
    right_contact_gate = torch.amax(right_region_contacts, dim=-1)
    return BimanualSupportProgress(
        distance=torch.maximum(left_distance, right_distance),
        left_distance=left_distance,
        right_distance=right_distance,
        left_support=left_support,
        right_support=right_support,
        left_contact_gate=left_contact_gate,
        right_contact_gate=right_contact_gate,
        couple_gate=torch.minimum(left_contact_gate, right_contact_gate),
        support_density=0.5 * (left_support + right_support),
    )
