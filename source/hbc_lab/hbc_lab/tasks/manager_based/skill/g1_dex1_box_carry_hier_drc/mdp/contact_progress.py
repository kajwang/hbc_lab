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
    bimanual_support: torch.Tensor


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
    left_support = torch.amax(left_region_contacts, dim=-1)
    right_support = torch.amax(right_region_contacts, dim=-1)
    return BimanualSupportProgress(
        distance=torch.maximum(left_distance, right_distance),
        left_distance=left_distance,
        right_distance=right_distance,
        left_support=left_support,
        right_support=right_support,
        bimanual_support=torch.minimum(left_support, right_support),
    )
