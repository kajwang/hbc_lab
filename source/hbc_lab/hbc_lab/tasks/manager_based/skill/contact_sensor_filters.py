from __future__ import annotations

import torch


def select_contact_filter_force(
    force_matrix_w: torch.Tensor,
    filter_indices: torch.Tensor,
) -> torch.Tensor:
    """Select one filtered contact-force vector per environment."""
    if force_matrix_w.ndim < 3 or force_matrix_w.shape[-1] != 3:
        raise ValueError("force_matrix_w must end in (num_filters, 3)")

    force_by_filter_w = force_matrix_w
    while force_by_filter_w.ndim > 3:
        force_by_filter_w = force_by_filter_w.sum(dim=1)

    filter_indices = filter_indices.to(device=force_by_filter_w.device, dtype=torch.long)
    if filter_indices.shape != (force_by_filter_w.shape[0],):
        raise ValueError("filter_indices must contain one index per environment")
    if torch.any(filter_indices < 0) or torch.any(filter_indices >= force_by_filter_w.shape[1]):
        raise ValueError("filter index is outside the contact sensor filter dimension")

    env_indices = torch.arange(force_by_filter_w.shape[0], device=force_by_filter_w.device)
    return force_by_filter_w[env_indices, filter_indices]
