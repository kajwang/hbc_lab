from __future__ import annotations

import torch


def compute_spatial_progress(
    initial_distance: torch.Tensor,
    current_distance: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    progress = torch.clamp(initial_distance - current_distance, min=0.0)
    valid_distance = initial_distance > eps
    return torch.where(
        valid_distance,
        progress / (initial_distance + eps),
        torch.zeros_like(progress),
    )
