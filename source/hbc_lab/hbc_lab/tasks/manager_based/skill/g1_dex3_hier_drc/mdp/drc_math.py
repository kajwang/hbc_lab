from __future__ import annotations

import torch


def update_ema(previous: torch.Tensor, current: torch.Tensor, alpha: float) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"EMA alpha must be in [0, 1], got {alpha}")
    return previous * (1.0 - alpha) + current * alpha


def compute_drc_weights(
    d_ee: torch.Tensor,
    c_couple: torch.Tensor,
    alpha: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    c_couple = torch.clamp(c_couple, 0.0, 1.0)
    w_app = (1.0 - c_couple) * (1.0 - torch.exp(-alpha * d_ee))
    w_couple = (1.0 - c_couple) * torch.exp(-alpha * d_ee)
    w_manip = c_couple
    total = w_app + w_couple + w_manip + eps
    return torch.stack((w_app / total, w_couple / total, w_manip / total), dim=-1)
