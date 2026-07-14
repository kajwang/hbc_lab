from __future__ import annotations

import torch


def balanced_active_hand_progress(
    w_manip: torch.Tensor,
    active_hand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-hand progress and the weaker present hand for curriculum updates."""
    if w_manip.ndim != 1 or active_hand.shape != w_manip.shape:
        raise ValueError(
            f"Expected matching 1-D tensors, got w_manip={tuple(w_manip.shape)} "
            f"and active_hand={tuple(active_hand.shape)}"
        )
    if w_manip.numel() == 0:
        raise ValueError("Cannot compute curriculum progress from an empty batch")

    active_hand = active_hand.to(device=w_manip.device)
    left_mask = active_hand == 0
    right_mask = active_hand == 1
    left_count = left_mask.sum()
    right_count = right_mask.sum()
    total_mean = w_manip.mean()
    left_mean = torch.where(
        left_count > 0,
        torch.sum(w_manip * left_mask) / left_count.clamp_min(1),
        total_mean,
    )
    right_mean = torch.where(
        right_count > 0,
        torch.sum(w_manip * right_mask) / right_count.clamp_min(1),
        total_mean,
    )
    both_hands_present = (left_count > 0) & (right_count > 0)
    balanced = torch.where(both_hands_present, torch.minimum(left_mean, right_mean), total_mean)
    return left_mean, right_mean, balanced


def object_mass_curriculum_parameters(
    w_manip: torch.Tensor,
    *,
    start_mass: float = 10.0,
    ref_w: float = 0.10,
    ref_mass: float = 5.0,
    anchor_w: float = 0.20,
    anchor_mass: float = 1.0,
    final_mass: float = 0.5,
    ref_log_std: float = 0.15,
    final_log_std: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-space mass center and noise scale for the global W_manip curriculum."""
    dtype = w_manip.dtype if w_manip.is_floating_point() else torch.float32
    device = w_manip.device
    w_manip = torch.clamp(w_manip.to(dtype=dtype), min=0.0)

    log_start = torch.log(torch.as_tensor(start_mass, device=device, dtype=dtype))
    log_ref = torch.log(torch.as_tensor(ref_mass, device=device, dtype=dtype))
    log_anchor = torch.log(torch.as_tensor(anchor_mass, device=device, dtype=dtype))
    log_final = torch.log(torch.as_tensor(final_mass, device=device, dtype=dtype))
    ref_w_t = torch.as_tensor(ref_w, device=device, dtype=dtype)
    anchor_w_t = torch.as_tensor(anchor_w, device=device, dtype=dtype)
    ref_log_std_t = torch.as_tensor(ref_log_std, device=device, dtype=dtype)
    final_log_std_t = torch.as_tensor(final_log_std, device=device, dtype=dtype)

    ref_log_ratio = torch.clamp((log_ref - log_final) / (log_start - log_final), min=1.0e-6, max=1.0)
    anchor_log_ratio = torch.clamp((log_anchor - log_final) / (log_start - log_final), min=1.0e-6, max=1.0)
    ref_decay_target = -torch.log(ref_log_ratio)
    anchor_decay_target = -torch.log(anchor_log_ratio)
    mass_power = torch.log(anchor_decay_target / ref_decay_target) / torch.log(anchor_w_t / ref_w_t)
    mass_decay = ref_decay_target / torch.pow(ref_w_t, mass_power)
    log_center = log_final + (log_start - log_final) * torch.exp(-mass_decay * torch.pow(w_manip, mass_power))

    std_ratio = torch.clamp(1.0 - ref_log_std_t / final_log_std_t, min=1.0e-6, max=1.0)
    std_decay = -torch.log(std_ratio) / ref_w_t
    log_std = final_log_std_t * (1.0 - torch.exp(-std_decay * w_manip))
    return torch.exp(log_center), log_std


def sample_object_masses(
    w_manip: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    min_mass: float = 0.15,
    max_mass: float = 25.0,
    **kwargs,
) -> torch.Tensor:
    w_manip = torch.as_tensor(w_manip, device=device, dtype=dtype)
    center, log_std = object_mass_curriculum_parameters(w_manip, **kwargs)
    log_center = torch.log(center)
    noise = torch.randn(num_envs, device=device, dtype=dtype, generator=generator)
    mass = torch.exp(log_center + noise * log_std)
    return torch.clamp(mass, min_mass, max_mass)
