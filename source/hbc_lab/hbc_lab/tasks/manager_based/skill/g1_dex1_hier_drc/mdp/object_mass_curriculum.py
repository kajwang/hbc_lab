from __future__ import annotations

import torch


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def object_mass_curriculum_parameters(
    w_manip: torch.Tensor,
    *,
    start_w: float = 0.10,
    mid_w: float = 0.20,
    end_w: float = 0.65,
    start_mass: float = 5.0,
    mid_mass: float = 1.0,
    final_mass: float = 0.5,
    mid_log_std: float = 0.15,
    final_log_std: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-space mass center and noise scale for the global W_manip curriculum."""
    dtype = w_manip.dtype if w_manip.is_floating_point() else torch.float32
    device = w_manip.device
    w_manip = w_manip.to(dtype=dtype)

    p_fast = _smoothstep((w_manip - start_w) / (mid_w - start_w))
    p_slow = _smoothstep((w_manip - mid_w) / (end_w - mid_w))

    log_start = torch.log(torch.as_tensor(start_mass, device=device, dtype=dtype))
    log_mid = torch.log(torch.as_tensor(mid_mass, device=device, dtype=dtype))
    log_final = torch.log(torch.as_tensor(final_mass, device=device, dtype=dtype))

    fast_log_center = (1.0 - p_fast) * log_start + p_fast * log_mid
    slow_log_center = (1.0 - p_slow) * log_mid + p_slow * log_final
    log_center = torch.where(w_manip <= mid_w, fast_log_center, slow_log_center)

    fast_log_std = p_fast * mid_log_std
    slow_log_std = mid_log_std + p_slow * (final_log_std - mid_log_std)
    log_std = torch.where(w_manip <= mid_w, fast_log_std, slow_log_std)
    return torch.exp(log_center), log_std


def sample_object_masses(
    w_manip: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    min_mass: float = 0.15,
    max_mass: float = 6.0,
    **kwargs,
) -> torch.Tensor:
    w_manip = torch.as_tensor(w_manip, device=device, dtype=dtype)
    center, log_std = object_mass_curriculum_parameters(w_manip, **kwargs)
    log_center = torch.log(center)
    noise = torch.randn(num_envs, device=device, dtype=dtype, generator=generator)
    mass = torch.exp(log_center + noise * log_std)
    return torch.clamp(mass, min_mass, max_mass)
