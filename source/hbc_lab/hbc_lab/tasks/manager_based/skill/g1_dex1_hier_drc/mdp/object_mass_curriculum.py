from __future__ import annotations

import torch


def object_mass_curriculum_parameters(
    w_manip: torch.Tensor,
    *,
    start_mass: float = 20.0,
    ref_w: float = 0.10,
    ref_mass: float = 5.0,
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
    log_final = torch.log(torch.as_tensor(final_mass, device=device, dtype=dtype))
    ref_w_t = torch.as_tensor(ref_w, device=device, dtype=dtype)
    ref_log_std_t = torch.as_tensor(ref_log_std, device=device, dtype=dtype)
    final_log_std_t = torch.as_tensor(final_log_std, device=device, dtype=dtype)

    mass_ratio = torch.clamp((log_ref - log_final) / (log_start - log_final), min=1.0e-6, max=1.0)
    mass_decay = -torch.log(mass_ratio) / ref_w_t
    log_center = log_final + (log_start - log_final) * torch.exp(-mass_decay * w_manip)

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
    max_mass: float = 6.0,
    **kwargs,
) -> torch.Tensor:
    w_manip = torch.as_tensor(w_manip, device=device, dtype=dtype)
    center, log_std = object_mass_curriculum_parameters(w_manip, **kwargs)
    log_center = torch.log(center)
    noise = torch.randn(num_envs, device=device, dtype=dtype, generator=generator)
    mass = torch.exp(log_center + noise * log_std)
    return torch.clamp(mass, min_mass, max_mass)
