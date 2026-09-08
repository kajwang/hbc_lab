from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PnpRandomizationRanges:
    init_x: tuple[float, float]
    init_y: tuple[float, float]
    init_z: tuple[float, float]
    goal_radius: tuple[float, float]


def _lerp_pair(
    start: tuple[float, float],
    end: tuple[float, float],
    level: float,
) -> tuple[float, float]:
    return (
        start[0] + level * (end[0] - start[0]),
        start[1] + level * (end[1] - start[1]),
    )


def randomization_ranges(
    level: float | torch.Tensor,
    *,
    start_init_x: tuple[float, float] = (1.5, 2.0),
    final_init_x: tuple[float, float] = (1.0, 2.7),
    start_init_y: tuple[float, float] = (-0.35, 0.35),
    final_init_y: tuple[float, float] = (-1.0, 1.0),
    start_init_z: tuple[float, float] = (0.5, 0.5),
    final_init_z: tuple[float, float] = (0.0, 0.7),
    start_goal_radius: tuple[float, float] = (1.5, 2.5),
    final_goal_radius: tuple[float, float] = (1.5, 4.0),
) -> PnpRandomizationRanges:
    """Interpolate the reset domain from the successful baseline to the full domain."""
    if isinstance(level, torch.Tensor):
        level = float(level.detach().clamp(0.0, 1.0).cpu())
    else:
        level = min(max(float(level), 0.0), 1.0)
    return PnpRandomizationRanges(
        init_x=_lerp_pair(start_init_x, final_init_x, level),
        init_y=_lerp_pair(start_init_y, final_init_y, level),
        init_z=_lerp_pair(start_init_z, final_init_z, level),
        goal_radius=_lerp_pair(start_goal_radius, final_goal_radius, level),
    )


def inactive_object_parking_offsets(
    num_objects: int,
    *,
    x: float = -4.0,
    y_spacing: float = 0.5,
) -> tuple[tuple[float, float], ...]:
    """Place inactive pool members behind the robot on the ground plane."""
    if num_objects < 1:
        raise ValueError("num_objects must be positive")
    center = 0.5 * (num_objects - 1)
    return tuple((x, (index - center) * y_spacing) for index in range(num_objects))


def allowed_scale_indices(
    level: float | torch.Tensor,
    scale_factors: tuple[float, ...],
) -> tuple[int, ...]:
    """Expand a symmetric discrete scale pool around the nominal 1.0 object."""
    if not scale_factors or 1.0 not in scale_factors:
        raise ValueError("scale_factors must contain the nominal 1.0 scale")
    if isinstance(level, torch.Tensor):
        level = float(level.detach().clamp(0.0, 1.0).cpu())
    else:
        level = min(max(float(level), 0.0), 1.0)

    nominal_index = scale_factors.index(1.0)
    max_radius = min(nominal_index, len(scale_factors) - 1 - nominal_index)
    active_radius = min(max_radius, int(level * max_radius + 1.0e-6))
    return tuple(range(nominal_index - active_radius, nominal_index + active_radius + 1))


def advance_curriculum_level(
    current_level: float,
    grasp_rate: float,
    *,
    threshold: float,
    step: float,
) -> float:
    """Advance a monotonic curriculum only after grasp performance has recovered."""
    current_level = min(max(float(current_level), 0.0), 1.0)
    if float(grasp_rate) < threshold:
        return current_level
    return min(current_level + step, 1.0)
