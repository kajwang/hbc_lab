from __future__ import annotations

import torch


GEOMETRY_FAMILY_NAMES = (
    "table_top",
    "table_under",
    "cabinet_top",
    "cabinet_middle",
    "cabinet_bottom",
    "pillar",
    "gap",
    "reach_over",
)
TABLE_TOP_FAMILY = 0
TABLE_UNDER_FAMILY = 1
CABINET_TOP_FAMILY = 2
CABINET_MIDDLE_FAMILY = 3
CABINET_BOTTOM_FAMILY = 4
PILLAR_FAMILY = 5
GAP_FAMILY = 6
REACH_OVER_FAMILY = 7


def preview_family_and_level(
    env_ids: torch.Tensor,
    family_count: int = 4,
    total_envs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_ids = torch.div(env_ids, 2, rounding_mode="floor")
    family = torch.remainder(pair_ids, family_count)
    group = torch.div(pair_ids, family_count, rounding_mode="floor")
    difficulty_group = torch.div(group, 2, rounding_mode="floor")
    if total_envs is None:
        max_pair_id = pair_ids.max()
    else:
        max_pair_id = torch.as_tensor((total_envs - 1) // 2, device=env_ids.device)
    max_group = torch.div(max_pair_id, family_count, rounding_mode="floor")
    max_difficulty_group = torch.clamp(torch.div(max_group, 2, rounding_mode="floor"), min=1)
    level = difficulty_group.to(torch.float32) / max_difficulty_group.to(torch.float32)
    return family, level.clamp(0.0, 1.0)


def paired_family_and_constraint(
    env_ids: torch.Tensor,
    family_count: int = len(GEOMETRY_FAMILY_NAMES),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign adjacent environments to the same task with/without geometry."""
    pair_ids = torch.div(env_ids, 2, rounding_mode="floor")
    family = torch.remainder(pair_ids, family_count)
    constrained = torch.remainder(env_ids, 2) == 1
    return family, constrained


def paired_active_hand(
    env_ids: torch.Tensor,
    family_count: int = len(GEOMETRY_FAMILY_NAMES),
) -> torch.Tensor:
    """Assign the same active hand to each pair and balance hands per family."""
    pair_ids = torch.div(env_ids, 2, rounding_mode="floor")
    group = torch.div(pair_ids, family_count, rounding_mode="floor")
    return torch.remainder(group, 2).long()


def paired_uniform(
    env_ids: torch.Tensor,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    salt: float,
) -> torch.Tensor:
    """Deterministic uniform samples shared by each adjacent environment pair."""
    pair_ids = torch.div(env_ids, 2, rounding_mode="floor").to(torch.float64)
    unit = torch.remainder(
        torch.sin((pair_ids + 1.0) * 12.9898 + salt * 78.233) * 43758.5453,
        1.0,
    )
    unit = unit.to(torch.float32)
    low_tensor = torch.as_tensor(low, device=env_ids.device, dtype=unit.dtype)
    high_tensor = torch.as_tensor(high, device=env_ids.device, dtype=unit.dtype)
    return low_tensor + unit * (high_tensor - low_tensor)


def lerp_range(level: torch.Tensor, easy: float, hard: float) -> torch.Tensor:
    return easy + level.clamp(0.0, 1.0) * (hard - easy)
