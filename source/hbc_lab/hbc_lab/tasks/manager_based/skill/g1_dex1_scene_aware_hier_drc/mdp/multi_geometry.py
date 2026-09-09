from __future__ import annotations

import torch


GEOMETRY_FAMILY_NAMES = (
    "open_platform",
    "table_under",
    "cabinet_top",
    "cabinet_middle",
    "cabinet_bottom",
    "pillar",
    "gap",
    "reach_over",
)
OPEN_PLATFORM_FAMILY = 0
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
    family = torch.remainder(env_ids, family_count)
    difficulty_group = torch.div(env_ids, family_count, rounding_mode="floor")
    if total_envs is None:
        max_env_id = env_ids.max()
    else:
        max_env_id = torch.as_tensor(total_envs - 1, device=env_ids.device)
    max_difficulty_group = torch.clamp(
        torch.div(max_env_id, family_count, rounding_mode="floor"), min=1
    )
    level = difficulty_group.to(torch.float32) / max_difficulty_group.to(torch.float32)
    return family, level.clamp(0.0, 1.0)


def balanced_family(
    env_ids: torch.Tensor,
    family_count: int = len(GEOMETRY_FAMILY_NAMES),
) -> torch.Tensor:
    """Distribute independent environments evenly across geometry families."""
    return torch.remainder(env_ids, family_count)


def balanced_active_hand(
    env_ids: torch.Tensor,
    family_count: int = len(GEOMETRY_FAMILY_NAMES),
) -> torch.Tensor:
    """Balance left and right active hands independently within each family."""
    group = torch.div(env_ids, family_count, rounding_mode="floor")
    return torch.remainder(group, 2).long()


def uniform_sample(
    env_ids: torch.Tensor,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
) -> torch.Tensor:
    """Sample independent values for reset environments."""
    unit = torch.rand(env_ids.shape[0], device=env_ids.device)
    low_tensor = torch.as_tensor(low, device=env_ids.device, dtype=unit.dtype)
    high_tensor = torch.as_tensor(high, device=env_ids.device, dtype=unit.dtype)
    return low_tensor + unit * (high_tensor - low_tensor)


def lerp_range(level: torch.Tensor, easy: float, hard: float) -> torch.Tensor:
    return easy + level.clamp(0.0, 1.0) * (hard - easy)
