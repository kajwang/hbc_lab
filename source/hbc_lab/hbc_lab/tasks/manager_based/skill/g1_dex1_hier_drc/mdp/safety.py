from __future__ import annotations

from typing import Any

import torch


def nonfinite_ratio(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return torch.zeros((), device=tensor.device)
    return (~torch.isfinite(tensor)).float().mean()


def nested_nonfinite_ratio(value: Any) -> torch.Tensor:
    tensors: list[torch.Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if torch.is_floating_point(item):
                tensors.append(item)
        elif isinstance(item, dict):
            for nested_item in item.values():
                collect(nested_item)
        elif isinstance(item, (list, tuple)):
            for nested_item in item:
                collect(nested_item)

    collect(value)
    if not tensors:
        return torch.zeros(())

    device = tensors[0].device
    nonfinite_count = torch.zeros((), device=device)
    total_count = 0
    for tensor in tensors:
        nonfinite_count += (~torch.isfinite(tensor)).sum().to(device=device, dtype=torch.float32)
        total_count += tensor.numel()
    return nonfinite_count / max(total_count, 1)


def sanitize_tensor(
    tensor: torch.Tensor,
    finite_clip: float | None = None,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor
    sanitized = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
    if finite_clip is not None and finite_clip > 0.0:
        sanitized = torch.clamp(sanitized, min=-finite_clip, max=finite_clip)
    return sanitized


def sanitize_nested_tensors(
    value: Any,
    finite_clip: float | None = None,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> Any:
    if isinstance(value, torch.Tensor):
        value.copy_(
            sanitize_tensor(
                value,
                finite_clip=finite_clip,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            )
        )
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = sanitize_nested_tensors(
                item,
                finite_clip=finite_clip,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            )
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = sanitize_nested_tensors(
                item,
                finite_clip=finite_clip,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            )
        return value
    if isinstance(value, tuple):
        return tuple(
            sanitize_nested_tensors(
                item,
                finite_clip=finite_clip,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            )
            for item in value
        )
    return value
