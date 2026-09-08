#!/usr/bin/env python3
"""Append zero-initialized directional-BPS columns to a PnP RSL-RL checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _append_zero_columns(weight: torch.Tensor, count: int) -> torch.Tensor:
    zeros = torch.zeros(weight.shape[0], count, dtype=weight.dtype, device=weight.device)
    return torch.cat((weight, zeros), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bps-dim", type=int, default=192)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    for key in ("actor.0.weight", "critic.0.weight"):
        old_shape = tuple(state_dict[key].shape)
        state_dict[key] = _append_zero_columns(state_dict[key], args.bps_dim)
        print(f"{key}: {old_shape} -> {tuple(state_dict[key].shape)}")

    checkpoint["infos"] = dict(checkpoint.get("infos") or {})
    checkpoint["infos"]["bps_zero_initialized_columns"] = args.bps_dim
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved expanded checkpoint to {args.output}")


if __name__ == "__main__":
    main()
