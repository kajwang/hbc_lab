#!/usr/bin/env python3
"""Append zero-initialized scene-perception inputs to a PnP RSL-RL checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def append_zero_columns(weight: torch.Tensor, count: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("The number of appended columns must be non-negative")
    zeros = torch.zeros(weight.shape[0], count, dtype=weight.dtype, device=weight.device)
    return torch.cat((weight, zeros), dim=1)


def expand_scene_perception_checkpoint(
    checkpoint: dict,
    actor_extra_dim: int = 190,
    critic_extra_dim: int = 195,
) -> dict:
    state_dict = checkpoint["model_state_dict"]
    for key, count in (("actor.0.weight", actor_extra_dim), ("critic.0.weight", critic_extra_dim)):
        old_shape = tuple(state_dict[key].shape)
        state_dict[key] = append_zero_columns(state_dict[key], count)
        print(f"{key}: {old_shape} -> {tuple(state_dict[key].shape)}")

    checkpoint["infos"] = dict(checkpoint.get("infos") or {})
    checkpoint["infos"]["scene_perception_zero_initialized_actor_columns"] = actor_extra_dim
    checkpoint["infos"]["scene_perception_zero_initialized_critic_columns"] = critic_extra_dim
    return checkpoint


def replace_observation_block(
    weight: torch.Tensor,
    *,
    prefix_dim: int,
    old_block_dim: int,
    new_block_dim: int,
    suffix_dim: int = 0,
) -> torch.Tensor:
    """Replace one observation block while preserving surrounding learned columns."""
    expected = prefix_dim + old_block_dim + suffix_dim
    if weight.shape[1] != expected:
        raise ValueError(f"Expected input width {expected}, got {weight.shape[1]}")
    result = torch.zeros(
        weight.shape[0],
        prefix_dim + new_block_dim + suffix_dim,
        dtype=weight.dtype,
        device=weight.device,
    )
    result[:, :prefix_dim] = weight[:, :prefix_dim]
    if suffix_dim:
        result[:, -suffix_dim:] = weight[:, -suffix_dim:]
    return result


def replace_scan_with_rolling_voxel(
    checkpoint: dict,
    *,
    old_environment_dim: int = 325,
    new_environment_dim: int = 9600,
    actor_prefix_dim: int = 1360,
    critic_prefix_dim: int = 146,
    critic_suffix_dim: int = 5,
) -> dict:
    """Discard old ray weights and zero-initialize the rolling-voxel input block."""
    state_dict = checkpoint["model_state_dict"]
    actor_key = next(
        key
        for key in ("actor.latent_actor.0.weight", "actor.0.weight")
        if key in state_dict
    )
    for key, prefix_dim, suffix_dim in (
        (actor_key, actor_prefix_dim, 0),
        ("critic.0.weight", critic_prefix_dim, critic_suffix_dim),
    ):
        old_shape = tuple(state_dict[key].shape)
        state_dict[key] = replace_observation_block(
            state_dict[key],
            prefix_dim=prefix_dim,
            old_block_dim=old_environment_dim,
            new_block_dim=new_environment_dim,
            suffix_dim=suffix_dim,
        )
        print(f"{key}: {old_shape} -> {tuple(state_dict[key].shape)}")

    checkpoint["infos"] = dict(checkpoint.get("infos") or {})
    checkpoint["infos"]["scene_perception_replaced_with_rolling_voxel"] = {
        "old_environment_dim": old_environment_dim,
        "new_environment_dim": new_environment_dim,
    }
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actor-extra-dim", type=int, default=190)
    parser.add_argument("--critic-extra-dim", type=int, default=195)
    parser.add_argument(
        "--rolling-voxel",
        action="store_true",
        help="Replace the old 325-D instantaneous scan with the 9600-D rolling voxel block.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if args.rolling_voxel:
        checkpoint = replace_scan_with_rolling_voxel(checkpoint)
    else:
        checkpoint = expand_scene_perception_checkpoint(
            checkpoint,
            actor_extra_dim=args.actor_extra_dim,
            critic_extra_dim=args.critic_extra_dim,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved expanded checkpoint to {args.output}")


if __name__ == "__main__":
    main()
