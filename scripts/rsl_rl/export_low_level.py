#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch


def export_actor(checkpoint_path: str, output_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor_critic = checkpoint["model_state_dict"]
    if "actor.0.weight" not in actor_critic:
        raise KeyError("Expected an RSL-RL actor MLP state dict with key 'actor.0.weight'")

    obs_dim = actor_critic["actor.0.weight"].shape[1]
    hidden0 = actor_critic["actor.0.weight"].shape[0]
    hidden1 = actor_critic["actor.2.weight"].shape[0]
    hidden2 = actor_critic["actor.4.weight"].shape[0]
    action_dim = actor_critic["actor.6.bias"].shape[0]

    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden0),
        torch.nn.ELU(),
        torch.nn.Linear(hidden0, hidden1),
        torch.nn.ELU(),
        torch.nn.Linear(hidden1, hidden2),
        torch.nn.ELU(),
        torch.nn.Linear(hidden2, action_dim),
    )
    actor_state = {key.replace("actor.", ""): value for key, value in actor_critic.items() if key.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval()

    scripted = torch.jit.script(actor)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scripted.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an HBC low-level RSL-RL actor to TorchScript.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output", required=True, help="Path to output TorchScript actor .pt")
    args = parser.parse_args()
    export_actor(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
