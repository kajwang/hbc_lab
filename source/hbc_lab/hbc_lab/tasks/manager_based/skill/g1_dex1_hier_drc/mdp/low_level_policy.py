from __future__ import annotations

import torch


class LowLevelPolicyWrapper:
    """TorchScript wrapper for an exported low-level tracking actor."""

    def __init__(self, policy_path: str, device: str | torch.device):
        self.device = torch.device(device)
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs.to(self.device)).detach()
