from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DoorManipulationProgress:
    handle_progress: torch.Tensor
    handle_gate: torch.Tensor
    hinge_progress: torch.Tensor
    reward: torch.Tensor


def compute_door_manipulation_progress(
    handle_angle: torch.Tensor,
    hinge_angle: torch.Tensor,
    latch_threshold: float,
    hinge_target: float,
) -> DoorManipulationProgress:
    handle_progress = torch.clamp(torch.abs(handle_angle) / latch_threshold, min=0.0, max=1.0)
    hinge_progress = torch.clamp(hinge_angle / hinge_target, min=0.0, max=1.0)

    gate_phase = torch.clamp((handle_progress - 0.8) / 0.2, min=0.0, max=1.0)
    handle_gate = gate_phase * gate_phase * (3.0 - 2.0 * gate_phase)
    reward = 0.3 + 0.25 * handle_progress + 0.75 * handle_gate * hinge_progress
    return DoorManipulationProgress(
        handle_progress=handle_progress,
        handle_gate=handle_gate,
        hinge_progress=hinge_progress,
        reward=reward,
    )
