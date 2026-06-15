"""Small reference-motion helpers for the first tracking-policy prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable


@dataclass(frozen=True)
class ReferenceMotionCfg:
    """Configuration for a procedural tracking reference."""

    kind: str = "stand"
    joint_names: tuple[str, ...] = ()
    amplitude: float = 0.25
    frequency_hz: float = 1.0


def build_reference_joint_offsets(cfg: ReferenceMotionCfg, phase):
    """Build joint offsets for a normalized phase in [0, 1).

    The function intentionally works without torch for unit tests. When a torch tensor is
    passed, it returns a tensor on the same device and dtype.
    """

    torch_result = _build_torch_offsets(cfg, phase)
    if torch_result is not None:
        return torch_result

    phases = _as_float_list(phase)
    offsets = [[0.0 for _ in cfg.joint_names] for _ in phases]

    if cfg.kind == "stand":
        return offsets
    if cfg.kind != "arm_swing":
        raise ValueError(f"Unsupported reference kind: {cfg.kind}")

    for row_id, phase_value in enumerate(phases):
        wave = cfg.amplitude * math.sin(2.0 * math.pi * cfg.frequency_hz * phase_value)
        for joint_id, joint_name in enumerate(cfg.joint_names):
            if "left_shoulder_pitch" in joint_name:
                offsets[row_id][joint_id] = wave
            elif "right_shoulder_pitch" in joint_name:
                offsets[row_id][joint_id] = -wave

    return offsets


def _build_torch_offsets(cfg: ReferenceMotionCfg, phase):
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if not isinstance(phase, torch.Tensor):
        return None

    phase_tensor = phase.reshape(-1)
    offsets = torch.zeros(
        (phase_tensor.shape[0], len(cfg.joint_names)),
        device=phase_tensor.device,
        dtype=phase_tensor.dtype,
    )
    if cfg.kind == "stand":
        return offsets
    if cfg.kind != "arm_swing":
        raise ValueError(f"Unsupported reference kind: {cfg.kind}")

    wave = cfg.amplitude * torch.sin(2.0 * math.pi * cfg.frequency_hz * phase_tensor)
    for joint_id, joint_name in enumerate(cfg.joint_names):
        if "left_shoulder_pitch" in joint_name:
            offsets[:, joint_id] = wave
        elif "right_shoulder_pitch" in joint_name:
            offsets[:, joint_id] = -wave
    return offsets


def _as_float_list(phase) -> list[float]:
    if isinstance(phase, (float, int)):
        return [float(phase)]
    if isinstance(phase, Iterable):
        return [float(item) for item in phase]
    raise TypeError(f"Unsupported phase type: {type(phase)!r}")

