"""Reference command generators for humanoid tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .reference_motion import ReferenceMotionCfg, build_reference_joint_offsets


class ReferenceMotionCommand(CommandTerm):
    """Low-frequency procedural reference command for whole-body tracking."""

    cfg: "ReferenceMotionCommandCfg"

    def __init__(self, cfg: "ReferenceMotionCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids, self.joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.reference_joint_offsets = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self._command_dt = 0.0
        self._reference_cfg = ReferenceMotionCfg(
            kind=cfg.reference_kind,
            joint_names=tuple(self.joint_names),
            amplitude=cfg.reference_amplitude,
            frequency_hz=cfg.reference_frequency_hz,
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.phase, self.reference_joint_offsets), dim=-1)

    def compute(self, dt: float):
        self._command_dt = dt
        super().compute(dt)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.randomize_phase:
            self.phase[env_ids, 0] = torch.rand(len(env_ids), device=self.device)
        else:
            self.phase[env_ids, 0] = 0.0
        self.reference_joint_offsets[env_ids] = build_reference_joint_offsets(
            self._reference_cfg,
            self.phase[env_ids, 0],
        )

    def _update_command(self):
        self.phase[:, 0] = torch.remainder(
            self.phase[:, 0] + self._command_dt * self.cfg.phase_rate_hz,
            1.0,
        )
        self.reference_joint_offsets[:] = build_reference_joint_offsets(
            self._reference_cfg,
            self.phase[:, 0],
        )

    def _update_metrics(self):
        self.metrics["tracking_phase"] = self.phase[:, 0]
        self.metrics["reference_joint_offset_norm"] = torch.linalg.norm(self.reference_joint_offsets, dim=-1)


@configclass
class ReferenceMotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`ReferenceMotionCommand`."""

    class_type: type = ReferenceMotionCommand
    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    reference_kind: str = "stand"
    reference_amplitude: float = 0.25
    reference_frequency_hz: float = 1.0
    phase_rate_hz: float = 0.5
    randomize_phase: bool = True


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Unitree-style velocity command with curriculum expansion limits."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
