from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformPostureCommand(CommandTerm):
    """Command generator for root height and torso pitch."""

    cfg: UniformLevelPostureCommandCfg

    def __init__(self, cfg: UniformLevelPostureCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.asset: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.asset.find_bodies(cfg.body_name)[0][0]

        self.posture_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["root_height_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["torso_pitch_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Posture command [root_height, torso_pitch]."""
        return self.posture_command

    def _current_posture(self) -> torch.Tensor:
        root_height = self.asset.data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        _, torso_pitch, _ = euler_xyz_from_quat(self.asset.data.body_quat_w[:, self.body_idx])
        return torch.stack((root_height, torso_pitch), dim=-1)

    def _update_metrics(self):
        error = self.posture_command - self._current_posture()
        self.metrics["root_height_error"] = torch.abs(error[:, 0])
        self.metrics["torso_pitch_error"] = torch.abs(error[:, 1])

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.posture_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.root_height)
        self.posture_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.torso_pitch)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "posture_command_visualizer"):
                self.posture_command_visualizer = VisualizationMarkers(self.cfg.posture_command_visualizer_cfg)
            self.posture_command_visualizer.set_visibility(True)
        else:
            if hasattr(self, "posture_command_visualizer"):
                self.posture_command_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.asset.is_initialized:
            return

        arrow_pos_w = self.asset.data.root_pos_w.clone()
        arrow_pos_w[:, 2] = self.env.scene.env_origins[:, 2] + self.posture_command[:, 0]

        zeros = torch.zeros(self.num_envs, device=self.device)
        pitch_quat = quat_from_euler_xyz(zeros, self.posture_command[:, 1], zeros)
        arrow_quat = quat_mul(yaw_quat(self.asset.data.root_quat_w), pitch_quat)
        self.posture_command_visualizer.visualize(arrow_pos_w, arrow_quat)


@configclass
class UniformLevelPostureCommandCfg(CommandTermCfg):
    """Configuration for root-height and torso-pitch posture commands."""

    class_type: type = UniformPostureCommand

    asset_name: str = MISSING
    body_name: str = MISSING

    @configclass
    class Ranges:
        root_height: tuple[float, float] = MISSING
        torso_pitch: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    limit_ranges: Ranges = MISSING

    posture_command_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/posture_command"
    )
    posture_command_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.4)
    posture_command_visualizer_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.05, 0.65),
        emissive_color=(0.25, 0.0, 0.12),
        roughness=0.35,
    )
