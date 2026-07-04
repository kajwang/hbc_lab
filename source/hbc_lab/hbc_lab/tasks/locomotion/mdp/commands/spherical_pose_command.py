from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SphericalPoseCommand(CommandTerm):
    """Shoulder-anchored spherical pose command with optional fixed world-height anchor."""

    cfg: SphericalLevelPoseCommandCfg

    def __init__(self, cfg: SphericalLevelPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.anchor_body_idx = self.robot.find_bodies(cfg.anchor_body_name)[0][0]
        self.tracked_frame = env.scene[cfg.tracked_frame_sensor_name] if cfg.tracked_frame_sensor_name else None

        self.spherical_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "SphericalPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired pose in the shoulder-yaw frame: [x, y, z, qw, qx, qy, qz]."""
        return self.pose_command_b

    def _spherical_to_anchor_xyz(self, env_ids: Sequence[int]):
        radius = self.spherical_command[env_ids, 0]
        pitch = self.spherical_command[env_ids, 1]
        azimuth = self.spherical_command[env_ids, 2]

        self.pose_command_b[env_ids, 0] = radius * torch.cos(pitch) * torch.cos(azimuth)
        self.pose_command_b[env_ids, 1] = radius * torch.cos(pitch) * torch.sin(azimuth)
        self.pose_command_b[env_ids, 2] = radius * torch.sin(pitch)

    def _anchor_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pos_w = self.robot.data.body_pos_w[:, self.anchor_body_idx].clone()
        root_yaw_quat = yaw_quat(self.robot.data.root_quat_w)
        anchor_quat_w = root_yaw_quat
        if self.cfg.anchor_pitch_command_name is not None:
            anchor_pitch_command = self.env.command_manager.get_command(self.cfg.anchor_pitch_command_name)
            anchor_pitch = (
                anchor_pitch_command[:, self.cfg.anchor_pitch_command_index] * self.cfg.anchor_pitch_scale
                + self.cfg.anchor_pitch_offset
            )
            zeros = torch.zeros(self.num_envs, device=self.device)
            pitch_quat = quat_from_euler_xyz(zeros, anchor_pitch, zeros)
            anchor_quat_w = quat_mul(anchor_quat_w, pitch_quat)
        if self.cfg.anchor_height_command_name is not None:
            anchor_height_command = self.env.command_manager.get_command(self.cfg.anchor_height_command_name)
            root_cmd_pos_w = self.robot.data.root_pos_w.clone()
            root_cmd_pos_w[:, 2] = (
                self.env.scene.env_origins[:, 2]
                + anchor_height_command[:, self.cfg.anchor_height_command_index]
            )
            root_to_anchor_w = anchor_pos_w - self.robot.data.root_pos_w
            root_to_anchor_yaw = quat_apply_inverse(root_yaw_quat, root_to_anchor_w)
            anchor_offset_b = torch.zeros_like(root_to_anchor_yaw)
            anchor_offset_b[:, 1] = root_to_anchor_yaw[:, 1]
            anchor_offset_b[:, 2] = self.cfg.anchor_height_offset
            anchor_pos_w = root_cmd_pos_w + quat_apply(anchor_quat_w, anchor_offset_b)
        elif self.cfg.fixed_anchor_height is not None:
            anchor_pos_w[:, 2] = self.cfg.fixed_anchor_height
        return anchor_pos_w, anchor_quat_w

    def _update_pose_command_w(self):
        anchor_pos_w, anchor_quat_w = self._anchor_pose_w()
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            anchor_pos_w,
            anchor_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )

    def _update_metrics(self):
        self._update_pose_command_w()
        current_pos_w, current_quat_w = self._current_pose_w()

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            current_pos_w,
            current_quat_w,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _current_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tracked_frame is not None:
            return (
                self.tracked_frame.data.target_pos_w[:, self.cfg.tracked_frame_index],
                self.tracked_frame.data.target_quat_w[:, self.cfg.tracked_frame_index],
            )
        return self.robot.data.body_pos_w[:, self.body_idx], self.robot.data.body_quat_w[:, self.body_idx]

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.spherical_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.l)
        self.spherical_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch)
        self.spherical_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.azimuth)

        self._spherical_to_anchor_xyz(env_ids)

        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.ee_pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(
            euler_angles[:, 0],
            euler_angles[:, 1],
            euler_angles[:, 2] + self.spherical_command[env_ids, 2] + self.cfg.orientation_yaw_offset,
        )
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self._update_pose_command_w()

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        current_pos_w, current_quat_w = self._current_pose_w()
        self.current_pose_visualizer.visualize(current_pos_w, current_quat_w)


@configclass
class SphericalLevelPoseCommandCfg(CommandTermCfg):
    """Configuration for a shoulder-anchored spherical pose command."""

    class_type: type = SphericalPoseCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    anchor_body_name: str = MISSING
    tracked_frame_sensor_name: str | None = None
    tracked_frame_index: int = 0
    anchor_height_command_name: str | None = None
    anchor_height_command_index: int = 0
    anchor_height_offset: float = 0.0
    anchor_pitch_command_name: str | None = None
    anchor_pitch_command_index: int = 1
    anchor_pitch_scale: float = 1.0
    anchor_pitch_offset: float = 0.0
    orientation_yaw_offset: float = 0.0
    fixed_anchor_height: float | None = None
    make_quat_unique: bool = False

    @configclass
    class Ranges:
        l: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        azimuth: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        ee_pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    limit_ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/spherical_goal_pose"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/spherical_body_pose"
    )

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
