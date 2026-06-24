from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import yaw_quat

from .contact_progress import sample_active_hands


class G1Dex3HierCommand(CommandTerm):
    cfg: "G1Dex3HierCommandCfg"

    def __init__(self, cfg: "G1Dex3HierCommandCfg", env):
        super().__init__(cfg, env)
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.posture_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.left_wrist_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.right_wrist_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.left_grip = torch.zeros(self.num_envs, 1, device=self.device)
        self.right_grip = torch.zeros(self.num_envs, 1, device=self.device)
        self.active_hand = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.left_anchor_body_id = self.robot.find_bodies("left_shoulder_pitch_link")[0][0]
        self.right_anchor_body_id = self.robot.find_bodies("right_shoulder_pitch_link")[0][0]
        self.anchor_height_offset = 0.43
        self._set_defaults(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (
                self.base_velocity,
                self.posture_command,
                self.left_wrist_pose_b,
                self.right_wrist_pose_b,
                self.left_grip,
                self.right_grip,
                self.active_hand.unsqueeze(-1).to(dtype=self.base_velocity.dtype),
            ),
            dim=-1,
        )

    def set_command(
        self,
        base_velocity: torch.Tensor,
        posture_command: torch.Tensor,
        left_wrist_pose_b: torch.Tensor,
        right_wrist_pose_b: torch.Tensor,
        left_grip: torch.Tensor,
        right_grip: torch.Tensor,
    ) -> None:
        self.base_velocity[:] = base_velocity
        self.posture_command[:] = posture_command
        self.left_wrist_pose_b[:] = left_wrist_pose_b
        self.right_wrist_pose_b[:] = right_wrist_pose_b
        self.left_grip[:] = left_grip
        self.right_grip[:] = right_grip

    def _set_defaults(self, env_ids: torch.Tensor):
        self.base_velocity[env_ids] = 0.0
        self.posture_command[env_ids, 0] = self.cfg.default_root_height
        self.posture_command[env_ids, 1] = self.cfg.default_torso_pitch
        self.left_wrist_pose_b[env_ids] = torch.tensor(self.cfg.default_left_wrist_pose_b, device=self.device)
        self.right_wrist_pose_b[env_ids] = torch.tensor(self.cfg.default_right_wrist_pose_b, device=self.device)
        self.left_grip[env_ids] = 0.0
        self.right_grip[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._set_defaults(env_ids_t)
        self.active_hand[env_ids_t] = sample_active_hands(
            len(env_ids_t),
            self.device,
            left_probability=self.cfg.left_hand_probability,
        )

    def _update_command(self):
        return

    def _update_metrics(self):
        self.metrics["base_velocity_norm"] = torch.norm(self.base_velocity[:, :2], dim=-1)
        self.metrics["left_grip"] = self.left_grip.squeeze(-1)
        self.metrics["right_grip"] = self.right_grip.squeeze(-1)
        self.metrics["active_left_ratio"] = (self.active_hand == 0).float()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_velocity_visualizer"):
                self.base_velocity_visualizer = VisualizationMarkers(self.cfg.base_velocity_visualizer_cfg)
                self.left_wrist_visualizer = VisualizationMarkers(self.cfg.left_wrist_visualizer_cfg)
                self.right_wrist_visualizer = VisualizationMarkers(self.cfg.right_wrist_visualizer_cfg)
                self.posture_command_visualizer = VisualizationMarkers(self.cfg.posture_command_visualizer_cfg)
                self.grip_visualizer = VisualizationMarkers(self.cfg.grip_visualizer_cfg)
                self.active_hand_visualizer = VisualizationMarkers(self.cfg.active_hand_visualizer_cfg)
            self.base_velocity_visualizer.set_visibility(True)
            self.left_wrist_visualizer.set_visibility(True)
            self.right_wrist_visualizer.set_visibility(True)
            self.posture_command_visualizer.set_visibility(True)
            self.grip_visualizer.set_visibility(True)
            self.active_hand_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_velocity_visualizer"):
                self.base_velocity_visualizer.set_visibility(False)
                self.left_wrist_visualizer.set_visibility(False)
                self.right_wrist_visualizer.set_visibility(False)
                self.posture_command_visualizer.set_visibility(False)
                self.grip_visualizer.set_visibility(False)
                self.active_hand_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        vel_marker_pos_w = base_pos_w.clone()
        vel_marker_pos_w[:, 2] += 0.5
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.base_velocity[:, :2])
        self.base_velocity_visualizer.visualize(vel_marker_pos_w, vel_arrow_quat, vel_arrow_scale)

        left_pos_w, left_quat_w = self._target_pose_w(self.left_wrist_pose_b, "left")
        right_pos_w, right_quat_w = self._target_pose_w(self.right_wrist_pose_b, "right")
        self.left_wrist_visualizer.visualize(left_pos_w, left_quat_w)
        self.right_wrist_visualizer.visualize(right_pos_w, right_quat_w)

        posture_arrow_pos_w, posture_arrow_quat_w = self._resolve_posture_arrow()
        self.posture_command_visualizer.visualize(posture_arrow_pos_w, posture_arrow_quat_w)

        grip_pos_w = torch.cat((left_pos_w, right_pos_w), dim=0)
        grip_scale = torch.cat(
            (
                self._resolve_grip_marker_scale(self.left_grip),
                self._resolve_grip_marker_scale(self.right_grip),
            ),
            dim=0,
        )
        grip_marker_indices = torch.cat(
            (
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs, dtype=torch.long, device=self.device),
            ),
            dim=0,
        )
        self.grip_visualizer.visualize(grip_pos_w, scales=grip_scale, marker_indices=grip_marker_indices)

        active_pos_w = torch.where((self.active_hand == 0).unsqueeze(-1), left_pos_w, right_pos_w)
        active_pos_w = active_pos_w.clone()
        active_pos_w[:, 2] += 0.12
        self.active_hand_visualizer.visualize(active_pos_w)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.base_velocity_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat_b = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat_w = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat_b)
        return arrow_scale, arrow_quat_w

    def _resolve_posture_arrow(self) -> tuple[torch.Tensor, torch.Tensor]:
        arrow_pos_w = self.robot.data.root_pos_w.clone()
        arrow_pos_w[:, 2] = self.env.scene.env_origins[:, 2] + self.posture_command[:, 0]
        zeros = torch.zeros(self.num_envs, device=self.device)
        pitch_quat = math_utils.quat_from_euler_xyz(zeros, self.posture_command[:, 1], zeros)
        arrow_quat_w = math_utils.quat_mul(yaw_quat(self.robot.data.root_quat_w), pitch_quat)
        return arrow_pos_w, arrow_quat_w

    def _anchor_pose_w(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_id = self.left_anchor_body_id if side == "left" else self.right_anchor_body_id
        anchor_pos_w = self.robot.data.body_pos_w[:, anchor_id].clone()
        root_yaw_quat = yaw_quat(self.robot.data.root_quat_w)
        zeros = torch.zeros(self.num_envs, device=self.device)
        pitch_quat = math_utils.quat_from_euler_xyz(zeros, self.posture_command[:, 1], zeros)
        anchor_quat_w = math_utils.quat_mul(root_yaw_quat, pitch_quat)

        root_cmd_pos_w = self.robot.data.root_pos_w.clone()
        root_cmd_pos_w[:, 2] = self.env.scene.env_origins[:, 2] + self.posture_command[:, 0]
        root_to_anchor_w = anchor_pos_w - self.robot.data.root_pos_w
        root_to_anchor_yaw = math_utils.quat_apply_inverse(root_yaw_quat, root_to_anchor_w)
        anchor_offset_b = torch.zeros_like(root_to_anchor_yaw)
        anchor_offset_b[:, 1] = root_to_anchor_yaw[:, 1]
        anchor_offset_b[:, 2] = self.anchor_height_offset
        anchor_pos_w = root_cmd_pos_w + math_utils.quat_apply(anchor_quat_w, anchor_offset_b)
        return anchor_pos_w, anchor_quat_w

    def _target_pose_w(self, pose_b: torch.Tensor, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.combine_frame_transforms(
            *self._anchor_pose_w(side),
            pose_b[:, :3],
            pose_b[:, 3:],
        )

    def _resolve_grip_marker_scale(self, grip: torch.Tensor) -> torch.Tensor:
        grip = torch.clamp(grip.reshape(-1), 0.0, 1.0)
        radius = self.cfg.grip_marker_open_radius + grip * (
            self.cfg.grip_marker_closed_radius - self.cfg.grip_marker_open_radius
        )
        return radius.unsqueeze(-1).repeat(1, 3)


@configclass
class G1Dex3HierCommandCfg(CommandTermCfg):
    class_type: type = G1Dex3HierCommand
    asset_name: str = MISSING
    left_hand_probability: float = 0.5
    default_root_height: float = 0.8
    default_torso_pitch: float = 0.0
    default_left_wrist_pose_b: tuple[float, float, float, float, float, float, float] = (
        0.25,
        0.15,
        -0.25,
        0.955177693375944,
        0.0,
        0.0,
        0.296033062472777,
    )
    default_right_wrist_pose_b: tuple[float, float, float, float, float, float, float] = (
        0.25,
        -0.15,
        -0.25,
        0.955177693375944,
        0.0,
        0.0,
        -0.296033062472777,
    )
    base_velocity_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/G1Dex3HierCommand/base_velocity"
    )
    left_wrist_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/G1Dex3HierCommand/left_wrist"
    )
    right_wrist_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/G1Dex3HierCommand/right_wrist"
    )
    posture_command_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/G1Dex3HierCommand/posture"
    )
    grip_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/G1Dex3HierCommand/grip",
        markers={
            "left_grip": sim_utils.SphereCfg(
                radius=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.35, 1.0),
                    emissive_color=(0.0, 0.04, 0.12),
                ),
            ),
            "right_grip": sim_utils.SphereCfg(
                radius=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.55, 0.05),
                    emissive_color=(0.12, 0.04, 0.0),
                ),
            ),
        },
    )
    active_hand_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/G1Dex3HierCommand/active_hand",
        markers={
            "active": sim_utils.SphereCfg(
                radius=0.055,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.25),
                    emissive_color=(0.0, 0.15, 0.02),
                ),
            ),
        },
    )
    grip_marker_open_radius: float = 0.025
    grip_marker_closed_radius: float = 0.075
    base_velocity_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    left_wrist_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    right_wrist_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    posture_command_visualizer_cfg.markers["arrow"].scale = (0.15, 0.15, 0.4)
    posture_command_visualizer_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.05, 0.65),
        emissive_color=(0.25, 0.0, 0.12),
        roughness=0.35,
    )


@configclass
class G1Dex3HierDrcCommandsCfg:
    high_level = G1Dex3HierCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=True,
    )
