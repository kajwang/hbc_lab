from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import math as math_utils

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import G1Dex1HierDrcEnv

from ..mdp.observations import (
    _compute_local_environment_scan,
    _compute_local_occupancy_voxel,
    clear_local_occupancy_voxel,
)
from ..mdp.scenes import ENVIRONMENT_VOXEL_SHAPE_XYZ


ENVIRONMENT_SCAN_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1SceneAware/environment_scan",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.012,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
        ),
    },
)

ENVIRONMENT_VOXEL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1SceneAware/environment_voxel_memory",
    markers={
        "recent": sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.82, 0.05)),
        ),
        "remembered": sim_utils.SphereCfg(
            radius=0.020,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.65)),
        ),
        "fading": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.75, 0.90)),
        ),
    },
)


class G1Dex1SceneAwareEnv(G1Dex1HierDrcEnv):
    """PnP environment with current-return and rolling-occupancy visualization."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.environment_scan_visualizer = None
        self.environment_voxel_visualizer = None

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        super()._reset_hier_buffers(env_ids)
        clear_local_occupancy_voxel(self, env_ids)

    def _update_target_pose_visualization(self) -> None:
        super()._update_target_pose_visualization()
        perception_enabled = getattr(self.cfg, "environment_perception_enabled", True)
        scan_enabled = bool(
            getattr(self.cfg, "environment_scan_debug_vis", False) and perception_enabled
        )
        voxel_enabled = bool(
            getattr(self.cfg, "environment_voxel_debug_vis", False) and perception_enabled
        )
        if not scan_enabled:
            if self.environment_scan_visualizer is not None:
                self.environment_scan_visualizer.set_visibility(False)
        else:
            if self.environment_scan_visualizer is None:
                self.environment_scan_visualizer = VisualizationMarkers(ENVIRONMENT_SCAN_MARKER_CFG)
            _, hits_w = _compute_local_environment_scan(self)
            finite_hits = hits_w[torch.isfinite(hits_w).all(dim=-1)]
            self.environment_scan_visualizer.set_visibility(finite_hits.numel() > 0)
            if finite_hits.numel() > 0:
                self.environment_scan_visualizer.visualize(translations=finite_hits)

        if not voxel_enabled:
            if self.environment_voxel_visualizer is not None:
                self.environment_voxel_visualizer.set_visibility(False)
            return
        if self.environment_voxel_visualizer is None:
            self.environment_voxel_visualizer = VisualizationMarkers(ENVIRONMENT_VOXEL_MARKER_CFG)

        occupancy = _compute_local_occupancy_voxel(self)
        centers_r = self._environment_voxel_centers_r.reshape(-1, 3)
        max_envs = min(self.num_envs, int(self.cfg.environment_voxel_debug_max_envs))
        max_points = int(self.cfg.environment_voxel_debug_max_points_per_env)
        threshold = float(self.cfg.environment_voxel_debug_threshold)
        translations = []
        marker_indices = []
        root = self.scene["robot"].data
        root_yaw_quat = math_utils.yaw_quat(root.root_quat_w)
        for env_id in range(max_envs):
            values = occupancy[env_id].flatten()
            valid_indices = torch.nonzero(values >= threshold, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                continue
            if valid_indices.numel() > max_points:
                selected_values, selected_order = torch.topk(values[valid_indices], max_points)
                valid_indices = valid_indices[selected_order]
            else:
                selected_values = values[valid_indices]
            local_points = centers_r[valid_indices]
            quat = root_yaw_quat[env_id].expand(local_points.shape[0], -1)
            world_points = root.root_pos_w[env_id] + math_utils.quat_apply(quat, local_points)
            indices = torch.full_like(valid_indices, 2)
            indices[selected_values >= 0.30] = 1
            indices[selected_values >= 0.75] = 0
            translations.append(world_points)
            marker_indices.append(indices)

        has_points = bool(translations)
        self.environment_voxel_visualizer.set_visibility(has_points)
        if has_points:
            self.environment_voxel_visualizer.visualize(
                translations=torch.cat(translations, dim=0),
                marker_indices=torch.cat(marker_indices, dim=0),
            )

    def _log_high_level_diagnostics(self) -> None:
        super()._log_high_level_diagnostics()
        action = self.last_high_level_action
        self.extras["log"]["HL/action_abs_mean"] = action.abs().mean()
        self.extras["log"]["HL/action_boundary_095_ratio"] = (action.abs() > 0.95).float().mean()
        self.extras["log"]["HL/base_action_boundary_095_ratio"] = (
            action[:, :3].abs() > 0.95
        ).float().mean()
        self.extras["log"]["HL/posture_action_boundary_095_ratio"] = (
            action[:, 3:5].abs() > 0.95
        ).float().mean()
        self.extras["log"]["HL/hand_action_boundary_095_ratio"] = (
            action[:, 5:].abs() > 0.95
        ).float().mean()
