#!/usr/bin/env python3
"""Preview a GALLANT-density 5 cm local voxel map without loading a policy checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WBC_ROOT = REPO_ROOT.parent
HBC_LAB_SOURCE_ROOT = Path(os.environ.get("HBC_LAB_SOURCE_ROOT", REPO_ROOT / "source" / "hbc_lab"))
sys.path.insert(0, str(HBC_LAB_SOURCE_ROOT))
for isaaclab_source in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks"):
    sys.path.insert(0, str(WBC_ROOT / "IsaacLab" / "source" / isaaclab_source))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="HBC-Isaac-G1-Dex1-MultiGeometry-Squashed-HierDrc-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_id", type=int, default=1, help="Geometry family id in [0, 7].")
parser.add_argument("--active_id", type=int, choices=(0, 1), default=1)
parser.add_argument("--geo_level", type=float, default=1.0)
parser.add_argument("--object_distance", type=float, default=1.0)
parser.add_argument("--history_frames", type=int, default=20)
parser.add_argument("--max_voxels", type=int, default=4096)
parser.add_argument("--real_time", action="store_true", default=True)
parser.add_argument(
    "--low_level_policy_path",
    default="logs/exported_policies/g1_dex1_handcenter_stickfigure_m49999.pt",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import math as math_utils

import hbc_lab.tasks  # noqa: F401,E402
from hbc_lab.tasks.manager_based.skill.g1_dex1_scene_aware_hier_drc.mdp.observations import (
    _compute_local_environment_scan,
    _hits_to_current_voxels,
    _voxel_center_grid,
)
from hbc_lab.utils.parser_cfg import parse_env_cfg


GALLANT_SHAPE_XYZ = (32, 32, 40)
# Preserve GALLANT's 1.6 x 1.6 x 2.0 m volume and 5 cm resolution, but shift
# its x extent forward for manipulation scenes whose geometry starts near 1 m.
GALLANT_MIN_XYZ = (-0.4, -0.8, -1.0)
GALLANT_MAX_XYZ = (1.2, 0.8, 1.0)

RAW_HIT_MARKERS = VisualizationMarkersCfg(
    prim_path="/Visuals/GallantVoxelPreview/raw_hits",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.008,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
        )
    },
)
VOXEL_MARKERS = VisualizationMarkersCfg(
    prim_path="/Visuals/GallantVoxelPreview/occupied_voxels",
    markers={
        "occupied": sim_utils.SphereCfg(
            radius=0.016,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.82, 0.05)),
        )
    },
)


def main() -> None:
    if not 0 <= args.env_id <= 7:
        raise ValueError(f"env_id must be in [0, 7], got {args.env_id}")
    if not 0.0 <= args.geo_level <= 1.0:
        raise ValueError(f"geo_level must be in [0, 1], got {args.geo_level}")

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, entry_point_key="play_env_cfg_entry_point")
    cfg.enable_debug_visualization = False
    cfg.environment_scan_debug_vis = False
    cfg.environment_voxel_debug_vis = False
    cfg.environment_perception_enabled = True
    cfg.play_active_id = args.active_id
    cfg.play_env_id = args.env_id
    cfg.play_geo_level = args.geo_level
    cfg.geometry_preview_sweep = False
    cfg.low_level_policy_path = args.low_level_policy_path
    cfg.events.reset_object.params["pose_range"]["x"] = (args.object_distance, args.object_distance)
    cfg.events.reset_object.params["pose_range"]["y"] = (0.0, 0.0)

    env = gym.make(args.task, cfg=cfg)
    unwrapped = env.unwrapped
    env.reset()
    raw_visualizer = VisualizationMarkers(RAW_HIT_MARKERS)
    voxel_visualizer = VisualizationMarkers(VOXEL_MARKERS)
    centers_r = _voxel_center_grid(
        unwrapped.device,
        torch.float32,
        GALLANT_SHAPE_XYZ,
        GALLANT_MIN_XYZ,
        GALLANT_MAX_XYZ,
    ).reshape(-1, 3)
    hit_history: list[torch.Tensor] = []
    actions = torch.zeros(
        unwrapped.num_envs,
        unwrapped.action_manager.total_action_dim,
        device=unwrapped.device,
    )

    print("[INFO] Red: current interlaced scan hits.")
    print("[INFO] Yellow: occupied 5 cm voxels fused from the recent scan history.")
    print(f"[INFO] Voxel grid: {GALLANT_SHAPE_XYZ}, bounds: {GALLANT_MIN_XYZ} -> {GALLANT_MAX_XYZ}")
    frame = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            _, hits_w = _compute_local_environment_scan(unwrapped)
            hit_history.append(hits_w.clone())
            del hit_history[:-max(args.history_frames, 1)]
            history_hits_w = torch.cat(hit_history, dim=1)
            robot = unwrapped.scene["robot"].data
            root_yaw = math_utils.euler_xyz_from_quat(robot.root_quat_w)[2]
            occupancy = _hits_to_current_voxels(
                history_hits_w,
                robot.root_pos_w,
                root_yaw,
                GALLANT_SHAPE_XYZ,
                GALLANT_MIN_XYZ,
                GALLANT_MAX_XYZ,
            )

            finite_hits = hits_w[torch.isfinite(hits_w).all(dim=-1)]
            raw_visualizer.set_visibility(finite_hits.numel() > 0)
            if finite_hits.numel() > 0:
                raw_visualizer.visualize(translations=finite_hits)

            world_points = []
            for env_id in range(unwrapped.num_envs):
                occupied = torch.nonzero(occupancy[env_id].flatten() > 0.5, as_tuple=False).squeeze(-1)
                if occupied.numel() > args.max_voxels:
                    occupied = occupied[: args.max_voxels]
                local_points = centers_r[occupied]
                yaw_quat = math_utils.yaw_quat(robot.root_quat_w[env_id : env_id + 1]).expand(
                    local_points.shape[0], -1
                )
                world_points.append(
                    robot.root_pos_w[env_id] + math_utils.quat_apply(yaw_quat, local_points)
                )
            occupied_points = torch.cat(world_points, dim=0)
            voxel_visualizer.set_visibility(occupied_points.numel() > 0)
            if occupied_points.numel() > 0:
                voxel_visualizer.visualize(translations=occupied_points)

            if frame % 20 == 0:
                print(
                    f"[VOXEL PREVIEW] frame={frame} current_hits={finite_hits.shape[0]} "
                    f"occupied_5cm={occupied_points.shape[0]}"
                )

            env.step(actions)
            frame += 1

        if args.real_time:
            remaining = unwrapped.step_dt - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
