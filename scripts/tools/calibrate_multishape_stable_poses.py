#!/usr/bin/env python3
"""Measure physically settled poses for the multishape grasp-reference assets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WBC_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "source" / "hbc_lab"))
for isaaclab_source in (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
):
    sys.path.insert(0, str(WBC_ROOT / "IsaacLab" / "source" / isaaclab_source))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="HBC-Isaac-G1-Dex1-HierDrc-MultiShape-Play-v0",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--settle-seconds", type=float, default=5.0)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--drop-height", type=float, default=0.02)
parser.add_argument("--linear-velocity-threshold", type=float, default=0.01)
parser.add_argument("--angular-velocity-threshold", type=float, default=0.05)
parser.add_argument("--disable-fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INFO] Isaac Sim application started; importing calibration dependencies.", flush=True)

import gymnasium as gym
import torch

import hbc_lab.tasks  # noqa: E402,F401
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import rotated_box_min_z
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.object_shape_bps import ALL_SHAPE_NAMES
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import SHALLOW_TRAY_SIZE
from hbc_lab.utils.parser_cfg import parse_env_cfg
from isaaclab.utils import math as math_utils

print("[INFO] Calibration dependencies imported.", flush=True)


def _active_object_value(env, collection_name: str, rigid_name: str) -> torch.Tensor:
    obj = env.scene["object"]
    collection_value = getattr(obj.data, collection_name, None)
    if collection_value is not None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        return collection_value[env_ids, env.active_object_index]
    return getattr(obj.data, rigid_name)


def _remove_world_yaw(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion_wxyz)
    del roll, pitch
    zeros = torch.zeros_like(yaw)
    inverse_yaw = math_utils.quat_inv(math_utils.quat_from_euler_xyz(zeros, zeros, yaw))
    canonical = math_utils.quat_mul(inverse_yaw, quaternion_wxyz)
    canonical = canonical / torch.linalg.vector_norm(canonical, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.where(canonical[:, :1] < 0.0, -canonical, canonical)


def _prepare_trial(env, trial_index: int) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset()
    shape_indices = env.active_object_index
    dtype = env._object_root_pos_w().dtype
    # Trial zero reproduces the task's exact spawn. Optional later trials only
    # break genuinely unstable equilibria without deliberately tipping objects.
    perturbations_deg = ((0.0, 0.0), (0.25, -0.25), (-0.25, 0.25))
    roll_deg, pitch_deg = perturbations_deg[trial_index % len(perturbations_deg)]
    roll = torch.full((env.num_envs,), math.radians(roll_deg), device=env.device, dtype=dtype)
    pitch = torch.full((env.num_envs,), math.radians(pitch_deg), device=env.device, dtype=dtype)
    yaw = torch.zeros_like(roll)
    perturbation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    quaternion_w = math_utils.quat_mul(perturbation, env.object_stable_quat_wxyz[shape_indices])

    scale = env.object_size_scale.unsqueeze(-1)
    bounds_min = env.object_geometry_bounds_min_o[shape_indices] * scale
    bounds_max = env.object_geometry_bounds_max_o[shape_indices] * scale
    root_pos_w = env._object_root_pos_w().clone()
    support_top_z = env.scene.env_origins[:, 2] + env.cfg.multishape_support_height
    root_pos_w[:, 2] = support_top_z - rotated_box_min_z(bounds_min, bounds_max, quaternion_w)
    root_pos_w[:, 2] += args.drop_height
    root_state = torch.cat(
        (
            root_pos_w,
            quaternion_w,
            torch.zeros(env.num_envs, 6, device=env.device, dtype=dtype),
        ),
        dim=-1,
    )
    env.scene["object"].write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def main() -> None:
    print("[INFO] Building stable-pose calibration environment.", flush=True)
    try:
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=None,
            use_fabric=not args.disable_fabric,
            entry_point_key="play_env_cfg_entry_point",
        )
    except BaseException as error:
        print(f"[ERROR] Environment configuration failed: {error!r}", flush=True)
        raise
    shape_names = tuple(env_cfg.object_shape_names)
    env_cfg.scene.num_envs = len(shape_names)
    print("[INFO] Stable-pose environment configuration parsed.", flush=True)
    env_cfg.allow_missing_low_level_policy = True
    env_cfg.low_level_policy_path = ""
    env_cfg.enable_debug_visualization = False
    env_cfg.platform_pose_debug_vis = False
    env_cfg.object_shape_bps_enabled = False
    env_cfg.object_mass_curriculum_enabled = False
    env_cfg.events.reset_object.params["pose_range"]["yaw"] = (0.0, 0.0)

    print("[INFO] Creating stable-pose simulation environments.", flush=True)
    wrapped_env = gym.make(args.task, cfg=env_cfg)
    print("[INFO] Stable-pose simulation environments created.", flush=True)
    env = wrapped_env.unwrapped
    steps = max(1, int(round(args.settle_seconds / env.physics_dt)))
    trial_results: list[list[dict[str, object]]] = [[] for _ in shape_names]
    try:
        for trial_index in range(args.trials):
            _prepare_trial(env, trial_index)
            for _ in range(steps):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            quaternion_w = _remove_world_yaw(env._object_root_quat_w()).detach().cpu()
            linear_velocity_w = _active_object_value(env, "object_lin_vel_w", "root_lin_vel_w")
            angular_velocity_w = _active_object_value(env, "object_ang_vel_w", "root_ang_vel_w")
            linear_speed = torch.linalg.vector_norm(linear_velocity_w, dim=-1).detach().cpu()
            angular_speed = torch.linalg.vector_norm(angular_velocity_w, dim=-1).detach().cpu()
            object_center_w = env._object_frame_pos_w().detach().cpu()
            support_center_w = env.scene["object_init_platform"].data.root_pos_w.detach().cpu()
            support_xy_error = torch.abs(object_center_w[:, :2] - support_center_w[:, :2])
            support_top_z = support_center_w[:, 2] + 0.5 * env.cfg.multishape_support_height
            on_support = (
                (support_xy_error[:, 0] <= 0.5 * SHALLOW_TRAY_SIZE[0])
                & (support_xy_error[:, 1] <= 0.5 * SHALLOW_TRAY_SIZE[1])
                & (object_center_w[:, 2] >= support_top_z - 0.03)
            )
            for shape_index, shape_name in enumerate(shape_names):
                settled = bool(
                    linear_speed[shape_index] <= args.linear_velocity_threshold
                    and angular_speed[shape_index] <= args.angular_velocity_threshold
                    and on_support[shape_index]
                )
                trial_results[shape_index].append(
                    {
                        "shape_name": shape_name,
                        "trial": trial_index,
                        "stable_quat_wxyz": quaternion_w[shape_index].tolist(),
                        "linear_speed": float(linear_speed[shape_index]),
                        "angular_speed": float(angular_speed[shape_index]),
                        "center_height": float(object_center_w[shape_index, 2]),
                        "support_center_w": support_center_w[shape_index].tolist(),
                        "support_xy_error": support_xy_error[shape_index].tolist(),
                        "on_support": bool(on_support[shape_index]),
                        "settled": settled,
                    }
                )

        output: dict[str, dict[str, object]] = {}
        for shape_name, candidates in zip(shape_names, trial_results):
            settled_candidates = [candidate for candidate in candidates if candidate["settled"]]
            if settled_candidates:
                selected = settled_candidates[0]
            else:
                selected = min(
                    candidates,
                    key=lambda candidate: (candidate["angular_speed"], candidate["linear_speed"]),
                )
            output[shape_name] = {**selected, "trials": candidates}
            print(
                f"{shape_name:16s} settled={selected['settled']} "
                f"support={selected['on_support']} xy={selected['support_xy_error']} "
                f"lin={selected['linear_speed']:.5f} ang={selected['angular_speed']:.5f} "
                f"quat={selected['stable_quat_wxyz']}"
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Saved stable-pose calibration to {args.output.resolve()}")
    finally:
        wrapped_env.close()


print(f"[INFO] Calibration module loaded as {__name__!r}.", flush=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
