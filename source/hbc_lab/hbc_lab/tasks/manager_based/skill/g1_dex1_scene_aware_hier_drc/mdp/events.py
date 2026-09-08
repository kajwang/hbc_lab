from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.assets.objects import APPLE_OBJECT_FRAME_OFFSET_Z
from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import (
    G1Dex1HierDrcEventCfg,
    _apply_object_mass_curriculum,
)

from .scenes import TABLE_NOMINAL_CLEARANCE, TABLE_TOP_SIZE
from .scenes import TABLE_LEG_CENTERS, TABLE_LEG_SIZE
from .geometry_shaping import write_obstacle_boxes


def _ensure_scene_context_buffers(env) -> None:
    if hasattr(env, "scene_is_constrained"):
        return
    env.scene_is_constrained = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.scene_table_clearance = torch.zeros(env.num_envs, device=env.device)
    env.scene_table_center_w = torch.zeros(env.num_envs, 3, device=env.device)


def _root_state(positions: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    velocities = torch.zeros(positions.shape[0], 6, device=positions.device, dtype=positions.dtype)
    return torch.cat((positions, orientations, velocities), dim=-1)


def reset_scene_aware_pnp(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    table_clearance_range: tuple[float, float],
    target_distance_range: tuple[float, float],
    target_lateral_range: tuple[float, float],
    object_root_height: float,
    target_root_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset matched open/table-under PnP scenes without exposing the scenario label to the actor."""
    _ensure_scene_context_buffers(env)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    obj = env.scene[asset_cfg.name]
    count = env_ids.numel()
    dtype = obj.data.root_pos_w.dtype

    object_root_pos_w = env.scene.env_origins[env_ids].clone()
    object_root_pos_w[:, 0] += torch.empty(count, device=env.device, dtype=dtype).uniform_(*pose_range["x"])
    object_root_pos_w[:, 1] += torch.empty(count, device=env.device, dtype=dtype).uniform_(*pose_range["y"])
    object_root_pos_w[:, 2] += object_root_height
    yaw = torch.empty(count, device=env.device, dtype=dtype).uniform_(*pose_range.get("yaw", (-torch.pi, torch.pi)))
    zeros = torch.zeros_like(yaw)
    object_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]
    object_to_robot_xy = robot_root_pos_w[:, :2] - object_root_pos_w[:, :2]
    object_to_robot_xy = object_to_robot_xy / torch.clamp(
        torch.linalg.norm(object_to_robot_xy, dim=-1, keepdim=True), min=1.0e-6
    )
    lateral_xy = torch.stack((-object_to_robot_xy[:, 1], object_to_robot_xy[:, 0]), dim=-1)
    target_distance = torch.empty(count, 1, device=env.device, dtype=dtype).uniform_(*target_distance_range)
    target_lateral = torch.empty(count, 1, device=env.device, dtype=dtype).uniform_(*target_lateral_range)
    target_root_pos_w = object_root_pos_w.clone()
    target_root_pos_w[:, :2] += target_distance * object_to_robot_xy + target_lateral * lateral_xy
    target_root_pos_w[:, 2] = env.scene.env_origins[env_ids, 2] + target_root_height

    object_velocity = torch.zeros(count, 6, device=env.device, dtype=dtype)
    obj.write_root_state_to_sim(
        torch.cat((object_root_pos_w, object_quat_w, object_velocity), dim=-1),
        env_ids=env_ids,
    )
    _apply_object_mass_curriculum(env, env_ids, obj)

    object_frame_pos_w = object_root_pos_w.clone()
    object_frame_pos_w[:, 2] += APPLE_OBJECT_FRAME_OFFSET_Z
    target_frame_pos_w = target_root_pos_w.clone()
    target_frame_pos_w[:, 2] += APPLE_OBJECT_FRAME_OFFSET_Z
    env.object_initial_pos_w[env_ids] = object_frame_pos_w
    env.object_initial_root_z_w[env_ids] = object_root_pos_w[:, 2]
    env.object_target_pos_w[env_ids] = target_frame_pos_w

    constrained = torch.remainder(env_ids, 2) == 1
    table_clearance = torch.empty(count, device=env.device, dtype=dtype).uniform_(*table_clearance_range)
    approach_xy = -object_to_robot_xy
    table_center_xy = object_root_pos_w[:, :2] + 0.10 * approach_xy
    table_yaw = torch.atan2(approach_xy[:, 1], approach_xy[:, 0])

    table_root_pos_w = env.scene.env_origins[env_ids].clone()
    table_root_pos_w[:, :2] = table_center_xy
    table_root_pos_w[:, 2] += table_clearance - TABLE_NOMINAL_CLEARANCE
    parked_table_pos_w = env.scene.env_origins[env_ids].clone()
    parked_table_pos_w[:, 0] -= 2.6
    table_pos_w = torch.where(constrained.unsqueeze(-1), table_root_pos_w, parked_table_pos_w)
    table_root_yaw = torch.where(constrained, table_yaw, torch.zeros_like(table_yaw))
    env.scene["scene_table"].write_root_state_to_sim(
        _root_state(table_pos_w, table_root_yaw),
        env_ids=env_ids,
    )

    table_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, table_root_yaw)
    local_centers = torch.tensor(
        ((0.0, 0.0, 0.76), *TABLE_LEG_CENTERS),
        device=env.device,
        dtype=dtype,
    ).unsqueeze(0).expand(count, -1, -1)
    box_sizes = torch.tensor(
        (TABLE_TOP_SIZE, *([TABLE_LEG_SIZE] * len(TABLE_LEG_CENTERS))),
        device=env.device,
        dtype=dtype,
    ).unsqueeze(0).expand(count, -1, -1)
    box_count = local_centers.shape[1]
    repeated_table_quat = table_quat_w[:, None, :].expand(-1, box_count, -1)
    centers_w = table_root_pos_w[:, None, :] + math_utils.quat_apply(
        repeated_table_quat.reshape(-1, 4), local_centers.reshape(-1, 3)
    ).reshape(count, box_count, 3)
    write_obstacle_boxes(
        env,
        env_ids,
        centers_w,
        repeated_table_quat,
        0.5 * box_sizes,
        constrained[:, None].expand(-1, box_count),
    )

    top_pos_w = table_root_pos_w.clone()
    top_pos_w[:, 2] += TABLE_NOMINAL_CLEARANCE + 0.5 * TABLE_TOP_SIZE[2]

    env.scene_is_constrained[env_ids] = constrained
    env.scene_table_clearance[env_ids] = torch.where(constrained, table_clearance, torch.zeros_like(table_clearance))
    env.scene_table_center_w[env_ids] = torch.where(
        constrained.unsqueeze(-1), top_pos_w, torch.zeros_like(top_pos_w)
    )
    env._environment_scan_cache_step = None


@configclass
class G1Dex1SceneAwareEventCfg(G1Dex1HierDrcEventCfg):
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
        },
    )
    reset_object = EventTerm(
        func=reset_scene_aware_pnp,
        mode="reset",
        params={
            "pose_range": {"x": (1.35, 1.65), "y": (-0.20, 0.20), "yaw": (-3.14, 3.14)},
            "table_clearance_range": (TABLE_NOMINAL_CLEARANCE, TABLE_NOMINAL_CLEARANCE),
            "target_distance_range": (0.90, 1.15),
            "target_lateral_range": (-0.45, 0.45),
            "object_root_height": 0.02,
            "target_root_height": 0.50,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
