from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.assets.objects import BOX_CUBE_CENTER_Z
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import (
    G1Dex1HierDrcEventCfg,
    _apply_object_mass_curriculum,
)


def reset_ground_box(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    object_goal_radius_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Place the box on the ground and sample a farther target away from the robot."""
    obj = env.scene[asset_cfg.name]
    num_envs = env_ids.shape[0]
    x = torch.empty(num_envs, device=obj.device).uniform_(*pose_range["x"])
    y = torch.empty(num_envs, device=obj.device).uniform_(*pose_range["y"])
    yaw = torch.empty(num_envs, device=obj.device).uniform_(*pose_range["yaw"])

    object_pos_w = env.scene.env_origins[env_ids].clone()
    object_pos_w[:, 0] += x
    object_pos_w[:, 1] += y
    object_pos_w[:, 2] += BOX_CUBE_CENTER_Z
    zeros = torch.zeros_like(yaw)
    object_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]
    away_xy = object_pos_w[:, :2] - robot_root_pos_w[:, :2]
    away_heading = torch.atan2(away_xy[:, 1], away_xy[:, 0])
    heading_jitter = torch.empty_like(away_heading).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)
    radius = torch.empty_like(away_heading).uniform_(*object_goal_radius_range)
    target_pos_w = object_pos_w.clone()
    target_pos_w[:, 0] += radius * torch.cos(away_heading + heading_jitter)
    target_pos_w[:, 1] += radius * torch.sin(away_heading + heading_jitter)

    object_velocity = torch.zeros(num_envs, 6, device=obj.device)
    obj.write_root_state_to_sim(
        torch.cat((object_pos_w, object_quat_w, object_velocity), dim=-1),
        env_ids=env_ids,
    )
    _apply_object_mass_curriculum(env, env_ids, obj)
    env.object_initial_pos_w[env_ids] = object_pos_w
    env.object_target_pos_w[env_ids] = target_pos_w


@configclass
class G1Dex1BoxCarryEventCfg(G1Dex1HierDrcEventCfg):
    reset_object = EventTerm(
        func=reset_ground_box,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.2, 1.8),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "object_goal_radius_range": (1.5, 2.5),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
