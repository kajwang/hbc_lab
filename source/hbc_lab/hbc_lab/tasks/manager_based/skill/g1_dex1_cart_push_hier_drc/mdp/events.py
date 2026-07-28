from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import G1Dex1HierDrcEventCfg

from .progress import compute_cart_goal, compute_planar_heading_quat


CART_JOINT_NAMES = (
    "RL_joint",
    "RR_joint",
    "FL_joint",
    "FR_joint",
    "RL_turn_joint",
    "RR_turn_joint",
)
CART_HANDLE_FRAME_OFFSET_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)


def reset_cart_articulation(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cart_goal_displacement_x: tuple[float, float],
    cart_goal_displacement_y: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    cart = env.scene[asset_cfg.name]
    root_state = cart.data.default_root_state[env_ids].clone()
    position_ranges = torch.tensor(
        [pose_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device=cart.device,
    )
    position_offset = math_utils.sample_uniform(
        position_ranges[:, 0],
        position_ranges[:, 1],
        (env_ids.numel(), 3),
        cart.device,
    )
    root_state[:, :3] += env.scene.env_origins[env_ids] + position_offset
    root_state[:, 7:] = 0.0
    cart.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    cart.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    joint_ids, joint_names = cart.find_joints(list(CART_JOINT_NAMES), preserve_order=True)
    if len(joint_ids) != len(CART_JOINT_NAMES):
        raise RuntimeError(f"Cart requires {CART_JOINT_NAMES}, resolved {joint_names}")
    joint_pos = torch.zeros(env_ids.numel(), len(joint_ids), device=cart.device)
    joint_vel = torch.zeros_like(joint_pos)
    cart.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    forward = torch.empty(env_ids.numel(), device=cart.device).uniform_(*cart_goal_displacement_x)
    lateral = torch.empty(env_ids.numel(), device=cart.device).uniform_(*cart_goal_displacement_y)
    env.cart_goal_forward[env_ids] = forward
    env.cart_goal_lateral[env_ids] = lateral
    handle_frame_offset_quat = torch.tensor(
        CART_HANDLE_FRAME_OFFSET_QUAT,
        device=cart.device,
        dtype=root_state.dtype,
    ).expand(env_ids.numel(), -1)
    handle_quat_w = math_utils.quat_mul(root_state[:, 3:7], handle_frame_offset_quat)
    env.cart_initial_quat_w[env_ids] = compute_planar_heading_quat(handle_quat_w)
    env.cart_target_root_pos_w[env_ids] = compute_cart_goal(
        root_state[:, :3],
        env.cart_initial_quat_w[env_ids],
        forward,
        lateral,
    )


@configclass
class G1Dex1CartPushEventCfg(G1Dex1HierDrcEventCfg):
    reset_object = EventTerm(
        func=reset_cart_articulation,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.5, 2.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
            },
            "cart_goal_displacement_x": (2.0, 4.0),
            "cart_goal_displacement_y": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
