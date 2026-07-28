from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.rewards import (
    both_hand_center_tracking_error_penalty,
    command_smoothness,
    root_object_facing_reward,
    task_success_reward,
)


def approach_reward(env) -> torch.Tensor:
    left_near = torch.exp(-env.left_hand_object_distance / 0.5)
    right_near = torch.exp(-env.right_hand_object_distance / 0.5)
    return 0.5 * (left_near + right_near)


def couple_reward(env) -> torch.Tensor:
    left_near = 1.0 - torch.tanh(env.left_hand_object_distance / 0.35)
    right_near = 1.0 - torch.tanh(env.right_hand_object_distance / 0.35)
    both_near = 0.5 * (left_near + right_near)
    left_close_window = 1.0 - torch.tanh(env.left_hand_object_distance / 0.1)
    right_close_window = 1.0 - torch.tanh(env.right_hand_object_distance / 0.1)
    gated_close = 0.5 * (
        env.left_grip * left_close_window + env.right_grip * right_close_window
    )
    early_close = 0.5 * (
        env.left_grip * (1.0 - left_close_window)
        + env.right_grip * (1.0 - right_close_window)
    )
    return (
        0.4 * both_near
        + 0.2 * env.bimanual_contact
        + 0.2 * env.bimanual_grasp
        + 0.2 * gated_close
        - 0.2 * early_close
    )


def manip_reward(env) -> torch.Tensor:
    progress = env.cart_manipulation_progress
    return 0.3 + progress.transport_progress


def hier_drc_reward(
    env,
    approach_scale: float = 2.0,
    couple_scale: float = 20.0,
    manip_scale: float = 200.0,
) -> torch.Tensor:
    r_app = approach_reward(env)
    r_couple = couple_reward(env)
    r_manip = manip_reward(env)
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    return (
        env.W_app * approach_scale * r_app
        + env.W_couple * couple_scale * r_couple
        + env.W_manip * manip_scale * r_manip
    )


@configclass
class G1Dex1CartPushRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    command_smoothness = RewTerm(func=command_smoothness, weight=-0.02)
    root_object_facing = RewTerm(func=root_object_facing_reward, weight=2.0)
    both_hand_center_tracking_error_penalty = RewTerm(
        func=both_hand_center_tracking_error_penalty,
        weight=-2.0,
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*|.*gripper.*|.*finger.*|.*hand.*).*"],
            ),
        },
    )
