from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.rewards import (
    both_hand_center_tracking_error_penalty,
    command_smoothness,
    object_fall_penalty,
    root_object_facing_reward,
    task_success_reward,
)

from .contact_progress import compute_lift_gated_transport_progress


def approach_reward(env) -> torch.Tensor:
    left_near = torch.exp(-env.left_hand_object_distance / 0.5)
    right_near = torch.exp(-env.right_hand_object_distance / 0.5)
    return 0.5 * (left_near + right_near)


def couple_reward(env) -> torch.Tensor:
    both_near = 1.0 - torch.tanh(env.d_active_hand / 0.5)
    return 0.5 * both_near + 0.5 * env.gated_support - 0.25 * env.early_contact


def manip_reward(env) -> torch.Tensor:
    initial_xy = torch.norm((env.object_initial_pos_w - env.object_target_pos_w)[:, :2], dim=-1)
    transport_progress = torch.clamp(initial_xy - env.d_goal_xy, min=0.0) / (initial_xy + 1.0e-5)
    progress = compute_lift_gated_transport_progress(
        lift_height=env.box_lift_height,
        transport_progress=transport_progress,
        lift_target_height=0.10,
    )
    env.extras["log"]["DRC/lift_progress_mean"] = progress.lift_progress.mean()
    env.extras["log"]["DRC/transport_progress_mean"] = progress.transport_progress.mean()
    env.extras["log"]["DRC/transport_gate_mean"] = progress.transport_gate.mean()
    return progress.reward


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
class G1Dex1BoxCarryRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    object_fall = RewTerm(func=object_fall_penalty, weight=-2.0)
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
