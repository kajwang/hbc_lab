from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.mdp.high_level_actions import HighLevelActionLimits


def approach_reward(env) -> torch.Tensor:
    return torch.exp(-env.d_active_hand / 0.5)


def couple_reward(env) -> torch.Tensor:
    grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.15)
    active_grip = env.active_grip
    gated_gripper_close = active_grip * grasp_window
    early_close_penalty = active_grip * (1.0 - grasp_window)
    return (
        0.35 * grasp_window
        + 0.35 * env.c_finger_count
        + 0.20 * env.c_pinch
        + 0.10 * gated_gripper_close
        - 0.20 * early_close_penalty
    )


def manip_reward(env) -> torch.Tensor:
    initial = torch.norm(env.object_initial_pos_w - env.object_target_pos_w, dim=-1)
    progress = torch.clamp(initial - env.d_goal, min=0.0) / (initial + 1e-5)
    return 0.7 * progress + 0.3 * env.c_couple


def hier_drc_reward(env, approach_scale: float = 2.0, couple_scale: float = 5.0, manip_scale: float = 20.0) -> torch.Tensor:
    r_app = approach_reward(env)
    r_couple = couple_reward(env)
    r_manip = manip_reward(env)
    grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.15)
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    env.extras["log"]["DRC/gated_gripper_close_mean"] = (env.active_grip * grasp_window).mean()
    env.extras["log"]["DRC/early_close_penalty_mean"] = (env.active_grip * (1.0 - grasp_window)).mean()
    return env.W_app * approach_scale * r_app + env.W_couple * couple_scale * r_couple + env.W_manip * manip_scale * r_manip


def command_smoothness(env) -> torch.Tensor:
    return torch.sum(torch.square(env.last_high_level_action - env.prev_high_level_action), dim=-1)


def _bound_violation(value: torch.Tensor, lower: torch.Tensor | float, upper: torch.Tensor | float) -> torch.Tensor:
    return torch.square(torch.clamp(lower - value, min=0.0)) + torch.square(torch.clamp(value - upper, min=0.0))


def active_wrist_workspace_penalty(env) -> torch.Tensor:
    limits = getattr(env, "action_limits", HighLevelActionLimits())
    left_active = env.active_hand == 0
    pose_b = torch.where(
        left_active.unsqueeze(-1),
        env.command_state.left_wrist_pose_b,
        env.command_state.right_wrist_pose_b,
    )
    pos_b = pose_b[:, :3]
    radius = torch.norm(pos_b, dim=-1)
    safe_radius = torch.clamp(radius, min=1.0e-6)
    pitch = torch.asin(torch.clamp(pos_b[:, 2] / safe_radius, min=-1.0, max=1.0))
    azimuth = torch.atan2(pos_b[:, 1], pos_b[:, 0])

    radius_min, radius_max = limits.wrist_radius_range
    pitch_min, pitch_max = limits.wrist_pitch_range
    left_azimuth_min, left_azimuth_max = limits.left_wrist_azimuth_range
    right_azimuth_min, right_azimuth_max = limits.right_wrist_azimuth_range
    azimuth_min = torch.where(
        left_active,
        torch.full_like(azimuth, left_azimuth_min),
        torch.full_like(azimuth, right_azimuth_min),
    )
    azimuth_max = torch.where(
        left_active,
        torch.full_like(azimuth, left_azimuth_max),
        torch.full_like(azimuth, right_azimuth_max),
    )

    return (
        _bound_violation(radius, radius_min, radius_max)
        + _bound_violation(pitch, pitch_min, pitch_max)
        + _bound_violation(azimuth, azimuth_min, azimuth_max)
    )


def task_success_reward(env) -> torch.Tensor:
    return env.task_succeeded.float() * 100.0


@configclass
class G1Dex3HierDrcRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    command_smoothness = RewTerm(func=command_smoothness, weight=-0.02)
    active_wrist_workspace_penalty = RewTerm(func=active_wrist_workspace_penalty, weight=-2.0)
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
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*|.*hand.*).*"]),
        },
    )
