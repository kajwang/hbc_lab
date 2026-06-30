from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp


def approach_reward(env) -> torch.Tensor:
    return torch.exp(-env.d_active_hand / 0.5)


def couple_reward(env) -> torch.Tensor:
    grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.35)
    gripper_close = env.active_grip
    gated_gripper_close = gripper_close * grasp_window
    early_close_penalty = gripper_close * (1.0 - grasp_window)
    return (
        0.35 * grasp_window
        + 0.35 * env.c_contact
        # Baseline A: require force-direction pinch in the couple reward.
        # + 0.20 * env.c_pinch
        + 0.20 * env.c_grasp
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
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    return env.W_app * approach_scale * r_app + env.W_couple * couple_scale * r_couple + env.W_manip * manip_scale * r_manip


def command_smoothness(env) -> torch.Tensor:
    return torch.sum(torch.square(env.last_high_level_action - env.prev_high_level_action), dim=-1)


def both_wrist_tracking_error_penalty(env) -> torch.Tensor:
    robot = env.scene["robot"]
    left_target_w = env.low_level_obs_builder._target_pos_w(
        env.command_state.left_wrist_pose_b,
        "left",
        env.command_state.posture_command,
    )
    right_target_w = env.low_level_obs_builder._target_pos_w(
        env.command_state.right_wrist_pose_b,
        "right",
        env.command_state.posture_command,
    )
    left_wrist_pos_w = robot.data.body_pos_w[:, env.left_wrist_body_id]
    right_wrist_pos_w = robot.data.body_pos_w[:, env.right_wrist_body_id]
    left_wrist_error = torch.norm(left_wrist_pos_w - left_target_w, dim=-1)
    right_wrist_error = torch.norm(right_wrist_pos_w - right_target_w, dim=-1)
    return left_wrist_error + right_wrist_error


def task_success_reward(env) -> torch.Tensor:
    return env.task_succeeded.float() * 100.0


def object_fall_penalty(env) -> torch.Tensor:
    return env.object_fallen.float()


@configclass
class G1Dex1HierDrcRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    object_fall = RewTerm(func=object_fall_penalty, weight=-2.0)
    command_smoothness = RewTerm(func=command_smoothness, weight=-0.02)
    both_wrist_tracking_error_penalty = RewTerm(func=both_wrist_tracking_error_penalty, weight=-2.0)
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
