from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp

from .scenes import HAND_CENTER_FRAME_NAME


INNER_PAD_CONTACT_GATE_SCALE = 0.05


def approach_reward(env) -> torch.Tensor:
    return torch.exp(-env.d_active_hand / 0.5)


def active_inner_pad_contact(env) -> torch.Tensor:
    if not hasattr(env, "_step_link_contact"):
        return torch.zeros_like(env.d_active_hand)

    left_active = env.active_hand == 0
    left_link1_3 = env._step_link_contact["left_Link1_3"]
    left_link2_3 = env._step_link_contact["left_Link2_3"]
    right_link1_3 = env._step_link_contact["right_Link1_3"]
    right_link2_3 = env._step_link_contact["right_Link2_3"]
    active_link1_3 = torch.where(left_active, left_link1_3, right_link1_3)
    active_link2_3 = torch.where(left_active, left_link2_3, right_link2_3)
    return torch.maximum(active_link1_3, active_link2_3)


def couple_reward(env) -> torch.Tensor:
    near = 1.0 - torch.tanh(env.d_active_hand / 0.35)
    grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.1)
    inner_pad_contact = active_inner_pad_contact(env)
    pad_gate = torch.clamp(inner_pad_contact / INNER_PAD_CONTACT_GATE_SCALE, min=0.0, max=1.0)
    gripper_close = env.active_grip
    gated_gripper_close1 = gripper_close * pad_gate
    early_close_penalty1 = gripper_close * (1.0 - pad_gate)
    gated_gripper_close2 = gripper_close * grasp_window
    early_close_penalty2 = gripper_close * (1.0 - grasp_window)
    env._couple_grasp_window = grasp_window.detach()
    env._couple_pad_gate = pad_gate.detach()
    env._couple_gated_gripper_close = (gated_gripper_close1 + gated_gripper_close2).detach()
    env._couple_early_close_penalty = (early_close_penalty1 + early_close_penalty2).detach()
    return (
        0.5 * near
        + 0.1 * pad_gate
        + 0.2 * env.c_grasp
        + 0.2 * gated_gripper_close2
        - 0.2 * early_close_penalty2
    )


def manip_reward(env) -> torch.Tensor:
    initial = torch.norm(env.object_initial_pos_w - env.object_target_pos_w, dim=-1)
    progress = torch.clamp(initial - env.d_goal, min=0.0) / (initial + 1e-5)
    return 0.7 * progress + 0.3 * env.c_couple


def hier_drc_reward(env, approach_scale: float = 2.0, couple_scale: float = 15.0, manip_scale: float = 50.0) -> torch.Tensor:
    r_app = approach_reward(env)
    r_couple = couple_reward(env)
    r_manip = manip_reward(env)
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    return env.W_app * approach_scale * r_app + env.W_couple * couple_scale * r_couple + env.W_manip * manip_scale * r_manip


def command_smoothness(env) -> torch.Tensor:
    return torch.sum(torch.square(env.last_high_level_action - env.prev_high_level_action), dim=-1)


def both_hand_center_tracking_error_penalty(env) -> torch.Tensor:
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
    hand_center_pos_w = env.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
    left_hand_center_error = torch.norm(hand_center_pos_w[:, 0, :] - left_target_w, dim=-1)
    right_hand_center_error = torch.norm(hand_center_pos_w[:, 1, :] - right_target_w, dim=-1)
    return left_hand_center_error + right_hand_center_error


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
    both_hand_center_tracking_error_penalty = RewTerm(func=both_hand_center_tracking_error_penalty, weight=-2.0)
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
