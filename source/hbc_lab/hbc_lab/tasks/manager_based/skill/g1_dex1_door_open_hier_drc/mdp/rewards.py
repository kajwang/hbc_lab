from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.rewards import (
    approach_reward,
    both_hand_center_tracking_error_penalty,
    command_smoothness,
    couple_reward,
    root_object_facing_reward,
    task_success_reward,
)


def door_manip_reward(env) -> torch.Tensor:
    return env.door_manipulation_progress.reward


def hier_drc_reward(
    env,
    approach_scale: float = 2.0,
    couple_scale: float = 20.0,
    manip_scale: float = 100.0,
) -> torch.Tensor:
    r_app = approach_reward(env)
    r_couple = couple_reward(env)
    r_manip = door_manip_reward(env)
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    return (
        env.W_app * approach_scale * r_app
        + env.W_couple * couple_scale * r_couple
        + env.W_manip * manip_scale * r_manip
    )


def inactive_left_contact_penalty(env) -> torch.Tensor:
    return env.inactive_left_contact


@configclass
class G1Dex1DoorOpenRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    inactive_left_contact = RewTerm(func=inactive_left_contact_penalty, weight=-2.0)
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
