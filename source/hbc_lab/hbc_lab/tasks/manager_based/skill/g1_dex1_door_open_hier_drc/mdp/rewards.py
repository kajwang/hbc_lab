from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.rewards import (
    approach_reward,
    contact_conditioned_motion_reward,
    couple_reward,
    root_object_facing_reward,
    task_success_reward,
)
from hbc_lab.tasks.manager_based.skill.pose_motion import generic_pose_manip_reward


def motion_manip_reward(env) -> torch.Tensor:
    reward = generic_pose_manip_reward(env.motion_progress)
    env.extras["log"]["Motion/progress_reward_mean"] = reward.mean()
    return reward


def approach_root_object_facing_reward(env) -> torch.Tensor:
    return env.W_app * root_object_facing_reward(env)


def hier_drc_reward(
    env,
    approach_scale: float = 2.0,
    couple_scale: float = 20.0,
    manip_scale: float = 100.0,
) -> torch.Tensor:
    r_app = approach_reward(env)
    r_couple = couple_reward(env)
    r_manip = motion_manip_reward(env)
    env.extras["log"]["DRC/R_app_raw"] = r_app.mean()
    env.extras["log"]["DRC/R_couple_raw"] = r_couple.mean()
    env.extras["log"]["DRC/R_manip_raw"] = r_manip.mean()
    return (
        env.W_app * approach_scale * r_app
        + env.W_couple * couple_scale * r_couple
        + env.W_manip * manip_scale * r_manip
    )


def inactive_hand_contact_penalty(env) -> torch.Tensor:
    return env.inactive_hand_contact


@configclass
class G1Dex1DoorOpenRewardsCfg:
    drc_total = RewTerm(func=hier_drc_reward, weight=1.0)
    motion_quality = RewTerm(
        func=contact_conditioned_motion_reward,
        weight=1.0,
        params={"approach_scale": 2.0, "couple_scale": 20.0, "manip_scale": 100.0},
    )
    task_success = RewTerm(func=task_success_reward, weight=1.0)
    inactive_hand_contact = RewTerm(func=inactive_hand_contact_penalty, weight=-2.0)
    root_object_facing = RewTerm(func=approach_root_object_facing_reward, weight=2.0)
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*|.*gripper.*|.*finger.*|.*hand.*).*"],
            ),
        },
    )
