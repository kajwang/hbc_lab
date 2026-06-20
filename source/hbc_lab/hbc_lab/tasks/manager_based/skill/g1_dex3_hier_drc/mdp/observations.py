from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hbc_lab.tasks.locomotion import mdp


def _resolve_wrist_body_ids(env) -> tuple[int, int]:
    if not hasattr(env, "left_wrist_body_id") or not hasattr(env, "right_wrist_body_id"):
        robot: Articulation = env.scene["robot"]
        env.left_wrist_body_id = robot.find_bodies("left_wrist_yaw_link")[0][0]
        env.right_wrist_body_id = robot.find_bodies("right_wrist_yaw_link")[0][0]
    return env.left_wrist_body_id, env.right_wrist_body_id


def object_goal_hand_obs(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    left_wrist_body_id, right_wrist_body_id = _resolve_wrist_body_ids(env)
    object_pos_w = obj.data.root_pos_w
    goal_pos_w = env.object_target_pos_w
    left_pos_w = robot.data.body_pos_w[:, left_wrist_body_id]
    right_pos_w = robot.data.body_pos_w[:, right_wrist_body_id]
    object_pos_b = quat_apply_inverse(robot.data.root_quat_w, object_pos_w - robot.data.root_pos_w)
    goal_pos_b = quat_apply_inverse(robot.data.root_quat_w, goal_pos_w - robot.data.root_pos_w)
    left_pos_b = quat_apply_inverse(robot.data.root_quat_w, left_pos_w - robot.data.root_pos_w)
    right_pos_b = quat_apply_inverse(robot.data.root_quat_w, right_pos_w - robot.data.root_pos_w)
    active_hand = torch.nn.functional.one_hot(env.active_hand, num_classes=2).to(dtype=object_pos_b.dtype)
    return torch.cat((object_pos_b, goal_pos_b, left_pos_b, right_pos_b, active_hand), dim=-1)


def grip_obs(env) -> torch.Tensor:
    if not hasattr(env, "command_state"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return torch.cat((env.command_state.left_grip, env.command_state.right_grip), dim=-1)


def last_high_level_action(env) -> torch.Tensor:
    return env.last_high_level_action


def privileged_drc_obs(env) -> torch.Tensor:
    return torch.stack(
        (
            env.d_active_hand,
            env.c_contact,
            env.c_couple,
            env.d_goal,
            env.W_app,
            env.W_couple,
            env.W_manip,
        ),
        dim=-1,
    )


@configclass
class G1Dex3HierDrcObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        task = ObsTerm(func=object_goal_hand_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        grip = ObsTerm(func=grip_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_high_action = ObsTerm(func=last_high_level_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        task = ObsTerm(func=object_goal_hand_obs)
        grip = ObsTerm(func=grip_obs)
        last_high_action = ObsTerm(func=last_high_level_action)
        drc = ObsTerm(func=privileged_drc_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
