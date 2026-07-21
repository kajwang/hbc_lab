from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hbc_lab.tasks.locomotion import mdp

from .scenes import HAND_CENTER_FRAME_NAME


def hand_center_positions_w(env) -> tuple[torch.Tensor, torch.Tensor]:
    hand_center_pos_w = env.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
    return hand_center_pos_w[:, 0, :], hand_center_pos_w[:, 1, :]


def object_goal_hand_obs(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    env._update_contact_target_regions()
    goal_pos_w = env.object_target_pos_w
    left_pos_w, right_pos_w = hand_center_positions_w(env)
    goal_pos_b = quat_apply_inverse(robot.data.root_quat_w, goal_pos_w - robot.data.root_pos_w)
    left_pos_b = quat_apply_inverse(robot.data.root_quat_w, left_pos_w - robot.data.root_pos_w)
    right_pos_b = quat_apply_inverse(robot.data.root_quat_w, right_pos_w - robot.data.root_pos_w)
    target_region_w = env.contact_label.target_region
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, target_region_w.shape[1], -1)
    target_region_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        (target_region_w - robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, -1, 3)
    effector_mask = env.contact_label.effector_mask.to(dtype=goal_pos_b.dtype)
    target_region_b = target_region_b * effector_mask.unsqueeze(-1)
    return torch.cat((goal_pos_b, left_pos_b, right_pos_b, target_region_b.flatten(1), effector_mask), dim=-1)


def grip_obs(env) -> torch.Tensor:
    if not hasattr(env, "command_state"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return torch.cat((env.command_state.left_grip, env.command_state.right_grip), dim=-1)


def high_level_command_state_obs(env) -> torch.Tensor:
    if not hasattr(env, "command_state"):
        return torch.zeros(env.num_envs, 23, device=env.device)

    command = env.command_state
    left_rot_6d = (
        matrix_from_quat(command.left_wrist_pose_b[:, 3:])[:, :, :2].transpose(1, 2).reshape(-1, 6)
    )
    right_rot_6d = (
        matrix_from_quat(command.right_wrist_pose_b[:, 3:])[:, :, :2].transpose(1, 2).reshape(-1, 6)
    )
    return torch.cat(
        (
            command.base_velocity,
            command.posture_command,
            command.left_wrist_pose_b[:, :3],
            left_rot_6d,
            command.right_wrist_pose_b[:, :3],
            right_rot_6d,
        ),
        dim=-1,
    )


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
class G1Dex1HierDrcObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        task = ObsTerm(func=object_goal_hand_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        grip = ObsTerm(func=grip_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        command_state = ObsTerm(func=high_level_command_state_obs)
        last_high_action = ObsTerm(func=last_high_level_action)

        def __post_init__(self):
            self.history_length = 10
            self.flatten_history_dim = True
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        task = ObsTerm(func=object_goal_hand_obs)
        grip = ObsTerm(func=grip_obs)
        command_state = ObsTerm(func=high_level_command_state_obs)
        last_high_action = ObsTerm(func=last_high_level_action)
        drc = ObsTerm(func=privileged_drc_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
