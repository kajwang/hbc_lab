from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.observations import (
    execution_state_obs,
    grip_obs,
    hand_center_positions_w,
    high_level_command_state_obs,
    last_high_level_action,
    privileged_drc_obs,
)
from hbc_lab.tasks.manager_based.skill.pose_motion import gather_keyframe


def _rotation_6d_in_root(root_quat_w: torch.Tensor, frame_quat_w: torch.Tensor) -> torch.Tensor:
    frame_quat_b = quat_mul(quat_inv(root_quat_w), frame_quat_w)
    return matrix_from_quat(frame_quat_b)[:, :, :2].transpose(1, 2).reshape(-1, 6)


def interaction_motion_obs(env) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    env._update_contact_target_regions()

    current_pos_w = env._selected_handle_pos_w()
    current_quat_w = env._selected_handle_quat_w()
    target_pos_w = gather_keyframe(env.motion_target_pos_w, env.motion_keyframe_index)
    target_quat_w = gather_keyframe(env.motion_target_quat_w, env.motion_keyframe_index)
    position_mask = gather_keyframe(env.motion_position_mask, env.motion_keyframe_index)
    rotation_mask = gather_keyframe(env.motion_rotation_mask, env.motion_keyframe_index)

    current_pos_b = quat_apply_inverse(robot.data.root_quat_w, current_pos_w - robot.data.root_pos_w)
    target_pos_b = quat_apply_inverse(robot.data.root_quat_w, target_pos_w - robot.data.root_pos_w)
    current_rot_6d = _rotation_6d_in_root(robot.data.root_quat_w, current_quat_w)
    target_rot_6d = _rotation_6d_in_root(robot.data.root_quat_w, target_quat_w)

    left_pos_w, right_pos_w = hand_center_positions_w(env)
    left_pos_b = quat_apply_inverse(robot.data.root_quat_w, left_pos_w - robot.data.root_pos_w)
    right_pos_b = quat_apply_inverse(robot.data.root_quat_w, right_pos_w - robot.data.root_pos_w)
    effector_mask = env.contact_label.effector_mask.to(dtype=current_pos_b.dtype)

    return torch.cat(
        (
            current_pos_b,
            current_rot_6d,
            target_pos_b,
            target_rot_6d,
            left_pos_b,
            right_pos_b,
            effector_mask,
            position_mask,
            rotation_mask,
        ),
        dim=-1,
    )


@configclass
class G1Dex1DrawerObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        task = ObsTerm(func=interaction_motion_obs)
        grip = ObsTerm(func=grip_obs, noise=Unoise(n_min=-0.01, n_max=0.01))
        command_state = ObsTerm(func=high_level_command_state_obs)
        execution = ObsTerm(func=execution_state_obs, params={"sensor_noise": True})
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
        task = ObsTerm(func=interaction_motion_obs)
        grip = ObsTerm(func=grip_obs)
        command_state = ObsTerm(func=high_level_command_state_obs)
        execution = ObsTerm(func=execution_state_obs)
        last_high_action = ObsTerm(func=last_high_level_action)
        drc = ObsTerm(func=privileged_drc_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
