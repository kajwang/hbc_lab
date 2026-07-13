from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.observations import object_goal_hand_obs


def box_object_goal_hand_obs(env) -> torch.Tensor:
    """Append the assigned opposing-face targets to the existing task observation."""
    base_task_obs = object_goal_hand_obs(env)
    env._update_face_targets()
    robot: Articulation = env.scene["robot"]
    left_target_b = quat_apply_inverse(
        robot.data.root_quat_w,
        env.left_face_target_pos_w - robot.data.root_pos_w,
    )
    right_target_b = quat_apply_inverse(
        robot.data.root_quat_w,
        env.right_face_target_pos_w - robot.data.root_pos_w,
    )
    return torch.cat((base_task_obs, left_target_b, right_target_b), dim=-1)
