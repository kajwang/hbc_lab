from __future__ import annotations

import torch
from isaaclab.utils.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    yaw_quat,
)


def posture_anchor_pose_w(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    shoulder_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    posture_command: torch.Tensor,
    anchor_height_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a shoulder anchor from commanded root height and torso pitch."""
    root_yaw_quat = yaw_quat(root_quat_w)
    zeros = torch.zeros_like(posture_command[:, 1])
    pitch_quat = quat_from_euler_xyz(zeros, posture_command[:, 1], zeros)
    anchor_quat_w = quat_mul(root_yaw_quat, pitch_quat)

    root_cmd_pos_w = root_pos_w.clone()
    root_cmd_pos_w[:, 2] = env_origins[:, 2] + posture_command[:, 0]

    root_to_shoulder_w = shoulder_pos_w - root_pos_w
    root_to_shoulder_yaw = quat_apply(quat_inv(root_yaw_quat), root_to_shoulder_w)
    anchor_offset = torch.zeros_like(root_to_shoulder_yaw)
    anchor_offset[:, 1] = root_to_shoulder_yaw[:, 1]
    anchor_offset[:, 2] = anchor_height_offset
    anchor_pos_w = root_cmd_pos_w + quat_apply(anchor_quat_w, anchor_offset)
    return anchor_pos_w, anchor_quat_w


def compose_wrist_chain_quat(wrist_angles: torch.Tensor, fixed_quat: torch.Tensor) -> torch.Tensor:
    """Compose wrist roll, pitch, yaw, and the fixed palm joint in asset-chain order."""
    zeros = torch.zeros_like(wrist_angles[:, 0])
    roll_quat = quat_from_euler_xyz(wrist_angles[:, 0], zeros, zeros)
    pitch_quat = quat_from_euler_xyz(zeros, wrist_angles[:, 1], zeros)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, wrist_angles[:, 2])
    if fixed_quat.ndim == 1:
        fixed_quat = fixed_quat.unsqueeze(0).expand_as(roll_quat)
    return quat_mul(quat_mul(quat_mul(roll_quat, pitch_quat), yaw_quat), fixed_quat)


def hand_base_to_hand_center_pose(
    hand_base_pos: torch.Tensor,
    hand_base_quat: torch.Tensor,
    hand_center_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the fixed hand-base offset and return the final hand-center pose."""
    if hand_center_offset.ndim == 1:
        hand_center_offset = hand_center_offset.unsqueeze(0).expand_as(hand_base_pos)
    hand_center_pos = hand_base_pos + quat_apply(hand_base_quat, hand_center_offset)
    return hand_center_pos, hand_base_quat
