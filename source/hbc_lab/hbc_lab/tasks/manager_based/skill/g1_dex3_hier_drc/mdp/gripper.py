from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


HandSide = Literal["left", "right"]

LEFT_DEX3_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
RIGHT_DEX3_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)

LEFT_DEX3_OPEN_POSE = torch.zeros(7, dtype=torch.float32)
RIGHT_DEX3_OPEN_POSE = torch.zeros(7, dtype=torch.float32)

# OASIS-style hand synergy. grip = 1.0 means fully closed.
LEFT_DEX3_CLOSE_POSE = torch.tensor([0.0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74], dtype=torch.float32)
RIGHT_DEX3_CLOSE_POSE = torch.tensor([0.0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74], dtype=torch.float32)


def _grip_column(grip: torch.Tensor) -> torch.Tensor:
    grip = torch.as_tensor(grip)
    if grip.ndim == 0:
        grip = grip.reshape(1, 1)
    elif grip.ndim == 1:
        grip = grip.unsqueeze(-1)
    elif grip.ndim != 2 or grip.shape[-1] != 1:
        raise ValueError(f"Expected grip shape (), (N,), or (N, 1), got {tuple(grip.shape)}")
    return torch.clamp(grip, 0.0, 1.0)


def _pose_pair(side: HandSide, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if side == "left":
        return LEFT_DEX3_OPEN_POSE.to(device=device, dtype=dtype), LEFT_DEX3_CLOSE_POSE.to(device=device, dtype=dtype)
    if side == "right":
        return RIGHT_DEX3_OPEN_POSE.to(device=device, dtype=dtype), RIGHT_DEX3_CLOSE_POSE.to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported Dex3 hand side: {side!r}")


def interpolate_dex3_hand_pose(grip: torch.Tensor, side: HandSide) -> torch.Tensor:
    """Map a scalar Dex3 grip command to seven hand joint targets.

    The command follows OASIS: grip = 0.0 is fully open, grip = 1.0 means fully closed.
    """
    grip = _grip_column(grip)
    open_pose, close_pose = _pose_pair(side, grip.device, grip.dtype)
    return open_pose.unsqueeze(0) + (close_pose - open_pose).unsqueeze(0) * grip


def build_dex3_hand_targets(left_grip: torch.Tensor, right_grip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left and right hand joint targets in Dex3 local joint order."""
    left_target = interpolate_dex3_hand_pose(left_grip, side="left")
    right_target = interpolate_dex3_hand_pose(right_grip, side="right")
    if left_target.shape[0] != right_target.shape[0]:
        raise ValueError(
            f"Left and right grip batch sizes must match, got {left_target.shape[0]} and {right_target.shape[0]}"
        )
    return left_target, right_target


@dataclass
class Dex3GripperCommand:
    """Continuous left/right hand synergy command."""

    left_grip: torch.Tensor
    right_grip: torch.Tensor

    @property
    def targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        return build_dex3_hand_targets(self.left_grip, self.right_grip)


class Dex3GripperController:
    """Small position-target controller for OASIS-style G1 Dex3 hands."""

    def __init__(self, robot, device: str | torch.device):
        self.robot = robot
        self.device = torch.device(device)
        self.left_joint_ids = self._find_joint_ids(LEFT_DEX3_JOINT_NAMES)
        self.right_joint_ids = self._find_joint_ids(RIGHT_DEX3_JOINT_NAMES)

    def _find_joint_ids(self, joint_names: tuple[str, ...]) -> list[int]:
        joint_ids, resolved_joint_names = self.robot.find_joints(list(joint_names), preserve_order=True)
        if tuple(resolved_joint_names) != joint_names:
            raise RuntimeError(f"Resolved Dex3 joint order does not match requested order: {resolved_joint_names}")
        return list(joint_ids)

    def apply(self, left_grip: torch.Tensor, right_grip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_target, right_target = build_dex3_hand_targets(left_grip, right_grip)
        left_target = left_target.to(device=self.device)
        right_target = right_target.to(device=self.device)
        self.robot.set_joint_position_target(left_target, joint_ids=self.left_joint_ids)
        self.robot.set_joint_position_target(right_target, joint_ids=self.right_joint_ids)
        return left_target, right_target
