from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from hbc_lab.assets.robots.unitree import G1_DEX1_LEFT_GRIPPER_JOINT_NAMES, G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES


HandSide = Literal["left", "right"]

LEFT_DEX1_JOINT_PATTERNS = tuple(G1_DEX1_LEFT_GRIPPER_JOINT_NAMES)
RIGHT_DEX1_JOINT_PATTERNS = tuple(G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES)
DEX1_OPEN_POSITION = 0.047
DEX1_CLOSE_POSITION = 0.0


def _grip_column(grip: torch.Tensor) -> torch.Tensor:
    grip = torch.as_tensor(grip)
    if grip.ndim == 0:
        grip = grip.reshape(1, 1)
    elif grip.ndim == 1:
        grip = grip.unsqueeze(-1)
    elif grip.ndim != 2 or grip.shape[-1] != 1:
        raise ValueError(f"Expected grip shape (), (N,), or (N, 1), got {tuple(grip.shape)}")
    return torch.clamp(grip, 0.0, 1.0)


def interpolate_dex1_gripper_position(
    grip: torch.Tensor,
    open_position: float = DEX1_OPEN_POSITION,
    close_position: float = DEX1_CLOSE_POSITION,
) -> torch.Tensor:
    """Map grip=0 open and grip=1 closed to a scalar gripper joint target."""
    grip = _grip_column(grip)
    return torch.full_like(grip, open_position) + (close_position - open_position) * grip


@dataclass
class Dex1GripperCommand:
    left_grip: torch.Tensor
    right_grip: torch.Tensor


class Dex1GripperController:
    """Position-target controller for Unitree Dex1 parallel grippers."""

    def __init__(self, robot, device: str | torch.device):
        self.robot = robot
        self.device = torch.device(device)
        self.left_joint_ids = self._find_joint_ids(LEFT_DEX1_JOINT_PATTERNS, side="left")
        self.right_joint_ids = self._find_joint_ids(RIGHT_DEX1_JOINT_PATTERNS, side="right")

    def _find_joint_ids(self, patterns: tuple[str, ...], side: HandSide) -> list[int]:
        seen: set[int] = set()
        joint_ids: list[int] = []
        for pattern in patterns:
            ids, _names = self.robot.find_joints(pattern, preserve_order=True)
            for joint_id in ids:
                joint_id = int(joint_id)
                if joint_id not in seen:
                    seen.add(joint_id)
                    joint_ids.append(joint_id)
        if not joint_ids:
            raise RuntimeError(f"Could not find any Dex1 {side} gripper joints from patterns: {patterns}")
        return joint_ids

    def _target(self, grip: torch.Tensor, count: int) -> torch.Tensor:
        target = interpolate_dex1_gripper_position(grip).to(device=self.device)
        return target.expand(-1, count).contiguous()

    def apply(self, left_grip: torch.Tensor, right_grip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_target = self._target(left_grip, len(self.left_joint_ids))
        right_target = self._target(right_grip, len(self.right_joint_ids))
        self.robot.set_joint_position_target(left_target, joint_ids=self.left_joint_ids)
        self.robot.set_joint_position_target(right_target, joint_ids=self.right_joint_ids)
        return left_target, right_target
