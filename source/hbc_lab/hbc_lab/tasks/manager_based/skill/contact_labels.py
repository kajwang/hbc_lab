from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


LEFT_HAND = 0
RIGHT_HAND = 1
NUM_HAND_EFFECTORS = 2
TARGET_REGION_DIM = 3
TARGET_ORIENTATION_DIM = 4


class ContactMode(IntEnum):
    GRASP = 0
    SUPPORT = 1


def active_hand_to_effector_mask(active_hand: torch.Tensor) -> torch.Tensor:
    active_hand = active_hand.reshape(-1).to(dtype=torch.long)
    valid = (active_hand == LEFT_HAND) | (active_hand == RIGHT_HAND)
    if not torch.all(valid):
        raise ValueError("active_hand must contain only LEFT_HAND=0 or RIGHT_HAND=1")
    return torch.nn.functional.one_hot(active_hand, num_classes=NUM_HAND_EFFECTORS).to(dtype=torch.float32)


@dataclass
class ContactLabel:
    """Minimal task contact label shared across interaction environments."""

    effector_mask: torch.Tensor
    target_region: torch.Tensor
    target_orientation: torch.Tensor
    contact_mode: torch.Tensor

    @classmethod
    def from_active_hand(
        cls,
        active_hand: torch.Tensor,
        contact_mode: ContactMode | int,
    ) -> "ContactLabel":
        effector_mask = active_hand_to_effector_mask(active_hand)
        num_envs = effector_mask.shape[0]
        target_region = torch.zeros(
            num_envs,
            NUM_HAND_EFFECTORS,
            TARGET_REGION_DIM,
            dtype=effector_mask.dtype,
            device=effector_mask.device,
        )
        target_orientation = torch.zeros(
            num_envs,
            NUM_HAND_EFFECTORS,
            TARGET_ORIENTATION_DIM,
            dtype=effector_mask.dtype,
            device=effector_mask.device,
        )
        target_orientation[..., 0] = 1.0
        mode_tensor = torch.full(
            (num_envs,),
            int(contact_mode),
            dtype=torch.long,
            device=effector_mask.device,
        )
        return cls(
            effector_mask=effector_mask,
            target_region=target_region,
            target_orientation=target_orientation,
            contact_mode=mode_tensor,
        )

    def set_single_active_hand(self, env_ids: torch.Tensor, active_hand: torch.Tensor) -> None:
        self.set_effector_mask(env_ids, active_hand_to_effector_mask(active_hand))

    def set_effector_mask(self, env_ids: torch.Tensor, effector_mask: torch.Tensor) -> None:
        env_ids = env_ids.reshape(-1).to(device=self.effector_mask.device, dtype=torch.long)
        effector_mask = effector_mask.to(device=self.effector_mask.device, dtype=self.effector_mask.dtype)
        if effector_mask.shape != (env_ids.numel(), NUM_HAND_EFFECTORS):
            raise ValueError(
                f"effector_mask must have shape ({env_ids.numel()}, {NUM_HAND_EFFECTORS}), "
                f"got {tuple(effector_mask.shape)}"
            )
        if not torch.all((effector_mask == 0.0) | (effector_mask == 1.0)):
            raise ValueError("effector_mask must be binary")
        if torch.any(effector_mask.sum(dim=-1) < 1.0):
            raise ValueError("contact label must select at least one effector")
        self.effector_mask[env_ids] = effector_mask
        self.target_region[env_ids] *= effector_mask.unsqueeze(-1)
        active = effector_mask.to(dtype=torch.bool).unsqueeze(-1)
        identity = torch.zeros_like(self.target_orientation[env_ids])
        identity[..., 0] = 1.0
        self.target_orientation[env_ids] = torch.where(
            active, self.target_orientation[env_ids], identity
        )

    def set_target_region(self, env_ids: torch.Tensor, target_region: torch.Tensor) -> None:
        env_ids = env_ids.reshape(-1).to(device=self.target_region.device, dtype=torch.long)
        target_region = target_region.to(device=self.target_region.device, dtype=self.target_region.dtype)
        expected_shape = (env_ids.numel(), NUM_HAND_EFFECTORS, TARGET_REGION_DIM)
        if target_region.shape != expected_shape:
            raise ValueError(f"target_region must have shape {expected_shape}, got {tuple(target_region.shape)}")
        self.target_region[env_ids] = target_region * self.effector_mask[env_ids].unsqueeze(-1)

    def set_target_region_pose(
        self,
        env_ids: torch.Tensor,
        target_region: torch.Tensor,
        target_orientation: torch.Tensor,
    ) -> None:
        env_ids = env_ids.reshape(-1).to(device=self.target_region.device, dtype=torch.long)
        target_region = target_region.to(device=self.target_region.device, dtype=self.target_region.dtype)
        target_orientation = target_orientation.to(
            device=self.target_orientation.device,
            dtype=self.target_orientation.dtype,
        )
        expected_position_shape = (env_ids.numel(), NUM_HAND_EFFECTORS, TARGET_REGION_DIM)
        expected_orientation_shape = (env_ids.numel(), NUM_HAND_EFFECTORS, TARGET_ORIENTATION_DIM)
        if target_region.shape != expected_position_shape:
            raise ValueError(
                f"target_region must have shape {expected_position_shape}, got {tuple(target_region.shape)}"
            )
        if target_orientation.shape != expected_orientation_shape:
            raise ValueError(
                "target_orientation must have shape "
                f"{expected_orientation_shape}, got {tuple(target_orientation.shape)}"
            )
        target_orientation = torch.nn.functional.normalize(target_orientation, dim=-1)
        active = self.effector_mask[env_ids].to(dtype=torch.bool).unsqueeze(-1)
        identity = torch.zeros_like(target_orientation)
        identity[..., 0] = 1.0
        self.target_region[env_ids] = target_region * active.to(dtype=target_region.dtype)
        self.target_orientation[env_ids] = torch.where(active, target_orientation, identity)

    def clear_target_region(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.reshape(-1).to(device=self.target_region.device, dtype=torch.long)
        self.target_region[env_ids] = 0.0
        self.target_orientation[env_ids] = 0.0
        self.target_orientation[env_ids, :, 0] = 1.0

    def set_contact_mode(self, env_ids: torch.Tensor, contact_mode: ContactMode | int) -> None:
        env_ids = env_ids.reshape(-1).to(device=self.contact_mode.device, dtype=torch.long)
        self.contact_mode[env_ids] = int(contact_mode)
