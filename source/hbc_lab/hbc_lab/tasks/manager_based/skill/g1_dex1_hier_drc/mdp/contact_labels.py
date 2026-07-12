from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


LEFT_HAND = 0
RIGHT_HAND = 1
NUM_HAND_EFFECTORS = 2


class ContactMode(IntEnum):
    """Semantic contact mode supplied by the task-level contact label."""

    INNER_PAD_GRASP = 0
    BIMANUAL_BOX_SUPPORT = 1


def active_hand_to_effector_mask(active_hand: torch.Tensor) -> torch.Tensor:
    """Convert a single-hand label to a future-compatible [left, right] mask."""
    active_hand = active_hand.reshape(-1).to(dtype=torch.long)
    valid = (active_hand == LEFT_HAND) | (active_hand == RIGHT_HAND)
    if not torch.all(valid):
        raise ValueError("active_hand must contain only LEFT_HAND=0 or RIGHT_HAND=1")
    return torch.nn.functional.one_hot(active_hand, num_classes=NUM_HAND_EFFECTORS).to(dtype=torch.float32)


@dataclass
class ContactLabelCommand:
    """Embodiment-facing contact command shared by the task and policy observations."""

    effector_mask: torch.Tensor
    mode: torch.Tensor

    @classmethod
    def from_active_hand(cls, active_hand: torch.Tensor, mode: ContactMode | int) -> "ContactLabelCommand":
        effector_mask = active_hand_to_effector_mask(active_hand)
        mode_tensor = torch.full(
            (effector_mask.shape[0],),
            int(mode),
            dtype=torch.long,
            device=effector_mask.device,
        )
        return cls(effector_mask=effector_mask, mode=mode_tensor)

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

    def set_mode(self, env_ids: torch.Tensor, mode: ContactMode | int) -> None:
        env_ids = env_ids.reshape(-1).to(device=self.mode.device, dtype=torch.long)
        self.mode[env_ids] = int(mode)
