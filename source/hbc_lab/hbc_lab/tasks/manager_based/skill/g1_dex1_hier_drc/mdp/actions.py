from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class HighLevelAction(ActionTerm):
    cfg: "HighLevelActionCfg"

    def __init__(self, cfg: "HighLevelActionCfg", env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, cfg.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)

    def apply_actions(self):
        return


@configclass
class HighLevelActionCfg(ActionTermCfg):
    class_type: type = HighLevelAction
    action_dim: int = MISSING


@configclass
class G1Dex1HierDrcActionsCfg:
    high_level = HighLevelActionCfg(asset_name="robot", action_dim=19)
