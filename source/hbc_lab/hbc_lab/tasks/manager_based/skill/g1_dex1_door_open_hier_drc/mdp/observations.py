from __future__ import annotations

import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.observations import (
    G1Dex1HierDrcObservationsCfg,
)


def door_articulation_state(env) -> torch.Tensor:
    return torch.stack(
        (
            env.handle_angle / env.cfg.door_latch_handle_threshold,
            env.hinge_angle / env.cfg.door_hinge_target,
        ),
        dim=-1,
    )


@configclass
class G1Dex1DoorOpenObservationsCfg(G1Dex1HierDrcObservationsCfg):
    @configclass
    class PolicyCfg(G1Dex1HierDrcObservationsCfg.PolicyCfg):
        door_state = ObsTerm(func=door_articulation_state)

    @configclass
    class CriticCfg(G1Dex1HierDrcObservationsCfg.CriticCfg):
        door_state = ObsTerm(func=door_articulation_state)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
