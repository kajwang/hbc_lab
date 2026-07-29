from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import G1Dex1HierDrcEventCfg


def set_default_root_state_from_current_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Preserve the spawned USD root-link pose as the reset default."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    if env_ids.numel() == 0:
        return

    root_pose = asset.data.root_link_pose_w[env_ids].clone()
    root_pose[:, :3] -= env.scene.env_origins[env_ids]
    asset.data.default_root_state[env_ids, :7] = root_pose


@configclass
class G1Dex1DoorOpenEventCfg(G1Dex1HierDrcEventCfg):
    object_physics_material = None

    set_default_object_root_state = EventTerm(
        func=set_default_root_state_from_current_pose,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("object")},
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (2.0, 2.0),
                "y": (-0.10, 0.10),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_object_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
