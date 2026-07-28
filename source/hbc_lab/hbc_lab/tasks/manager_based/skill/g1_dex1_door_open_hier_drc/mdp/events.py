from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import G1Dex1HierDrcEventCfg


def reset_door_articulation(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    door = env.scene[asset_cfg.name]
    root_state = door.data.default_root_state[env_ids].clone()
    position_ranges = torch.tensor(
        [pose_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device=door.device,
    )
    position_offset = math_utils.sample_uniform(
        position_ranges[:, 0],
        position_ranges[:, 1],
        (env_ids.numel(), 3),
        door.device,
    )
    root_state[:, :3] += env.scene.env_origins[env_ids] + position_offset
    root_state[:, 7:] = 0.0
    door.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    door.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    joint_ids, joint_names = door.find_joints(["joint_1", "joint_2"], preserve_order=True)
    if len(joint_ids) != 2:
        raise RuntimeError(f"Door requires joint_1 and joint_2, resolved {joint_names}")
    joint_pos = torch.zeros(env_ids.numel(), 2, device=door.device)
    joint_vel = torch.zeros_like(joint_pos)
    door.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    door.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)


@configclass
class G1Dex1DoorOpenEventCfg(G1Dex1HierDrcEventCfg):
    object_physics_material = None
    reset_object = EventTerm(
        func=reset_door_articulation,
        mode="reset",
        params={
            "pose_range": {
                "x": (2.0, 2.0),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
