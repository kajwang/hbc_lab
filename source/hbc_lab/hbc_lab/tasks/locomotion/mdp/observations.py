from __future__ import annotations

import torch
from typing import Literal
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def _resolve_body_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    if asset_cfg.body_ids is None:
        return list(range(asset.data.body_pos_w.shape[1]))
    if isinstance(asset_cfg.body_ids, int):
        return [asset_cfg.body_ids]
    return list(asset_cfg.body_ids)


def body_pose_in_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """Return selected body poses expressed in the robot root frame.

    This is proprioceptive: on hardware the same signal can be reconstructed from
    joint encoders and the robot kinematic model, without global localization.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = _resolve_body_ids(asset, asset_cfg)
    num_bodies = len(body_ids)

    body_pos_w = asset.data.body_pos_w[:, body_ids].reshape(-1, 3)
    body_quat_w = asset.data.body_quat_w[:, body_ids].reshape(-1, 4)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 3)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)

    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_pos_b = body_pos_b.reshape(env.num_envs, num_bodies, 3)
    body_quat_b = body_quat_b.reshape(env.num_envs, num_bodies, 4)

    if return_key == "pos":
        return body_pos_b.reshape(env.num_envs, -1)
    if return_key == "quat":
        return body_quat_b.reshape(env.num_envs, -1)
    return torch.cat((body_pos_b, body_quat_b), dim=-1).reshape(env.num_envs, -1)


def body_pose_command_position_error_in_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root-frame target minus current body position for a pose command."""
    target_pos_b = env.command_manager.get_command(command_name)[:, :3]
    current_pos_b = body_pose_in_root_frame(env, asset_cfg=asset_cfg, return_key="pos")[:, :3]
    return target_pos_b - current_pos_b
