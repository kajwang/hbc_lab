from __future__ import annotations

import torch
from typing import Literal
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, euler_xyz_from_quat, subtract_frame_transforms

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


def frame_transformer_pose_in_root_frame(
    env: ManagerBasedRLEnv,
    frame_sensor_name: str,
    frame_index: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """Return a FrameTransformer target frame pose expressed in the robot root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    frame_sensor = env.scene[frame_sensor_name]

    frame_pos_w = frame_sensor.data.target_pos_w[:, frame_index]
    frame_quat_w = frame_sensor.data.target_quat_w[:, frame_index]
    frame_pos_b, frame_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        frame_pos_w,
        frame_quat_w,
    )

    if return_key == "pos":
        return frame_pos_b
    if return_key == "quat":
        return frame_quat_b
    return torch.cat((frame_pos_b, frame_quat_b), dim=-1)


def body_pose_command_position_error_in_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root-frame target minus current body position for a pose command."""
    target_pos_b = env.command_manager.get_command(command_name)[:, :3]
    current_pos_b = body_pose_in_root_frame(env, asset_cfg=asset_cfg, return_key="pos")[:, :3]
    return target_pos_b - current_pos_b


def frame_transformer_pose_command_position_error_w_in_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    frame_sensor_name: str,
    frame_index: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root-frame command target minus current FrameTransformer target-frame position."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w = _body_pose_command_target_pos_w(env, command_name, asset)
    target_pos_b, _ = subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_w)
    current_pos_b = frame_transformer_pose_in_root_frame(
        env,
        frame_sensor_name=frame_sensor_name,
        frame_index=frame_index,
        asset_cfg=asset_cfg,
        return_key="pos",
    )
    return target_pos_b - current_pos_b


def _body_pose_command_target_pos_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset: Articulation,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "_update_pose_command_w"):
        command_term._update_pose_command_w()
    if hasattr(command_term, "pose_command_w"):
        return command_term.pose_command_w[:, :3]

    command = env.command_manager.get_command(command_name)
    target_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3])
    return target_pos_w


def body_pose_command_position_error_w_in_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root-frame position error to a command term's world-frame target."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_w = _body_pose_command_target_pos_w(env, command_name, asset)
    target_pos_b, _ = subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_w)
    current_pos_b = body_pose_in_root_frame(env, asset_cfg=asset_cfg, return_key="pos")[:, :3]
    return target_pos_b - current_pos_b


def current_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Return current [root_height, torso_pitch]."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    _, torso_pitch, _ = euler_xyz_from_quat(asset.data.body_quat_w[:, asset_cfg.body_ids[0]])
    return torch.stack((root_height, torso_pitch), dim=-1)


def posture_command_error(
    env: ManagerBasedRLEnv,
    command_name: str = "posture_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Return posture command minus current posture."""
    return env.command_manager.get_command(command_name) - current_posture(env, asset_cfg)
