from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adjust terrain difficulty from distance traveled under velocity command."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def _expand_uniform_range(
    current: tuple[float, float], limit: tuple[float, float], delta: float, device: str | torch.device
) -> list[float]:
    expanded = torch.tensor(current, device=device) + torch.tensor([-delta, delta], device=device)
    return torch.clamp(expanded, limit[0], limit[1]).tolist()


def _expand_lower_bound(
    current: tuple[float, float], limit: tuple[float, float], delta: float, device: str | torch.device
) -> list[float]:
    lower = torch.clamp(torch.tensor(current[0], device=device) - delta, limit[0], current[1])
    upper = torch.clamp(torch.tensor(current[1], device=device), lower, limit[1])
    return torch.stack((lower, upper)).tolist()


def _expand_upper_bound(
    current: tuple[float, float], limit: tuple[float, float], delta: float, device: str | torch.device
) -> list[float]:
    lower = torch.clamp(torch.tensor(current[0], device=device), limit[0], current[1])
    upper = torch.clamp(torch.tensor(current[1], device=device) + delta, lower, limit[1])
    return torch.stack((lower, upper)).tolist()


def _episode_duration_s(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return each resetting environment's actual elapsed episode time."""
    episode_steps = torch.clamp(env.episode_length_buf[env_ids].float(), min=1.0)
    return episode_steps * env.step_dt


def _episode_survival_ratio(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    episode_steps = env.episode_length_buf[env_ids].float()
    return torch.mean(episode_steps / env.max_episode_length)


def _mean_episode_command_error(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_names: tuple[str, ...],
    error_sum_names: tuple[str, ...],
) -> torch.Tensor:
    duration_s = _episode_duration_s(env, env_ids)
    errors = []
    for command_name in command_names:
        command_term = env.command_manager.get_term(command_name)
        for error_sum_name in error_sum_names:
            error_sum = getattr(command_term, error_sum_name)
            errors.append(torch.mean(error_sum[env_ids] / duration_s))
    return torch.mean(torch.stack(errors)) if errors else torch.tensor(float("inf"), device=env.device)


def _curriculum_update_ready(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    min_episode_fraction: float,
) -> bool:
    return bool(
        env.common_step_counter % env.max_episode_length == 0
        and _episode_survival_ratio(env, env_ids) >= min_episode_fraction
    )


def pose_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_names: tuple[str, ...] = ("left_wrist_pose", "right_wrist_pose"),
    reward_term_names: tuple[str, ...] = ("track_left_wrist_pose", "track_right_wrist_pose"),
    success_threshold: float = 0.75,
    position_delta: float = 0.03,
    rotation_delta: float = 0.05,
) -> torch.Tensor:
    """Backward-compatible position-only pose-command curriculum."""
    rewards = []
    for reward_term_name in reward_term_names:
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        if reward_term.weight <= 0.0:
            continue
        episode_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        rewards.append(episode_reward / env.max_episode_length_s / reward_term.weight)

    if rewards:
        tracking_score = torch.mean(torch.stack(rewards))
    else:
        tracking_score = torch.tensor(0.0, device=env.device)

    if env.common_step_counter % env.max_episode_length == 0 and tracking_score > success_threshold:
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            ranges = command_term.cfg.ranges
            limit_ranges = command_term.cfg.limit_ranges

            ranges.pos_x = _expand_uniform_range(ranges.pos_x, limit_ranges.pos_x, position_delta, env.device)
            ranges.pos_y = _expand_uniform_range(ranges.pos_y, limit_ranges.pos_y, position_delta, env.device)
            ranges.pos_z = _expand_uniform_range(ranges.pos_z, limit_ranges.pos_z, position_delta, env.device)

    progress = []
    for command_name in command_names:
        command_term = env.command_manager.get_term(command_name)
        ranges = command_term.cfg.ranges
        limit_ranges = command_term.cfg.limit_ranges
        for range_name in ("pos_x", "pos_y", "pos_z"):
            current_range = getattr(ranges, range_name)
            limit_range = getattr(limit_ranges, range_name)
            current_width = torch.tensor(current_range[1] - current_range[0], device=env.device)
            limit_width = torch.tensor(limit_range[1] - limit_range[0], device=env.device)
            progress.append(current_width / limit_width)

    return torch.mean(torch.stack(progress)) if progress else torch.tensor(0.0, device=env.device)


def pose_position_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_names: tuple[str, ...] = ("left_wrist_pose", "right_wrist_pose"),
    penalty_term_names: tuple[str, ...] = ("penalty_left_wrist_pose_error", "penalty_right_wrist_pose_error"),
    success_threshold: float = 0.08,
    position_delta: float = 0.03,
) -> torch.Tensor:
    """Expand wrist pose-command position ranges when position tracking is reliable."""
    distances = []
    for penalty_term_name in penalty_term_names:
        penalty_term = env.reward_manager.get_term_cfg(penalty_term_name)
        if penalty_term.weight >= 0.0:
            continue
        episode_reward = torch.mean(env.reward_manager._episode_sums[penalty_term_name][env_ids])
        distance = episode_reward / env.max_episode_length_s / penalty_term.weight
        distances.append(torch.clamp(distance, min=0.0))

    tracking_error = torch.mean(torch.stack(distances)) if distances else torch.tensor(float("inf"), device=env.device)

    if env.common_step_counter % env.max_episode_length == 0 and tracking_error < success_threshold:
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            ranges = command_term.cfg.ranges
            limit_ranges = command_term.cfg.limit_ranges

            ranges.pos_x = _expand_uniform_range(ranges.pos_x, limit_ranges.pos_x, position_delta, env.device)
            ranges.pos_y = _expand_uniform_range(ranges.pos_y, limit_ranges.pos_y, position_delta, env.device)
            ranges.pos_z = _expand_uniform_range(ranges.pos_z, limit_ranges.pos_z, position_delta, env.device)

    progress = []
    for command_name in command_names:
        command_term = env.command_manager.get_term(command_name)
        ranges = command_term.cfg.ranges
        limit_ranges = command_term.cfg.limit_ranges
        for range_name in ("pos_x", "pos_y", "pos_z"):
            current_range = getattr(ranges, range_name)
            limit_range = getattr(limit_ranges, range_name)
            current_width = torch.tensor(current_range[1] - current_range[0], device=env.device)
            limit_width = torch.tensor(limit_range[1] - limit_range[0], device=env.device)
            progress.append(current_width / limit_width)

    return torch.mean(torch.stack(progress)) if progress else torch.tensor(0.0, device=env.device)


def spherical_pose_radius_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_names: tuple[str, ...] = ("left_wrist_pose", "right_wrist_pose"),
    error_sum_name: str = "position_error_sum",
    success_threshold: float = 0.08,
    radius_delta: float = 0.03,
    min_episode_fraction: float = 0.5,
) -> torch.Tensor:
    """Expand spherical wrist-command radius after tracking is reliable."""
    tracking_error = _mean_episode_command_error(env, env_ids, command_names, (error_sum_name,))

    if _curriculum_update_ready(env, env_ids, min_episode_fraction) and tracking_error < success_threshold:
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            ranges = command_term.cfg.ranges
            limit_ranges = command_term.cfg.limit_ranges
            ranges.l = _expand_uniform_range(ranges.l, limit_ranges.l, radius_delta, env.device)

    progress = []
    for command_name in command_names:
        command_term = env.command_manager.get_term(command_name)
        ranges = command_term.cfg.ranges
        limit_ranges = command_term.cfg.limit_ranges
        for range_name in ("l",):
            current_range = getattr(ranges, range_name)
            limit_range = getattr(limit_ranges, range_name)
            current_width = torch.tensor(current_range[1] - current_range[0], device=env.device)
            limit_width = torch.tensor(limit_range[1] - limit_range[0], device=env.device)
            progress.append(current_width / limit_width)

    return torch.mean(torch.stack(progress)) if progress else torch.tensor(0.0, device=env.device)


def spherical_pose_orientation_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_names: tuple[str, ...] = ("left_wrist_pose", "right_wrist_pose"),
    error_sum_name: str = "orientation_error_sum",
    success_threshold: float = 0.30,
    roll_delta: float = 0.35,
    ee_pitch_delta: float = 0.04,
    yaw_delta: float = 0.04,
    min_episode_fraction: float = 0.5,
) -> torch.Tensor:
    """Expand spherical pose-command orientation ranges after orientation tracking is reliable."""
    tracking_error = _mean_episode_command_error(env, env_ids, command_names, (error_sum_name,))

    if _curriculum_update_ready(env, env_ids, min_episode_fraction) and tracking_error < success_threshold:
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            ranges = command_term.cfg.ranges
            limit_ranges = command_term.cfg.limit_ranges
            ranges.roll = _expand_uniform_range(ranges.roll, limit_ranges.roll, roll_delta, env.device)
            ranges.ee_pitch = _expand_uniform_range(ranges.ee_pitch, limit_ranges.ee_pitch, ee_pitch_delta, env.device)
            ranges.yaw = _expand_uniform_range(ranges.yaw, limit_ranges.yaw, yaw_delta, env.device)

    progress = []
    for command_name in command_names:
        command_term = env.command_manager.get_term(command_name)
        ranges = command_term.cfg.ranges
        limit_ranges = command_term.cfg.limit_ranges
        for range_name in ("roll", "ee_pitch", "yaw"):
            current_range = getattr(ranges, range_name)
            limit_range = getattr(limit_ranges, range_name)
            current_width = torch.tensor(current_range[1] - current_range[0], device=env.device)
            limit_width = torch.tensor(limit_range[1] - limit_range[0], device=env.device)
            progress.append(current_width / limit_width)

    return torch.mean(torch.stack(progress)) if progress else torch.tensor(0.0, device=env.device)


def posture_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "posture_command",
    error_sum_names: tuple[str, ...] = ("root_height_error_sum", "torso_pitch_error_sum"),
    success_threshold: float = 0.06,
    root_height_delta: float = 0.03,
    torso_pitch_delta: float = 0.04,
    min_episode_fraction: float = 0.5,
) -> torch.Tensor:
    """Expand root-height and torso-pitch command ranges after posture tracking is reliable."""
    tracking_error = _mean_episode_command_error(env, env_ids, (command_name,), error_sum_names)

    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    if _curriculum_update_ready(env, env_ids, min_episode_fraction) and tracking_error < success_threshold:
        ranges.root_height = _expand_lower_bound(
            ranges.root_height, limit_ranges.root_height, root_height_delta, env.device
        )
        ranges.torso_pitch = _expand_upper_bound(
            ranges.torso_pitch, limit_ranges.torso_pitch, torso_pitch_delta, env.device
        )

    progress = []
    for range_name in ("root_height", "torso_pitch"):
        current_range = getattr(ranges, range_name)
        limit_range = getattr(limit_ranges, range_name)
        current_width = torch.tensor(current_range[1] - current_range[0], device=env.device)
        limit_width = torch.tensor(limit_range[1] - limit_range[0], device=env.device)
        progress.append(current_width / limit_width)

    return torch.mean(torch.stack(progress)) if progress else torch.tensor(0.0, device=env.device)
