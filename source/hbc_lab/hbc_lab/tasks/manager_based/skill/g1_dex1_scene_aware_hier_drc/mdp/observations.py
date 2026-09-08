from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.observations import (
    G1Dex1HierDrcObservationsCfg,
    _enable_dynamic_term_history,
)

from .scenes import (
    ENVIRONMENT_LIDAR_HORIZONTAL_FOV,
    ENVIRONMENT_LIDAR_HORIZONTAL_SAMPLES,
    ENVIRONMENT_LIDAR_INTERLACE_PHASES,
    ENVIRONMENT_LIDAR_OFFSET,
    ENVIRONMENT_LIDAR_RAY_COUNT,
    ENVIRONMENT_LIDAR_VERTICAL_CHANNELS,
    ENVIRONMENT_LIDAR_VERTICAL_FOV,
    ENVIRONMENT_VOXEL_COUNT,
    ENVIRONMENT_VOXEL_MAX_XYZ,
    ENVIRONMENT_VOXEL_MIN_XYZ,
    ENVIRONMENT_VOXEL_SHAPE_XYZ,
    TABLE_LEG_CENTERS,
    TABLE_LEG_SIZE,
    TABLE_TOP_SIZE,
)


def proximity_from_ray_hits(
    ray_hits_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Convert ray hits to a bounded proximity image: no return=0, near obstacle=1."""
    relative_hits = ray_hits_w - sensor_pos_w.unsqueeze(1)
    distance = torch.linalg.norm(relative_hits, dim=-1)
    finite = torch.isfinite(distance)
    distance = torch.where(finite, distance, torch.full_like(distance, max_distance))
    return torch.clamp(1.0 - distance / max_distance, min=0.0, max=1.0)


def _interlaced_angles(
    angle_range: tuple[float, float],
    count: int,
    half_step: bool,
    device: str,
) -> torch.Tensor:
    if not half_step:
        return torch.linspace(*angle_range, count, device=device)
    step = (angle_range[1] - angle_range[0]) / (count - 1)
    return torch.linspace(
        angle_range[0] + 0.5 * step,
        angle_range[1] - 0.5 * step,
        count - 1,
        device=device,
    )


def _interaction_scan_directions(device: str, dtype: torch.dtype, phase: int = 0) -> torch.Tensor:
    vertical = _interlaced_angles(
        ENVIRONMENT_LIDAR_VERTICAL_FOV,
        ENVIRONMENT_LIDAR_VERTICAL_CHANNELS,
        bool(phase & 1),
        device,
    )
    horizontal = _interlaced_angles(
        ENVIRONMENT_LIDAR_HORIZONTAL_FOV,
        ENVIRONMENT_LIDAR_HORIZONTAL_SAMPLES,
        bool(phase & 2),
        device,
    )
    vertical, horizontal = torch.meshgrid(
        torch.deg2rad(vertical),
        torch.deg2rad(horizontal),
        indexing="ij",
    )
    cos_vertical = torch.cos(vertical)
    return torch.stack(
        (
            torch.cos(horizontal) * cos_vertical,
            torch.sin(horizontal) * cos_vertical,
            torch.sin(vertical),
        ),
        dim=-1,
    ).reshape(-1, 3).to(dtype=dtype)


def _ray_box_distance(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    center: tuple[float, float, float] | torch.Tensor,
    size: tuple[float, float, float] | torch.Tensor,
) -> torch.Tensor:
    center_t = torch.as_tensor(center, device=ray_origins.device, dtype=ray_origins.dtype)
    half_size = 0.5 * torch.as_tensor(size, device=ray_origins.device, dtype=ray_origins.dtype)
    if center_t.ndim == 2:
        center_t = center_t.unsqueeze(1)
    if half_size.ndim == 2:
        half_size = half_size.unsqueeze(1)
    bounds_min = center_t - half_size
    bounds_max = center_t + half_size
    parallel = torch.abs(ray_directions) < 1.0e-7
    safe_directions = torch.where(parallel, torch.ones_like(ray_directions), ray_directions)
    t_min = (bounds_min - ray_origins) / safe_directions
    t_max = (bounds_max - ray_origins) / safe_directions
    entry = torch.minimum(t_min, t_max)
    exit = torch.maximum(t_min, t_max)
    entry = torch.where(parallel, torch.full_like(entry, -torch.inf), entry)
    exit = torch.where(parallel, torch.full_like(exit, torch.inf), exit)
    parallel_outside = (
        parallel & ((ray_origins < bounds_min) | (ray_origins > bounds_max))
    ).any(dim=-1)
    near = entry.amax(dim=-1)
    far = exit.amin(dim=-1)
    valid = (~parallel_outside) & (far >= torch.clamp(near, min=0.0))
    return torch.where(valid, torch.clamp(near, min=0.0), torch.full_like(near, torch.inf))


def _compute_local_environment_scan(env) -> tuple[torch.Tensor, torch.Tensor]:
    step = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_environment_scan_cache_step", None) == step:
        return env._environment_scan_proximity, env._environment_scan_hits_w

    robot = env.scene["robot"]
    if not hasattr(env, "_environment_scan_torso_body_id"):
        body_ids, _ = robot.find_bodies("torso_link")
        env._environment_scan_torso_body_id = int(body_ids[0])
    torso_id = env._environment_scan_torso_body_id
    torso_pos_w = robot.data.body_pos_w[:, torso_id]
    torso_quat_w = robot.data.body_quat_w[:, torso_id]
    dtype = torso_pos_w.dtype

    direction_phases = getattr(env, "_environment_scan_direction_phases_s", None)
    if (
        direction_phases is None
        or direction_phases[0].device != torso_pos_w.device
        or direction_phases[0].dtype != dtype
    ):
        direction_phases = tuple(
            _interaction_scan_directions(env.device, dtype, phase)
            for phase in range(ENVIRONMENT_LIDAR_INTERLACE_PHASES)
        )
        env._environment_scan_direction_phases_s = direction_phases
    directions_s = direction_phases[step % ENVIRONMENT_LIDAR_INTERLACE_PHASES]
    ray_count = directions_s.shape[0]
    torso_quat_expanded = torso_quat_w.unsqueeze(1).expand(-1, ray_count, -1).reshape(-1, 4)
    directions_w = quat_apply(
        torso_quat_expanded,
        directions_s.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3),
    ).reshape(env.num_envs, ray_count, 3)
    offset_s = torch.tensor(
        ENVIRONMENT_LIDAR_OFFSET,
        device=env.device,
        dtype=dtype,
    ).expand(env.num_envs, -1)
    sensor_pos_w = torso_pos_w + quat_apply(torso_quat_w, offset_s)

    if hasattr(env, "scene_obstacle_box_centers_w"):
        distances = torch.full(
            (env.num_envs, ray_count),
            torch.inf,
            device=env.device,
            dtype=dtype,
        )
        box_count = env.scene_obstacle_box_centers_w.shape[1]
        for box_index in range(box_count):
            box_quat_w = env.scene_obstacle_box_quats_w[:, box_index]
            box_quat_expanded = box_quat_w.unsqueeze(1).expand(-1, ray_count, -1).reshape(-1, 4)
            origins_box = quat_apply_inverse(
                box_quat_w,
                sensor_pos_w - env.scene_obstacle_box_centers_w[:, box_index],
            ).unsqueeze(1).expand(-1, ray_count, -1)
            directions_box = quat_apply_inverse(
                box_quat_expanded,
                directions_w.reshape(-1, 3),
            ).reshape(env.num_envs, ray_count, 3)
            box_distance = _ray_box_distance(
                origins_box,
                directions_box,
                torch.zeros(env.num_envs, 3, device=env.device, dtype=dtype),
                2.0 * env.scene_obstacle_box_half_extents[:, box_index],
            )
            box_distance = torch.where(
                env.scene_obstacle_box_active[:, box_index, None],
                box_distance,
                torch.full_like(box_distance, torch.inf),
            )
            distances = torch.minimum(distances, box_distance)
    else:
        table = env.scene["scene_table"]
        table_pos_w = table.data.root_pos_w
        table_quat_w = table.data.root_quat_w
        table_quat_expanded = table_quat_w.unsqueeze(1).expand(-1, ray_count, -1).reshape(-1, 4)
        origins_table = quat_apply_inverse(
            table_quat_w,
            sensor_pos_w - table_pos_w,
        ).unsqueeze(1).expand(-1, ray_count, -1)
        directions_table = quat_apply_inverse(
            table_quat_expanded,
            directions_w.reshape(-1, 3),
        ).reshape(env.num_envs, ray_count, 3)

        distances = _ray_box_distance(
            origins_table,
            directions_table,
            (0.0, 0.0, 0.76),
            TABLE_TOP_SIZE,
        )
        for leg_center in TABLE_LEG_CENTERS:
            distances = torch.minimum(
                distances,
                _ray_box_distance(origins_table, directions_table, leg_center, TABLE_LEG_SIZE),
            )
    max_distance = env.cfg.environment_scan_max_distance
    valid = torch.isfinite(distances) & (distances <= max_distance)
    proximity = torch.where(valid, 1.0 - distances / max_distance, torch.zeros_like(distances))
    hits_w = sensor_pos_w.unsqueeze(1) + distances.unsqueeze(-1) * directions_w
    hits_w = torch.where(valid.unsqueeze(-1), hits_w, torch.full_like(hits_w, torch.inf))

    env._environment_scan_cache_step = step
    env._environment_scan_proximity = proximity
    env._environment_scan_hits_w = hits_w
    return proximity, hits_w


def _voxel_center_grid(
    device: str,
    dtype: torch.dtype,
    shape_xyz: tuple[int, int, int] = ENVIRONMENT_VOXEL_SHAPE_XYZ,
    lower_xyz: tuple[float, float, float] = ENVIRONMENT_VOXEL_MIN_XYZ,
    upper_xyz: tuple[float, float, float] = ENVIRONMENT_VOXEL_MAX_XYZ,
) -> torch.Tensor:
    """Return root-yaw-frame voxel centers in storage order ``(z, y, x, xyz)``."""
    shape = shape_xyz
    lower = lower_xyz
    upper = upper_xyz
    axes = tuple(
        torch.linspace(
            lower[axis] + 0.5 * (upper[axis] - lower[axis]) / shape[axis],
            upper[axis] - 0.5 * (upper[axis] - lower[axis]) / shape[axis],
            shape[axis],
            device=device,
            dtype=dtype,
        )
        for axis in range(3)
    )
    z, y, x = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return torch.stack((x, y, z), dim=-1)


def _warp_previous_occupancy(
    occupancy: torch.Tensor,
    voxel_centers_r: torch.Tensor,
    previous_root_pos_w: torch.Tensor,
    previous_root_yaw: torch.Tensor,
    current_root_pos_w: torch.Tensor,
    current_root_yaw: torch.Tensor,
) -> torch.Tensor:
    """Reproject a previous root-local map into the current root-yaw frame."""
    x = voxel_centers_r[..., 0]
    y = voxel_centers_r[..., 1]
    z = voxel_centers_r[..., 2]
    yaw_delta = current_root_yaw - previous_root_yaw
    cos_delta = torch.cos(yaw_delta)[:, None, None, None]
    sin_delta = torch.sin(yaw_delta)[:, None, None, None]

    root_delta_w = current_root_pos_w - previous_root_pos_w
    cos_previous = torch.cos(previous_root_yaw)
    sin_previous = torch.sin(previous_root_yaw)
    translation_x = (
        cos_previous * root_delta_w[:, 0] + sin_previous * root_delta_w[:, 1]
    )[:, None, None, None]
    translation_y = (
        -sin_previous * root_delta_w[:, 0] + cos_previous * root_delta_w[:, 1]
    )[:, None, None, None]
    translation_z = root_delta_w[:, 2, None, None, None]

    sample_x = cos_delta * x - sin_delta * y + translation_x
    sample_y = sin_delta * x + cos_delta * y + translation_y
    sample_z = z + translation_z
    lower = ENVIRONMENT_VOXEL_MIN_XYZ
    upper = ENVIRONMENT_VOXEL_MAX_XYZ
    sample_grid = torch.stack(
        (
            2.0 * (sample_x - lower[0]) / (upper[0] - lower[0]) - 1.0,
            2.0 * (sample_y - lower[1]) / (upper[1] - lower[1]) - 1.0,
            2.0 * (sample_z - lower[2]) / (upper[2] - lower[2]) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        occupancy.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)


def _hits_to_current_voxels(
    hits_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_yaw: torch.Tensor,
    shape_xyz: tuple[int, int, int] = ENVIRONMENT_VOXEL_SHAPE_XYZ,
    lower_xyz: tuple[float, float, float] = ENVIRONMENT_VOXEL_MIN_XYZ,
    upper_xyz: tuple[float, float, float] = ENVIRONMENT_VOXEL_MAX_XYZ,
) -> torch.Tensor:
    """Rasterize current finite scan returns into a root-yaw-frame occupancy volume."""
    relative_w = hits_w - root_pos_w[:, None, :]
    cos_yaw = torch.cos(root_yaw)[:, None]
    sin_yaw = torch.sin(root_yaw)[:, None]
    points_r = torch.stack(
        (
            cos_yaw * relative_w[..., 0] + sin_yaw * relative_w[..., 1],
            -sin_yaw * relative_w[..., 0] + cos_yaw * relative_w[..., 1],
            relative_w[..., 2],
        ),
        dim=-1,
    )
    finite = torch.isfinite(points_r).all(dim=-1)
    safe_points = torch.where(finite[..., None], points_r, torch.zeros_like(points_r))
    lower = torch.tensor(
        lower_xyz, device=hits_w.device, dtype=hits_w.dtype
    )
    upper = torch.tensor(
        upper_xyz, device=hits_w.device, dtype=hits_w.dtype
    )
    shape_xyz = torch.tensor(
        shape_xyz, device=hits_w.device, dtype=torch.long
    )
    indices_xyz = torch.floor(
        (safe_points - lower) / (upper - lower) * shape_xyz.to(hits_w.dtype)
    ).long()
    inside = finite & ((indices_xyz >= 0) & (indices_xyz < shape_xyz)).all(dim=-1)
    indices_xyz = torch.minimum(torch.maximum(indices_xyz, torch.zeros_like(indices_xyz)), shape_xyz - 1)
    flat_indices = (
        indices_xyz[..., 2] * (shape_xyz[1] * shape_xyz[0])
        + indices_xyz[..., 1] * shape_xyz[0]
        + indices_xyz[..., 0]
    )
    current = torch.zeros(
        hits_w.shape[0], shape_xyz[0] * shape_xyz[1] * shape_xyz[2], device=hits_w.device, dtype=hits_w.dtype
    )
    current.scatter_reduce_(
        1,
        flat_indices,
        inside.to(hits_w.dtype),
        reduce="amax",
        include_self=True,
    )
    shape = shape_xyz
    return current.reshape(hits_w.shape[0], shape[2], shape[1], shape[0])


def _compute_local_occupancy_voxel(env) -> torch.Tensor:
    """Fuse current scan returns into a decaying, ego-motion-compensated local map."""
    step = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_environment_voxel_cache_step", None) == step:
        return env._environment_voxel_occupancy

    _, hits_w = _compute_local_environment_scan(env)
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2]
    dtype = root_pos_w.dtype
    shape = ENVIRONMENT_VOXEL_SHAPE_XYZ
    volume_shape = (shape[2], shape[1], shape[0])

    if not hasattr(env, "_environment_voxel_occupancy"):
        env._environment_voxel_occupancy = torch.zeros(
            env.num_envs, *volume_shape, device=env.device, dtype=dtype
        )
        env._environment_voxel_initialized = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        env._environment_voxel_previous_root_pos_w = root_pos_w.clone()
        env._environment_voxel_previous_root_yaw = root_yaw.clone()
        env._environment_voxel_centers_r = _voxel_center_grid(env.device, dtype)

    occupancy = env._environment_voxel_occupancy
    initialized = env._environment_voxel_initialized
    if bool(initialized.any()):
        warped = _warp_previous_occupancy(
            occupancy,
            env._environment_voxel_centers_r,
            env._environment_voxel_previous_root_pos_w,
            env._environment_voxel_previous_root_yaw,
            root_pos_w,
            root_yaw,
        )
        warped = torch.where(initialized[:, None, None, None], warped, torch.zeros_like(warped))
    else:
        warped = torch.zeros_like(occupancy)

    half_life = max(float(env.cfg.environment_voxel_half_life_s), 1.0e-3)
    decay = math.exp(-math.log(2.0) * float(env.step_dt) / half_life)
    current = _hits_to_current_voxels(hits_w, root_pos_w, root_yaw)
    occupancy = torch.maximum(decay * warped, current)
    minimum = float(env.cfg.environment_voxel_min_occupancy)
    occupancy = torch.where(occupancy >= minimum, occupancy, torch.zeros_like(occupancy))

    env._environment_voxel_occupancy = occupancy
    env._environment_voxel_initialized.fill_(True)
    env._environment_voxel_previous_root_pos_w.copy_(root_pos_w)
    env._environment_voxel_previous_root_yaw.copy_(root_yaw)
    env._environment_voxel_cache_step = step
    return occupancy


def clear_local_occupancy_voxel(env, env_ids: torch.Tensor) -> None:
    """Clear temporal geometry state for reset environments."""
    if not hasattr(env, "_environment_voxel_occupancy"):
        return
    env._environment_voxel_occupancy[env_ids] = 0.0
    env._environment_voxel_initialized[env_ids] = False
    env._environment_voxel_cache_step = None


def local_environment_scan_obs(env) -> torch.Tensor:
    if not getattr(env.cfg, "environment_perception_enabled", True):
        return torch.zeros(env.num_envs, ENVIRONMENT_VOXEL_COUNT, device=env.device)
    return _compute_local_occupancy_voxel(env).flatten(start_dim=1)


def privileged_environment_context_obs(env) -> torch.Tensor:
    robot = env.scene["robot"]
    constrained = getattr(
        env,
        "scene_is_constrained",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    clearance = getattr(
        env,
        "scene_geometry_scalar",
        getattr(
            env,
            "scene_table_clearance",
            torch.zeros(env.num_envs, device=env.device),
        ),
    )
    geometry_center_w = getattr(
        env,
        "scene_geometry_center_w",
        getattr(
            env,
            "scene_table_center_w",
            torch.zeros(env.num_envs, 3, device=env.device),
        ),
    )
    geometry_center_b = quat_apply_inverse(
        robot.data.root_quat_w,
        geometry_center_w - robot.data.root_pos_w,
    )
    geometry_center_b = geometry_center_b * constrained.to(geometry_center_b.dtype).unsqueeze(-1)
    return torch.cat(
        (
            constrained.to(clearance.dtype).unsqueeze(-1),
            clearance.unsqueeze(-1),
            geometry_center_b,
        ),
        dim=-1,
    )


@configclass
class G1Dex1SceneAwareObservationsCfg:
    @configclass
    class PolicyCfg(G1Dex1HierDrcObservationsCfg.PolicyCfg):
        environment = ObsTerm(func=local_environment_scan_obs, history_length=0)

        def __post_init__(self):
            super().__post_init__()
            self.history_length = None
            _enable_dynamic_term_history(self)

    @configclass
    class CriticCfg(G1Dex1HierDrcObservationsCfg.CriticCfg):
        environment = ObsTerm(func=local_environment_scan_obs, history_length=0)
        environment_privileged = ObsTerm(func=privileged_environment_context_obs, history_length=0)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
