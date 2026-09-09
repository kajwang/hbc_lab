from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.assets.objects import APPLE_OBJECT_FRAME_OFFSET_Z
from hbc_lab.tasks.locomotion import mdp
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.events import (
    _apply_object_mass_curriculum,
    reset_robot_body_and_hand_joints,
)

from .events import G1Dex1SceneAwareEventCfg, _root_state
from .geometry_shaping import write_obstacle_boxes
from .multi_geometry import (
    CABINET_BOTTOM_FAMILY,
    CABINET_MIDDLE_FAMILY,
    CABINET_TOP_FAMILY,
    GAP_FAMILY,
    GEOMETRY_FAMILY_NAMES,
    OPEN_PLATFORM_FAMILY,
    PILLAR_FAMILY,
    REACH_OVER_FAMILY,
    TABLE_UNDER_FAMILY,
    balanced_family,
    lerp_range,
    preview_family_and_level,
    uniform_sample,
)
from .multi_geometry_scenes import (
    CABINET_BOX_SIZES,
    CABINET_LOCAL_CENTERS,
    CABINET_SHELF_SURFACE_HEIGHT,
    GAP_WALL_SIZE,
    OPEN_PLATFORM_SIZE,
    OOD_ARCH_CLEARANCE,
    OOD_ARCH_OPENING_WIDTH,
    OOD_ARCH_POST_SIZE,
    OOD_ARCH_TOP_SIZE,
    PILLAR_SIZE,
    REACH_OVER_BARRIER_HEIGHTS,
    REACH_OVER_BARRIER_THICKNESS,
    REACH_OVER_BARRIER_WIDTH,
    REACH_OVER_SUPPORT_SIZE,
    REACH_OVER_SUPPORT_SURFACE_HEIGHT,
)
from .scenes import TABLE_LEG_CENTERS, TABLE_LEG_SIZE, TABLE_NOMINAL_CLEARANCE, TABLE_TOP_SIZE


_TABLE_BOX_COUNT = 1 + len(TABLE_LEG_CENTERS)
_CABINET_BOX_COUNT = len(CABINET_LOCAL_CENTERS)
_TABLE_BOX_SLICE = slice(0, _TABLE_BOX_COUNT)
_CABINET_BOX_SLICE = slice(_TABLE_BOX_COUNT, _TABLE_BOX_COUNT + _CABINET_BOX_COUNT)
_PILLAR_BOX_INDEX = _TABLE_BOX_COUNT + _CABINET_BOX_COUNT
_GAP_BOX_SLICE = slice(_PILLAR_BOX_INDEX + 1, _PILLAR_BOX_INDEX + 3)
_OOD_ARCH_BOX_SLICE = slice(_GAP_BOX_SLICE.stop, _GAP_BOX_SLICE.stop + 3)
_REACH_OVER_BARRIER_BOX_SLICE = slice(_OOD_ARCH_BOX_SLICE.stop, _OOD_ARCH_BOX_SLICE.stop + 3)
_SUPPORT_BOX_INDEX = _REACH_OVER_BARRIER_BOX_SLICE.stop
_OBSTACLE_BOX_COUNT = _SUPPORT_BOX_INDEX + 1


def _ensure_multi_geometry_buffers(env) -> None:
    if not hasattr(env, "geometry_family_id"):
        env.geometry_family_id = torch.full(
            (env.num_envs,), OPEN_PLATFORM_FAMILY, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "geometry_family_level"):
        env.geometry_family_level = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "scene_geometry_scalar"):
        env.scene_geometry_scalar = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "scene_geometry_center_w"):
        env.scene_geometry_center_w = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "scene_table_clearance"):
        env.scene_table_clearance = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "scene_table_center_w"):
        env.scene_table_center_w = torch.zeros(env.num_envs, 3, device=env.device)


def _family_and_level(env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(env.cfg, "geometry_ood_mode", False):
        family = torch.full_like(env_ids, GAP_FAMILY)
        return family, torch.ones(env_ids.shape[0], device=env.device)
    play_env_id = int(getattr(env.cfg, "play_env_id", -1))
    if play_env_id >= 0:
        if not 0 <= play_env_id < len(GEOMETRY_FAMILY_NAMES):
            raise ValueError(
                f"play_env_id must be in [0, {len(GEOMETRY_FAMILY_NAMES) - 1}], got {play_env_id}."
            )
        family = torch.full_like(env_ids, play_env_id)
        if getattr(env.cfg, "geometry_preview_sweep", False):
            max_env_id = max(env.num_envs - 1, 1)
            level = (env_ids.float() / float(max_env_id)).clamp_(0.0, 1.0)
        else:
            level = torch.full(
                (env_ids.shape[0],),
                float(getattr(env.cfg, "play_geo_level", 1.0)),
                device=env.device,
            ).clamp_(0.0, 1.0)
        return family, level
    train_family_id = int(getattr(env.cfg, "train_family_id", -1))
    if train_family_id >= 0:
        if not 0 <= train_family_id < len(GEOMETRY_FAMILY_NAMES):
            raise ValueError(
                f"train_family_id must be in [0, {len(GEOMETRY_FAMILY_NAMES) - 1}], got {train_family_id}."
            )
        family = torch.full_like(env_ids, train_family_id)
        levels = getattr(
            env,
            "geometry_curriculum_levels",
            torch.zeros(len(GEOMETRY_FAMILY_NAMES), device=env.device),
        )
        return family, levels[family]
    family = balanced_family(env_ids, len(GEOMETRY_FAMILY_NAMES))
    if getattr(env.cfg, "geometry_preview_sweep", False):
        preview_family, level = preview_family_and_level(
            env_ids,
            family_count=len(GEOMETRY_FAMILY_NAMES),
            total_envs=env.num_envs,
        )
        return preview_family, level
    levels = getattr(
        env,
        "geometry_curriculum_levels",
        torch.zeros(len(GEOMETRY_FAMILY_NAMES), device=env.device),
    )
    return family, levels[family]


def _parked_positions(
    env,
    env_ids: torch.Tensor,
    xy_offset: tuple[float, float],
    z: float,
) -> torch.Tensor:
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 0] += xy_offset[0]
    positions[:, 1] += xy_offset[1]
    positions[:, 2] += z
    return positions


def _masked_write(
    destination: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    slots: slice,
) -> None:
    destination[:, slots] = torch.where(
        mask[:, None, None],
        values,
        destination[:, slots],
    )


def reset_multi_geometry_pnp(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    target_distance: float,
    object_ground_root_height: float,
    target_root_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset geometry-conditioned PnP families under one task interface."""
    _ensure_multi_geometry_buffers(env)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    obj = env.scene[asset_cfg.name]
    count = env_ids.numel()
    dtype = obj.data.root_pos_w.dtype
    family, level = _family_and_level(env, env_ids)

    object_root_pos_w = env.scene.env_origins[env_ids].clone()
    object_root_pos_w[:, 0] += uniform_sample(env_ids, *pose_range["x"]).to(dtype)
    object_root_pos_w[:, 1] += uniform_sample(env_ids, *pose_range["y"]).to(dtype)
    object_root_pos_w[:, 2] += object_ground_root_height
    open_platform = family == OPEN_PLATFORM_FAMILY
    cabinet_top = family == CABINET_TOP_FAMILY
    cabinet_middle = family == CABINET_MIDDLE_FAMILY
    cabinet_bottom = family == CABINET_BOTTOM_FAMILY
    cabinet = cabinet_top | cabinet_middle | cabinet_bottom
    reach_over_family = family == REACH_OVER_FAMILY
    platform_surface_height = uniform_sample(
        env_ids,
        OPEN_PLATFORM_SIZE[2],
        lerp_range(level, 0.30, 0.70),
    ).to(dtype)
    object_root_pos_w[:, 2] += open_platform * platform_surface_height
    object_root_pos_w[:, 2] += cabinet_top * CABINET_SHELF_SURFACE_HEIGHT["top"]
    object_root_pos_w[:, 2] += cabinet_middle * CABINET_SHELF_SURFACE_HEIGHT["middle"]
    object_root_pos_w[:, 2] += cabinet_bottom * CABINET_SHELF_SURFACE_HEIGHT["bottom"]
    object_root_pos_w[:, 2] += reach_over_family * REACH_OVER_SUPPORT_SURFACE_HEIGHT

    yaw = uniform_sample(env_ids, *pose_range.get("yaw", (-torch.pi, torch.pi))).to(dtype)
    zeros = torch.zeros_like(yaw)
    object_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]
    toward_robot_xy = robot_root_pos_w[:, :2] - object_root_pos_w[:, :2]
    toward_robot_xy = toward_robot_xy / torch.clamp(
        torch.linalg.norm(toward_robot_xy, dim=-1, keepdim=True), min=1.0e-6
    )
    away_xy = -toward_robot_xy
    lateral_xy = torch.stack((-toward_robot_xy[:, 1], toward_robot_xy[:, 0]), dim=-1)
    structure_yaw = torch.atan2(away_xy[:, 1], away_xy[:, 0])
    structure_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, structure_yaw)

    target_root_pos_w = object_root_pos_w.clone()
    target_root_pos_w[:, :2] += float(target_distance) * toward_robot_xy
    target_root_pos_w[:, 2] = env.scene.env_origins[env_ids, 2] + target_root_height

    object_velocity = torch.zeros(count, 6, device=env.device, dtype=dtype)
    obj.write_root_state_to_sim(
        torch.cat((object_root_pos_w, object_quat_w, object_velocity), dim=-1),
        env_ids=env_ids,
    )
    mass_levels = getattr(
        env,
        "object_mass_curriculum_levels",
        env.object_mass_curriculum_level.expand(len(GEOMETRY_FAMILY_NAMES)),
    )
    _apply_object_mass_curriculum(env, env_ids, obj, curriculum_level=mass_levels[family])

    object_frame_pos_w = object_root_pos_w.clone()
    object_frame_pos_w[:, 2] += APPLE_OBJECT_FRAME_OFFSET_Z
    target_frame_pos_w = target_root_pos_w.clone()
    target_frame_pos_w[:, 2] += APPLE_OBJECT_FRAME_OFFSET_Z
    env.object_initial_pos_w[env_ids] = object_frame_pos_w
    env.object_initial_root_z_w[env_ids] = object_root_pos_w[:, 2]
    env.object_target_pos_w[env_ids] = target_frame_pos_w

    table_under = family == TABLE_UNDER_FAMILY
    table = table_under
    table_depth = lerp_range(level, 0.12, 0.38)
    table_center_xy = object_root_pos_w[:, :2] + away_xy * (
        0.5 * TABLE_TOP_SIZE[0] - table_depth
    )[:, None]
    table_pos_w = env.scene.env_origins[env_ids].clone()
    table_pos_w[:, :2] = table_center_xy
    parked_table_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 6.0)
    actual_table_pos_w = torch.where(table[:, None], table_pos_w, parked_table_pos_w)
    actual_table_yaw = torch.where(table, structure_yaw, torch.zeros_like(structure_yaw))
    env.scene["scene_table"].write_root_state_to_sim(
        _root_state(actual_table_pos_w, actual_table_yaw), env_ids=env_ids
    )

    cabinet_obstacle = cabinet
    cabinet_depth = lerp_range(level, 0.10, 0.28)
    cabinet_center_xy = object_root_pos_w[:, :2] + away_xy * (0.25 - cabinet_depth)[:, None]
    cabinet_pos_w = env.scene.env_origins[env_ids].clone()
    cabinet_pos_w[:, :2] = cabinet_center_xy
    parked_cabinet_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 9.0)
    actual_cabinet_pos_w = torch.where(
        cabinet_obstacle[:, None], cabinet_pos_w, parked_cabinet_pos_w
    )
    actual_cabinet_yaw = torch.where(
        cabinet_obstacle, structure_yaw, torch.zeros_like(structure_yaw)
    )
    env.scene["three_level_cabinet"].write_root_state_to_sim(
        _root_state(actual_cabinet_pos_w, actual_cabinet_yaw), env_ids=env_ids
    )

    side_sign = torch.where(
        torch.remainder(
            torch.div(
                torch.div(env_ids, 2, rounding_mode="floor"),
                len(GEOMETRY_FAMILY_NAMES),
                rounding_mode="floor",
            ),
            2,
        )
        == 0,
        torch.ones(count, device=env.device, dtype=dtype),
        -torch.ones(count, device=env.device, dtype=dtype),
    )
    pillar_family = family == PILLAR_FAMILY
    gap_family = family == GAP_FAMILY
    ood_arch = torch.full(
        (count,), bool(getattr(env.cfg, "geometry_ood_mode", False)), device=env.device, dtype=torch.bool
    )
    pillar = pillar_family & ~ood_arch
    gap = gap_family & ~ood_arch
    pillar_distance = lerp_range(level, 0.62, 0.32)
    pillar_lateral = lerp_range(level, 0.20, 0.03)
    pillar_center_xy = (
        object_root_pos_w[:, :2]
        + pillar_distance[:, None] * toward_robot_xy
        + side_sign[:, None] * pillar_lateral[:, None] * lateral_xy
    )
    pillar_pos_w = env.scene.env_origins[env_ids].clone()
    pillar_pos_w[:, :2] = pillar_center_xy
    pillar_pos_w[:, 2] += 0.5 * PILLAR_SIZE[2]
    parked_pillar_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 12.0)
    actual_pillar_pos_w = torch.where(pillar[:, None], pillar_pos_w, parked_pillar_pos_w)
    env.scene["front_pillar"].write_root_state_to_sim(
        _root_state(actual_pillar_pos_w, torch.zeros_like(yaw)), env_ids=env_ids
    )

    gap_width = lerp_range(level, 1.00, 0.68)
    robot_object_distance = torch.linalg.norm(
        object_root_pos_w[:, :2] - robot_root_pos_w[:, :2], dim=-1
    )
    gap_center_xy = robot_root_pos_w[:, :2] + 0.52 * robot_object_distance[:, None] * away_xy
    gap_side_offset = 0.5 * (gap_width + GAP_WALL_SIZE[1])
    gap_left_pos_w = env.scene.env_origins[env_ids].clone()
    gap_right_pos_w = env.scene.env_origins[env_ids].clone()
    gap_left_pos_w[:, :2] = gap_center_xy + gap_side_offset[:, None] * lateral_xy
    gap_right_pos_w[:, :2] = gap_center_xy - gap_side_offset[:, None] * lateral_xy
    gap_left_pos_w[:, 2] += 0.5 * GAP_WALL_SIZE[2]
    gap_right_pos_w[:, 2] += 0.5 * GAP_WALL_SIZE[2]
    parked_gap_left_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 14.0)
    parked_gap_right_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 16.0)
    actual_gap_left_pos_w = torch.where(gap[:, None], gap_left_pos_w, parked_gap_left_pos_w)
    actual_gap_right_pos_w = torch.where(gap[:, None], gap_right_pos_w, parked_gap_right_pos_w)
    actual_gap_yaw = torch.where(gap, structure_yaw, torch.zeros_like(structure_yaw))
    env.scene["gap_wall_left"].write_root_state_to_sim(
        _root_state(actual_gap_left_pos_w, actual_gap_yaw), env_ids=env_ids
    )
    env.scene["gap_wall_right"].write_root_state_to_sim(
        _root_state(actual_gap_right_pos_w, actual_gap_yaw), env_ids=env_ids
    )

    # Reach-over places a taller frontal wall between the robot and object,
    # requiring the active hand to approach from above.
    reach_over = reach_over_family
    reach_depth = lerp_range(level, 0.20, 0.38)
    reach_barrier_center_xy = object_root_pos_w[:, :2] + toward_robot_xy * (
        reach_depth + 0.5 * REACH_OVER_BARRIER_THICKNESS
    )[:, None]
    reach_height_bin = torch.clamp(
        torch.floor(level * len(REACH_OVER_BARRIER_HEIGHTS)).long(),
        max=len(REACH_OVER_BARRIER_HEIGHTS) - 1,
    )
    reach_barrier_positions = []
    reach_barrier_masks = []
    reach_barrier_names = (
        "reach_over_barrier_low",
        "reach_over_barrier_mid",
        "reach_over_barrier_high",
    )
    for bin_index, (barrier_name, barrier_height) in enumerate(
        zip(reach_barrier_names, REACH_OVER_BARRIER_HEIGHTS)
    ):
        barrier_mask = reach_over & (reach_height_bin == bin_index)
        barrier_pos_w = env.scene.env_origins[env_ids].clone()
        barrier_pos_w[:, :2] = reach_barrier_center_xy
        barrier_pos_w[:, 2] += 0.5 * barrier_height
        # Inactive variants must not overlap.  Parking them below ground makes
        # PhysX eject them upward, while parking them side-by-side still overlaps
        # their 1.5 m-wide colliders.  Isolated overhead layers avoid both cases.
        parked_barrier_pos_w = _parked_positions(
            env, env_ids, (0.0, 0.0), 26.0 + 2.0 * bin_index
        )
        actual_barrier_pos_w = torch.where(
            barrier_mask[:, None], barrier_pos_w, parked_barrier_pos_w
        )
        env.scene[barrier_name].write_root_state_to_sim(
            _root_state(actual_barrier_pos_w, structure_yaw), env_ids=env_ids
        )
        reach_barrier_positions.append(barrier_pos_w)
        reach_barrier_masks.append(barrier_mask)

    reach_support_pos_w = object_root_pos_w.clone()
    reach_support_pos_w[:, 2] = (
        env.scene.env_origins[env_ids, 2]
        + REACH_OVER_SUPPORT_SURFACE_HEIGHT
        - 0.5 * REACH_OVER_SUPPORT_SIZE[2]
    )
    parked_reach_support_pos_w = _parked_positions(
        env, env_ids, (0.0, 0.0), 32.0
    )
    actual_reach_support_pos_w = torch.where(
        reach_over_family[:, None], reach_support_pos_w, parked_reach_support_pos_w
    )
    env.scene["reach_over_support"].write_root_state_to_sim(
        _root_state(actual_reach_support_pos_w, torch.zeros_like(yaw)), env_ids=env_ids
    )

    # OOD-only low arch: a new composition of lateral and overhead constraints.
    # The object and transport target lie on opposite sides, so both approach and
    # loaded return motion must react to the unseen geometry.
    arch_center_xy = robot_root_pos_w[:, :2] + 0.52 * robot_object_distance[:, None] * away_xy
    arch_post_offset = 0.5 * (OOD_ARCH_OPENING_WIDTH + OOD_ARCH_POST_SIZE[1])
    arch_left_pos_w = env.scene.env_origins[env_ids].clone()
    arch_right_pos_w = env.scene.env_origins[env_ids].clone()
    arch_top_pos_w = env.scene.env_origins[env_ids].clone()
    arch_left_pos_w[:, :2] = arch_center_xy + arch_post_offset * lateral_xy
    arch_right_pos_w[:, :2] = arch_center_xy - arch_post_offset * lateral_xy
    arch_top_pos_w[:, :2] = arch_center_xy
    arch_left_pos_w[:, 2] += 0.5 * OOD_ARCH_POST_SIZE[2]
    arch_right_pos_w[:, 2] += 0.5 * OOD_ARCH_POST_SIZE[2]
    arch_top_pos_w[:, 2] += OOD_ARCH_CLEARANCE + 0.5 * OOD_ARCH_TOP_SIZE[2]
    parked_arch_left_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 18.0)
    parked_arch_right_pos_w = _parked_positions(env, env_ids, (0.0, 0.0), 20.0)
    parked_arch_top_pos_w = _parked_positions(
        env, env_ids, (0.0, 0.0), 22.0
    )
    actual_arch_left_pos_w = torch.where(ood_arch[:, None], arch_left_pos_w, parked_arch_left_pos_w)
    actual_arch_right_pos_w = torch.where(ood_arch[:, None], arch_right_pos_w, parked_arch_right_pos_w)
    actual_arch_top_pos_w = torch.where(ood_arch[:, None], arch_top_pos_w, parked_arch_top_pos_w)
    arch_yaw = torch.where(ood_arch, structure_yaw, torch.zeros_like(structure_yaw))
    env.scene["ood_arch_left_post"].write_root_state_to_sim(
        _root_state(actual_arch_left_pos_w, arch_yaw), env_ids=env_ids
    )
    env.scene["ood_arch_right_post"].write_root_state_to_sim(
        _root_state(actual_arch_right_pos_w, arch_yaw), env_ids=env_ids
    )
    env.scene["ood_arch_top"].write_root_state_to_sim(
        _root_state(actual_arch_top_pos_w, arch_yaw), env_ids=env_ids
    )

    open_platform_pos_w = object_root_pos_w.clone()
    open_platform_pos_w[:, 2] = (
        env.scene.env_origins[env_ids, 2]
        + platform_surface_height
        - 0.5 * OPEN_PLATFORM_SIZE[2]
    )
    parked_open_platform_pos_w = _parked_positions(
        env, env_ids, (0.0, 0.0), 24.0
    )
    actual_open_platform_pos_w = torch.where(
        open_platform[:, None], open_platform_pos_w, parked_open_platform_pos_w
    )
    env.scene["open_platform"].write_root_state_to_sim(
        _root_state(actual_open_platform_pos_w, torch.zeros_like(yaw)), env_ids=env_ids
    )

    centers_w = torch.zeros(count, _OBSTACLE_BOX_COUNT, 3, device=env.device, dtype=dtype)
    quats_w = torch.zeros(count, _OBSTACLE_BOX_COUNT, 4, device=env.device, dtype=dtype)
    quats_w[..., 0] = 1.0
    half_extents = torch.zeros_like(centers_w)
    active = torch.zeros(count, _OBSTACLE_BOX_COUNT, device=env.device, dtype=torch.bool)

    table_local_centers = torch.tensor(
        ((0.0, 0.0, 0.76), *TABLE_LEG_CENTERS), device=env.device, dtype=dtype
    ).unsqueeze(0).expand(count, -1, -1)
    table_sizes = torch.tensor(
        (TABLE_TOP_SIZE, *([TABLE_LEG_SIZE] * len(TABLE_LEG_CENTERS))),
        device=env.device,
        dtype=dtype,
    ).unsqueeze(0).expand(count, -1, -1)
    repeated_table_quat = structure_quat_w[:, None, :].expand(-1, _TABLE_BOX_COUNT, -1)
    table_centers_w = table_pos_w[:, None, :] + math_utils.quat_apply(
        repeated_table_quat.reshape(-1, 4), table_local_centers.reshape(-1, 3)
    ).reshape(count, _TABLE_BOX_COUNT, 3)
    _masked_write(centers_w, table_centers_w, table, _TABLE_BOX_SLICE)
    _masked_write(quats_w, repeated_table_quat, table, _TABLE_BOX_SLICE)
    _masked_write(half_extents, 0.5 * table_sizes, table, _TABLE_BOX_SLICE)
    active[:, _TABLE_BOX_SLICE] |= table[:, None]

    cabinet_local_centers = torch.tensor(
        CABINET_LOCAL_CENTERS, device=env.device, dtype=dtype
    ).unsqueeze(0).expand(count, -1, -1)
    cabinet_sizes = torch.tensor(
        CABINET_BOX_SIZES, device=env.device, dtype=dtype
    ).unsqueeze(0).expand(count, -1, -1)
    repeated_cabinet_quat = structure_quat_w[:, None, :].expand(-1, _CABINET_BOX_COUNT, -1)
    cabinet_centers_w = cabinet_pos_w[:, None, :] + math_utils.quat_apply(
        repeated_cabinet_quat.reshape(-1, 4), cabinet_local_centers.reshape(-1, 3)
    ).reshape(count, _CABINET_BOX_COUNT, 3)
    _masked_write(centers_w, cabinet_centers_w, cabinet_obstacle, _CABINET_BOX_SLICE)
    _masked_write(quats_w, repeated_cabinet_quat, cabinet_obstacle, _CABINET_BOX_SLICE)
    _masked_write(half_extents, 0.5 * cabinet_sizes, cabinet_obstacle, _CABINET_BOX_SLICE)
    active[:, _CABINET_BOX_SLICE] |= cabinet_obstacle[:, None]

    pillar_half_extent = 0.5 * torch.tensor(PILLAR_SIZE, device=env.device, dtype=dtype)
    centers_w[:, _PILLAR_BOX_INDEX] = torch.where(
        pillar[:, None], pillar_pos_w, centers_w[:, _PILLAR_BOX_INDEX]
    )
    half_extents[:, _PILLAR_BOX_INDEX] = torch.where(
        pillar[:, None], pillar_half_extent, half_extents[:, _PILLAR_BOX_INDEX]
    )
    active[:, _PILLAR_BOX_INDEX] |= pillar

    gap_half_extent = 0.5 * torch.tensor(GAP_WALL_SIZE, device=env.device, dtype=dtype)
    centers_w[:, _GAP_BOX_SLICE] = torch.where(
        gap[:, None, None],
        torch.stack((gap_left_pos_w, gap_right_pos_w), dim=1),
        centers_w[:, _GAP_BOX_SLICE],
    )
    quats_w[:, _GAP_BOX_SLICE] = torch.where(
        gap[:, None, None],
        structure_quat_w[:, None, :].expand(-1, 2, -1),
        quats_w[:, _GAP_BOX_SLICE],
    )
    half_extents[:, _GAP_BOX_SLICE] = torch.where(
        gap[:, None, None],
        gap_half_extent[None, None, :].expand(count, 2, -1),
        half_extents[:, _GAP_BOX_SLICE],
    )
    active[:, _GAP_BOX_SLICE] |= gap[:, None]

    arch_centers_w = torch.stack((arch_left_pos_w, arch_right_pos_w, arch_top_pos_w), dim=1)
    arch_quats_w = structure_quat_w[:, None, :].expand(-1, 3, -1)
    arch_sizes = torch.tensor(
        (OOD_ARCH_POST_SIZE, OOD_ARCH_POST_SIZE, OOD_ARCH_TOP_SIZE),
        device=env.device,
        dtype=dtype,
    ).unsqueeze(0).expand(count, -1, -1)
    _masked_write(centers_w, arch_centers_w, ood_arch, _OOD_ARCH_BOX_SLICE)
    _masked_write(quats_w, arch_quats_w, ood_arch, _OOD_ARCH_BOX_SLICE)
    _masked_write(half_extents, 0.5 * arch_sizes, ood_arch, _OOD_ARCH_BOX_SLICE)
    active[:, _OOD_ARCH_BOX_SLICE] |= ood_arch[:, None]

    reach_barrier_centers_w = torch.stack(reach_barrier_positions, dim=1)
    reach_barrier_quats_w = structure_quat_w[:, None, :].expand(-1, 3, -1)
    reach_barrier_sizes = torch.tensor(
        tuple(
            (REACH_OVER_BARRIER_THICKNESS, REACH_OVER_BARRIER_WIDTH, height)
            for height in REACH_OVER_BARRIER_HEIGHTS
        ),
        device=env.device,
        dtype=dtype,
    ).unsqueeze(0).expand(count, -1, -1)
    reach_barrier_active = torch.stack(reach_barrier_masks, dim=1)
    centers_w[:, _REACH_OVER_BARRIER_BOX_SLICE] = torch.where(
        reach_barrier_active[:, :, None],
        reach_barrier_centers_w,
        centers_w[:, _REACH_OVER_BARRIER_BOX_SLICE],
    )
    quats_w[:, _REACH_OVER_BARRIER_BOX_SLICE] = torch.where(
        reach_barrier_active[:, :, None],
        reach_barrier_quats_w,
        quats_w[:, _REACH_OVER_BARRIER_BOX_SLICE],
    )
    half_extents[:, _REACH_OVER_BARRIER_BOX_SLICE] = torch.where(
        reach_barrier_active[:, :, None],
        0.5 * reach_barrier_sizes,
        half_extents[:, _REACH_OVER_BARRIER_BOX_SLICE],
    )
    active[:, _REACH_OVER_BARRIER_BOX_SLICE] |= reach_barrier_active

    reach_support_half_extent = 0.5 * torch.tensor(REACH_OVER_SUPPORT_SIZE, device=env.device, dtype=dtype)
    open_platform_half_extent = 0.5 * torch.tensor(OPEN_PLATFORM_SIZE, device=env.device, dtype=dtype)
    support_center_w = torch.where(open_platform[:, None], open_platform_pos_w, reach_support_pos_w)
    support_half_extent = torch.where(
        open_platform[:, None], open_platform_half_extent, reach_support_half_extent
    )
    support_active = open_platform | reach_over_family
    centers_w[:, _SUPPORT_BOX_INDEX] = torch.where(
        support_active[:, None], support_center_w, centers_w[:, _SUPPORT_BOX_INDEX]
    )
    half_extents[:, _SUPPORT_BOX_INDEX] = torch.where(
        support_active[:, None], support_half_extent, half_extents[:, _SUPPORT_BOX_INDEX]
    )
    active[:, _SUPPORT_BOX_INDEX] |= support_active
    write_obstacle_boxes(env, env_ids, centers_w, quats_w, half_extents, active)

    table_top_center_w = table_pos_w.clone()
    table_top_center_w[:, 2] += TABLE_NOMINAL_CLEARANCE + 0.5 * TABLE_TOP_SIZE[2]
    cabinet_opening_center_w = cabinet_pos_w.clone()
    cabinet_opening_center_w[:, :2] -= 0.25 * away_xy
    cabinet_opening_center_w[:, 2] = object_root_pos_w[:, 2] + 0.20
    gap_center_w = torch.cat(
        (gap_center_xy, (env.scene.env_origins[env_ids, 2] + 0.5 * GAP_WALL_SIZE[2])[:, None]),
        dim=-1,
    )
    main_center_w = torch.zeros_like(object_root_pos_w)
    main_center_w = torch.where(table[:, None], table_top_center_w, main_center_w)
    main_center_w = torch.where(cabinet_obstacle[:, None], cabinet_opening_center_w, main_center_w)
    main_center_w = torch.where(open_platform[:, None], open_platform_pos_w, main_center_w)
    main_center_w = torch.where(pillar_family[:, None], pillar_pos_w, main_center_w)
    main_center_w = torch.where(gap_family[:, None], gap_center_w, main_center_w)
    reach_barrier_center_w = torch.cat(
        (
            reach_barrier_center_xy,
            (
                env.scene.env_origins[env_ids, 2]
                + 0.5
                * torch.tensor(
                    REACH_OVER_BARRIER_HEIGHTS, device=env.device, dtype=dtype
                )[reach_height_bin]
            )[:, None],
        ),
        dim=-1,
    )
    main_center_w = torch.where(reach_over[:, None], reach_barrier_center_w, main_center_w)
    arch_center_w = torch.cat(
        (
            arch_center_xy,
            (env.scene.env_origins[env_ids, 2] + 0.5 * OOD_ARCH_CLEARANCE)[:, None],
        ),
        dim=-1,
    )
    main_center_w = torch.where(ood_arch[:, None], arch_center_w, main_center_w)
    geometry_scalar = torch.where(open_platform, platform_surface_height, torch.zeros_like(level))
    geometry_scalar = torch.where(table, table_depth, geometry_scalar)
    geometry_scalar = torch.where(cabinet_obstacle, cabinet_depth, geometry_scalar)
    geometry_scalar = torch.where(pillar_family, pillar_distance, geometry_scalar)
    geometry_scalar = torch.where(gap_family, gap_width, geometry_scalar)
    reach_barrier_height = torch.tensor(
        REACH_OVER_BARRIER_HEIGHTS, device=env.device, dtype=dtype
    )[reach_height_bin]
    geometry_scalar = torch.where(reach_over, reach_barrier_height, geometry_scalar)
    geometry_scalar = torch.where(
        ood_arch, torch.full_like(geometry_scalar, OOD_ARCH_OPENING_WIDTH), geometry_scalar
    )

    env.geometry_family_id[env_ids] = family
    env.geometry_family_level[env_ids] = level
    env.scene_geometry_scalar[env_ids] = geometry_scalar
    env.scene_geometry_center_w[env_ids] = main_center_w
    env.scene_table_clearance[env_ids] = torch.where(
        table, torch.full_like(level, TABLE_NOMINAL_CLEARANCE), torch.zeros_like(level)
    )
    env.scene_table_center_w[env_ids] = torch.where(
        table[:, None], table_top_center_w, torch.zeros_like(main_center_w)
    )
    env._environment_scan_cache_step = None


@configclass
class G1Dex1MultiGeometryEventCfg(G1Dex1SceneAwareEventCfg):
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=reset_robot_body_and_hand_joints,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_object = EventTerm(
        func=reset_multi_geometry_pnp,
        mode="reset",
        params={
            "pose_range": {"x": (2.0, 2.5), "y": (-0.35, 0.35), "yaw": (-3.14, 3.14)},
            "target_distance": 2.0,
            "object_ground_root_height": 0.02,
            "target_root_height": 0.50,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
