from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from hbc_lab.assets.objects import APPLE_OBJECT_FRAME_OFFSET_Z, OBJECT_PLATFORM_HEIGHT, OBJECT_ROOT_ON_PLATFORM_Z
from hbc_lab.assets.robots.unitree import (
    G1_29DOF_BODY_JOINT_NAMES,
    G1_DEX1_LEFT_GRIPPER_JOINT_NAMES,
    G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES,
)
from hbc_lab.tasks.locomotion import mdp

from .object_mass_curriculum import sample_object_masses
from .grasp_references import candidate_evaluation_yaws
from .domain_randomization_curriculum import (
    allowed_scale_indices,
    inactive_object_parking_offsets,
    randomization_ranges,
)
from .scenes import PNP_OBJECT_SCALE_FACTORS
from .object_shape_bps import rotated_box_min_z


def _find_joint_ids(asset: Articulation, joint_names: list[str]) -> list[int]:
    joint_ids, resolved_joint_names = asset.find_joints(joint_names, preserve_order=False)
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(f"Expected {len(joint_names)} joints, got {len(joint_ids)}: {resolved_joint_names}")
    return list(joint_ids)


def _find_gripper_joint_ids(asset: Articulation) -> list[int]:
    return _find_joint_ids(asset, [*G1_DEX1_LEFT_GRIPPER_JOINT_NAMES, *G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES])


def set_rigid_object_collection_material(
    env,
    env_ids: torch.Tensor | None,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Set one material for every shape in a rigid-object collection at startup."""
    del env_ids
    asset = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = static_friction
    materials[..., 1] = dynamic_friction
    materials[..., 2] = restitution
    indices = torch.arange(materials.shape[0], device="cpu")
    asset.root_physx_view.set_material_properties(materials, indices)


def reset_robot_body_and_hand_joints(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset G1 body joints like the low-level task while keeping Dex1 hands stable."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_joint_ids = _find_joint_ids(asset, G1_29DOF_BODY_JOINT_NAMES)
    hand_joint_ids = _find_gripper_joint_ids(asset)

    joint_ids = body_joint_ids + hand_joint_ids
    joint_pos = asset.data.default_joint_pos[env_ids[:, None], joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids[:, None], joint_ids].clone()

    body_shape = joint_pos[:, : len(body_joint_ids)].shape
    joint_pos[:, : len(body_joint_ids)] *= math_utils.sample_uniform(
        *position_range,
        body_shape,
        joint_pos.device,
    )
    joint_vel[:, : len(body_joint_ids)] *= math_utils.sample_uniform(
        *velocity_range,
        body_shape,
        joint_vel.device,
    )
    joint_vel[:, len(body_joint_ids) :] = 0.0

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids[:, None], joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
    hand_default_pos = asset.data.default_joint_pos[env_ids[:, None], hand_joint_ids]
    asset.set_joint_position_target(hand_default_pos, joint_ids=hand_joint_ids, env_ids=env_ids)


def _platform_pose_under_object_root(
    object_root_pos_w: torch.Tensor,
    platform_height: float = OBJECT_PLATFORM_HEIGHT,
) -> torch.Tensor:
    platform_pos_w = object_root_pos_w.clone()
    platform_pos_w[:, 2] = object_root_pos_w[:, 2] - 0.5 * platform_height
    platform_quat_w = torch.zeros(object_root_pos_w.shape[0], 4, device=object_root_pos_w.device)
    platform_quat_w[:, 0] = 1.0
    return torch.cat((platform_pos_w, platform_quat_w), dim=-1)


def _object_frame_pos_from_root(object_root_pos_w: torch.Tensor) -> torch.Tensor:
    object_frame_pos_w = object_root_pos_w.clone()
    object_frame_pos_w[:, 2] += APPLE_OBJECT_FRAME_OFFSET_Z
    return object_frame_pos_w


def _apply_object_mass_curriculum(
    env,
    env_ids: torch.Tensor,
    obj,
    curriculum_level: torch.Tensor | float | None = None,
) -> None:
    if not getattr(env.cfg, "object_mass_curriculum_enabled", False):
        return
    if not hasattr(env, "object_mass_curriculum_level"):
        return

    state_dtype = obj.data.object_pos_w.dtype if hasattr(obj.data, "object_pos_w") else obj.data.root_pos_w.dtype
    level = env.object_mass_curriculum_level if curriculum_level is None else curriculum_level
    mass = sample_object_masses(
        level,
        num_envs=env_ids.shape[0],
        device=obj.device,
        dtype=state_dtype,
        start_mass=env.cfg.object_mass_start_mass,
        ref_w=env.cfg.object_mass_ref_w,
        ref_mass=env.cfg.object_mass_ref_mass,
        anchor_w=env.cfg.object_mass_anchor_w,
        anchor_mass=env.cfg.object_mass_anchor_mass,
        final_mass=env.cfg.object_mass_final_mass,
        ref_log_std=env.cfg.object_mass_ref_log_std,
        final_log_std=env.cfg.object_mass_final_log_std,
        min_mass=env.cfg.object_mass_min,
        max_mass=env.cfg.object_mass_max,
    )
    if not bool(torch.isfinite(mass).all()):
        raise RuntimeError("Object mass curriculum produced a non-finite PhysX mass")

    env_ids_cpu = env_ids.cpu()
    mass_cpu = mass.detach().cpu()
    if hasattr(obj.data, "object_pos_w"):
        masses = obj.reshape_view_to_data(obj.root_physx_view.get_masses())
        masses[env_ids_cpu] = mass_cpu[:, None, None]
        view_masses = obj.reshape_data_to_view(masses)
        view_indices = torch.arange(view_masses.shape[0], device="cpu")
        obj.root_physx_view.set_masses(view_masses, view_indices)

        ratios = masses[env_ids_cpu] / obj.data.default_mass[env_ids_cpu]
        inertias = obj.reshape_view_to_data(obj.root_physx_view.get_inertias())
        inertias[env_ids_cpu] = obj.data.default_inertia[env_ids_cpu] * ratios
        obj.root_physx_view.set_inertias(obj.reshape_data_to_view(inertias), view_indices)
    else:
        masses = obj.root_physx_view.get_masses()
        masses[env_ids_cpu] = mass_cpu.unsqueeze(-1).expand(-1, masses.shape[1])
        obj.root_physx_view.set_masses(masses, env_ids.cpu())

        ratios = masses[env_ids_cpu] / obj.data.default_mass[env_ids_cpu].cpu()
        inertias = obj.root_physx_view.get_inertias()
        inertias[env_ids_cpu] = obj.data.default_inertia[env_ids_cpu].cpu() * ratios
        obj.root_physx_view.set_inertias(inertias, env_ids.cpu())

    if hasattr(env, "object_mass"):
        env.object_mass[env_ids] = mass.to(env.object_mass.device)


def reset_object_and_support_platforms(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_goal_radius_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    init_platform_cfg: SceneEntityCfg = SceneEntityCfg("object_init_platform"),
    target_platform_cfg: SceneEntityCfg = SceneEntityCfg("object_target_platform"),
) -> None:
    obj = env.scene[asset_cfg.name]
    root_states = obj.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=obj.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=obj.device)
    object_root_pos_w = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    object_quat_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    object_quat_w = math_utils.quat_mul(root_states[:, 3:7], object_quat_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=obj.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=obj.device)
    object_velocity = root_states[:, 7:13] + rand_samples

    radius_range = object_goal_radius_range or env.cfg.object_goal_radius_range
    target_offset = torch.zeros_like(object_root_pos_w)
    radius = torch.empty(object_root_pos_w.shape[0], 1, device=obj.device).uniform_(*radius_range)
    robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]
    away_xy = object_root_pos_w[:, :2] - robot_root_pos_w[:, :2]
    away_heading = torch.atan2(away_xy[:, 1:2], away_xy[:, 0:1])
    heading_jitter = torch.empty_like(radius).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)
    heading = away_heading + heading_jitter
    target_offset[:, 0:1] = radius * torch.cos(heading)
    target_offset[:, 1:2] = radius * torch.sin(heading)
    target_root_pos_w = object_root_pos_w + target_offset

    zero_platform_velocity = torch.zeros(env_ids.shape[0], 6, device=obj.device)
    env.scene[init_platform_cfg.name].write_root_state_to_sim(
        torch.cat((_platform_pose_under_object_root(object_root_pos_w), zero_platform_velocity), dim=-1),
        env_ids=env_ids,
    )
    env.scene[target_platform_cfg.name].write_root_state_to_sim(
        torch.cat((_platform_pose_under_object_root(target_root_pos_w), zero_platform_velocity), dim=-1),
        env_ids=env_ids,
    )

    obj.write_root_state_to_sim(torch.cat((object_root_pos_w, object_quat_w, object_velocity), dim=-1), env_ids=env_ids)
    _apply_object_mass_curriculum(env, env_ids, obj)
    env.object_initial_pos_w[env_ids] = _object_frame_pos_from_root(object_root_pos_w)
    env.object_target_pos_w[env_ids] = _object_frame_pos_from_root(target_root_pos_w)


def reset_randomized_object_and_support_platforms(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_goal_radius_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    init_platform_cfg: SceneEntityCfg = SceneEntityCfg("object_init_platform"),
    target_platform_cfg: SceneEntityCfg = SceneEntityCfg("object_target_platform"),
) -> None:
    """Reset one active member of the pre-scaled apple pool and park the others behind the robot."""
    del velocity_range, object_goal_radius_range
    obj = env.scene[asset_cfg.name]
    if not hasattr(obj.data, "object_pos_w"):
        raise TypeError("The randomized PnP reset requires a RigidObjectCollection")

    level = env.domain_randomization_curriculum_level
    ranges = randomization_ranges(
        level,
        start_init_x=env.cfg.domain_randomization_start_init_x,
        final_init_x=env.cfg.domain_randomization_final_init_x,
        start_init_y=env.cfg.domain_randomization_start_init_y,
        final_init_y=env.cfg.domain_randomization_final_init_y,
        start_init_z=env.cfg.domain_randomization_start_init_z,
        final_init_z=env.cfg.domain_randomization_final_init_z,
        start_goal_radius=env.cfg.domain_randomization_start_goal_radius,
        final_goal_radius=env.cfg.domain_randomization_final_goal_radius,
    )
    allowed_indices = torch.tensor(
        allowed_scale_indices(level, PNP_OBJECT_SCALE_FACTORS),
        device=obj.device,
        dtype=torch.long,
    )
    sampled_slots = torch.randint(0, allowed_indices.numel(), (env_ids.numel(),), device=obj.device)
    active_indices = allowed_indices[sampled_slots]
    env.active_object_index[env_ids] = active_indices
    scale_factors = torch.tensor(PNP_OBJECT_SCALE_FACTORS, device=obj.device)
    env.object_size_scale[env_ids] = scale_factors[active_indices]

    states = obj.data.default_object_state[env_ids].clone()
    num_objects = states.shape[1]
    states[..., :3] = env.scene.env_origins[env_ids, None, :]
    parking_xy = torch.tensor(
        inactive_object_parking_offsets(num_objects),
        device=obj.device,
        dtype=states.dtype,
    )
    states[..., :2] += parking_xy.unsqueeze(0)
    states[..., 3:7] = 0.0
    states[..., 3] = 1.0
    states[..., 7:13] = 0.0

    count = env_ids.numel()
    row_ids = torch.arange(count, device=obj.device)
    object_root_pos_w = env.scene.env_origins[env_ids].clone()
    object_root_pos_w[:, 0] += torch.empty(count, device=obj.device).uniform_(*ranges.init_x)
    object_root_pos_w[:, 1] += torch.empty(count, device=obj.device).uniform_(*ranges.init_y)
    object_root_pos_w[:, 2] += torch.empty(count, device=obj.device).uniform_(*ranges.init_z)
    yaw_range = pose_range.get("yaw", (-torch.pi, torch.pi))
    yaw = torch.empty(count, device=obj.device).uniform_(*yaw_range)
    zeros = torch.zeros_like(yaw)
    object_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    states[row_ids, active_indices, :3] = object_root_pos_w
    states[row_ids, active_indices, 3:7] = object_quat_w
    obj.write_object_state_to_sim(states, env_ids=env_ids)

    radius = torch.empty(count, 1, device=obj.device).uniform_(*ranges.goal_radius)
    robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]
    away_xy = object_root_pos_w[:, :2] - robot_root_pos_w[:, :2]
    away_heading = torch.atan2(away_xy[:, 1:2], away_xy[:, 0:1])
    heading = away_heading + torch.empty_like(radius).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)
    target_root_pos_w = object_root_pos_w.clone()
    target_root_pos_w[:, 0:1] += radius * torch.cos(heading)
    target_root_pos_w[:, 1:2] += radius * torch.sin(heading)

    env.scene[init_platform_cfg.name].write_root_pose_to_sim(
        _platform_pose_under_object_root(
            object_root_pos_w,
            platform_height=env.cfg.domain_randomization_support_height,
        ),
        env_ids=env_ids,
    )
    env.scene[target_platform_cfg.name].write_root_pose_to_sim(
        _platform_pose_under_object_root(
            target_root_pos_w,
            platform_height=env.cfg.domain_randomization_support_height,
        ),
        env_ids=env_ids,
    )

    _apply_object_mass_curriculum(env, env_ids, obj)
    frame_offset = APPLE_OBJECT_FRAME_OFFSET_Z * env.object_size_scale[env_ids]
    env.object_initial_pos_w[env_ids] = object_root_pos_w
    env.object_initial_pos_w[env_ids, 2] += frame_offset
    env.object_target_pos_w[env_ids] = target_root_pos_w
    env.object_target_pos_w[env_ids, 2] += frame_offset


def reset_multishape_object_and_support_platforms(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_goal_radius_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    init_platform_cfg: SceneEntityCfg = SceneEntityCfg("object_init_platform"),
    target_platform_cfg: SceneEntityCfg = SceneEntityCfg("object_target_platform"),
) -> None:
    """Place each selected USD on a fixed platform using its measured mesh bounds."""
    del velocity_range
    obj = env.scene[asset_cfg.name]
    count = env_ids.numel()
    dtype = obj.data.root_pos_w.dtype

    shape_indices = env.active_object_index[env_ids]
    size_scale = env.object_size_scale[env_ids].unsqueeze(-1)
    bounds_min_o = env.object_geometry_bounds_min_o[shape_indices] * size_scale
    center_offset_o = env.object_geometry_center_offsets_o[shape_indices] * size_scale

    object_root_pos_w = env.scene.env_origins[env_ids].clone()
    object_root_pos_w[:, 0] += torch.empty(count, device=obj.device, dtype=dtype).uniform_(
        *pose_range.get("x", (0.0, 0.0))
    )
    object_root_pos_w[:, 1] += torch.empty(count, device=obj.device, dtype=dtype).uniform_(
        *pose_range.get("y", (0.0, 0.0))
    )
    support_top_z = env.scene.env_origins[env_ids, 2] + env.cfg.multishape_support_height
    if env.cfg.grasp_candidate_yaw_count > 0:
        yaw = candidate_evaluation_yaws(env_ids, env.cfg.grasp_candidate_yaw_count).to(device=obj.device, dtype=dtype)
    else:
        yaw = torch.empty(count, device=obj.device, dtype=dtype).uniform_(
            *pose_range.get("yaw", (-torch.pi, torch.pi))
        )
    zeros = torch.zeros_like(yaw)
    yaw_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    stable_quat_w = env.object_stable_quat_wxyz[shape_indices]
    object_quat_w = math_utils.quat_mul(yaw_quat_w, stable_quat_w)
    object_root_pos_w[:, 2] = support_top_z - rotated_box_min_z(
        bounds_min_o,
        env.object_geometry_bounds_max_o[shape_indices] * size_scale,
        object_quat_w,
    )
    object_velocity = torch.zeros(count, 6, device=obj.device, dtype=dtype)

    center_offset_w = math_utils.quat_apply(object_quat_w, center_offset_o)
    object_center_w = object_root_pos_w + center_offset_w
    radius_range = object_goal_radius_range or env.cfg.object_goal_radius_range
    radius = torch.empty(count, 1, device=obj.device, dtype=dtype).uniform_(*radius_range)
    robot_root_pos_w = env.scene["robot"].data.root_pos_w[env_ids]
    away_xy = object_center_w[:, :2] - robot_root_pos_w[:, :2]
    away_heading = torch.atan2(away_xy[:, 1:2], away_xy[:, 0:1])
    heading = away_heading + torch.empty_like(radius).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)
    target_offset = torch.zeros_like(object_center_w)
    target_offset[:, 0:1] = radius * torch.cos(heading)
    target_offset[:, 1:2] = radius * torch.sin(heading)
    target_center_w = object_center_w + target_offset

    def platform_pose(center_w: torch.Tensor) -> torch.Tensor:
        pose = torch.zeros(count, 7, device=obj.device, dtype=dtype)
        pose[:, :2] = center_w[:, :2]
        pose[:, 2] = support_top_z - 0.5 * env.cfg.multishape_support_height
        pose[:, 3] = 1.0
        return pose

    env.scene[init_platform_cfg.name].write_root_pose_to_sim(platform_pose(object_center_w), env_ids=env_ids)
    env.scene[target_platform_cfg.name].write_root_pose_to_sim(platform_pose(target_center_w), env_ids=env_ids)
    obj.write_root_state_to_sim(
        torch.cat((object_root_pos_w, object_quat_w, object_velocity), dim=-1),
        env_ids=env_ids,
    )
    _apply_object_mass_curriculum(env, env_ids, obj)
    env.object_initial_pos_w[env_ids] = object_center_w
    env.object_initial_root_z_w[env_ids] = object_root_pos_w[:, 2]
    env.object_target_pos_w[env_ids] = target_center_w


@configclass
class G1Dex1HierDrcEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*hand.*).*"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*hand.*"),
            "static_friction_range": (1.2, 2.0),
            "dynamic_friction_range": (1.0, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=reset_robot_body_and_hand_joints,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.1, 0.1)},
    )
    reset_object = EventTerm(
        func=reset_object_and_support_platforms,
        mode="reset",
        params={
            "pose_range": {
                "x": (1.5, 2.0),
                "y": (-0.35, 0.35),
                "z": (OBJECT_ROOT_ON_PLATFORM_Z, OBJECT_ROOT_ON_PLATFORM_Z),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )
