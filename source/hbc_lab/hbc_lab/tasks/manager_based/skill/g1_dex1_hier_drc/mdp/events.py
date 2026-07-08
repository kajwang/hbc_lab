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


def _find_joint_ids(asset: Articulation, joint_names: list[str]) -> list[int]:
    joint_ids, resolved_joint_names = asset.find_joints(joint_names, preserve_order=False)
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(f"Expected {len(joint_names)} joints, got {len(joint_ids)}: {resolved_joint_names}")
    return list(joint_ids)


def _find_gripper_joint_ids(asset: Articulation) -> list[int]:
    return _find_joint_ids(asset, [*G1_DEX1_LEFT_GRIPPER_JOINT_NAMES, *G1_DEX1_RIGHT_GRIPPER_JOINT_NAMES])


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


def _platform_pose_under_object_root(object_root_pos_w: torch.Tensor) -> torch.Tensor:
    platform_pos_w = object_root_pos_w.clone()
    platform_pos_w[:, 2] = object_root_pos_w[:, 2] - 0.5 * OBJECT_PLATFORM_HEIGHT
    platform_quat_w = torch.zeros(object_root_pos_w.shape[0], 4, device=object_root_pos_w.device)
    platform_quat_w[:, 0] = 1.0
    return torch.cat((platform_pos_w, platform_quat_w), dim=-1)


def _object_frame_pos_from_root(object_root_pos_w: torch.Tensor) -> torch.Tensor:
    object_frame_pos_w = object_root_pos_w.clone()
    object_frame_pos_w[:, 2] += APPLE_OBJECT_FRAME_OFFSET_Z
    return object_frame_pos_w


def _apply_object_mass_curriculum(env, env_ids: torch.Tensor, obj) -> None:
    if not getattr(env.cfg, "object_mass_curriculum_enabled", False):
        return
    if not hasattr(env, "object_mass_curriculum_level"):
        return

    mass = sample_object_masses(
        env.object_mass_curriculum_level,
        num_envs=env_ids.shape[0],
        device=obj.device,
        dtype=obj.data.root_pos_w.dtype,
        start_w=env.cfg.object_mass_start_w,
        mid_w=env.cfg.object_mass_mid_w,
        end_w=env.cfg.object_mass_end_w,
        start_mass=env.cfg.object_mass_start_mass,
        mid_mass=env.cfg.object_mass_mid_mass,
        final_mass=env.cfg.object_mass_final_mass,
        mid_log_std=env.cfg.object_mass_mid_log_std,
        final_log_std=env.cfg.object_mass_final_log_std,
        min_mass=env.cfg.object_mass_min,
        max_mass=env.cfg.object_mass_max,
    )

    env_ids_cpu = env_ids.cpu()
    mass_cpu = mass.detach().cpu()
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
    heading = torch.empty(object_root_pos_w.shape[0], 1, device=obj.device).uniform_(-3.14159, 3.14159)
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
