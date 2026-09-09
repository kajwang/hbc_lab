from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils

from .multi_geometry import GEOMETRY_FAMILY_NAMES


# The group is "body", "left_hand", or "right_hand". Feet remain excluded
# because their terrain contacts are locomotion-relevant.
BODY_PROBE_SPECS = (
    ("pelvis", 0.17, "body"),
    ("waist_yaw_link", 0.14, "body"),
    ("waist_roll_link", 0.14, "body"),
    ("torso_link", 0.20, "body"),
    ("left_shoulder_roll_link", 0.11, "body"),
    ("right_shoulder_roll_link", 0.11, "body"),
    ("left_elbow_link", 0.10, "body"),
    ("right_elbow_link", 0.10, "body"),
    ("left_hip_pitch_link", 0.12, "body"),
    ("right_hip_pitch_link", 0.12, "body"),
    ("left_knee_link", 0.11, "body"),
    ("right_knee_link", 0.11, "body"),
    ("left_wrist_yaw_link", 0.065, "left_hand"),
    ("right_wrist_yaw_link", 0.065, "right_hand"),
    ("left_hand_Link1_2", 0.030, "left_hand"),
    ("left_hand_Link2_2", 0.030, "left_hand"),
    ("right_hand_Link1_2", 0.030, "right_hand"),
    ("right_hand_Link2_2", 0.030, "right_hand"),
)
_PROBE_GROUP_ID = {"body": 0, "left_hand": 1, "right_hand": 2}


def initialize_obstacle_box_buffers(env, box_count: int) -> None:
    """Allocate the task-independent oriented-box geometry interface."""
    if hasattr(env, "scene_obstacle_box_centers_w"):
        if env.scene_obstacle_box_centers_w.shape[1] != box_count:
            raise ValueError("Obstacle box count cannot change after environment initialization.")
        return
    dtype = env.scene["robot"].data.root_pos_w.dtype
    env.scene_obstacle_box_centers_w = torch.zeros(env.num_envs, box_count, 3, device=env.device, dtype=dtype)
    env.scene_obstacle_box_quats_w = torch.zeros(env.num_envs, box_count, 4, device=env.device, dtype=dtype)
    env.scene_obstacle_box_quats_w[..., 0] = 1.0
    env.scene_obstacle_box_half_extents = torch.zeros(
        env.num_envs, box_count, 3, device=env.device, dtype=dtype
    )
    env.scene_obstacle_box_active = torch.zeros(env.num_envs, box_count, device=env.device, dtype=torch.bool)


def write_obstacle_boxes(
    env,
    env_ids: torch.Tensor,
    centers_w: torch.Tensor,
    quats_w: torch.Tensor,
    half_extents: torch.Tensor,
    active: torch.Tensor,
) -> None:
    initialize_obstacle_box_buffers(env, centers_w.shape[1])
    env.scene_obstacle_box_centers_w[env_ids] = centers_w
    env.scene_obstacle_box_quats_w[env_ids] = quats_w
    env.scene_obstacle_box_half_extents[env_ids] = half_extents
    env.scene_obstacle_box_active[env_ids] = active


def point_oriented_box_signed_distance(
    points_w: torch.Tensor,
    box_centers_w: torch.Tensor,
    box_quats_w: torch.Tensor,
    box_half_extents: torch.Tensor,
) -> torch.Tensor:
    """Return signed distances with shape ``(env, box, point)``."""
    env_count, point_count, _ = points_w.shape
    box_count = box_centers_w.shape[1]
    relative_w = points_w[:, None, :, :] - box_centers_w[:, :, None, :]
    quats = box_quats_w[:, :, None, :].expand(-1, -1, point_count, -1)
    relative_b = math_utils.quat_apply_inverse(
        quats.reshape(-1, 4), relative_w.reshape(-1, 3)
    ).reshape(env_count, box_count, point_count, 3)
    q = torch.abs(relative_b) - box_half_extents[:, :, None, :]
    outside = torch.linalg.norm(torch.relu(q), dim=-1)
    inside = torch.clamp(q.amax(dim=-1), max=0.0)
    return outside + inside


def _body_probe_metadata(
    env, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = getattr(env, "_geometry_body_probe_cache", None)
    if cached is not None and cached[1].dtype == dtype:
        return cached

    body_names = env.scene["robot"].body_names
    body_ids = []
    radii = []
    groups = []
    for requested_name, radius, group in BODY_PROBE_SPECS:
        if requested_name in body_names:
            body_ids.append(body_names.index(requested_name))
            radii.append(radius)
            groups.append(_PROBE_GROUP_ID[group])
    if not body_ids:
        raise RuntimeError("No geometry body probes matched the robot body names.")
    result = (
        torch.tensor(body_ids, device=env.device, dtype=torch.long),
        torch.tensor(radii, device=env.device, dtype=dtype),
        torch.tensor(groups, device=env.device, dtype=torch.long),
    )
    env._geometry_body_probe_cache = result
    return result


def body_obstacle_clearance(env, include_hands: bool = False) -> torch.Tensor:
    """Minimum surface clearance for each body probe, shaped ``(env, probe)``."""
    robot = env.scene["robot"]
    body_ids, radii, groups = _body_probe_metadata(env, robot.data.body_pos_w.dtype)
    if not include_hands:
        body_mask = groups == _PROBE_GROUP_ID["body"]
        body_ids = body_ids[body_mask]
        radii = radii[body_mask]
    points_w = robot.data.body_pos_w[:, body_ids]
    if not hasattr(env, "scene_obstacle_box_centers_w"):
        return torch.full(points_w.shape[:2], 3.0, device=env.device, dtype=points_w.dtype)

    signed_distance = point_oriented_box_signed_distance(
        points_w,
        env.scene_obstacle_box_centers_w,
        env.scene_obstacle_box_quats_w,
        env.scene_obstacle_box_half_extents,
    )
    clearance = signed_distance - radii[None, None, :]
    clearance = torch.where(
        env.scene_obstacle_box_active[:, :, None],
        clearance,
        torch.full_like(clearance, torch.inf),
    )
    return clearance.amin(dim=1)


def geometry_conditioned_posture_reward(
    env,
    safe_margin: float,
    transition_width: float,
    relative_coefficient: float,
    active_hand_relax_distance: float,
) -> torch.Tensor:
    """Avoid scene geometry while releasing the active hand near its task target."""
    clearance = body_obstacle_clearance(env, include_hands=True)
    _, _, groups = _body_probe_metadata(env, clearance.dtype)
    has_geometry = getattr(
        env,
        "scene_obstacle_box_active",
        torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device),
    ).any(dim=-1)
    normalized_violation = torch.relu(safe_margin - clearance) / transition_width
    probe_risk = normalized_violation.clamp(max=1.0).square()
    probe_risk = probe_risk * has_geometry[:, None]
    active_hand_weight = torch.tanh(
        env.d_active_hand.detach() / max(active_hand_relax_distance, 1.0e-6)
    )
    probe_weights = torch.ones_like(probe_risk)
    left_active = env.active_hand == 0
    left_probe = groups == _PROBE_GROUP_ID["left_hand"]
    right_probe = groups == _PROBE_GROUP_ID["right_hand"]
    probe_weights[:, left_probe] = torch.where(
        left_active[:, None], active_hand_weight[:, None], torch.ones_like(active_hand_weight[:, None])
    )
    probe_weights[:, right_probe] = torch.where(
        (~left_active)[:, None], active_hand_weight[:, None], torch.ones_like(active_hand_weight[:, None])
    )
    weighted_risk = probe_risk * probe_weights
    risk = 0.7 * weighted_risk.amax(dim=-1) + 0.3 * weighted_risk.mean(dim=-1)
    stage_scale = 2.0 * env.W_app + 20.0 * env.W_couple + 200.0 * env.W_manip
    reward = -relative_coefficient * stage_scale.detach() * risk

    minimum_clearance = clearance.amin(dim=-1)
    minimum_clearance = torch.where(has_geometry, minimum_clearance, torch.full_like(minimum_clearance, 3.0))
    unsafe = has_geometry & (minimum_clearance < safe_margin)
    robot = env.scene["robot"]
    root_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    _, torso_pitch, _ = math_utils.euler_xyz_from_quat(robot.data.body_quat_w[:, env.torso_body_id])

    log = env.extras["log"]
    log["Geometry/min_clearance_mean"] = minimum_clearance.clamp(-0.5, 3.0).mean()
    log["Geometry/unsafe_ratio"] = unsafe.float().mean()
    log["Geometry/risk_mean"] = risk.mean()
    log["Geometry/active_hand_collision_weight"] = active_hand_weight.mean()
    log["Geometry/scaled_reward_mean"] = reward.mean()
    family_id = getattr(env, "geometry_family_id", None)
    if family_id is None:
        family_masks = (("all", torch.ones_like(has_geometry)),)
    else:
        family_masks = tuple(
            (name, family_id == family_index)
            for family_index, name in enumerate(GEOMETRY_FAMILY_NAMES)
        )
    for name, mask in family_masks:
        if bool(mask.any()):
            log[f"Geometry/{name}_root_height_actual"] = root_height[mask].mean()
            log[f"Geometry/{name}_torso_pitch_actual"] = torso_pitch[mask].mean()
            log[f"Geometry/{name}_root_height_command"] = env.command_state.posture_command[mask, 0].mean()
            log[f"Geometry/{name}_torso_pitch_command"] = env.command_state.posture_command[mask, 1].mean()
            log[f"Geometry/{name}_min_clearance"] = minimum_clearance[mask].clamp(-0.5, 3.0).mean()
            log[f"Geometry/{name}_unsafe_ratio"] = unsafe[mask].float().mean()
    return reward
