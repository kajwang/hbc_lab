from __future__ import annotations

import torch


def _require_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def reachability_gate(
    distance: torch.Tensor,
    release_radius: float,
    release_width: float,
) -> torch.Tensor:
    _require_positive(release_width, "release_width")
    return torch.sigmoid((release_radius - distance) / release_width)


def smooth_pose_guidance(
    position_error: torch.Tensor,
    orientation_error: torch.Tensor,
    *,
    position_scale: float,
    orientation_gate_scale: float,
    position_weight: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if position_error.shape != orientation_error.shape:
        raise ValueError(
            "position_error and orientation_error must have the same shape, "
            f"got {position_error.shape} and {orientation_error.shape}"
    )
    _require_positive(position_scale, "position_scale")
    _require_positive(orientation_gate_scale, "orientation_gate_scale")
    if not 0.0 <= position_weight <= 1.0:
        raise ValueError(f"position_weight must be in [0, 1], got {position_weight}")

    position_score = 1.0 - torch.tanh(position_error / position_scale)
    orientation_near_gate = 1.0 - torch.tanh(position_error / orientation_gate_scale)
    normalized_orientation_error = torch.clamp(orientation_error, min=0.0, max=torch.pi) / torch.pi
    orientation_score = orientation_near_gate * (1.0 - normalized_orientation_error)
    guidance = position_weight * position_score + (1.0 - position_weight) * orientation_score
    return guidance, position_score, orientation_score


def stable_grasp_released_pose_guidance(
    position_error: torch.Tensor,
    orientation_error: torch.Tensor,
    physical_grasp: torch.Tensor,
    *,
    position_scale: float,
    release_threshold: float,
    release_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep position and orientation guidance independent until grasp is stable."""
    if position_error.shape != orientation_error.shape or position_error.shape != physical_grasp.shape:
        raise ValueError("position_error, orientation_error, and physical_grasp must have the same shape")
    _require_positive(position_scale, "position_scale")
    _require_positive(release_width, "release_width")
    if not 0.0 <= release_threshold <= 1.0:
        raise ValueError(f"release_threshold must be in [0, 1], got {release_threshold}")

    position_score = 1.0 - torch.tanh(position_error / position_scale)
    orientation_score = 1.0 - torch.clamp(orientation_error, min=0.0, max=torch.pi) / torch.pi
    release_gate = torch.sigmoid((physical_grasp - release_threshold) / release_width)
    position_guidance = torch.lerp(position_score, torch.ones_like(position_score), release_gate)
    orientation_guidance = torch.lerp(orientation_score, torch.ones_like(orientation_score), release_gate)
    return position_guidance, orientation_guidance, position_score, orientation_score, release_gate


def comfort_weights(effector_mask: torch.Tensor, reach_gate: torch.Tensor) -> torch.Tensor:
    if effector_mask.shape != reach_gate.shape:
        raise ValueError(
            f"effector_mask and reach_gate must have the same shape, got {effector_mask.shape} and {reach_gate.shape}"
        )
    return (1.0 - effector_mask) + effector_mask * (1.0 - reach_gate)


def walking_posture_weight(
    effector_mask: torch.Tensor,
    reach_gate: torch.Tensor,
    near_floor: float,
) -> torch.Tensor:
    if not 0.0 <= near_floor <= 1.0:
        raise ValueError(f"near_floor must be in [0, 1], got {near_floor}")
    if effector_mask.shape != reach_gate.shape:
        raise ValueError(
            f"effector_mask and reach_gate must have the same shape, got {effector_mask.shape} and {reach_gate.shape}"
        )
    body_gate = torch.amax(effector_mask * reach_gate, dim=-1)
    return near_floor + (1.0 - near_floor) * (1.0 - body_gate)


def normalized_deadzone_square(value: torch.Tensor, deadzone: float, scale: float) -> torch.Tensor:
    if deadzone < 0.0:
        raise ValueError(f"deadzone must be non-negative, got {deadzone}")
    _require_positive(scale, "scale")
    return torch.square(torch.relu(torch.abs(value) - deadzone) / scale)


def radial_workspace_violation(
    position: torch.Tensor,
    radius_range: tuple[float, float],
    scale: float,
) -> torch.Tensor:
    _require_positive(scale, "scale")
    radius_min, radius_max = radius_range
    if radius_min < 0.0 or radius_min >= radius_max:
        raise ValueError(f"invalid radius_range {radius_range}")
    radius = torch.linalg.norm(position, dim=-1)
    inner = torch.relu(radius_min - radius)
    outer = torch.relu(radius - radius_max)
    return torch.square((inner + outer) / scale)


def normalized_joint_limit_proximity(
    joint_position: torch.Tensor,
    joint_limits: torch.Tensor,
    margin_fraction: float = 0.10,
) -> torch.Tensor:
    if not 0.0 < margin_fraction <= 1.0:
        raise ValueError(f"margin_fraction must be in (0, 1], got {margin_fraction}")
    if joint_limits.shape[-1] != 2 or joint_limits.shape[:-1] != joint_position.shape:
        raise ValueError(
            f"joint_limits must have shape {joint_position.shape + (2,)}, got {joint_limits.shape}"
        )
    lower = joint_limits[..., 0]
    upper = joint_limits[..., 1]
    half_range = torch.clamp(0.5 * (upper - lower), min=1.0e-6)
    center = 0.5 * (upper + lower)
    normalized_offset = torch.abs(joint_position - center) / half_range
    return torch.square(torch.relu(normalized_offset - (1.0 - margin_fraction)) / margin_fraction)


def quaternion_angular_distance(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape[-1] != 4 or rhs.shape[-1] != 4:
        raise ValueError(f"quaternions must have (..., 4) shapes, got {lhs.shape} and {rhs.shape}")
    try:
        lhs, rhs = torch.broadcast_tensors(lhs, rhs)
    except RuntimeError as error:
        raise ValueError(f"quaternion shapes are not broadcastable: {lhs.shape} and {rhs.shape}") from error
    lhs = lhs / torch.clamp(torch.linalg.norm(lhs, dim=-1, keepdim=True), min=1.0e-6)
    rhs = rhs / torch.clamp(torch.linalg.norm(rhs, dim=-1, keepdim=True), min=1.0e-6)
    cosine_half_angle = torch.abs(torch.sum(lhs * rhs, dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(cosine_half_angle)


def mask_normalized_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError(f"values and weights must have the same shape, got {values.shape} and {weights.shape}")
    return torch.sum(values * weights, dim=-1) / torch.clamp(torch.sum(weights, dim=-1), min=1.0e-6)


def scaled_quality_reward(
    stage_scale: torch.Tensor,
    quality_loss: torch.Tensor,
    coefficient: float = 0.10,
    loss_cap: float = 2.0,
) -> torch.Tensor:
    if coefficient < 0.0:
        raise ValueError(f"coefficient must be non-negative, got {coefficient}")
    _require_positive(loss_cap, "loss_cap")
    return -coefficient * stage_scale.detach() * torch.clamp(quality_loss, 0.0, loss_cap)
