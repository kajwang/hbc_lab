from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BimanualSupportProgress:
    distance: torch.Tensor
    left_distance: torch.Tensor
    right_distance: torch.Tensor
    left_support: torch.Tensor
    right_support: torch.Tensor
    left_contact_gate: torch.Tensor
    right_contact_gate: torch.Tensor
    couple_gate: torch.Tensor
    support_density: torch.Tensor


@dataclass
class BimanualPositionRelation:
    left_distance: torch.Tensor
    right_distance: torch.Tensor
    opposition: torch.Tensor
    radial_balance: torch.Tensor
    height_alignment: torch.Tensor
    score: torch.Tensor


@dataclass
class LiftGatedTransportProgress:
    lift_progress: torch.Tensor
    transport_progress: torch.Tensor
    transport_gate: torch.Tensor
    reward: torch.Tensor


def compute_lift_gated_transport_progress(
    lift_height: torch.Tensor,
    transport_progress: torch.Tensor,
    lift_target_height: float,
) -> LiftGatedTransportProgress:
    if lift_target_height <= 0.0:
        raise ValueError(f"lift_target_height must be positive, got {lift_target_height}")
    if lift_height.shape != transport_progress.shape:
        raise ValueError("lift_height and transport_progress must have identical shapes")

    lift_progress = torch.clamp(lift_height / lift_target_height, min=0.0, max=1.0)
    transport_progress = torch.clamp(transport_progress, min=0.0, max=1.0)
    transport_gate = lift_progress
    reward = 0.3 + 0.35 * lift_progress + 0.35 * transport_gate * transport_progress
    return LiftGatedTransportProgress(
        lift_progress=lift_progress,
        transport_progress=transport_progress,
        transport_gate=transport_gate,
        reward=reward,
    )


def compute_bimanual_position_relation(
    object_pos: torch.Tensor,
    left_hand_pos: torch.Tensor,
    right_hand_pos: torch.Tensor,
    radial_balance_scale: float,
    height_alignment_scale: float,
) -> BimanualPositionRelation:
    """Score an axis-free horizontal opposing grasp around the object center."""
    if radial_balance_scale <= 0.0:
        raise ValueError(f"radial_balance_scale must be positive, got {radial_balance_scale}")
    if height_alignment_scale <= 0.0:
        raise ValueError(f"height_alignment_scale must be positive, got {height_alignment_scale}")
    if not (object_pos.shape == left_hand_pos.shape == right_hand_pos.shape):
        raise ValueError("object and hand positions must have identical shapes")
    if object_pos.ndim != 2 or object_pos.shape[-1] != 3:
        raise ValueError("object and hand positions must have shape (num_envs, 3)")

    left_vector = left_hand_pos - object_pos
    right_vector = right_hand_pos - object_pos
    left_distance = torch.linalg.vector_norm(left_vector, dim=-1)
    right_distance = torch.linalg.vector_norm(right_vector, dim=-1)
    left_radius_xy = torch.linalg.vector_norm(left_vector[:, :2], dim=-1)
    right_radius_xy = torch.linalg.vector_norm(right_vector[:, :2], dim=-1)
    norm_product = left_radius_xy * right_radius_xy
    cosine = torch.sum(left_vector[:, :2] * right_vector[:, :2], dim=-1) / norm_product.clamp_min(
        1.0e-6
    )
    opposition = 0.5 * (1.0 - torch.clamp(cosine, min=-1.0, max=1.0))
    opposition = opposition * (norm_product > 1.0e-6).to(dtype=opposition.dtype)
    radial_balance = torch.exp(-torch.abs(left_radius_xy - right_radius_xy) / radial_balance_scale)
    height_alignment = torch.exp(
        -torch.abs(left_vector[:, 2] - right_vector[:, 2]) / height_alignment_scale
    )
    return BimanualPositionRelation(
        left_distance=left_distance,
        right_distance=right_distance,
        opposition=opposition,
        radial_balance=radial_balance,
        height_alignment=height_alignment,
        score=opposition * radial_balance * height_alignment,
    )


def compute_independent_support_reward_terms(
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
    left_support: torch.Tensor,
    right_support: torch.Tensor,
    distance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate each hand's support using its own clearance from the grasp region."""
    if distance_scale <= 0.0:
        raise ValueError(f"distance_scale must be positive, got {distance_scale}")
    if not (left_distance.shape == right_distance.shape == left_support.shape == right_support.shape):
        raise ValueError("distance and support tensors must have identical shapes")

    left_gate = 1.0 - torch.tanh(left_distance / distance_scale)
    right_gate = 1.0 - torch.tanh(right_distance / distance_scale)
    gated_support = 0.5 * (left_gate * left_support + right_gate * right_support)
    early_contact = 0.5 * ((1.0 - left_gate) * left_support + (1.0 - right_gate) * right_support)
    return left_gate, right_gate, gated_support, early_contact


def compute_bimanual_distance_gated_couple(
    left_gate: torch.Tensor,
    right_gate: torch.Tensor,
    raw_couple: torch.Tensor,
) -> torch.Tensor:
    """Allow couple progress only where both hands satisfy their distance gates."""
    if not (left_gate.shape == right_gate.shape == raw_couple.shape):
        raise ValueError("distance gates and raw_couple must have identical shapes")
    return torch.minimum(left_gate, right_gate) * raw_couple


def compute_bimanual_support_progress(
    left_distance: torch.Tensor,
    right_distance: torch.Tensor,
    left_region_contacts: torch.Tensor,
    right_region_contacts: torch.Tensor,
) -> BimanualSupportProgress:
    """Compute center approach and two-hand support from semantic contact regions."""
    if left_distance.shape != right_distance.shape:
        raise ValueError("left_distance and right_distance must have identical shapes")
    if left_region_contacts.shape != right_region_contacts.shape:
        raise ValueError("left and right region contacts must have identical shapes")
    if left_region_contacts.ndim != 2 or left_region_contacts.shape[0] != left_distance.shape[0]:
        raise ValueError("region contacts must have shape (num_envs, num_regions)")
    left_support = torch.mean(left_region_contacts, dim=-1)
    right_support = torch.mean(right_region_contacts, dim=-1)
    left_contact_gate = torch.amax(left_region_contacts, dim=-1)
    right_contact_gate = torch.amax(right_region_contacts, dim=-1)
    return BimanualSupportProgress(
        distance=torch.maximum(left_distance, right_distance),
        left_distance=left_distance,
        right_distance=right_distance,
        left_support=left_support,
        right_support=right_support,
        left_contact_gate=left_contact_gate,
        right_contact_gate=right_contact_gate,
        couple_gate=torch.minimum(left_contact_gate, right_contact_gate),
        support_density=0.5 * (left_support + right_support),
    )
