from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


GRASP_MODE_TOP = 0
GRASP_MODE_SIDE = 1
GRASP_MODE_OBLIQUE = 2
NUM_GRASP_MODES = 3
NUM_GRASP_CANDIDATES = 32

GRASP_REFERENCE_DATA_PATH = (
    Path(__file__).resolve().parents[5]
    / "assets/models/objects/grasp_references/multishape_grasp_references_k32.npz"
)


@dataclass(frozen=True)
class GraspReferenceLibrary:
    shape_names: tuple[str, ...]
    positions_o: torch.Tensor
    quaternions_o: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor
    modes: torch.Tensor

    def __post_init__(self) -> None:
        num_shapes = len(self.shape_names)
        if self.positions_o.ndim != 3 or self.positions_o.shape[:1] != (num_shapes,) or self.positions_o.shape[-1] != 3:
            raise ValueError("positions_o must have shape (num_shapes, num_candidates, 3)")
        num_candidates = self.positions_o.shape[1]
        expected_pose = (num_shapes, num_candidates)
        if self.quaternions_o.shape != (*expected_pose, 4):
            raise ValueError("quaternions_o must match positions_o and contain scalar-first quaternions")
        for name, value in (("scores", self.scores), ("valid", self.valid), ("modes", self.modes)):
            if value.shape != expected_pose:
                raise ValueError(f"{name} must have shape {expected_pose}")
        if len(set(self.shape_names)) != num_shapes:
            raise ValueError("shape_names must be unique")
        if torch.any(self.valid & ((self.modes < 0) | (self.modes >= NUM_GRASP_MODES))):
            raise ValueError("valid candidates must use a known grasp mode")


@dataclass(frozen=True)
class CandidateEvaluationMetadata:
    shape_name: str
    candidate_ids: tuple[int, ...]
    scores: tuple[float, ...]
    modes: tuple[int, ...]


def load_candidate_evaluation_metadata(
    data_path: str | Path,
    shape_name: str,
) -> CandidateEvaluationMetadata:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Grasp reference library does not exist: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        shape_names = tuple(str(name) for name in data["shape_names"].tolist())
        if shape_name not in shape_names:
            raise ValueError(f"Unknown grasp-reference shape '{shape_name}'")
        shape_index = shape_names.index(shape_name)
        valid = np.asarray(data["grasp_valid"][shape_index], dtype=np.bool_)
        candidate_ids = tuple(int(index) for index in np.flatnonzero(valid))
        if not candidate_ids:
            raise ValueError(f"Shape '{shape_name}' has no valid grasp candidates")
        scores = tuple(float(value) for value in data["grasp_scores"][shape_index, valid])
        modes = tuple(int(value) for value in data["grasp_modes"][shape_index, valid])
    return CandidateEvaluationMetadata(
        shape_name=shape_name,
        candidate_ids=candidate_ids,
        scores=scores,
        modes=modes,
    )


def assign_candidate_indices(
    env_ids: torch.Tensor,
    candidate_ids: tuple[int, ...],
) -> torch.Tensor:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if env_ids.numel() and int(env_ids.min()) < 0:
        raise ValueError("env_ids must be non-negative")
    candidates = torch.as_tensor(candidate_ids, device=env_ids.device, dtype=torch.long)
    return candidates[torch.remainder(env_ids, candidates.numel())]


def candidate_evaluation_yaws(env_ids: torch.Tensor, yaw_count: int) -> torch.Tensor:
    """Map global environment IDs to an evenly spaced yaw sweep in ``[0, 2 pi)``."""
    if yaw_count < 1:
        raise ValueError("yaw_count must be positive")
    env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
    return torch.remainder(env_ids, yaw_count).to(dtype=torch.float32) * (2.0 * torch.pi / yaw_count)


def valid_rigid_transforms(transforms: torch.Tensor, atol: float = 1.0e-4) -> torch.Tensor:
    """Return a mask selecting finite homogeneous transforms with proper rotation matrices."""
    if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape (num_candidates, 4, 4)")
    finite = torch.isfinite(transforms).all(dim=(-2, -1))
    rotations = torch.nan_to_num(transforms[:, :3, :3])
    identity = torch.eye(3, device=transforms.device, dtype=transforms.dtype).expand_as(rotations)
    orthogonal = torch.isclose(
        rotations.transpose(-1, -2) @ rotations,
        identity,
        atol=atol,
        rtol=0.0,
    ).all(dim=(-2, -1))
    proper = torch.isclose(torch.linalg.det(rotations), torch.ones_like(finite, dtype=transforms.dtype), atol=atol, rtol=0.0)
    expected_bottom = torch.tensor((0.0, 0.0, 0.0, 1.0), device=transforms.device, dtype=transforms.dtype)
    homogeneous = torch.isclose(transforms[:, 3], expected_bottom, atol=atol, rtol=0.0).all(dim=-1)
    return finite & orthogonal & proper & homogeneous


def matrix_to_quaternion_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized scalar-first quaternions."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end with shape (3, 3)")
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    quat = torch.stack(
        (
            0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)),
            0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)), m21 - m12),
            0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)), m02 - m20),
            0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)), m10 - m01),
        ),
        dim=-1,
    )
    quat = torch.nn.functional.normalize(quat, dim=-1)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose scalar-first quaternions as ``left * right``."""
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion inputs must have equal shape ending in 4")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors using scalar-first quaternions."""
    if quaternion.shape[:-1] != vector.shape[:-1] or quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("quaternion/vector batch shapes must match")
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def compose_object_grasp_pose(
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    grasp_pos_o: torch.Tensor,
    grasp_quat_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform object-local grasp poses into world coordinates using the live object pose."""
    grasp_pos_w = object_pos_w + quaternion_apply_wxyz(object_quat_w, grasp_pos_o)
    grasp_quat_w = torch.nn.functional.normalize(
        quaternion_multiply_wxyz(object_quat_w, grasp_quat_o),
        dim=-1,
    )
    return grasp_pos_w, grasp_quat_w


def classify_approach_modes(
    rotations_o: torch.Tensor,
    *,
    support_up_o: torch.Tensor,
    top_angle_rad: float = 0.6108652381980153,
    side_angle_rad: float = 0.9599310885968813,
) -> torch.Tensor:
    """Classify the grasp +Z approach axis relative to the support-plane normal."""
    if rotations_o.ndim != 3 or rotations_o.shape[-2:] != (3, 3):
        raise ValueError("rotations_o must have shape (num_candidates, 3, 3)")
    support_up_o = torch.as_tensor(support_up_o, device=rotations_o.device, dtype=rotations_o.dtype)
    if support_up_o.shape != (3,):
        raise ValueError("support_up_o must have shape (3,)")
    support_up_o = torch.nn.functional.normalize(support_up_o, dim=-1)
    approach_o = rotations_o[..., :, 2]
    vertical_alignment = torch.abs(torch.sum(approach_o * support_up_o, dim=-1))
    top_threshold = torch.cos(torch.as_tensor(top_angle_rad, device=rotations_o.device, dtype=rotations_o.dtype))
    side_threshold = torch.cos(torch.as_tensor(side_angle_rad, device=rotations_o.device, dtype=rotations_o.dtype))
    modes = torch.full_like(vertical_alignment, GRASP_MODE_OBLIQUE, dtype=torch.long)
    modes[vertical_alignment >= top_threshold] = GRASP_MODE_TOP
    modes[vertical_alignment <= side_threshold] = GRASP_MODE_SIDE
    return modes


def ergonomic_wrist_mask(
    rotations_o: torch.Tensor,
    *,
    support_up_o: torch.Tensor,
    max_roll_rad: float | torch.Tensor,
    top_down_projection_threshold: float = 0.25,
) -> torch.Tensor:
    """Keep grasps whose hand +Z is upright after fixing the hand +Y approach axis.

    For a top-down approach the world-up projection is ill-defined, so roll is
    intentionally left unconstrained. This removes the 180-degree jaw-swap
    symmetry that is geometrically valid for a parallel gripper but awkward for
    the humanoid wrist.
    """
    if rotations_o.ndim != 3 or rotations_o.shape[-2:] != (3, 3):
        raise ValueError("rotations_o must have shape (num_candidates, 3, 3)")
    support_up_o = torch.as_tensor(support_up_o, device=rotations_o.device, dtype=rotations_o.dtype)
    if support_up_o.shape != (3,):
        raise ValueError("support_up_o must have shape (3,)")
    support_up_o = torch.nn.functional.normalize(support_up_o, dim=-1)
    max_roll_rad = torch.as_tensor(max_roll_rad, device=rotations_o.device, dtype=rotations_o.dtype)
    if max_roll_rad.numel() != 1 or not 0.0 <= float(max_roll_rad) <= torch.pi:
        raise ValueError("max_roll_rad must be a scalar in [0, pi]")

    approach_o = rotations_o[..., :, 1]
    hand_up_o = rotations_o[..., :, 2]
    up_along_approach = torch.sum(approach_o * support_up_o, dim=-1, keepdim=True)
    projected_up_o = support_up_o - up_along_approach * approach_o
    projected_norm = torch.linalg.vector_norm(projected_up_o, dim=-1)
    preferred_up_o = torch.nn.functional.normalize(projected_up_o, dim=-1, eps=1.0e-6)
    upright_alignment = torch.sum(hand_up_o * preferred_up_o, dim=-1)
    top_down = projected_norm < float(top_down_projection_threshold)
    return top_down | (upright_alignment >= torch.cos(max_roll_rad))


def _rotation_angle(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    relative = left.transpose(-1, -2) @ right
    cosine = torch.clamp((torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5, -1.0, 1.0)
    return torch.acos(cosine)


def se3_nms_indices(
    transforms: torch.Tensor,
    scores: torch.Tensor,
    *,
    translation_threshold: float,
    rotation_threshold_rad: float,
) -> torch.Tensor:
    """Greedily retain score-ordered candidates that are distinct in both translation and rotation."""
    if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape (num_candidates, 4, 4)")
    if scores.shape != (transforms.shape[0],):
        raise ValueError("scores must have one value per transform")
    order = torch.argsort(scores, descending=True, stable=True)
    kept: list[torch.Tensor] = []
    for index in order:
        if not kept:
            kept.append(index)
            continue
        previous = torch.stack(kept)
        translation_distance = torch.linalg.vector_norm(
            transforms[previous, :3, 3] - transforms[index, :3, 3],
            dim=-1,
        )
        rotation_distance = _rotation_angle(
            transforms[previous, :3, :3],
            transforms[index, :3, :3].expand(previous.numel(), -1, -1),
        )
        duplicate = (translation_distance < translation_threshold) & (rotation_distance < rotation_threshold_rad)
        if not bool(torch.any(duplicate)):
            kept.append(index)
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=transforms.device)


def balanced_candidate_indices(
    transforms: torch.Tensor,
    scores: torch.Tensor,
    modes: torch.Tensor,
    *,
    max_candidates: int = NUM_GRASP_CANDIDATES,
) -> torch.Tensor:
    """Select a score-ranked but mode-balanced subset from already deduplicated candidates."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if scores.shape != modes.shape or scores.shape != (transforms.shape[0],):
        raise ValueError("transforms, scores, and modes must contain the same number of candidates")
    order = torch.argsort(scores, descending=True, stable=True)
    quota = max_candidates // NUM_GRASP_MODES
    selected: list[int] = []
    for mode in range(NUM_GRASP_MODES):
        mode_indices = order[modes[order] == mode][:quota]
        selected.extend(int(index) for index in mode_indices)
    selected_set = set(selected)
    for index in order:
        scalar_index = int(index)
        if scalar_index not in selected_set:
            selected.append(scalar_index)
            selected_set.add(scalar_index)
        if len(selected) >= max_candidates:
            break
    selected = selected[:max_candidates]
    selected.sort(key=lambda index: (-float(scores[index]), index))
    return torch.as_tensor(selected, dtype=torch.long, device=transforms.device)


def sample_valid_candidate_indices(
    valid_by_shape: torch.Tensor,
    shape_indices: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniformly sample one non-padded candidate for each requested shape."""
    if valid_by_shape.ndim != 2 or valid_by_shape.dtype != torch.bool:
        raise ValueError("valid_by_shape must be a boolean (num_shapes, num_candidates) tensor")
    shape_indices = shape_indices.to(device=valid_by_shape.device, dtype=torch.long).reshape(-1)
    if shape_indices.numel() and (int(shape_indices.min()) < 0 or int(shape_indices.max()) >= valid_by_shape.shape[0]):
        raise ValueError("shape_indices contains an out-of-range shape")
    candidate_weights = valid_by_shape[shape_indices].to(dtype=torch.float32)
    if torch.any(candidate_weights.sum(dim=-1) == 0.0):
        raise ValueError("every selected shape must contain at least one valid candidate")
    return torch.multinomial(candidate_weights, num_samples=1, replacement=True, generator=generator).squeeze(-1)


def accessible_candidate_weights(
    *,
    candidate_positions_w: torch.Tensor,
    candidate_quaternions_w: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    object_positions_w: torch.Tensor,
    robot_positions_w: torch.Tensor,
    robot_quaternions_w: torch.Tensor,
    active_hand: torch.Tensor,
    support_top_z: torch.Tensor,
    active_shoulder_positions_w: torch.Tensor | None = None,
    pregrasp_retreat: float = 0.12,
    hand_center_floor_clearance: float = 0.02,
    max_top_down_wrist_roll_rad: float = 1.3089969389957472,
) -> torch.Tensor:
    """Weight collision-free grasp references by score, robot near side, and active-hand comfort."""
    if candidate_positions_w.ndim != 3 or candidate_positions_w.shape[-1] != 3:
        raise ValueError("candidate_positions_w must have shape (num_envs, num_candidates, 3)")
    num_envs, num_candidates, _ = candidate_positions_w.shape
    if candidate_quaternions_w.shape != (num_envs, num_candidates, 4):
        raise ValueError("candidate_quaternions_w must match candidate positions")
    if candidate_scores.shape != (num_envs, num_candidates):
        raise ValueError("candidate_scores must have shape (num_envs, num_candidates)")
    if candidate_valid.shape != (num_envs, num_candidates) or candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be a boolean (num_envs, num_candidates) tensor")
    if object_positions_w.shape != (num_envs, 3) or robot_positions_w.shape != (num_envs, 3):
        raise ValueError("object and robot positions must have shape (num_envs, 3)")
    if robot_quaternions_w.shape != (num_envs, 4):
        raise ValueError("robot_quaternions_w must have shape (num_envs, 4)")
    if active_shoulder_positions_w is not None and active_shoulder_positions_w.shape != (num_envs, 3):
        raise ValueError("active_shoulder_positions_w must have shape (num_envs, 3)")

    dtype = candidate_positions_w.dtype
    device = candidate_positions_w.device
    active_hand = active_hand.to(device=device, dtype=torch.long).reshape(-1)
    support_top_z = support_top_z.to(device=device, dtype=dtype).reshape(-1)
    if active_hand.shape[0] != num_envs or support_top_z.shape[0] != num_envs:
        raise ValueError("active_hand and support_top_z must contain one value per environment")

    local_approach = torch.zeros_like(candidate_positions_w)
    local_approach[..., 1] = 1.0
    approach_w = quaternion_apply_wxyz(
        torch.nn.functional.normalize(candidate_quaternions_w, dim=-1),
        local_approach,
    )
    pregrasp_positions_w = candidate_positions_w - float(pregrasp_retreat) * approach_w
    minimum_center_z = support_top_z.unsqueeze(-1) + float(hand_center_floor_clearance)
    support_clear = (candidate_positions_w[..., 2] >= minimum_center_z) & (
        pregrasp_positions_w[..., 2] >= minimum_center_z
    )

    object_to_robot_xy = robot_positions_w[:, :2] - object_positions_w[:, :2]
    object_to_robot_xy = torch.nn.functional.normalize(object_to_robot_xy, dim=-1, eps=1.0e-6)
    object_to_pregrasp_xy = pregrasp_positions_w[..., :2] - object_positions_w[:, None, :2]
    object_to_pregrasp_xy = torch.nn.functional.normalize(object_to_pregrasp_xy, dim=-1, eps=1.0e-6)
    near_alignment = torch.sum(object_to_pregrasp_xy * object_to_robot_xy.unsqueeze(1), dim=-1)
    near_side_weight = 0.05 + 0.95 * torch.clamp((near_alignment + 0.2) / 1.2, 0.0, 1.0).square()

    approach_xy_norm = torch.linalg.vector_norm(approach_w[..., :2], dim=-1)
    approach_xy = torch.nn.functional.normalize(approach_w[..., :2], dim=-1, eps=1.0e-6)
    toward_object_alignment = torch.sum(approach_xy * -object_to_robot_xy.unsqueeze(1), dim=-1)
    top_down = approach_xy_norm < 0.25
    local_hand_up = torch.zeros_like(candidate_positions_w)
    local_hand_up[..., 2] = 1.0
    hand_up_w = quaternion_apply_wxyz(
        torch.nn.functional.normalize(candidate_quaternions_w, dim=-1),
        local_hand_up,
    )
    shoulder_positions_w = robot_positions_w if active_shoulder_positions_w is None else active_shoulder_positions_w
    # A natural G1 top-down wrist pitch leaves hand +Z pointing from the active shoulder toward the grasp.
    shoulder_to_candidate_xy = candidate_positions_w[..., :2] - shoulder_positions_w[:, None, :2]
    shoulder_to_candidate_xy = torch.nn.functional.normalize(shoulder_to_candidate_xy, dim=-1, eps=1.0e-6)
    hand_up_xy = torch.nn.functional.normalize(hand_up_w[..., :2], dim=-1, eps=1.0e-6)
    top_down_wrist_alignment = torch.sum(hand_up_xy * shoulder_to_candidate_xy, dim=-1)
    top_down_ergonomic = top_down & (
        top_down_wrist_alignment >= torch.cos(torch.as_tensor(max_top_down_wrist_roll_rad, device=device, dtype=dtype))
    )
    near_side_approach = (~top_down) & (near_alignment >= 0.0) & (toward_object_alignment >= 0.0)
    accessible = top_down_ergonomic | near_side_approach

    local_left = torch.zeros(num_envs, 3, device=device, dtype=dtype)
    local_left[:, 1] = 1.0
    robot_left_w = quaternion_apply_wxyz(
        torch.nn.functional.normalize(robot_quaternions_w, dim=-1),
        local_left,
    )[:, :2]
    robot_left_w = torch.nn.functional.normalize(robot_left_w, dim=-1, eps=1.0e-6)
    lateral_alignment = torch.sum(object_to_pregrasp_xy * robot_left_w.unsqueeze(1), dim=-1)
    preferred_side = torch.where(active_hand == 0, 1.0, -1.0).to(dtype=dtype).unsqueeze(-1)
    active_hand_weight = 0.75 + 0.25 * preferred_side * lateral_alignment

    score_offset = candidate_scores - torch.where(
        candidate_valid,
        candidate_scores,
        torch.full_like(candidate_scores, -torch.inf),
    ).amax(dim=-1, keepdim=True)
    score_weight = torch.exp(2.0 * torch.nan_to_num(score_offset, neginf=-20.0))
    weights = score_weight * near_side_weight * active_hand_weight
    weights = torch.where(candidate_valid & support_clear & accessible, weights, torch.zeros_like(weights))

    empty = weights.sum(dim=-1, keepdim=True) <= 0.0
    top_fallback = candidate_valid & top_down_ergonomic
    accessible_fallback = candidate_valid & accessible
    has_top = top_fallback.any(dim=-1, keepdim=True)
    has_accessible = accessible_fallback.any(dim=-1, keepdim=True)
    fallback_mask = torch.where(
        has_top,
        top_fallback,
        torch.where(has_accessible, accessible_fallback, candidate_valid),
    )
    fallback = torch.where(
        fallback_mask,
        score_weight * near_side_weight * active_hand_weight,
        torch.zeros_like(weights),
    )
    return torch.where(empty, fallback, weights)


def select_candidate_indices(
    candidate_weights: torch.Tensor,
    *,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one candidate per row, either by argmax or weighted sampling."""
    if candidate_weights.ndim != 2 or torch.any(candidate_weights < 0.0):
        raise ValueError("candidate_weights must be a non-negative 2-D tensor")
    if torch.any(candidate_weights.sum(dim=-1) <= 0.0):
        raise ValueError("every row must contain at least one positive candidate weight")
    if deterministic:
        return torch.argmax(candidate_weights, dim=-1)
    return torch.multinomial(candidate_weights, num_samples=1, replacement=True, generator=generator).squeeze(-1)


def sample_weighted_candidate_indices(
    candidate_weights: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one candidate from each row without selecting zero-weight entries."""
    return select_candidate_indices(candidate_weights, generator=generator)


def save_grasp_reference_library(path: str | Path, library: GraspReferenceLibrary) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shape_names=np.asarray(library.shape_names),
        grasp_positions_o=library.positions_o.detach().cpu().numpy(),
        grasp_quaternions_o=library.quaternions_o.detach().cpu().numpy(),
        grasp_scores=library.scores.detach().cpu().numpy(),
        grasp_valid=library.valid.detach().cpu().numpy(),
        grasp_modes=library.modes.detach().cpu().numpy(),
    )


def load_grasp_reference_library(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GraspReferenceLibrary:
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Grasp reference library not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "shape_names",
            "grasp_positions_o",
            "grasp_quaternions_o",
            "grasp_scores",
            "grasp_valid",
            "grasp_modes",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Grasp reference library is missing arrays: {sorted(missing)}")
        return GraspReferenceLibrary(
            shape_names=tuple(str(name) for name in data["shape_names"].tolist()),
            positions_o=torch.as_tensor(data["grasp_positions_o"], device=device, dtype=dtype),
            quaternions_o=torch.as_tensor(data["grasp_quaternions_o"], device=device, dtype=dtype),
            scores=torch.as_tensor(data["grasp_scores"], device=device, dtype=dtype),
            valid=torch.as_tensor(data["grasp_valid"], device=device, dtype=torch.bool),
            modes=torch.as_tensor(data["grasp_modes"], device=device, dtype=torch.long),
        )
