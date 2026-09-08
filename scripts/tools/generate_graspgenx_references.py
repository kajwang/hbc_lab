#!/usr/bin/env python3
"""Run GraspGenX on exported hbc objects and build a compact grasp-reference library."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MODULE_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/grasp_references.py"
)
DEFAULT_LIBRARY_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/assets/models/objects/grasp_references"
    / "multishape_grasp_references_k32.npz"
)
DEFAULT_SUPPORT_SIZE_XY = (0.34, 0.40)

# GraspGenX uses +X as the closing axis and +Z as the approach axis.  Dex1's
# hand-base frame uses +X for closing and +Y for approach.  The translation is
# the measured midpoint of Link1_3 and Link2_3 in the shipped Dex1 USD.
DEX1_SWEEP_VOLUME = {
    "extents_open": np.asarray([0.084, 0.030, 0.040], dtype=np.float32),
    "offset_open": np.asarray([0.0, -0.0142, 0.080], dtype=np.float32),
    "extents_mid": np.asarray([0.042, 0.030, 0.040], dtype=np.float32),
    "offset_mid": np.asarray([0.0, -0.0142, 0.080], dtype=np.float32),
    "gripper_type": 0,
    "fingertip_depth": 0.09734,
}
DEX1_GRIPPER_TO_HAND_CENTER = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.0142],
        [0.0, 1.0, 0.0, 0.09734],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("hbc_grasp_references", REFERENCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load grasp reference helpers from {REFERENCE_MODULE_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_centered_point_cloud(mesh_file: Path, num_points: int, seed: int):
    import trimesh

    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected one triangle mesh in {mesh_file}")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    finally:
        np.random.set_state(state)
    center = points.mean(axis=0)
    return mesh, np.asarray(points - center, dtype=np.float32), np.asarray(center, dtype=np.float32)


def _make_dex1_open_mesh_gripper_frame():
    """Build a compact open-state Dex1 proxy in the GraspGenX gripper frame."""
    import trimesh

    palm = trimesh.creation.box(extents=(0.070, 0.055, 0.035))
    palm.apply_translation((0.0, 0.0, 0.020))
    fingers = []
    for x in (-0.050, 0.050):
        finger = trimesh.creation.box(extents=(0.016, 0.035, 0.055))
        finger.apply_translation((x, -0.0142, 0.075))
        fingers.append(finger)
    return trimesh.util.concatenate([palm, *fingers])


def _gripper_to_hand_center_transforms(transforms_o_g: torch.Tensor) -> torch.Tensor:
    transform_g_h = torch.as_tensor(
        DEX1_GRIPPER_TO_HAND_CENTER,
        dtype=transforms_o_g.dtype,
        device=transforms_o_g.device,
    )
    return transforms_o_g @ transform_g_h


def _support_filter(
    transforms_o: np.ndarray,
    object_vertices_o: np.ndarray,
    gripper_vertices_g: np.ndarray,
    stable_quat_wxyz: np.ndarray,
    floor_tolerance: float,
    pregrasp_retreat: float,
    pregrasp_sweep_steps: int,
) -> np.ndarray:
    stable_rotation = _quaternion_to_matrix_wxyz(stable_quat_wxyz)
    object_floor = (object_vertices_o @ stable_rotation.T)[:, 2].min()
    accepted = np.ones(len(transforms_o), dtype=bool)
    retreat_offsets = np.linspace(0.0, pregrasp_retreat, pregrasp_sweep_steps)
    for index, transform in enumerate(transforms_o):
        approach_o = transform[:3, 2]
        for retreat in retreat_offsets:
            swept_translation_o = transform[:3, 3] - retreat * approach_o
            vertices_o = gripper_vertices_g @ transform[:3, :3].T + swept_translation_o
            gripper_floor = (vertices_o @ stable_rotation.T)[:, 2].min()
            if gripper_floor < object_floor - floor_tolerance:
                accepted[index] = False
                break
    return accepted


def _sample_support_scene_points(
    object_vertices_o: np.ndarray,
    stable_quat_wxyz: np.ndarray,
    support_size_xy: tuple[float, float],
    spacing: float,
) -> np.ndarray:
    """Sample the finite support top in object coordinates without mixing it into the target cloud."""
    if spacing <= 0.0:
        raise ValueError("support scene spacing must be positive")
    stable_rotation = _quaternion_to_matrix_wxyz(stable_quat_wxyz)
    object_vertices_s = np.asarray(object_vertices_o, dtype=np.float64) @ stable_rotation.T
    support_z_s = float(object_vertices_s[:, 2].min())
    size_x, size_y = (float(value) for value in support_size_xy)
    count_x = max(2, int(np.ceil(size_x / spacing)) + 1)
    count_y = max(2, int(np.ceil(size_y / spacing)) + 1)
    coordinates_x = np.linspace(-0.5 * size_x, 0.5 * size_x, count_x)
    coordinates_y = np.linspace(-0.5 * size_y, 0.5 * size_y, count_y)
    grid_x, grid_y = np.meshgrid(coordinates_x, coordinates_y, indexing="xy")
    points_s = np.stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.full(grid_x.size, support_z_s),
        ),
        axis=-1,
    )
    return np.asarray(points_s @ stable_rotation, dtype=np.float32)


def _swept_scene_collision_filter(
    transforms_o: np.ndarray,
    gripper_points_g: np.ndarray,
    scene_points_o: np.ndarray,
    pregrasp_retreat: float,
    pregrasp_sweep_steps: int,
    collision_threshold: float,
) -> np.ndarray:
    """Reject candidates whose complete gripper proxy touches the support scene over the approach sweep."""
    from scipy.spatial import cKDTree

    if pregrasp_sweep_steps < 1:
        raise ValueError("pregrasp_sweep_steps must be positive")
    if collision_threshold <= 0.0:
        raise ValueError("collision_threshold must be positive")
    scene_tree = cKDTree(np.asarray(scene_points_o, dtype=np.float64))
    accepted = np.ones(len(transforms_o), dtype=bool)
    retreat_offsets = np.linspace(0.0, pregrasp_retreat, pregrasp_sweep_steps)
    for index, transform in enumerate(np.asarray(transforms_o)):
        approach_o = transform[:3, 2]
        rotated_gripper_o = np.asarray(gripper_points_g) @ transform[:3, :3].T
        for retreat in retreat_offsets:
            translation_o = transform[:3, 3] - retreat * approach_o
            gripper_points_o = rotated_gripper_o + translation_o
            distances, _ = scene_tree.query(gripper_points_o, k=1, workers=-1)
            if np.any(distances < collision_threshold):
                accepted[index] = False
                break
    return accepted


def _filter_shape(
    raw_path: Path,
    manifest_entry: dict[str, object],
    gripper_vertices_g: np.ndarray,
    gripper_collision_points_g: np.ndarray,
    refs,
    max_candidates: int,
    floor_tolerance: float,
    min_score: float,
    pregrasp_retreat: float,
    pregrasp_sweep_steps: int,
    support_size_xy: tuple[float, float],
    support_scene_spacing: float,
    scene_collision_threshold: float,
    max_wrist_roll_rad: float,
):
    import trimesh

    with np.load(raw_path, allow_pickle=False) as raw:
        transforms = torch.as_tensor(raw["grasps_o"], dtype=torch.float32)
        scores = torch.as_tensor(raw["scores"], dtype=torch.float32).reshape(-1)
    mesh = trimesh.load(manifest_entry["mesh_file"], force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    geometry_center = torch.as_tensor(bounds.mean(axis=0))
    geometry_diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))

    valid = refs.valid_rigid_transforms(transforms) & torch.isfinite(scores)
    valid &= scores >= min_score
    distance = torch.linalg.vector_norm(transforms[:, :3, 3] - geometry_center, dim=-1)
    valid &= distance <= 0.5 * geometry_diagonal + 0.18
    support_valid = _support_filter(
        transforms.detach().cpu().numpy(),
        np.asarray(mesh.vertices),
        gripper_vertices_g,
        np.asarray(manifest_entry["stable_quat_wxyz"]),
        floor_tolerance,
        pregrasp_retreat,
        pregrasp_sweep_steps,
    )
    valid &= torch.as_tensor(support_valid)
    support_scene_points_o = _sample_support_scene_points(
        np.asarray(mesh.vertices),
        np.asarray(manifest_entry["stable_quat_wxyz"]),
        support_size_xy,
        support_scene_spacing,
    )
    scene_valid = _swept_scene_collision_filter(
        transforms.detach().cpu().numpy(),
        gripper_collision_points_g,
        support_scene_points_o,
        pregrasp_retreat,
        pregrasp_sweep_steps,
        scene_collision_threshold,
    )
    valid &= torch.as_tensor(scene_valid)

    transforms = transforms[valid]
    scores = scores[valid]
    if transforms.shape[0] == 0:
        return transforms, scores, torch.empty(0, dtype=torch.long)

    stable_rotation = torch.as_tensor(
        _quaternion_to_matrix_wxyz(np.asarray(manifest_entry["stable_quat_wxyz"])),
        dtype=torch.float32,
    )
    support_up_o = stable_rotation.transpose(0, 1) @ torch.tensor([0.0, 0.0, 1.0])
    modes = refs.classify_approach_modes(transforms[:, :3, :3], support_up_o=support_up_o)
    transforms = _gripper_to_hand_center_transforms(transforms)
    ergonomic = refs.ergonomic_wrist_mask(
        transforms[:, :3, :3],
        support_up_o=support_up_o,
        max_roll_rad=max_wrist_roll_rad,
    )
    transforms = transforms[ergonomic]
    scores = scores[ergonomic]
    modes = modes[ergonomic]
    if transforms.shape[0] == 0:
        return transforms, scores, modes
    translation_threshold = min(0.03, max(0.008, 0.12 * geometry_diagonal))
    nms_indices = refs.se3_nms_indices(
        transforms,
        scores,
        translation_threshold=translation_threshold,
        rotation_threshold_rad=np.deg2rad(20.0),
    )
    transforms = transforms[nms_indices]
    scores = scores[nms_indices]
    modes = modes[nms_indices]
    selected = refs.balanced_candidate_indices(
        transforms,
        scores,
        modes,
        max_candidates=max_candidates,
    )
    return transforms[selected], scores[selected], modes[selected]


def _save_runtime_library(
    manifest: list[dict[str, object]],
    raw_root: Path,
    output_path: Path,
    gripper_vertices_g: np.ndarray,
    gripper_collision_points_g: np.ndarray,
    refs,
    max_candidates: int,
    floor_tolerance: float,
    min_score: float,
    pregrasp_retreat: float,
    pregrasp_sweep_steps: int,
    support_size_xy: tuple[float, float],
    support_scene_spacing: float,
    scene_collision_threshold: float,
    max_wrist_roll_rad: float,
) -> None:
    num_shapes = len(manifest)
    positions = torch.zeros(num_shapes, max_candidates, 3)
    quaternions = torch.zeros(num_shapes, max_candidates, 4)
    quaternions[..., 0] = 1.0
    scores = torch.zeros(num_shapes, max_candidates)
    valid = torch.zeros(num_shapes, max_candidates, dtype=torch.bool)
    modes = torch.full((num_shapes, max_candidates), -1, dtype=torch.long)

    for shape_index, entry in enumerate(manifest):
        raw_path = raw_root / f"{entry['shape_name']}.npz"
        transforms, selected_scores, selected_modes = _filter_shape(
            raw_path,
            entry,
            gripper_vertices_g,
            gripper_collision_points_g,
            refs,
            max_candidates,
            floor_tolerance,
            min_score,
            pregrasp_retreat,
            pregrasp_sweep_steps,
            support_size_xy,
            support_scene_spacing,
            scene_collision_threshold,
            max_wrist_roll_rad,
        )
        count = transforms.shape[0]
        if count:
            positions[shape_index, :count] = transforms[:, :3, 3]
            quaternions[shape_index, :count] = refs.matrix_to_quaternion_wxyz(transforms[:, :3, :3])
            scores[shape_index, :count] = selected_scores
            valid[shape_index, :count] = True
            modes[shape_index, :count] = selected_modes
        mode_counts = torch.bincount(selected_modes, minlength=3).tolist() if count else [0, 0, 0]
        print(
            f"{entry['shape_name']:16s} kept={count:2d}/{max_candidates} "
            f"top={mode_counts[0]:2d} side={mode_counts[1]:2d} oblique={mode_counts[2]:2d}"
        )

    refs.save_grasp_reference_library(
        output_path,
        refs.GraspReferenceLibrary(
            shape_names=tuple(str(entry["shape_name"]) for entry in manifest),
            positions_o=positions,
            quaternions_o=quaternions,
            scores=scores,
            valid=valid,
            modes=modes,
        ),
    )
    print(f"Saved runtime grasp library to {output_path}")


def main() -> None:
    import trimesh

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graspgenx-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--library-output", type=Path, default=DEFAULT_LIBRARY_PATH)
    parser.add_argument("--planner", choices=("graspmoe", "diffusion"), default="graspmoe")
    parser.add_argument("--num-grasps", type=int, default=400)
    parser.add_argument("--num-sample-points", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--floor-tolerance", type=float, default=0.005)
    parser.add_argument("--pregrasp-retreat", type=float, default=0.12)
    parser.add_argument("--pregrasp-sweep-steps", type=int, default=5)
    parser.add_argument("--support-size-xy", type=float, nargs=2, default=DEFAULT_SUPPORT_SIZE_XY)
    parser.add_argument("--support-scene-spacing", type=float, default=0.01)
    parser.add_argument("--scene-collision-threshold", type=float, default=0.002)
    parser.add_argument("--max-wrist-roll-deg", type=float, default=75.0)
    parser.add_argument("--num-gripper-collision-points", type=int, default=1000)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--filter-only", action="store_true")
    args = parser.parse_args()

    refs = _load_reference_module()
    manifest = json.loads(args.manifest.read_text())
    for entry in manifest:
        mesh_file = Path(entry["mesh_file"])
        if not mesh_file.is_absolute():
            mesh_file = args.manifest.resolve().parent / mesh_file
        entry["mesh_file"] = str(mesh_file.resolve())
    raw_root = args.output_root.resolve() / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    gripper_mesh_g = _make_dex1_open_mesh_gripper_frame()
    gripper_mesh_h = gripper_mesh_g.copy()
    gripper_mesh_h.apply_transform(np.linalg.inv(DEX1_GRIPPER_TO_HAND_CENTER))
    gripper_mesh_path = args.output_root.resolve() / "dex1_open_hand_center_mesh.obj"
    gripper_mesh_h.export(gripper_mesh_path)
    print(f"Exported visualization gripper mesh to {gripper_mesh_path}")
    random_state = np.random.get_state()
    np.random.seed(args.seed)
    try:
        sampled_gripper_points_g, _ = trimesh.sample.sample_surface(gripper_mesh_g, args.num_gripper_collision_points)
    finally:
        np.random.set_state(random_state)
    gripper_collision_points_g = np.concatenate(
        (np.asarray(gripper_mesh_g.vertices), np.asarray(sampled_gripper_points_g)),
        axis=0,
    ).astype(np.float32)

    if not args.filter_only:
        graspgenx_root = args.graspgenx_root.resolve()
        sys.path.insert(0, str(graspgenx_root))
        from graspgenx import get_checkpoints_version_dir
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.samplers.planner import run_planner_on_batch
        from graspgenx.utils.checkpoint_io import load_model_cfg
        from graspgenx.x_grippers import make_sweep_volume_gripper_info

        checkpoint_root = Path(get_checkpoints_version_dir())
        model_cfg = load_model_cfg(
            str(checkpoint_root / "gen"),
            str(checkpoint_root / "dis"),
            None,
            None,
        )
        dex1_gripper_info = make_sweep_volume_gripper_info(
            extents_open=DEX1_SWEEP_VOLUME["extents_open"],
            offset_open=DEX1_SWEEP_VOLUME["offset_open"],
            extents_mid=DEX1_SWEEP_VOLUME["extents_mid"],
            offset_mid=DEX1_SWEEP_VOLUME["offset_mid"],
            gripper_type=DEX1_SWEEP_VOLUME["gripper_type"],
            fingertip_depth=DEX1_SWEEP_VOLUME["fingertip_depth"],
            name="unitree_g1_dex1",
        )
        sampler = GraspGenXSampler(model_cfg, gripper_info=dex1_gripper_info)
        pending: list[tuple[dict[str, object], object, np.ndarray]] = []
        for shape_index, entry in enumerate(manifest):
            raw_path = raw_root / f"{entry['shape_name']}.npz"
            if raw_path.is_file() and not args.overwrite:
                continue
            mesh, point_cloud, center = _load_centered_point_cloud(
                Path(entry["mesh_file"]),
                args.num_sample_points,
                args.seed + shape_index,
            )
            pending.append((entry, mesh, center))
            entry["_point_cloud"] = point_cloud

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            results = run_planner_on_batch(
                [entry["_point_cloud"] for entry, _, _ in batch],
                sampler,
                planner=args.planner,
                grasp_threshold=-1.0,
                num_grasps=args.num_grasps,
                topk_num_grasps=200,
            )
            for (entry, _, center), (grasps, confidence, branch_tags, _) in zip(batch, results):
                grasps = np.asarray(grasps, dtype=np.float32)
                grasps[:, :3, 3] += center
                raw_path = raw_root / f"{entry['shape_name']}.npz"
                np.savez_compressed(
                    raw_path,
                    grasps_o=grasps,
                    scores=np.asarray(confidence, dtype=np.float32),
                    branch_tags=np.asarray(branch_tags),
                )
                print(f"{entry['shape_name']:16s} raw={len(grasps):4d} -> {raw_path}")

    missing = [entry["shape_name"] for entry in manifest if not (raw_root / f"{entry['shape_name']}.npz").is_file()]
    if missing:
        raise RuntimeError(f"Missing raw GraspGenX output for: {missing}")
    _save_runtime_library(
        manifest,
        raw_root,
        args.library_output.resolve(),
        np.asarray(gripper_mesh_g.vertices),
        gripper_collision_points_g,
        refs,
        args.max_candidates,
        args.floor_tolerance,
        args.min_score,
        args.pregrasp_retreat,
        args.pregrasp_sweep_steps,
        tuple(args.support_size_xy),
        args.support_scene_spacing,
        args.scene_collision_threshold,
        np.deg2rad(args.max_wrist_roll_deg),
    )


if __name__ == "__main__":
    main()
