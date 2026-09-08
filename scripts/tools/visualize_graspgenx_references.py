#!/usr/bin/env python3
"""Render per-object contact sheets of filtered GraspGenX references."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SHAPE_MODULE_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_shape_bps.py"
)
DEFAULT_LIBRARY_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/assets/models/objects/grasp_references"
    / "multishape_grasp_references_k32.npz"
)
ALL_SHAPE_NAMES: tuple[str, ...] = ()


def _load_shape_names() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("hbc_grasp_vis_shapes", SHAPE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load shape metadata from {SHAPE_MODULE_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ALL_SHAPE_NAMES


def _set_equal_axes(axis, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float((upper - lower).max()) * 0.58, 0.03)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _visualization_triangles(mesh, max_faces: int) -> np.ndarray:
    """Return a deterministic face subset for fast contact-sheet rendering."""
    triangles = np.asarray(mesh.triangles)
    if len(triangles) <= max_faces:
        return triangles
    indices = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
    return triangles[indices]


def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY_PATH)
    parser.add_argument("--gripper-mesh", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    global ALL_SHAPE_NAMES
    ALL_SHAPE_NAMES = _load_shape_names()
    manifest_entries = json.loads(args.manifest.read_text())
    for entry in manifest_entries:
        mesh_file = Path(entry["mesh_file"])
        if not mesh_file.is_absolute():
            mesh_file = args.manifest.resolve().parent / mesh_file
        entry["mesh_file"] = str(mesh_file.resolve())
    manifest = {entry["shape_name"]: entry for entry in manifest_entries}
    with np.load(args.library, allow_pickle=False) as library:
        shape_names = tuple(str(name) for name in library["shape_names"].tolist())
        positions = library["grasp_positions_o"]
        quaternions = library["grasp_quaternions_o"]
        scores = library["grasp_scores"]
        valid = library["grasp_valid"]
        grasp_modes = library["grasp_modes"]

    if shape_names != ALL_SHAPE_NAMES:
        raise ValueError("Grasp library shape order does not match ALL_SHAPE_NAMES")
    gripper_mesh = trimesh.load(args.gripper_mesh, force="mesh", process=False)
    gripper_triangles = _visualization_triangles(gripper_mesh, max_faces=600)
    mode_colors = {0: "#e83e8c", 1: "#00a6d6", 2: "#ffb000"}
    args.output_root.mkdir(parents=True, exist_ok=True)

    for shape_index, shape_name in enumerate(shape_names):
        entry = manifest[shape_name]
        object_mesh = trimesh.load(entry["mesh_file"], force="mesh", process=False)
        stable_rotation = _rotation_from_wxyz(np.asarray(entry["stable_quat_wxyz"], dtype=np.float64))
        object_triangles = _visualization_triangles(object_mesh, max_faces=1800)
        object_triangles_stable = np.einsum("ij,fvj->fvi", stable_rotation, object_triangles)
        object_vertices_stable = np.einsum("ij,vj->vi", stable_rotation, np.asarray(object_mesh.vertices))
        candidate_ids = np.flatnonzero(valid[shape_index])
        columns = 8
        rows = max(1, int(np.ceil(len(candidate_ids) / columns)))
        figure = plt.figure(figsize=(4 * columns, 4 * rows), constrained_layout=True)
        figure.suptitle(f"{shape_name}: filtered GraspGenX Dex1 references", fontsize=18)
        support_z = float(object_vertices_stable[:, 2].min())
        support_x, support_y = np.meshgrid(np.asarray([-0.17, 0.17]), np.asarray([-0.20, 0.20]))
        support_points = np.stack(
            (support_x.reshape(-1), support_y.reshape(-1), np.full(4, support_z)),
            axis=-1,
        )
        for plot_index in range(rows * columns):
            axis = figure.add_subplot(rows, columns, plot_index + 1, projection="3d")
            axis.set_axis_off()
            axis.plot_surface(
                support_x,
                support_y,
                np.full_like(support_x, support_z),
                color="#d9dde1",
                alpha=0.30,
                shade=False,
            )
            object_poly = Poly3DCollection(
                object_triangles_stable,
                facecolor="#c8ccd0",
                edgecolor="#888888",
                linewidth=0.15,
                alpha=0.55,
            )
            axis.add_collection3d(object_poly)
            points = [object_vertices_stable, support_points]
            if plot_index < len(candidate_ids):
                candidate_id = int(candidate_ids[plot_index])
                quat = quaternions[shape_index, candidate_id]
                rotation = _rotation_from_wxyz(quat)
                stable_position = stable_rotation @ positions[shape_index, candidate_id]
                stable_gripper_rotation = stable_rotation @ rotation
                transform = np.eye(4)
                transform[:3, :3] = stable_gripper_rotation
                transform[:3, 3] = stable_position
                candidate_mesh = gripper_mesh.copy()
                candidate_mesh.apply_transform(transform)
                candidate_triangles = np.einsum(
                    "ij,fvj->fvi",
                    stable_gripper_rotation,
                    gripper_triangles,
                ) + stable_position
                color = mode_colors[int(grasp_modes[shape_index, candidate_id])]
                gripper_poly = Poly3DCollection(
                    candidate_triangles,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.3,
                    alpha=0.35,
                )
                axis.add_collection3d(gripper_poly)
                points.append(np.asarray(candidate_mesh.vertices))
                axis.set_title(
                    f"#{candidate_id:02d} score={scores[shape_index, candidate_id]:.3f} "
                    f"mode={int(grasp_modes[shape_index, candidate_id])}",
                    fontsize=9,
                )
            _set_equal_axes(axis, np.concatenate(points, axis=0))
            axis.view_init(elev=24, azim=-55)
        contact_sheet = args.output_root / f"{shape_index:02d}_{shape_name}.png"
        figure.savefig(contact_sheet, dpi=130)
        plt.close(figure)
        print(contact_sheet)


if __name__ == "__main__":
    main()
