#!/usr/bin/env python3
"""Generate fixed directional-BPS descriptors from the vendored object USD meshes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "hbc_lab"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples-per-shape", type=int, default=50000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import trimesh  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.object_shape_bps import (  # noqa: E402
    ALL_SHAPE_NAMES,
    BPS_DATA_PATH,
    BPS_RADIUS,
    NUM_BPS_POINTS,
    OBJECT_SHAPE_SPECS,
    directional_bps_from_surface_points,
    fibonacci_ball_basis,
)


def _triangulate(face_counts: list[int], face_indices: list[int]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts:
        face = face_indices[offset : offset + count]
        offset += count
        for index in range(1, count - 1):
            triangles.append((face[0], face[index], face[index + 1]))
    return np.asarray(triangles, dtype=np.int64)


def _transform_points(points: np.ndarray, transform: Gf.Matrix4d) -> np.ndarray:
    return np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in points], dtype=np.float64)


def _load_usd_mesh(path: Path, scale: tuple[float, float, float]) -> trimesh.Trimesh:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"USD has no default prim: {path}")
    meshes: list[trimesh.Trimesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if points.size == 0 or not counts or not indices:
            continue
        # UsdFileCfg authors the spawn transform on the referenced default prim,
        # overriding that prim's authored xform ops. Keep only child transforms so
        # offline BPS geometry matches the runtime object exactly.
        mesh_to_default, _ = xform_cache.ComputeRelativeTransform(prim, default_prim)
        points = _transform_points(points, mesh_to_default) * np.asarray(scale)
        faces = _triangulate(counts, indices)
        if faces.size:
            meshes.append(trimesh.Trimesh(vertices=points, faces=faces, process=False))
    if not meshes:
        raise RuntimeError(f"No triangle mesh found in USD: {path}")
    return trimesh.util.concatenate(meshes)


def main() -> None:
    np.random.seed(7)
    basis = fibonacci_ball_basis(NUM_BPS_POINTS, radius=BPS_RADIUS, dtype=torch.float64)
    descriptors = []
    centers = []
    bounds_min = []
    bounds_max = []
    principal_axes = []

    for name in ALL_SHAPE_NAMES:
        spec = OBJECT_SHAPE_SPECS[name]
        mesh = _load_usd_mesh(spec.usd_path, spec.scale)
        mesh_vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float64)
        exact_lower = mesh_vertices.amin(dim=0)
        exact_upper = mesh_vertices.amax(dim=0)
        points, _ = trimesh.sample.sample_surface(mesh, args.samples_per_shape)
        points_tensor = torch.as_tensor(points, dtype=torch.float64)
        descriptor, center, lower, upper = directional_bps_from_surface_points(
            points_tensor,
            basis,
            bounds_min_o=exact_lower,
            bounds_max_o=exact_upper,
        )
        centered = points_tensor - center
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        axes = torch.flip(eigenvectors, dims=(-1,))

        descriptors.append(descriptor)
        centers.append(center)
        bounds_min.append(lower)
        bounds_max.append(upper)
        principal_axes.append(axes)
        size = upper - lower
        print(f"{name:16s} scale={spec.scale} size={size.tolist()} center={center.tolist()}")

    BPS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BPS_DATA_PATH,
        shape_names=np.asarray(ALL_SHAPE_NAMES),
        basis_points_o=basis.numpy().astype(np.float32),
        surface_offsets_o=torch.stack(descriptors).numpy().astype(np.float32),
        geometry_center_offsets_o=torch.stack(centers).numpy().astype(np.float32),
        bounds_min_o=torch.stack(bounds_min).numpy().astype(np.float32),
        bounds_max_o=torch.stack(bounds_max).numpy().astype(np.float32),
        principal_axes_o=torch.stack(principal_axes).numpy().astype(np.float32),
    )
    print(f"Saved {len(ALL_SHAPE_NAMES)} descriptors to {BPS_DATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
