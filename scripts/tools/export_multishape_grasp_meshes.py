#!/usr/bin/env python3
"""Export runtime-equivalent multishape USD geometry for GraspGenX inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
HBC_SOURCE_ROOT = REPO_ROOT / "source" / "hbc_lab"
SHAPE_MODULE_PATH = (
    HBC_SOURCE_ROOT
    / "hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_shape_bps.py"
)


def _load_shape_module():
    spec = importlib.util.spec_from_file_location("hbc_grasp_shape_specs", SHAPE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load shape metadata from {SHAPE_MODULE_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _triangulate(face_counts: list[int], face_indices: list[int]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts:
        face = face_indices[offset : offset + count]
        offset += count
        for index in range(1, count - 1):
            triangles.append((face[0], face[index], face[index + 1]))
    return np.asarray(triangles, dtype=np.int64)


def main() -> None:
    sys.path.insert(0, str(HBC_SOURCE_ROOT))
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import trimesh
        from pxr import Gf, Usd, UsdGeom

        shape_module = _load_shape_module()
        mesh_root = args.output_root.resolve() / "meshes"
        mesh_root.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, object]] = []

        for shape_name in shape_module.ALL_SHAPE_NAMES:
            shape_spec = shape_module.OBJECT_SHAPE_SPECS[shape_name]
            stage = Usd.Stage.Open(str(shape_spec.usd_path))
            if stage is None:
                raise RuntimeError(f"Unable to open USD: {shape_spec.usd_path}")
            default_prim = stage.GetDefaultPrim()
            if not default_prim.IsValid():
                raise RuntimeError(f"USD has no default prim: {shape_spec.usd_path}")
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            meshes: list[trimesh.Trimesh] = []
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                usd_mesh = UsdGeom.Mesh(prim)
                points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
                counts = list(usd_mesh.GetFaceVertexCountsAttr().Get() or [])
                indices = list(usd_mesh.GetFaceVertexIndicesAttr().Get() or [])
                if points.size == 0 or not counts or not indices:
                    continue
                mesh_to_default, _ = xform_cache.ComputeRelativeTransform(prim, default_prim)
                points = np.asarray(
                    [mesh_to_default.Transform(Gf.Vec3d(*point)) for point in points],
                    dtype=np.float64,
                )
                points *= np.asarray(shape_spec.scale, dtype=np.float64)
                faces = _triangulate(counts, indices)
                if faces.size:
                    meshes.append(trimesh.Trimesh(vertices=points, faces=faces, process=False))
            if not meshes:
                raise RuntimeError(f"No triangle mesh found in USD: {shape_spec.usd_path}")

            mesh = trimesh.util.concatenate(meshes)
            mesh_file = mesh_root / f"{shape_name}.obj"
            mesh.export(mesh_file)
            manifest.append(
                {
                    "shape_name": shape_name,
                    "mesh_file": str(mesh_file.relative_to(args.output_root.resolve())),
                    "source_usd": str(shape_spec.usd_path),
                    "nominal_scale": list(shape_spec.scale),
                    "stable_quat_wxyz": list(shape_spec.stable_quat_wxyz),
                    "bounds_min_o": mesh.bounds[0].tolist(),
                    "bounds_max_o": mesh.bounds[1].tolist(),
                }
            )
            print(f"{shape_name:16s} mesh={mesh_file} extents={mesh.extents.tolist()}")

        manifest_path = args.output_root.resolve() / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {len(manifest)} runtime-equivalent meshes to {mesh_root}")
        print(f"Manifest: {manifest_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
