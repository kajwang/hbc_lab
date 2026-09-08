#!/usr/bin/env python3
"""Compact an existing grasp library after removing wrist-inverted candidates."""

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


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("hbc_grasp_references", REFERENCE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load grasp reference helpers from {REFERENCE_MODULE_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quaternion_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-wrist-roll-deg", type=float, default=75.0)
    args = parser.parse_args()

    refs = _load_reference_module()
    library = refs.load_grasp_reference_library(args.input)
    manifest = {
        str(entry["shape_name"]): entry
        for entry in json.loads(args.manifest.read_text())
    }
    shape_count, candidate_count = library.valid.shape
    positions = torch.zeros_like(library.positions_o)
    quaternions = torch.zeros_like(library.quaternions_o)
    quaternions[..., 0] = 1.0
    scores = torch.zeros_like(library.scores)
    valid = torch.zeros_like(library.valid)
    modes = torch.full_like(library.modes, -1)

    world_up = torch.tensor([0.0, 0.0, 1.0])
    for shape_index, shape_name in enumerate(library.shape_names):
        if shape_name not in manifest:
            raise ValueError(f"Manifest is missing shape '{shape_name}'")
        stable_quat = torch.as_tensor(manifest[shape_name]["stable_quat_wxyz"], dtype=torch.float32)
        stable_rotation = _quaternion_to_matrix_wxyz(stable_quat)
        support_up_o = stable_rotation.transpose(-1, -2) @ world_up
        rotations_o = _quaternion_to_matrix_wxyz(library.quaternions_o[shape_index])
        ergonomic = refs.ergonomic_wrist_mask(
            rotations_o,
            support_up_o=support_up_o,
            max_roll_rad=np.deg2rad(args.max_wrist_roll_deg),
        )
        kept = torch.nonzero(library.valid[shape_index] & ergonomic, as_tuple=False).flatten()
        if kept.numel() == 0:
            raise RuntimeError(f"Ergonomic filtering removed every candidate for '{shape_name}'")
        count = int(kept.numel())
        positions[shape_index, :count] = library.positions_o[shape_index, kept]
        quaternions[shape_index, :count] = library.quaternions_o[shape_index, kept]
        scores[shape_index, :count] = library.scores[shape_index, kept]
        valid[shape_index, :count] = True
        modes[shape_index, :count] = library.modes[shape_index, kept]
        print(f"{shape_name:16s} kept={count:2d}/{int(library.valid[shape_index].sum()):2d}")

    output = args.output.resolve()
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}") if output == args.input.resolve() else output
    refs.save_grasp_reference_library(
        temporary,
        refs.GraspReferenceLibrary(
            shape_names=library.shape_names,
            positions_o=positions,
            quaternions_o=quaternions,
            scores=scores,
            valid=valid,
            modes=modes,
        ),
    )
    if temporary != output:
        temporary.replace(output)
    print(f"Saved ergonomic grasp library to {output} ({shape_count} shapes, K={candidate_count})")


if __name__ == "__main__":
    main()
