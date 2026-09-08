#!/usr/bin/env python3
"""Play the highest-weight grasp candidate across 12 yaws for one asset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PLAY_SCRIPT = Path(__file__).with_name("play.py")
MODE_TASKS = {
    "graspref": "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-Play-v0",
    "object_center": "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-Play-v0",
}


def build_play_arguments(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_TASKS), required=True)
    parser.add_argument("--asset_id", type=int, required=True)
    args, forwarded = parser.parse_known_args(argv)
    if args.asset_id < 0:
        parser.error("--asset_id must be non-negative")
    if any(argument == "--task" or argument.startswith("--task=") for argument in forwarded):
        parser.error("--task is selected by --mode and must not be supplied")
    if any(argument == "--num_envs" or argument.startswith("--num_envs=") for argument in forwarded):
        parser.error("--num_envs is fixed by the 12-yaw diagnostic sweep")

    return [
        f"--task={MODE_TASKS[args.mode]}",
        *forwarded,
        f"env.grasp_candidate_asset_id={args.asset_id}",
        "env.object_shape_bps_enabled=true",
        "env.commands.high_level.left_hand_probability=0.0",
        "env.enable_debug_visualization=false",
        "env.target_pose_debug_vis=false",
        "env.grasp_reference_pose_debug_vis=true",
        "env.platform_pose_debug_vis=true",
    ]


def main() -> None:
    arguments = build_play_arguments(sys.argv[1:])
    os.execv(sys.executable, [sys.executable, str(PLAY_SCRIPT), *arguments])


if __name__ == "__main__":
    main()
