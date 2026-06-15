#!/usr/bin/env python3
"""List registered HBC gym environments."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WBC_ROOT = REPO_ROOT.parent
ROBOT_LAB_ROOT = WBC_ROOT / "robot_lab"

for path in (
    REPO_ROOT / "source" / "hbc_lab",
    ROBOT_LAB_ROOT / "source" / "robot_lab",
):
    sys.path.insert(0, str(path))

import gymnasium as gym  # noqa: E402

import hbc_lab.tasks  # noqa: E402,F401


for env_id in sorted(gym.registry.keys()):
    if env_id.startswith("HBC-"):
        print(env_id)

