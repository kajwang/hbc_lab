"""Task registration entry point for HBC Lab."""

from __future__ import annotations

import importlib

try:
    importlib.import_module("hbc_lab.tasks.locomotion.robots.g1.29dof")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_box_carry_hier_drc")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_door_open_hier_drc")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_cart_push_hier_drc")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_drawer_hier_drc")
    importlib.import_module("hbc_lab.tasks.manager_based.skill.g1_dex1_scene_aware_hier_drc")
except ModuleNotFoundError as exc:
    if exc.name not in {"gymnasium", "isaaclab", "isaaclab_rl", "torch"}:
        raise
