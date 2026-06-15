"""Compatibility shim for the migrated Unitree G1 task registration."""

from __future__ import annotations

import importlib


try:
    importlib.import_module("hbc_lab.tasks.locomotion.robots.g1.29dof")
except ModuleNotFoundError as exc:
    if exc.name not in {"gymnasium", "isaaclab", "isaaclab_rl", "torch"}:
        raise
