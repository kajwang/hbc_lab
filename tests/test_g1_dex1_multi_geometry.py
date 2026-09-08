from __future__ import annotations

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_scene_aware_hier_drc"
)


def _load_pure_functions(path: Path, names: set[str]) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch, "GEOMETRY_FAMILY_NAMES": tuple(range(8))}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_review_layout_pairs_conditions_and_balances_hands_at_four_levels():
    functions = _load_pure_functions(
        PACKAGE / "mdp/multi_geometry.py",
        {"preview_family_and_level", "paired_family_and_constraint", "paired_active_hand"},
    )
    env_ids = torch.arange(128)
    family, level = functions["preview_family_and_level"](
        env_ids,
        family_count=8,
        total_envs=128,
    )
    paired_family, constrained = functions["paired_family_and_constraint"](env_ids, family_count=8)
    active_hand = functions["paired_active_hand"](env_ids, family_count=8)

    expected_families = [family_id for _ in range(8) for family_id in range(8) for _ in range(2)]
    assert family.tolist() == expected_families
    assert torch.equal(family, paired_family)
    assert constrained.tolist() == [condition for _ in range(64) for condition in (False, True)]
    assert active_hand.tolist() == [hand for group in range(8) for hand in [group % 2] * 16]
    expected_levels = torch.tensor(
        [level_value for level_value in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0) for _ in range(32)]
    )
    assert torch.allclose(level, expected_levels)


def test_paired_uniform_matches_adjacent_counterfactual_envs():
    functions = _load_pure_functions(PACKAGE / "mdp/multi_geometry.py", {"paired_uniform"})
    samples = functions["paired_uniform"](torch.arange(64), -0.35, 0.75, salt=3.0)

    assert torch.allclose(samples[0::2], samples[1::2])
    assert bool(((samples >= -0.35) & (samples <= 0.75)).all())


def test_multi_geometry_task_uses_one_obstacle_interface_and_balanced_curricula():
    reset_source = (PACKAGE / "mdp/multi_geometry_events.py").read_text()
    env_source = (PACKAGE / "config/multi_geometry_env.py").read_text()
    registration_source = (PACKAGE / "__init__.py").read_text()

    assert "write_obstacle_boxes(" in reset_source
    assert "TABLE_NOMINAL_CLEARANCE + TABLE_TOP_SIZE[2]" in reset_source
    assert "object_mass_curriculum_levels" in env_source
    assert "_condition_and_hand_balanced_progress" in env_source
    assert "geometry_safe_reach_rate" in env_source
    assert "task_curriculum_success_rate" not in env_source
    assert '"target_distance": 2.0' in reset_source
    assert "object_mass_family_progress[present].amin()" not in env_source
    assert "geometry_curriculum_levels" in env_source
    assert "HBC-Isaac-G1-Dex1-MultiGeometry-Squashed-HierDrc-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-MultiGeometry-OOD-Squashed-HierDrc-Play-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-ReachOver-Squashed-HierDrc-v0" in registration_source


def test_exact_play_selector_and_ood_arch_are_wired_to_the_scene_scan():
    cfg_source = (PACKAGE / "config/env_cfg.py").read_text()
    env_source = (PACKAGE / "config/multi_geometry_env.py").read_text()
    reset_source = (PACKAGE / "mdp/multi_geometry_events.py").read_text()
    scene_source = (PACKAGE / "mdp/multi_geometry_scenes.py").read_text()

    for field in ("play_active_id", "play_env_id", "play_geo_level"):
        assert field in cfg_source
    assert "play_active_id" in env_source
    assert "play_env_id" in reset_source
    assert "OOD_ARCH_BOX_SLICE" in reset_source
    assert 'env.scene["ood_arch_top"]' in reset_source
    assert "ood_arch_left_post" in scene_source
    assert "ood_arch_right_post" in scene_source
    assert "reach_over_barrier_high" in scene_source
    assert "reach_over_support" in scene_source


def test_reach_over_wall_is_above_the_object_support() -> None:
    scene_source = (PACKAGE / "mdp/multi_geometry_scenes.py").read_text()
    assert "REACH_OVER_BARRIER_HEIGHTS = (0.48, 0.60, 0.72)" in scene_source
    assert "REACH_OVER_SUPPORT_SURFACE_HEIGHT = 0.28" in scene_source
    assert "reach_depth = lerp_range(level, 0.20, 0.38)" in (
        PACKAGE / "mdp/multi_geometry_events.py"
    ).read_text()
