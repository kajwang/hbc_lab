import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc"
MODULE_PATH = TASK_ROOT / "mdp/domain_randomization_curriculum.py"
ENV_PATH = TASK_ROOT / "config/g1_dex1_env.py"
EVENTS_PATH = TASK_ROOT / "mdp/events.py"
SCENES_PATH = TASK_ROOT / "mdp/scenes.py"
RANDOMIZED_CFG_PATH = TASK_ROOT / "config/randomized_env_cfg.py"
REGISTRATION_PATH = TASK_ROOT / "__init__.py"


def _load_curriculum_module():
    spec = importlib.util.spec_from_file_location("g1_dex1_pnp_randomization_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grasp_rate_curriculum_is_monotonic_and_thresholded():
    module = _load_curriculum_module()

    assert module.advance_curriculum_level(0.3, 0.59, threshold=0.6, step=0.1) == pytest.approx(0.3)
    assert module.advance_curriculum_level(0.3, 0.61, threshold=0.6, step=0.1) == pytest.approx(0.4)
    assert module.advance_curriculum_level(0.95, 0.9, threshold=0.6, step=0.1) == pytest.approx(1.0)


def test_randomization_ranges_expand_from_baseline_to_full_domain():
    module = _load_curriculum_module()

    easy = module.randomization_ranges(torch.tensor(0.0))
    full = module.randomization_ranges(torch.tensor(1.0))

    assert easy.init_x == pytest.approx((1.5, 2.0))
    assert easy.init_y == pytest.approx((-0.35, 0.35))
    assert easy.init_z == pytest.approx((0.5, 0.5))
    assert easy.goal_radius == pytest.approx((1.5, 2.5))
    assert full.init_x == pytest.approx((1.0, 2.7))
    assert full.init_y == pytest.approx((-1.0, 1.0))
    assert full.init_z == pytest.approx((0.0, 0.7))
    assert full.goal_radius == pytest.approx((1.5, 4.0))


def test_scale_pool_expands_around_nominal_size():
    module = _load_curriculum_module()
    scales = (0.8, 0.9, 1.0, 1.1, 1.2)

    assert module.allowed_scale_indices(0.0, scales) == (2,)
    assert module.allowed_scale_indices(0.5, scales) == (1, 2, 3)
    assert module.allowed_scale_indices(1.0, scales) == (0, 1, 2, 3, 4)


def test_inactive_object_pool_is_parked_behind_the_robot_not_below_ground():
    module = _load_curriculum_module()

    offsets = module.inactive_object_parking_offsets(5)

    assert len(offsets) == 5
    assert all(x <= -4.0 for x, _ in offsets)
    assert len(set(offsets)) == 5


def test_randomized_pnp_task_uses_active_object_pool_and_grasp_rate_curriculum():
    env_source = ENV_PATH.read_text()
    events_source = EVENTS_PATH.read_text()
    scenes_source = SCENES_PATH.read_text()
    cfg_source = RANDOMIZED_CFG_PATH.read_text()
    registration_source = REGISTRATION_PATH.read_text()

    assert "RigidObjectCollectionCfg" in scenes_source
    assert "PNP_OBJECT_SCALE_FACTORS" in scenes_source
    assert "object_size_" in scenes_source
    assert "active_object_index" in env_source
    assert "domain_randomization_curriculum_level" in env_source
    assert "Curriculum/grasp_rate_ema" in env_source
    assert "Curriculum/domain_randomization_level" in env_source
    assert "domain_randomization_start_init_z" in (TASK_ROOT / "config/g1_dex1_env_cfg.py").read_text()
    assert 'DRC/object_mass_mean' in env_source
    assert "reset_randomized_object_and_support_platforms" in events_source
    assert "parked_depth" not in events_source
    assert "domain_randomization_support_height" in events_source
    assert "RANDOMIZED_OBJECT_PLATFORM_HEIGHT" in scenes_source
    assert "kinematic_enabled=True" in scenes_source
    assert "visible=False" in scenes_source
    assert "write_root_pose_to_sim" in events_source
    assert "PLATFORM_MARKER_CFG" in env_source
    assert "platform_pose_debug_vis" in env_source
    assert "platform_pose_debug_vis = True" in cfg_source
    assert "platform_visualizer.visualize" in env_source
    assert "G1Dex1HierDrcRandomizedEnvCfg" in cfg_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-Randomized-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-Randomized-Play-v0" in registration_source
