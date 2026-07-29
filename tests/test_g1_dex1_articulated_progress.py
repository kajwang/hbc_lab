import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill"
DOOR_PROGRESS_PATH = SKILL_ROOT / "g1_dex1_door_open_hier_drc/mdp/progress.py"
CART_PROGRESS_PATH = SKILL_ROOT / "g1_dex1_cart_push_hier_drc/mdp/progress.py"


def _load_module(name: str, path: Path):
    assert path.exists(), f"Missing progress module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_door_progress_requires_near_complete_handle_turn_before_hinge_reward():
    module = _load_module("door_progress_under_test", DOOR_PROGRESS_PATH)
    progress = module.compute_door_manipulation_progress(
        handle_angle=torch.tensor([0.0, 0.25, 0.5]),
        hinge_angle=torch.tensor([0.0, math.radians(30.0), math.radians(60.0)]),
        latch_threshold=0.5,
        hinge_target=math.radians(60.0),
    )

    assert torch.allclose(progress.handle_progress, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(progress.hinge_progress, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(progress.handle_gate, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(progress.reward, torch.tensor([0.3, 0.425, 1.3]), atol=1.0e-6)


def test_door_handle_gate_is_smooth_near_latch_release():
    module = _load_module("door_progress_gate_under_test", DOOR_PROGRESS_PATH)
    progress = module.compute_door_manipulation_progress(
        handle_angle=torch.tensor([0.40, 0.45, 0.50]),
        hinge_angle=torch.zeros(3),
        latch_threshold=0.5,
        hinge_target=math.radians(60.0),
    )

    assert torch.allclose(progress.handle_gate, torch.tensor([0.0, 0.5, 1.0]), atol=1.0e-6)


def test_bimanual_grasp_uses_geometric_mean():
    module = _load_module("cart_progress_grasp_under_test", CART_PROGRESS_PATH)
    confidence = module.bimanual_grasp_confidence(
        torch.tensor([1.0, 0.25, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    )

    assert torch.allclose(confidence, torch.tensor([1.0, 0.5, 0.0]))


def test_handle_targets_follow_horizontal_handle_bar_axis():
    module = _load_module("cart_progress_targets_under_test", CART_PROGRESS_PATH)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    yaw_90 = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    handle_pos = torch.tensor([[1.0, 2.0, 0.8], [0.0, 0.0, 0.8]])

    left, right = module.compute_handle_targets(
        handle_pos,
        torch.stack((identity, yaw_90)),
        half_width=0.18,
    )

    assert torch.allclose(left[0], torch.tensor([1.0, 2.18, 0.8]), atol=1.0e-6)
    assert torch.allclose(right[0], torch.tensor([1.0, 1.82, 0.8]), atol=1.0e-6)
    assert torch.allclose(left[1], torch.tensor([-0.18, 0.0, 0.8]), atol=1.0e-6)
    assert torch.allclose(right[1], torch.tensor([0.18, 0.0, 0.8]), atol=1.0e-6)


def test_cart_planar_heading_uses_projected_handle_forward_axis():
    module = _load_module("cart_progress_heading_under_test", CART_PROGRESS_PATH)
    roll_90 = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]])
    yaw_90_roll_90 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    heading = module.compute_planar_heading_quat(torch.cat((roll_90, yaw_90_roll_90), dim=0))
    forward = module._quat_apply(
        heading,
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    assert torch.allclose(forward[0], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(forward[1], torch.tensor([0.0, 1.0, 0.0]), atol=1.0e-6)


def test_cart_goal_uses_initial_cart_local_forward_and_lateral_axes():
    module = _load_module("cart_progress_goal_under_test", CART_PROGRESS_PATH)
    yaw_90 = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    goal = module.compute_cart_goal(
        initial_pos_w=torch.tensor([[1.0, 2.0, 0.5]]),
        initial_quat_w=yaw_90,
        forward=torch.tensor([3.0]),
        lateral=torch.tensor([0.5]),
    )

    assert torch.allclose(goal, torch.tensor([[0.5, 5.0, 0.5]]), atol=1.0e-6)


def test_transport_progress_clamps_regression_and_normalizes_distance():
    module = _load_module("cart_progress_transport_under_test", CART_PROGRESS_PATH)
    progress = module.compute_transport_progress(
        initial_distance=torch.tensor([4.0, 2.0, 0.0]),
        current_distance=torch.tensor([2.0, 3.0, 0.0]),
    )

    assert torch.allclose(progress, torch.tensor([0.5, 0.0, 0.0]))
