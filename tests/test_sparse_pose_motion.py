import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/pose_motion.py"


def _load_module():
    assert MODULE_PATH.exists(), f"Missing sparse pose-motion module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("sparse_pose_motion_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_axis_angle_and_quaternion_log_preserve_rotation_axis():
    module = _load_module()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = module.quat_from_axis_angle(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([math.pi / 2.0]),
    )

    error = module.quaternion_log_error(identity, target)

    assert torch.allclose(error.abs(), torch.tensor([[0.0, 0.0, math.pi / 2.0]]), atol=1.0e-5)


def test_local_axis_rotation_matches_door_handle_joint_frame():
    module = _load_module()
    initial_handle_quat = torch.tensor(
        [[-0.70710678, -0.70710678, 0.0, 0.0]],
    )
    expected_unlocked_quat = torch.tensor(
        [[-0.88448924, -0.46656057, 0.0, 0.0]],
    )

    target = module.compose_local_axis_rotation(
        initial_handle_quat,
        axis_local=torch.tensor([[-1.0, 0.0, 0.0]]),
        angle=torch.tensor([0.6]),
    )

    assert torch.allclose(target, expected_unlocked_quat, atol=1.0e-5)


def test_local_axis_pose_rotation_moves_point_around_pivot():
    module = _load_module()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    target_pos, target_quat = module.compose_local_axis_pose(
        frame_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        frame_quat=identity,
        pivot_pos=torch.zeros(1, 3),
        axis_local=torch.tensor([[0.0, 0.0, 1.0]]),
        angle=torch.tensor([math.pi / 2.0]),
    )

    expected_quat = module.quat_from_axis_angle(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([math.pi / 2.0]),
    )
    assert torch.allclose(target_pos, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-5)
    assert torch.allclose(target_quat, expected_quat, atol=1.0e-5)


def test_pose_error_respects_position_and_rotation_masks():
    module = _load_module()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    yaw = module.quat_from_axis_angle(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([1.0]),
    )

    total, position, orientation = module.compute_masked_pose_error(
        current_pos=torch.tensor([[1.0, 2.0, 3.0]]),
        current_quat=identity,
        target_pos=torch.zeros(1, 3),
        target_quat=yaw,
        position_mask=torch.tensor([[1.0, 0.0, 0.0]]),
        rotation_mask=torch.tensor([[0.0, 0.0, 1.0]]),
        position_scale=1.0,
        rotation_scale=1.0,
    )

    assert torch.allclose(position, torch.tensor([1.0]), atol=1.0e-5)
    assert torch.allclose(orientation, torch.tensor([1.0]), atol=1.0e-5)
    assert torch.allclose(total, torch.tensor([2.0]), atol=1.0e-5)


def test_gather_keyframe_selects_per_environment_targets():
    module = _load_module()
    values = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    selected = module.gather_keyframe(values, torch.tensor([1, 0]))

    assert torch.equal(selected, torch.tensor([[2.0], [3.0]]))


def test_flatten_motion_frame_markers_keeps_world_poses_in_environment_order():
    module = _load_module()
    current_pos_w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    current_quat_w = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    keyframe_pos_w = torch.tensor(
        [
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
            [[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]],
        ]
    )
    keyframe_quat_w = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5]],
        ]
    )

    positions, orientations, marker_indices = module.flatten_motion_frame_markers(
        current_pos_w,
        current_quat_w,
        keyframe_pos_w,
        keyframe_quat_w,
    )

    assert torch.equal(
        positions,
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
                [4.0, 5.0, 6.0],
                [30.0, 31.0, 32.0],
                [40.0, 41.0, 42.0],
            ]
        ),
    )
    assert torch.equal(
        orientations,
        torch.stack(
            (
                current_quat_w[0],
                keyframe_quat_w[0, 0],
                keyframe_quat_w[0, 1],
                current_quat_w[1],
                keyframe_quat_w[1, 0],
                keyframe_quat_w[1, 1],
            )
        ),
    )
    assert torch.equal(marker_indices, torch.tensor([0, 1, 2, 0, 1, 2]))


def test_cumulative_keyframe_progress_is_continuous_across_phases():
    module = _load_module()

    raw, normalized = module.compute_cumulative_keyframe_progress(
        current_error=torch.tensor([2.0, 1.0, 0.5]),
        phase_initial_error=torch.tensor([2.0, 2.0, 2.0]),
        keyframe_index=torch.tensor([0, 0, 1]),
        num_keyframes=2,
    )

    assert torch.allclose(raw, torch.tensor([0.0, 0.5, 1.75]))
    assert torch.allclose(normalized, torch.tensor([0.0, 0.25, 0.875]))


def test_cumulative_keyframe_progress_clamps_regression():
    module = _load_module()

    raw, normalized = module.compute_cumulative_keyframe_progress(
        current_error=torch.tensor([3.0, 0.0]),
        phase_initial_error=torch.tensor([2.0, 2.0]),
        keyframe_index=torch.tensor([0, 1]),
        num_keyframes=2,
    )

    assert torch.allclose(raw, torch.tensor([0.0, 2.0]))
    assert torch.allclose(normalized, torch.tensor([0.0, 1.0]))


def test_generic_pose_manip_reward_retains_accumulated_state_progress():
    module = _load_module()

    reward = module.generic_pose_manip_reward(
        normalized_progress=torch.tensor([0.0, 0.25, 0.5, 1.0]),
    )

    assert torch.allclose(reward, torch.tensor([0.3, 0.55, 0.8, 1.3]))
