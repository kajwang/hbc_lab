from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/motion_regularization.py"
)
SPEC = importlib.util.spec_from_file_location("g1_dex1_motion_regularization_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE


def _load_module():
    SPEC.loader.exec_module(MODULE)
    return MODULE


def test_contact_conditioned_weights_release_only_active_effectors():
    motion = _load_module()
    mask = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    gate = torch.tensor([[0.9, 0.9], [0.25, 0.75]])

    weight = motion.comfort_weights(mask, gate)

    assert torch.allclose(weight, torch.tensor([[1.0, 0.1], [0.75, 0.25]]))


def test_release_gate_increases_as_target_enters_reach():
    motion = _load_module()
    distance = torch.tensor([[0.75, 0.60, 0.45]])

    gate = motion.reachability_gate(distance, release_radius=0.60, release_width=0.06)

    assert gate[0, 0] < gate[0, 1] < gate[0, 2]
    assert torch.allclose(gate[0, 1], torch.tensor(0.5))


def test_walking_weight_never_disappears_near_target():
    motion = _load_module()
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    gate = torch.tensor([[1.0, 0.0], [0.4, 0.8]])

    weight = motion.walking_posture_weight(mask, gate, near_floor=0.2)

    assert torch.allclose(weight, torch.tensor([0.2, 0.36]))


def test_normalized_deadzone_is_zero_inside_tolerance():
    motion = _load_module()

    value = motion.normalized_deadzone_square(torch.tensor([0.04, 0.15]), deadzone=0.05, scale=0.10)

    assert torch.allclose(value, torch.tensor([0.0, 1.0]))


def test_radial_workspace_penalizes_both_inner_and_outer_violation():
    motion = _load_module()
    position = torch.tensor([[0.10, 0.0, 0.0], [0.30, 0.0, 0.0], [0.70, 0.0, 0.0]])

    loss = motion.radial_workspace_violation(position, radius_range=(0.20, 0.58), scale=0.10)

    assert torch.allclose(loss, torch.tensor([1.0, 0.0, 1.44]), atol=1.0e-6)


def test_quaternion_angular_distance_is_sign_invariant():
    motion = _load_module()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    same_rotation = -identity
    half_turn_x = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    assert torch.allclose(motion.quaternion_angular_distance(identity, same_rotation), torch.zeros(1))
    assert torch.allclose(
        motion.quaternion_angular_distance(identity, half_turn_x), torch.tensor([torch.pi]), atol=1.0e-6
    )


def test_quaternion_angular_distance_broadcasts_reference_pose_across_environments():
    motion = _load_module()
    current = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]).repeat(8, 1, 1)
    reference = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])

    distance = motion.quaternion_angular_distance(current, reference)

    assert distance.shape == (8, 2)
    assert torch.allclose(distance, torch.zeros(8, 2))


def test_mask_normalized_mean_does_not_dilute_single_active_hand():
    motion = _load_module()
    values = torch.tensor([[2.0, 10.0], [4.0, 8.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    mean = motion.mask_normalized_mean(values, mask)

    assert torch.allclose(mean, torch.tensor([2.0, 6.0]))


def test_quality_reward_is_capped_at_twenty_percent_of_stage_scale():
    motion = _load_module()

    reward = motion.scaled_quality_reward(
        torch.tensor([10.0]), torch.tensor([5.0]), coefficient=0.10, loss_cap=2.0
    )

    assert torch.allclose(reward, torch.tensor([-2.0]))


def test_smooth_pose_guidance_improves_as_position_and_orientation_errors_shrink():
    motion = _load_module()
    position_error = torch.tensor([0.30, 0.15, 0.03])
    orientation_error = torch.tensor([1.00, 0.50, 0.10])

    guidance, position_score, orientation_score = motion.smooth_pose_guidance(
        position_error,
        orientation_error,
        position_scale=0.15,
        orientation_gate_scale=0.50,
        position_weight=0.65,
    )

    assert guidance[0] < guidance[1] < guidance[2]
    assert position_score[0] < position_score[1] < position_score[2]
    assert orientation_score[0] < orientation_score[1] < orientation_score[2]


def test_smooth_pose_guidance_keeps_broad_orientation_signal_before_contact():
    motion = _load_module()
    position_error = torch.tensor([0.50, 0.02])
    orientation_error = torch.zeros(2)

    _, _, orientation_score = motion.smooth_pose_guidance(
        position_error,
        orientation_error,
        position_scale=0.15,
        orientation_gate_scale=0.50,
        position_weight=0.65,
    )

    assert orientation_score[0] > 0.20
    assert orientation_score[1] > 0.90


def test_smooth_pose_guidance_uses_linear_normalized_orientation_score():
    motion = _load_module()
    position_error = torch.full((3,), 0.50)
    orientation_error = torch.tensor([torch.pi, 0.5 * torch.pi, 0.0])

    guidance, _, orientation_score = motion.smooth_pose_guidance(
        position_error,
        orientation_error,
        position_scale=0.15,
        orientation_gate_scale=0.50,
        position_weight=0.65,
    )

    near_gate = 1.0 - torch.tanh(torch.tensor(1.0))
    assert torch.allclose(
        orientation_score,
        near_gate * torch.tensor([0.0, 0.5, 1.0]),
        atol=1.0e-6,
    )
    assert guidance[0] < guidance[1] < guidance[2]


def test_pose_guidance_keeps_orientation_independent_and_releases_after_stable_grasp():
    motion = _load_module()
    position_error = torch.tensor([0.30, 0.30, 0.30])
    orientation_error = torch.tensor([torch.pi, 0.5 * torch.pi, 0.0])
    physical_grasp = torch.tensor([0.0, 0.2, 0.8])

    position_guidance, orientation_guidance, position_score, orientation_score, release_gate = (
        motion.stable_grasp_released_pose_guidance(
            position_error,
            orientation_error,
            physical_grasp,
            position_scale=0.15,
            release_threshold=0.50,
            release_width=0.08,
        )
    )

    assert torch.allclose(orientation_score, torch.tensor([0.0, 0.5, 1.0]), atol=1.0e-6)
    assert torch.allclose(position_score, position_score[0].expand_as(position_score))
    assert release_gate[0] < release_gate[1] < release_gate[2]
    assert orientation_guidance[0] < orientation_guidance[1] < orientation_guidance[2]
    assert position_guidance[2] > position_guidance[1] > position_guidance[0]


def test_invalid_widths_and_scales_are_rejected():
    motion = _load_module()

    for call in (
        lambda: motion.reachability_gate(torch.ones(1), release_radius=0.6, release_width=0.0),
        lambda: motion.normalized_deadzone_square(torch.ones(1), deadzone=0.1, scale=0.0),
        lambda: motion.radial_workspace_violation(torch.ones(1, 3), radius_range=(0.2, 0.6), scale=-1.0),
        lambda: motion.smooth_pose_guidance(
            torch.ones(1),
            torch.ones(1),
            position_scale=0.0,
            orientation_gate_scale=0.50,
        ),
        lambda: motion.stable_grasp_released_pose_guidance(
            torch.ones(1),
            torch.ones(1),
            torch.zeros(1),
            position_scale=0.15,
            release_threshold=0.5,
            release_width=0.0,
        ),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
