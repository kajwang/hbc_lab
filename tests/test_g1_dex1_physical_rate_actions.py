from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/high_level_actions.py"
)
SPEC = importlib.util.spec_from_file_location("g1_dex1_physical_rate_actions_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

HighLevelActionLimits = MODULE.HighLevelActionLimits
HighLevelCommandState = MODULE.HighLevelCommandState
decode_high_level_action = MODULE.decode_high_level_action


def _command_state(left_position: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return HighLevelCommandState(
        base_velocity=torch.zeros(1, 3),
        posture_command=torch.tensor([[0.70, 0.0]]),
        left_hand_center_pose_a=torch.cat((torch.tensor([left_position]), quat), dim=-1),
        right_hand_center_pose_a=torch.cat((torch.zeros(1, 3), quat), dim=-1),
        left_grip=torch.zeros(1, 1),
        right_grip=torch.zeros(1, 1),
    )


def test_translation_and_posture_are_invariant_to_high_level_frequency():
    action = torch.zeros(1, 19)
    action[:, 3] = -0.5
    action[:, 4] = 0.5
    action[:, 5] = 1.0

    one, _ = decode_high_level_action(action, _command_state(), HighLevelActionLimits(), dt=0.1)
    half, _ = decode_high_level_action(action, _command_state(), HighLevelActionLimits(), dt=0.05)
    two, _ = decode_high_level_action(action, half, HighLevelActionLimits(), dt=0.05)

    assert torch.allclose(one.left_hand_center_pose_a[:, :3], two.left_hand_center_pose_a[:, :3], atol=1.0e-6)
    assert torch.allclose(one.posture_command, two.posture_command, atol=1.0e-6)
    assert torch.allclose(one.left_hand_center_pose_a[:, 0], torch.tensor([0.03]), atol=1.0e-6)


def test_orientation_is_invariant_to_high_level_frequency():
    action = torch.zeros(1, 19)
    action[:, 8:11] = torch.tensor([0.5, -0.25, 0.75])

    one, _ = decode_high_level_action(action, _command_state(), HighLevelActionLimits(), dt=0.1)
    half, _ = decode_high_level_action(action, _command_state(), HighLevelActionLimits(), dt=0.05)
    two, _ = decode_high_level_action(action, half, HighLevelActionLimits(), dt=0.05)

    assert torch.allclose(one.left_hand_center_pose_a[:, 3:], two.left_hand_center_pose_a[:, 3:], atol=1.0e-6)


def test_decoded_rate_reports_physical_units():
    action = torch.zeros(1, 19)
    action[:, 0] = 1.0
    action[:, 2] = 1.0
    action[:, 3] = 1.0
    action[:, 4] = 1.0
    action[:, 5] = 1.0
    action[:, 8] = 1.0

    _, rate = decode_high_level_action(action, _command_state(), HighLevelActionLimits(), dt=0.1)

    assert torch.allclose(rate.base_acceleration, torch.tensor([[1.5, 0.0, 1.5]]))
    assert torch.allclose(rate.posture_velocity, torch.tensor([[0.12, 0.40]]))
    assert torch.allclose(rate.left_hand_twist[:, 0], torch.tensor([0.30]))
    assert torch.allclose(rate.left_hand_twist[:, 3], torch.tensor([1.00]))
    assert rate.as_tensor().shape == (1, 17)


def test_posture_rate_reports_actual_motion_after_range_clamp():
    previous = _command_state()
    previous.posture_command[:, 0] = HighLevelActionLimits().root_height_range[1]
    action = torch.zeros(1, 19)
    action[:, 3] = 1.0

    decoded, rate = decode_high_level_action(action, previous, HighLevelActionLimits(), dt=0.1)

    assert torch.allclose(decoded.posture_command[:, 0], previous.posture_command[:, 0])
    assert torch.allclose(rate.posture_velocity[:, 0], torch.zeros(1))


def test_physical_rate_decoder_does_not_hard_clamp_workspace():
    previous = _command_state(left_position=(0.80, 0.20, -0.10))
    action = torch.zeros(1, 19)
    action[:, 5] = 1.0

    decoded, _ = decode_high_level_action(action, previous, HighLevelActionLimits(), dt=0.1)

    assert decoded.left_hand_center_pose_a[0, 0] > 0.80
