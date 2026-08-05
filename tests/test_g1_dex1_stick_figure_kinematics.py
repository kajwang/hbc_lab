import importlib.util
import math
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp/stick_figure_kinematics.py"


def _load_module():
    assert MODULE_PATH.exists(), "stick-figure kinematics module has not been implemented"
    spec = importlib.util.spec_from_file_location("stick_figure_kinematics_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_elbow_flexion_couples_hand_position_and_orientation():
    module = _load_module()

    center_pos, center_quat = module.stick_figure_hand_center_pose(
        upper_pitch=torch.zeros(1),
        upper_azimuth=torch.zeros(1),
        upper_roll=torch.zeros(1),
        elbow_flexion=torch.full((1,), 0.5 * math.pi),
        wrist_angles=torch.zeros(1, 3),
        upper_arm_length=0.30,
        forearm_length=0.25,
        fixed_palm_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        hand_center_offset=torch.zeros(3),
    )

    torch.testing.assert_close(center_pos, torch.tensor([[0.30, 0.0, -0.25]]), atol=1.0e-5, rtol=1.0e-5)
    expected_quat = torch.tensor([[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]])
    torch.testing.assert_close(center_quat, expected_quat, atol=1.0e-5, rtol=1.0e-5)


def test_upper_roll_rotates_elbow_bending_plane_without_moving_elbow():
    module = _load_module()

    center_pos, _ = module.stick_figure_hand_center_pose(
        upper_pitch=torch.zeros(1),
        upper_azimuth=torch.zeros(1),
        upper_roll=torch.full((1,), 0.5 * math.pi),
        elbow_flexion=torch.full((1,), 0.5 * math.pi),
        wrist_angles=torch.zeros(1, 3),
        upper_arm_length=0.30,
        forearm_length=0.25,
        fixed_palm_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        hand_center_offset=torch.zeros(3),
    )

    torch.testing.assert_close(center_pos, torch.tensor([[0.30, 0.25, 0.0]]), atol=1.0e-5, rtol=1.0e-5)


def test_lowered_upper_arm_uses_forward_natural_elbow_bend():
    module = _load_module()

    center_pos, _ = module.stick_figure_hand_center_pose(
        upper_pitch=torch.full((1,), -0.5 * math.pi),
        upper_azimuth=torch.zeros(1),
        upper_roll=torch.zeros(1),
        elbow_flexion=torch.full((1,), 0.5 * math.pi),
        wrist_angles=torch.zeros(1, 3),
        upper_arm_length=0.30,
        forearm_length=0.25,
        fixed_palm_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        hand_center_offset=torch.zeros(3),
    )

    torch.testing.assert_close(center_pos, torch.tensor([[0.25, 0.0, -0.30]]), atol=1.0e-5, rtol=1.0e-5)


def test_hand_center_offset_follows_sampled_wrist_orientation():
    module = _load_module()

    center_pos, _ = module.stick_figure_hand_center_pose(
        upper_pitch=torch.zeros(1),
        upper_azimuth=torch.zeros(1),
        upper_roll=torch.zeros(1),
        elbow_flexion=torch.zeros(1),
        wrist_angles=torch.tensor([[0.0, 0.0, 0.5 * math.pi]]),
        upper_arm_length=0.30,
        forearm_length=0.25,
        fixed_palm_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        hand_center_offset=torch.tensor([0.10, 0.0, 0.0]),
    )

    torch.testing.assert_close(center_pos, torch.tensor([[0.55, 0.10, 0.0]]), atol=1.0e-5, rtol=1.0e-5)
