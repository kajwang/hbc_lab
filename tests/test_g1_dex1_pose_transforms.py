import importlib.util
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.utils.math")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/locomotion/mdp/pose_transforms.py"
SPEC = importlib.util.spec_from_file_location("g1_dex1_pose_transforms_under_test", MODULE_PATH)
pose_transforms = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pose_transforms)


def test_posture_anchor_rotates_fixed_shoulder_height_with_pitch():
    root_pos = torch.zeros(1, 3)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    shoulder_pos = torch.tensor([[0.0, 0.2, 0.43]])
    posture_command = torch.tensor([[0.6, 0.5 * math.pi]])

    anchor_pos, _ = pose_transforms.posture_anchor_pose_w(
        root_pos,
        root_quat,
        shoulder_pos,
        torch.zeros_like(root_pos),
        posture_command,
        anchor_height_offset=0.43,
    )

    torch.testing.assert_close(anchor_pos, torch.tensor([[0.43, 0.2, 0.6]]), atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ((0.5 * math.pi, 0.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)),
        ((0.0, 0.5 * math.pi, 0.0), (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)),
        ((0.0, 0.0, 0.5 * math.pi), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
    ],
)
def test_wrist_chain_uses_physical_xyz_joint_axes(angles, expected):
    wrist_angles = torch.tensor([angles])
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])

    quat = pose_transforms.compose_wrist_chain_quat(wrist_angles, identity)

    torch.testing.assert_close(quat, torch.tensor([expected]), atol=1.0e-6, rtol=1.0e-6)


def test_hand_center_offset_rotates_with_fixed_palm_orientation():
    fixed_palm_quat = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]])

    center_pos, center_quat = pose_transforms.hand_base_to_hand_center_pose(
        torch.zeros(1, 3),
        fixed_palm_quat,
        torch.tensor([0.0, 0.1, 0.0]),
    )

    torch.testing.assert_close(center_pos, torch.tensor([[0.1, 0.0, 0.0]]), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(center_quat, fixed_palm_quat)
