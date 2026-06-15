import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source/hbc_lab"))


from hbc_lab.tasks.manager_based.humanoid_tracking.mdp.reference_motion import (  # noqa: E402
    ReferenceMotionCfg,
    build_reference_joint_offsets,
)


def test_stand_reference_returns_zero_offsets_for_all_joints():
    cfg = ReferenceMotionCfg(kind="stand", joint_names=("a", "b", "c"))

    offsets = build_reference_joint_offsets(cfg, phase=[0.0, 0.5])

    assert offsets == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_arm_swing_reference_only_moves_selected_arm_joints():
    cfg = ReferenceMotionCfg(
        kind="arm_swing",
        joint_names=("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_knee_joint"),
        amplitude=0.25,
    )

    offsets = build_reference_joint_offsets(cfg, phase=[0.25])

    assert len(offsets) == 1
    assert len(offsets[0]) == 3
    assert offsets[0][2] == 0.0
    assert offsets[0][0] > 0.20
    assert offsets[0][1] < -0.20
