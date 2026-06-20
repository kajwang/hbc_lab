import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source/hbc_lab"
sys.path.insert(0, str(SOURCE_ROOT))


from hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.mdp.contact_progress import (  # noqa: E402
    LEFT_HAND,
    RIGHT_HAND,
    compute_active_hand_grasp_progress,
    compute_hand_contact_confidence,
    sample_active_hands,
    select_active_hand_value,
)
from hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.mdp.gripper import (  # noqa: E402
    LEFT_DEX3_CLOSE_POSE,
    RIGHT_DEX3_CLOSE_POSE,
    interpolate_dex3_hand_pose,
)


def test_interpolate_dex3_hand_pose_maps_zero_to_open_and_one_to_closed():
    grip = torch.tensor([[0.0], [0.5], [1.0]])

    left_pose = interpolate_dex3_hand_pose(grip, side="left")
    right_pose = interpolate_dex3_hand_pose(grip.squeeze(-1), side="right")

    assert left_pose.shape == (3, 7)
    assert right_pose.shape == (3, 7)
    assert torch.allclose(left_pose[0], torch.zeros(7))
    assert torch.allclose(left_pose[1], 0.5 * LEFT_DEX3_CLOSE_POSE)
    assert torch.allclose(left_pose[2], LEFT_DEX3_CLOSE_POSE)
    assert torch.allclose(right_pose[2], RIGHT_DEX3_CLOSE_POSE)


def test_active_hand_selection_uses_only_the_sampled_hand_values():
    left_value = torch.tensor([1.0, 2.0, 3.0])
    right_value = torch.tensor([10.0, 20.0, 30.0])
    active_hand = torch.tensor([LEFT_HAND, RIGHT_HAND, LEFT_HAND])

    selected = select_active_hand_value(left_value, right_value, active_hand)

    assert torch.allclose(selected, torch.tensor([1.0, 20.0, 3.0]))


def test_sample_active_hands_can_force_one_side_for_curriculum_or_contact_labels():
    assert torch.equal(sample_active_hands(4, torch.device("cpu"), left_probability=1.0), torch.zeros(4, dtype=torch.long))
    assert torch.equal(sample_active_hands(4, torch.device("cpu"), left_probability=0.0), torch.ones(4, dtype=torch.long))


def test_hand_contact_confidence_uses_whole_hand_object_force():
    hand_force = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    components = compute_hand_contact_confidence(hand_force, force_threshold=1.0)

    assert components.contact[0] > 0.7
    assert components.force[0] == 2.0
    assert components.contact[1] == 0.0


def test_active_hand_grasp_progress_ignores_non_active_hand_even_if_it_contacts():
    left_components = compute_hand_contact_confidence(
        torch.tensor([[2.0, 0.0, 0.0]]),
        force_threshold=1.0,
    )
    right_components = compute_hand_contact_confidence(
        torch.zeros(1, 3),
        force_threshold=1.0,
    )

    progress = compute_active_hand_grasp_progress(
        left_components=left_components,
        right_components=right_components,
        active_hand=torch.tensor([RIGHT_HAND]),
        left_grip=torch.tensor([[1.0]]),
        right_grip=torch.tensor([[1.0]]),
        left_distance=torch.tensor([0.05]),
        right_distance=torch.tensor([0.05]),
    )

    assert progress.contact.item() == 0.0
    assert progress.grasp.item() == 0.0
