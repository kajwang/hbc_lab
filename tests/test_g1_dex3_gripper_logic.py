import sys
import types
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source/hbc_lab"
MDP_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex3_hier_drc/mdp"
sys.path.insert(0, str(SOURCE_ROOT))


from hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.mdp.contact_progress import (  # noqa: E402
    LEFT_HAND,
    RIGHT_HAND,
    compute_active_hand_grasp_progress,
    compute_dex3_hand_contact_components,
    compute_hand_contact_confidence,
    sample_active_hands,
    select_active_hand_value,
)
from hbc_lab.tasks.manager_based.skill.g1_dex3_hier_drc.mdp.gripper import (  # noqa: E402
    LEFT_DEX3_CLOSE_POSE,
    RIGHT_DEX3_CLOSE_POSE,
    interpolate_dex3_hand_pose,
)


def _load_rewards_module(monkeypatch):
    dummy_mdp = types.SimpleNamespace(
        is_alive=lambda *args, **kwargs: None,
        is_terminated=lambda *args, **kwargs: None,
        lin_vel_z_l2=lambda *args, **kwargs: None,
        ang_vel_xy_l2=lambda *args, **kwargs: None,
        joint_vel_l2=lambda *args, **kwargs: None,
        undesired_contacts=lambda *args, **kwargs: None,
    )

    class _DummyCfg:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.managers",
        types.SimpleNamespace(RewardTermCfg=_DummyCfg, SceneEntityCfg=_DummyCfg),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils",
        types.SimpleNamespace(configclass=lambda cls: cls),
    )
    monkeypatch.setitem(
        sys.modules,
        "hbc_lab.tasks.locomotion",
        types.SimpleNamespace(mdp=dummy_mdp),
    )

    module_path = MDP_ROOT / "rewards.py"
    spec = importlib.util.spec_from_file_location("g1_dex3_rewards_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _reward_env(distance: float, active_grip: float):
    return types.SimpleNamespace(
        d_active_hand=torch.tensor([distance]),
        active_grip=torch.tensor([active_grip]),
        c_contact=torch.zeros(1),
        c_finger_count=torch.zeros(1),
        c_opposition=torch.zeros(1),
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


def test_dex3_grasp_requires_thumb_and_index_or_middle_opposition():
    strong_force = torch.tensor([[3.0, 0.0, 0.0]])
    no_force = torch.zeros(1, 3)

    palm_only = compute_dex3_hand_contact_components(
        palm_force_w=strong_force,
        thumb_force_w=no_force,
        index_force_w=no_force,
        middle_force_w=no_force,
        force_threshold=1.0,
    )
    opposing_fingers = compute_dex3_hand_contact_components(
        palm_force_w=no_force,
        thumb_force_w=strong_force,
        index_force_w=strong_force,
        middle_force_w=no_force,
        force_threshold=1.0,
    )

    palm_progress = compute_active_hand_grasp_progress(
        left_components=palm_only,
        right_components=palm_only,
        active_hand=torch.tensor([LEFT_HAND]),
        left_grip=torch.tensor([[1.0]]),
        right_grip=torch.tensor([[1.0]]),
        left_distance=torch.tensor([0.05]),
        right_distance=torch.tensor([0.05]),
    )
    pinch_progress = compute_active_hand_grasp_progress(
        left_components=opposing_fingers,
        right_components=palm_only,
        active_hand=torch.tensor([LEFT_HAND]),
        left_grip=torch.tensor([[1.0]]),
        right_grip=torch.tensor([[1.0]]),
        left_distance=torch.tensor([0.05]),
        right_distance=torch.tensor([0.05]),
    )

    assert palm_progress.contact.item() > 0.9
    assert palm_progress.opposition.item() == 0.0
    assert palm_progress.grasp.item() == 0.0
    assert pinch_progress.finger_count.item() > 0.9
    assert pinch_progress.opposition.item() > 0.9
    assert pinch_progress.grasp.item() > 0.9


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


def test_couple_reward_does_not_discourage_active_grip_closure_near_object(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)

    open_reward = rewards.couple_reward(_reward_env(distance=0.32, active_grip=0.0))
    closed_reward = rewards.couple_reward(_reward_env(distance=0.32, active_grip=1.0))

    assert closed_reward.item() > open_reward.item()
