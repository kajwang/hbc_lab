import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/contact_labels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("contact_label_pose_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contact_label_initializes_identity_target_orientations():
    module = _load_module()
    labels = module.ContactLabel.from_active_hand(
        torch.tensor([0, 1]),
        contact_mode=module.ContactMode.GRASP,
    )

    expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(2, 2, 4)
    assert torch.equal(labels.target_orientation, expected)


def test_contact_label_sets_pose_and_masks_inactive_effector():
    module = _load_module()
    labels = module.ContactLabel.from_active_hand(
        torch.tensor([1]),
        contact_mode=module.ContactMode.GRASP,
    )
    positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    orientations = torch.tensor(
        [[[0.7071068, 0.0, 0.0, 0.7071068], [0.7071068, 0.0, 0.7071068, 0.0]]]
    )

    labels.set_target_region_pose(torch.tensor([0]), positions, orientations)

    assert torch.equal(labels.target_region[0, 0], torch.zeros(3))
    assert torch.allclose(labels.target_region[0, 1], positions[0, 1])
    assert torch.equal(labels.target_orientation[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(labels.target_orientation[0, 1], orientations[0, 1])
