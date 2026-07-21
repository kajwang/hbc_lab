import sys
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source/hbc_lab"
MODULE_PATH = (
    SOURCE_ROOT
    / "hbc_lab/tasks/manager_based/skill/contact_labels.py"
)
sys.path.insert(0, str(SOURCE_ROOT))


def _load_contact_labels_module():
    assert MODULE_PATH.exists(), f"Missing contact-label module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("g1_dex1_contact_labels_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contact_label_encodes_effectors_targets_and_internal_mode():
    module = _load_contact_labels_module()
    labels = module.ContactLabel.from_active_hand(
        torch.tensor([0, 1, 0]),
        contact_mode=module.ContactMode.GRASP,
    )

    assert torch.equal(labels.effector_mask, torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    assert torch.equal(labels.target_region, torch.zeros(3, 2, 3))
    assert torch.equal(labels.contact_mode, torch.zeros(3, dtype=torch.long))


def test_contact_label_interface_accepts_future_bimanual_effectors():
    module = _load_contact_labels_module()
    labels = module.ContactLabel.from_active_hand(
        torch.tensor([0, 1]),
        contact_mode=module.ContactMode.GRASP,
    )

    labels.set_effector_mask(torch.tensor([1]), torch.tensor([[1.0, 1.0]]))
    labels.set_contact_mode(torch.tensor([1]), module.ContactMode.SUPPORT)

    assert torch.equal(labels.effector_mask[1], torch.tensor([1.0, 1.0]))
    assert labels.contact_mode[1].item() == int(module.ContactMode.SUPPORT)


def test_contact_label_masks_target_regions_for_inactive_effectors():
    module = _load_contact_labels_module()
    labels = module.ContactLabel.from_active_hand(
        torch.tensor([0, 1]),
        contact_mode=module.ContactMode.GRASP,
    )
    targets = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    labels.set_target_region(torch.tensor([0, 1]), targets)

    assert torch.equal(labels.target_region[0], torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(labels.target_region[1], torch.tensor([[0.0, 0.0, 0.0], [10.0, 11.0, 12.0]]))


def test_contact_label_rejects_empty_effector_selection():
    module = _load_contact_labels_module()
    labels = module.ContactLabel.from_active_hand(
        torch.tensor([0]),
        contact_mode=module.ContactMode.GRASP,
    )

    with pytest.raises(ValueError, match="at least one effector"):
        labels.set_effector_mask(torch.tensor([0]), torch.tensor([[0.0, 0.0]]))
