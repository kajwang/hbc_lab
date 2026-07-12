import sys
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source/hbc_lab"
MODULE_PATH = (
    SOURCE_ROOT
    / "hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/contact_labels.py"
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


def test_contact_label_encodes_single_active_hands_without_changing_two_value_interface():
    module = _load_contact_labels_module()
    labels = module.ContactLabelCommand.from_active_hand(
        torch.tensor([0, 1, 0]),
        mode=module.ContactMode.INNER_PAD_GRASP,
    )

    assert torch.equal(labels.effector_mask, torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    assert torch.equal(labels.mode, torch.zeros(3, dtype=torch.long))


def test_contact_label_interface_accepts_future_bimanual_effectors():
    module = _load_contact_labels_module()
    labels = module.ContactLabelCommand.from_active_hand(
        torch.tensor([0, 1]),
        mode=module.ContactMode.INNER_PAD_GRASP,
    )

    labels.set_effector_mask(torch.tensor([1]), torch.tensor([[1.0, 1.0]]))

    assert torch.equal(labels.effector_mask[1], torch.tensor([1.0, 1.0]))


def test_contact_label_rejects_empty_effector_selection():
    module = _load_contact_labels_module()
    labels = module.ContactLabelCommand.from_active_hand(
        torch.tensor([0]),
        mode=module.ContactMode.INNER_PAD_GRASP,
    )

    with pytest.raises(ValueError, match="at least one effector"):
        labels.set_effector_mask(torch.tensor([0]), torch.tensor([[0.0, 0.0]]))
