import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/contact_sensor_filters.py"
)


def _load_module():
    assert MODULE_PATH.exists(), f"Missing contact-filter module: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("contact_sensor_filters_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_contact_filter_force_uses_per_environment_filter_without_cross_level_leakage():
    module = _load_module()
    force_matrix_w = torch.tensor(
        [
            [[[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
            [[[20.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        ]
    )

    selected = module.select_contact_filter_force(
        force_matrix_w,
        filter_indices=torch.tensor([0, 1]),
    )

    assert torch.equal(selected, torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
