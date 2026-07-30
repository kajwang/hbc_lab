import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFETY_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/safety.py"
)


def _load_safety_module():
    spec = importlib.util.spec_from_file_location("g1_dex1_safety_under_test", SAFETY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sanitize_tensor_replaces_nonfinite_values_and_clips_finite_outliers():
    safety = _load_safety_module()
    value = torch.tensor([float("nan"), float("inf"), -float("inf"), 250.0, -250.0, 2.0])

    result = safety.sanitize_tensor(value, finite_clip=100.0)

    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.0, 100.0, -100.0, 2.0]))


def test_nonfinite_ratio_and_nested_sanitization_cover_actor_and_critic_observations():
    safety = _load_safety_module()
    observations = {
        "policy": torch.tensor([[1.0, float("nan")], [float("inf"), 3.0]]),
        "critic": torch.tensor([[4.0, -float("inf")]]),
    }

    ratio = safety.nested_nonfinite_ratio(observations)
    result = safety.sanitize_nested_tensors(observations, finite_clip=10.0)

    torch.testing.assert_close(ratio, torch.tensor(3.0 / 6.0))
    assert torch.isfinite(result["policy"]).all()
    assert torch.isfinite(result["critic"]).all()
