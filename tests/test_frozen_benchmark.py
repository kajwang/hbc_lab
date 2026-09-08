import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.benchmark_metrics import BenchmarkAccumulator, write_benchmark_summary


def test_benchmark_accumulator_counts_successful_episodes_and_scalar_logs(tmp_path: Path):
    accumulator = BenchmarkAccumulator()
    accumulator.update(
        rewards=[1.0, 3.0],
        dones=[0, 1],
        extras={"log": {"Task/success_count": 1.0, "DRC/d_goal_mean": 0.8}},
    )
    accumulator.update(
        rewards=[2.0, 4.0],
        dones=[1, 0],
        extras={"log": {"Task/success_count": 0.0, "DRC/d_goal_mean": 0.4}},
    )

    summary = accumulator.summary(elapsed_seconds=2.0)
    assert summary["rollout_steps"] == 2
    assert summary["environment_steps"] == 4
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["mean_reward"] == 2.5
    assert math.isclose(summary["metrics"]["DRC/d_goal_mean"], 0.6)

    output_path = write_benchmark_summary(tmp_path / "result.json", summary)
    assert json.loads(output_path.read_text()) == summary


def test_frozen_baseline_manifest_has_complete_local_artifacts():
    manifest = json.loads((REPO_ROOT / "benchmarks/frozen_baselines_v1.json").read_text())

    assert set(manifest["baselines"]) == {"pnp", "box_carry", "cart_push", "door_open", "drawer"}
    assert "TO_BE_FILLED" not in json.dumps(manifest)
    for entry in manifest["repository"]["source_trees"].values():
        assert (REPO_ROOT / entry["path"]).exists()
        assert len(entry["sha256"]) == 64
    for entry in manifest["repository"]["source_files"].values():
        assert (REPO_ROOT / entry["path"]).is_file()
        assert len(entry["sha256"]) == 64
    for baseline in manifest["baselines"].values():
        assert (REPO_ROOT / baseline["checkpoint"]["path"]).is_file()
        assert len(baseline["checkpoint"]["sha256"]) == 64
        if "source_root" in baseline:
            assert (REPO_ROOT / baseline["source_root"]).is_dir()
