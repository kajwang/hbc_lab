"""Small, framework-agnostic helpers for fixed-horizon policy evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _sum_and_count(value: Any) -> tuple[float, int] | None:
    """Return the numeric sum and element count for scalars or tensor-like values."""
    if isinstance(value, (bool, int, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"Benchmark metric is not finite: {scalar}")
        return scalar, 1

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel") and hasattr(value, "sum"):
        count = int(value.numel())
        if count == 0:
            return None
        total = float(value.sum().item())
        if not math.isfinite(total):
            raise ValueError("Benchmark tensor contains non-finite values.")
        return total, count

    if isinstance(value, (list, tuple)):
        total = 0.0
        count = 0
        for item in value:
            result = _sum_and_count(item)
            if result is not None:
                total += result[0]
                count += result[1]
        return (total, count) if count else None

    return None


@dataclass
class BenchmarkAccumulator:
    """Accumulate rollout and environment log metrics without depending on PyTorch."""

    steps: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    episodes: int = 0
    successes: int = 0
    success_metric_seen: bool = False
    metric_sums: dict[str, float] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)

    def update(self, rewards: Any, dones: Any, extras: dict[str, Any] | None) -> None:
        reward_stats = _sum_and_count(rewards)
        done_stats = _sum_and_count(dones)
        if reward_stats is None or done_stats is None:
            raise ValueError("Benchmark rewards and dones must be non-empty numeric values.")

        self.steps += 1
        self.reward_sum += reward_stats[0]
        self.reward_count += reward_stats[1]
        self.episodes += int(round(done_stats[0]))

        log_values = (extras or {}).get("log", {})
        if not isinstance(log_values, dict):
            return

        for name, value in log_values.items():
            stats = _sum_and_count(value)
            if stats is None:
                continue
            self.metric_sums[name] = self.metric_sums.get(name, 0.0) + stats[0]
            self.metric_counts[name] = self.metric_counts.get(name, 0) + stats[1]
            if name == "Task/success_count":
                self.success_metric_seen = True
                self.successes += int(round(stats[0]))

    def summary(self, elapsed_seconds: float) -> dict[str, Any]:
        metrics = {
            name: self.metric_sums[name] / self.metric_counts[name]
            for name in sorted(self.metric_sums)
            if self.metric_counts[name] > 0
        }
        return {
            "rollout_steps": self.steps,
            "environment_steps": self.reward_count,
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.successes / self.episodes if self.episodes else None,
            "success_metric_seen": self.success_metric_seen,
            "mean_reward": self.reward_sum / self.reward_count if self.reward_count else None,
            "elapsed_seconds": elapsed_seconds,
            "throughput_environment_steps_per_second": (
                self.reward_count / elapsed_seconds if elapsed_seconds > 0.0 else None
            ),
            "metrics": metrics,
        }


def write_benchmark_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    """Write one benchmark result as strict, deterministic JSON."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output_path
