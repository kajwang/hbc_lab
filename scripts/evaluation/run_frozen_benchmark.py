#!/usr/bin/env python3
"""Run and verify the frozen HBC Lab interaction baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/frozen_baselines_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_python_tree(path: Path) -> str:
    """Hash Python sources using relative paths as well as file contents."""
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*.py") if item.is_file())
    for file_path in files:
        relative_path = file_path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _verify_digest(path: Path, expected: str, actual: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if actual != expected:
        raise RuntimeError(f"{label} drifted: {path}\nexpected {expected}\nactual   {actual}")


def verify_manifest_sources(manifest: dict[str, Any], allow_source_drift: bool) -> None:
    errors: list[str] = []
    for name, entry in manifest["repository"]["source_trees"].items():
        path = _resolve_repo_path(entry["path"])
        try:
            _verify_digest(path, entry["sha256"], sha256_python_tree(path), f"source tree '{name}'")
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(str(exc))

    for name, entry in manifest["repository"]["source_files"].items():
        path = _resolve_repo_path(entry["path"])
        try:
            actual = sha256_file(path) if path.exists() else "missing"
            _verify_digest(path, entry["sha256"], actual, f"source file '{name}'")
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(str(exc))

    if errors and not allow_source_drift:
        raise RuntimeError("Frozen source verification failed:\n\n" + "\n\n".join(errors))
    for error in errors:
        print(f"[WARN] {error}", file=sys.stderr)


def verify_checkpoint(entry: dict[str, Any], label: str) -> Path:
    path = _resolve_repo_path(entry["path"])
    actual = sha256_file(path) if path.exists() else "missing"
    _verify_digest(path, entry["sha256"], actual, label)
    return path


def _selected_names(manifest: dict[str, Any], requested: list[str]) -> list[str]:
    available = list(manifest["baselines"])
    if requested == ["all"] or "all" in requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {', '.join(unknown)}. Available: {', '.join(available)}")
    return requested


def build_command(
    manifest: dict[str, Any],
    name: str,
    args: argparse.Namespace,
    low_level_path: Path,
    output_path: Path,
) -> list[str]:
    baseline = manifest["baselines"][name]
    defaults = manifest["evaluation"]
    command = [
        args.python,
        str(REPO_ROOT / "scripts/rsl_rl/play.py"),
        f"--task={baseline['task_id']}",
        f"--checkpoint={_resolve_repo_path(baseline['checkpoint']['path'])}",
        f"--num_envs={args.num_envs or defaults['num_envs']}",
        f"--seed={args.seed if args.seed is not None else defaults['seed']}",
        f"--benchmark_steps={args.steps or defaults['steps']}",
        f"--benchmark_warmup_steps={args.warmup_steps if args.warmup_steps is not None else defaults['warmup_steps']}",
        f"--benchmark_output={output_path}",
    ]
    if not args.visual:
        command.append("--headless")
    command.extend(
        [
            f"env.low_level_policy_path={low_level_path}",
            "env.enable_debug_visualization=False",
            *baseline.get("overrides", []),
            *args.extra_override,
        ]
    )
    return command


def _baseline_environment(baseline: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    source_root = baseline.get("source_root")
    if source_root:
        environment["HBC_LAB_SOURCE_ROOT"] = str(_resolve_repo_path(source_root).resolve())
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", action="append", default=None, help="Baseline name; repeat or use 'all'.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/benchmarks/frozen_v1")
    parser.add_argument("--python", default=sys.executable, help="Python executable that owns the Isaac Lab environment.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visual", action="store_true", help="Launch Isaac Sim with rendering instead of headless mode.")
    parser.add_argument("--allow-source-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify files and print commands without launching Isaac Sim.")
    parser.add_argument("--list", action="store_true", help="List frozen baselines and exit.")
    parser.add_argument("--extra-override", action="append", default=[], help="Additional env.* or agent.* override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    requested = args.baseline or ["all"]
    selected = _selected_names(manifest, requested)

    if args.list:
        for name in manifest["baselines"]:
            baseline = manifest["baselines"][name]
            print(f"{name:12s} {baseline['task_id']}  [{baseline['capability_status']}]")
        return 0

    verify_manifest_sources(manifest, args.allow_source_drift)
    low_level_path = verify_checkpoint(manifest["shared"]["low_level_policy"], "low-level policy")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in selected:
        baseline = manifest["baselines"][name]
        verify_checkpoint(baseline["checkpoint"], f"checkpoint '{name}'")
        output_path = args.output_dir / f"{name}.json"
        output_path.unlink(missing_ok=True)
        command = build_command(manifest, name, args, low_level_path, output_path)
        source_prefix = ""
        if baseline.get("source_root"):
            source_prefix = f"HBC_LAB_SOURCE_ROOT={_resolve_repo_path(baseline['source_root']).resolve()} "
        print(f"[INFO] {name}: {source_prefix}{shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, env=_baseline_environment(baseline), check=True)
            if not output_path.is_file():
                raise RuntimeError(f"Benchmark process did not produce its result: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
