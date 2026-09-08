#!/usr/bin/env python3
"""Play an HBC Lab RSL-RL checkpoint with the registered play environment config."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WBC_ROOT = REPO_ROOT.parent
HBC_LAB_SOURCE_ROOT = Path(os.environ.get("HBC_LAB_SOURCE_ROOT", REPO_ROOT / "source" / "hbc_lab"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(HBC_LAB_SOURCE_ROOT))
for isaaclab_source in (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
):
    sys.path.insert(0, str(WBC_ROOT / "IsaacLab" / "source" / isaaclab_source))

import hbc_lab.tasks  # noqa: E402,F401
from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an HBC Lab RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for deterministic environment evaluation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--benchmark_steps",
    type=int,
    default=0,
    help="Run exactly this many measured policy steps, write a summary, and exit. Zero disables benchmark mode.",
)
parser.add_argument(
    "--benchmark_warmup_steps",
    type=int,
    default=0,
    help="Number of unmeasured policy steps before a benchmark rollout.",
)
parser.add_argument("--benchmark_output", type=str, default=None, help="Optional JSON output path for benchmark mode.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.benchmark_steps < 0 or args_cli.benchmark_warmup_steps < 0:
    parser.error("--benchmark_steps and --benchmark_warmup_steps must be non-negative.")
if args_cli.benchmark_steps > 0 and args_cli.video:
    parser.error("Benchmark mode does not support --video; run a separate visual play rollout.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from evaluation.benchmark_metrics import BenchmarkAccumulator, write_benchmark_summary
from hbc_lab.learning import register_rsl_rl_extensions
from hbc_lab.utils.parser_cfg import parse_env_cfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path


register_rsl_rl_extensions()


def _parse_override_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return raw_value


def _set_nested_cfg_value(cfg, dotted_key: str, value) -> None:
    target = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        target = target[key] if isinstance(target, dict) else getattr(target, key)

    leaf_key = keys[-1]
    if isinstance(target, dict):
        current_value = target.get(leaf_key)
        target[leaf_key] = tuple(value) if isinstance(current_value, tuple) and isinstance(value, list) else value
        return

    current_value = getattr(target, leaf_key)
    if isinstance(current_value, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(target, leaf_key, value)


def _apply_cfg_overrides(env_cfg, agent_cfg, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Unsupported play override '{override}'. Expected key=value.")
        key, raw_value = override.split("=", 1)
        if key.startswith("env."):
            _set_nested_cfg_value(env_cfg, key.removeprefix("env."), _parse_override_value(raw_value))
        elif key.startswith("agent."):
            _set_nested_cfg_value(agent_cfg, key.removeprefix("agent."), _parse_override_value(raw_value))
        else:
            raise ValueError(
                f"Unsupported play override '{override}'. Use 'env.<field>=...' or 'agent.<field>=...'."
            )


def main():
    """Play with RSL-RL."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    _apply_cfg_overrides(env_cfg, agent_cfg, hydra_args)
    if hasattr(env_cfg, "apply_grasp_candidate_evaluation"):
        env_cfg.apply_grasp_candidate_evaluation()
    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.seed is not None:
        env.unwrapped.seed(agent_cfg.seed)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    timestep = 0
    rollout_step = 0
    benchmark_accumulator = BenchmarkAccumulator() if args_cli.benchmark_steps > 0 else None
    benchmark_start_time = None
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
        if benchmark_accumulator is not None:
            if rollout_step >= args_cli.benchmark_warmup_steps:
                if benchmark_start_time is None:
                    benchmark_start_time = time.perf_counter()
                benchmark_accumulator.update(rewards, dones, extras)
            rollout_step += 1
            if benchmark_accumulator.steps >= args_cli.benchmark_steps:
                break
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if benchmark_accumulator is not None:
        elapsed_seconds = time.perf_counter() - benchmark_start_time if benchmark_start_time is not None else 0.0
        checkpoint_hash = hashlib.sha256(Path(resume_path).read_bytes()).hexdigest()
        summary = {
            "schema_version": 1,
            "task": args_cli.task,
            "checkpoint": str(Path(resume_path).resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "seed": agent_cfg.seed,
            "num_envs": env.unwrapped.num_envs,
            "warmup_steps": args_cli.benchmark_warmup_steps,
            **benchmark_accumulator.summary(elapsed_seconds),
        }
        print("[INFO] Benchmark summary:")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args_cli.benchmark_output:
            output_path = write_benchmark_summary(args_cli.benchmark_output, summary)
            print(f"[INFO] Wrote benchmark summary to: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
