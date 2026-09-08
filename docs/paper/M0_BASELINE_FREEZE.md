# M0: Frozen Interaction Baselines

This snapshot is the control group for all environment-perception experiments. It freezes the policy interface and
the five interaction tasks before height maps, obstacle representations, or a larger backbone are introduced.

## Frozen Components

- Low-level HBC policy: `logs/exported_policies/g1_dex1_handcenter_stickfigure_m49999.pt`
- High-level command interface and contact-label interface
- PnP, Box Carry, Cart Push, Door Open, and Drawer task implementations
- One reference high-level checkpoint per task
- Evaluation seed, rollout horizon, environment count, and metric aggregation rules

The exact artifact hashes and Python source-tree hashes are stored in
`benchmarks/frozen_baselines_v1.json`. The repository was already a dirty research working tree when this snapshot
was created, so the source hashes are authoritative; the recorded Git commit alone is not.

PnP uses the current physical-rate motion interface. The other four checkpoints were trained before that interface
change and are evaluated against the preserved `legacy/door-model-3500-play` source snapshot at commit
`9dfe6493949820927ca08a0b4868c63900ab1a3a`. Loading them through the current `1360/1510`-dimensional interface
would be semantically wrong even if their old `670/820`-dimensional inputs were padded. The benchmark runner selects
the matching source root automatically.

## Capability Status

| Baseline | Status | Frozen purpose |
| --- | --- | --- |
| PnP | validated | Single-object approach, grasp, and transport reference |
| Box Carry | validated | Bimanual support, lift, and transport reference |
| Cart Push | validated | Bimanual articulated transport reference |
| Door Open | partial | Handle contact and articulated-motion diagnostic; not a final solved baseline |
| Drawer | partial | Open/close keyframe diagnostic; not a final solved baseline |

The shape/BPS/GraspRef branch is intentionally not promoted to a primary baseline. The experiments showed a modest
BPS benefit in visual behavior, but GraspRef guidance did not produce a robust qualitative improvement. Those runs
remain an auxiliary ablation rather than a dependency of the environment-perception stage.

## Unified Evaluation

Run all five baselines headlessly:

```bash
/home/kaijun/anaconda3/envs/robot_lab/bin/python \
  scripts/evaluation/run_frozen_benchmark.py \
  --python /home/kaijun/anaconda3/envs/robot_lab/bin/python
```

Run a quick smoke evaluation of one task:

```bash
/home/kaijun/anaconda3/envs/robot_lab/bin/python \
  scripts/evaluation/run_frozen_benchmark.py \
  --baseline pnp --num-envs 16 --steps 100 --allow-source-drift
```

Results are written to `artifacts/benchmarks/frozen_v1/<task>.json`. Each result contains:

- `success_rate`: successful task terminations divided by all episode terminations
- `mean_reward`: mean reward over every environment step
- `metrics`: time-averaged values already exposed through `extras["log"]`
- `throughput_environment_steps_per_second`: rollout throughput excluding simulator startup
- Checkpoint path/hash, task id, seed, environment count, and measured horizon

Source drift or a checkpoint mismatch fails before Isaac Sim starts. `--allow-source-drift` is intended only for
explicit comparisons against a later implementation; it does not bypass checkpoint verification.

If the legacy worktree is absent, recreate it before running all five baselines:

```bash
git worktree add --detach .worktrees/door-model-3500-play \
  9dfe6493949820927ca08a0b4868c63900ab1a3a
```

## M1 Handoff

The next stage should keep these policies and metrics fixed while adding environment geometry. The first two
environment-aware benchmarks should be:

1. Retrieve the same object from an open tabletop and from beneath an overhang.
2. Carry the same box through open space and through a narrow passage.

This isolates whether geometry observations change whole-body posture and collision avoidance before multi-task
training or a larger temporal backbone is introduced.
