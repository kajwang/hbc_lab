# G1 Dex1 Multi-Shape Visual Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two matched local visual-play entries that expose one deterministic training or OOD object per environment for direct BPS versus no-BPS grasp inspection.

**Architecture:** Reuse the existing parameterized MultiShape play task and its ordered multi-asset spawner. Make the play config an 11-environment deterministic evaluator, print the environment-to-shape mapping once, and select BPS behavior only through `env.object_shape_bps_enabled`. Copy the latest common checkpoint iteration, `model_11300.pt`, from JC and JD into stable local paths.

**Tech Stack:** Isaac Lab manager-based environments, Hydra overrides, RSL-RL, VS Code debugpy launch configurations, pytest.

---

### Task 1: Deterministic visual-evaluation contract

**Files:**
- Modify: `tests/test_g1_dex1_multishape_bps.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_shape_bps.py`
- Modify: `source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/multishape_bps_env_cfg.py`

- [ ] Add a failing test that requires `format_shape_assignment(ALL_SHAPE_NAMES)` to list environments 0 through 10, label the final three entries as OOD, and preserve the exact asset order.
- [ ] Add static assertions that the play scene contains 11 environments at 6 m spacing and that the duplicate `@configclass` decorator is absent.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py` and confirm the new assertions fail.
- [ ] Implement:

```python
def format_shape_assignment(shape_names: tuple[str, ...]) -> str:
    lines = ["[MultiShape Visual Eval] deterministic asset assignment:"]
    for env_id, shape_name in enumerate(shape_names):
        split = "OOD" if shape_name in HELD_OUT_SHAPE_NAMES else "train"
        lines.append(f"  env {env_id:02d}: {shape_name} ({split})")
    return "\n".join(lines)
```

- [ ] Set `G1Dex1HierDrcMultiShapePlayEnvCfg.scene` and `self.scene.num_envs` to 11, use 6 m spacing, and print the mapping once in `__post_init__`.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py` and confirm it passes.

### Task 2: Separate matched BPS and no-BPS launch entries

**Files:**
- Modify: `tests/test_g1_dex1_multishape_bps.py`
- Modify: `.vscode/launch.json`
- Modify: `docs/superpowers/specs/2026-08-20-g1-dex1-multishape-visual-eval-design.md`

- [ ] Add failing static tests requiring both launch names, 11 environments, fixed right-hand evaluation, 6 m spacing, BPS toggles, and the two stable `model_11300.pt` paths.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py` and confirm the launch assertions fail.
- [ ] Replace the old generic play entries with:

```text
g1_dex1_multishape_bps_visual_play
g1_dex1_multishape_no_bps_visual_play
```

Each entry uses the same play task, `--num_envs=11`, `env.scene.env_spacing=6.0`, `env.commands.high_level.left_hand_probability=0.0`, the same low-level checkpoint, and its corresponding BPS switch and high-level checkpoint.
- [ ] Update the design document checkpoint contract from `model_10500.pt` to the verified latest common `model_11300.pt`.
- [ ] Parse `.vscode/launch.json` as JSON-with-comments through the existing static test source checks and rerun the focused test.

### Task 3: Checkpoint synchronization and verification

**Files:**
- Create runtime artifacts only under:
  - `logs/5090/g1_dex1/multishape/bps/model_11300.pt`
  - `logs/5090/g1_dex1/multishape/no_bps/model_11300.pt`

- [ ] Copy the BPS checkpoint from:

```text
jc_5090:/home/kaijun/hbc_lab/logs/rsl_rl/g1_dex1_hier_drc_multishape/2026-08-19_10-45-40_multishape8_directionalbps64_from_motion_m8200_50k/model_11300.pt
```

- [ ] Copy the no-BPS checkpoint from:

```text
jd_5090:/home/zju/code/hbc_lab/logs/rsl_rl/g1_dex1_hier_drc_multishape/2026-08-19_20-28-33_multishape8_no_bps_from_motion_m8200_50k/model_11300.pt
```

- [ ] Compare remote and local SHA-256 hashes for both files.
- [ ] Load each checkpoint with `torch.load(..., map_location="cpu", weights_only=False)` and verify the first actor layer expects 1552 inputs for BPS and 1360 for no-BPS.
- [ ] Run `pytest -q tests/test_g1_dex1_multishape_bps.py tests/test_g1_dex1_hier_drc_static.py`.
- [ ] Run `python -m compileall -q source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc`.
- [ ] If local Isaac Sim can launch without monopolizing the user's display, run one short Play smoke test per entry; otherwise report that visual runtime verification remains for the user.
