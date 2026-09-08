# M1: Environment-Aware Whole-Body Interaction

## Question

Can the same frozen HBC motion base execute an identical PnP objective differently when the object is in open space or under a table?

## Controlled setup

- Even environment IDs: object on open ground.
- Odd environment IDs: the same object distribution under a low table.
- The target is outside the table, toward the robot side.
- Existing 19-D high-level command interface, DRC reward, motion regularization, and frozen low-level policy are unchanged.
- The actor never observes the scenario ID or table pose.

## Perception

A torso-mounted body-frame ray caster produces a 19 x 10 near-field range image:

- horizontal FOV: `[-100, 100]` degrees;
- 19 vertical channels over `[-55, 35]` degrees;
- maximum range: `3.0 m`;
- observation: `1 - clipped_range / max_range`, so a missed ray is zero and a nearby obstacle approaches one.

The 190-D scan is appended once to the policy observation instead of being repeated in the 10-frame task history. The policy input grows from 1360 to 1550 dimensions.

The no-perception ablation keeps the same 1550-D actor input and replaces only these 190 values with zeros.

## Warm start

The new observation changes actor input `1360 -> 1550` and critic input `146 -> 341`. Use
`scripts/rsl_rl/expand_checkpoint_for_scene_perception.py` to append zero-initialized columns to a trained PnP checkpoint.
The expanded model initially reproduces the old policy exactly, while both scene-aware conditions start from identical
network weights. Optimizer state loading is disabled because its tensors still have the old shapes.

Example:

```bash
python scripts/rsl_rl/expand_checkpoint_for_scene_perception.py \
  logs/5090/g1_dex1/motion/model_5700.pt \
  logs/rsl_rl/g1_dex1_scene_aware_hier_drc/warmstart/model_0.pt
```

Generate the same expanded checkpoint under the no-perception experiment directory, then enable the commented
`--load_run`, `--checkpoint`, and `--resume` entries in `.vscode/launch.json` for a matched warm-start ablation.

## Reward and metrics

The original task reward is preserved. One smooth penalty discourages contacts made by robot bodies other than feet, wrists, and hands:

`tanh(max(force - 5 N, 0) / 40 N)` with weight `-5.0`.

TensorBoard records open/table splits for scan proximity, body collision, active-hand distance, physical grasp, and success.

## Task IDs

- `HBC-Isaac-G1-Dex1-SceneAware-HierDrc-v0`
- `HBC-Isaac-G1-Dex1-SceneAware-HierDrc-Play-v0`
- `HBC-Isaac-G1-Dex1-SceneAware-NoPerception-HierDrc-v0`
- `HBC-Isaac-G1-Dex1-SceneAware-NoPerception-HierDrc-Play-v0`
