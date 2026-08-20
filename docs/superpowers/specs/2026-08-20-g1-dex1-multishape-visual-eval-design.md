# G1 Dex1 Multi-Shape Visual Evaluation Design

## Goal

Provide a local visual evaluation for comparing the BPS and no-BPS PnP policies on every training and OOD object. The evaluation must make per-shape grasp behavior easy to inspect without creating a duplicate task implementation.

## Architecture

Reuse `HBC-Isaac-G1-Dex1-HierDrc-MultiShape-Play-v0` and its deterministic `MultiAssetSpawnerCfg`. Add two independent VS Code launch entries because the BPS and no-BPS actors have different observation dimensions and cannot load into the same runner:

- `g1_dex1_multishape_bps_visual_play`
- `g1_dex1_multishape_no_bps_visual_play`

Both entries launch 11 environments with identical scene configuration, random seed, low-level policy, and matched high-level checkpoint iteration. They differ only in the shape-observation switch and high-level checkpoint.

## Object Assignment

Disable random asset selection for visual evaluation. Assign one asset to every environment in this fixed order:

| Environment | Shape | Split |
| --- | --- | --- |
| 0 | egg | train |
| 1 | milk | train |
| 2 | ketchup | train |
| 3 | butter | train |
| 4 | hotdog | train |
| 5 | shaker | train |
| 6 | cheese | train |
| 7 | cucumber | train |
| 8 | donut | OOD |
| 9 | bread_bag | OOD |
| 10 | toy_gun | OOD |

Print this mapping once at startup. In-world text labels are intentionally omitted because they add custom visualization code without improving policy evaluation.

## Evaluation Controls

- Fix the right hand as the active hand with `left_hand_probability=0.0` so hand choice does not confound the shape comparison.
- Keep object orientation randomization enabled. This is necessary to test whether BPS helps select a grasp orientation rather than memorize an asset-specific pose.
- Use the same seed for both launch entries so their resets are directly comparable.
- Use 11 environments and 6 m environment spacing for a readable local overview without scene overlap.
- Keep command, hand-center, and object-frame debug visualization enabled.
- Use the existing local low-level HBC checkpoint for both policies.

## Checkpoint Contract

Copy matched-iteration BPS and no-BPS checkpoints from the two remote training runs into stable local paths:

```text
logs/5090/g1_dex1/multishape/bps/model_10500.pt
logs/5090/g1_dex1/multishape/no_bps/model_10500.pt
```

The launch entries must reference these stable paths rather than timestamped remote run directories. The BPS checkpoint must be loaded only with BPS observations enabled; the no-BPS checkpoint must be loaded only with BPS observations disabled.

## Verification

1. Check that both launch configurations resolve to the same play task and 11 environments.
2. Verify the fixed environment-to-shape assignment and train/OOD labels.
3. Verify actor observation dimensions against each matched checkpoint before launching simulation.
4. Run each configuration locally for a short visual smoke test and confirm all 11 assets render, reset, and receive the intended active-hand command.
5. Compare grasp approach, gripper orientation, stable contact, lift, and transport behavior per shape. The visual evaluation does not alter rewards or report success metrics; quantitative per-shape evaluation remains a separate experiment.

## Non-Goals

- Running BPS and no-BPS policies in one Isaac Sim process.
- Adding in-world text labels.
- Changing training randomization, reward shaping, BPS encoding, or policy architecture.
- Treating visual inspection as a replacement for per-shape success-rate evaluation.
