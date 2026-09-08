# Grasp Candidate Play Design

## Goal

Provide one diagnostic Play command that selects a single multishape asset by ID and evaluates every valid GraspGen candidate in a separate Isaac Lab environment. The same command must preserve the observation semantics of both the GraspRef and object-center control policies.

## Interface

```bash
python scripts/rsl_rl/play_grasp_candidates.py \
  --mode graspref \
  --asset_id 8 \
  --checkpoint=path/to/model.pt
```

`asset_id` indexes `GRASP_REF_ALL_SHAPE_NAMES`. The script chooses the matching Play task, enables BPS, fixes the active hand to the right hand, and forwards all ordinary Play arguments to `scripts/rsl_rl/play.py`.

## Environment Layout

The number of environments equals the number of valid library candidates for the selected asset. Every environment loads the same asset at scale 1.0 and the same object-relative initial pose. Environment index `i` permanently selects valid candidate `i`, including after resets. The console prints asset name and the mapping from environment index to candidate ID, score, and approach mode.

## Mode Semantics

- `graspref`: the selected candidate position and orientation are supplied to the actor and used by the existing pose reward/gates.
- `object_center`: the actor continues to receive object center plus identity orientation. The selected GraspGen candidate is loaded only for the reference marker and never changes reward, contact gates, or observations.

Both modes render the selected reference frame and actual active hand-center frame. Generic command markers remain disabled and platform visualization remains enabled.

## Implementation

A thin wrapper translates `mode` and `asset_id` into the existing Play script and Hydra overrides. The Play runner calls a post-override configuration hook before constructing the environment. The multishape config hook validates the asset, loads valid candidate metadata, replaces the scene object spawner with the selected asset, and sizes the scene. The environment uses fixed candidate assignment and computes marker poses directly from the diagnostic candidate library, independently of whether candidates control the policy target.

## Validation

Pure tests cover candidate metadata lookup, deterministic environment-to-candidate assignment, wrapper argument forwarding, invalid IDs, and target/control isolation. Static integration tests cover the Play hook and launch entries. A one-step local Isaac launch validates both modes against matching checkpoint dimensions.
