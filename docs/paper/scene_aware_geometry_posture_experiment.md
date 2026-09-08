# Geometry-Conditioned Posture Shaping Experiment

Date: 2026-09-01

## Hypothesis

A scene representation is useful only if the high-level policy can trade posture
against task progress. A bounded body-clearance potential should let the same
policy crouch below an overhang, turn through a narrow passage, avoid an obstacle,
or stay clear of a door frame without prescribing a task-specific posture.

## Interface

- Actor perception: 325 torso-local rays, horizontal FOV `[-75, 75]` degrees and
  vertical FOV `[-60, 60]` degrees.
- Reward geometry: per-environment oriented obstacle boxes. The current table
  provider writes one tabletop and four legs. Door frames, corridor walls, and
  other obstacles can populate the same tensors later.
- Robot occupancy: spheres attached to pelvis, waist, torso, shoulders, elbows,
  hips, and knees. Hands and feet are excluded because their contacts may be
  task- or locomotion-relevant.
- Clearance risk: a smooth function of signed body-to-obstacle clearance with an
  8 cm safety margin and a 6 cm transition width.

## Reward Scale

The geometry term uses the same detached DRC phase scale as the task reward:

`-0.05 * (2 W_app + 20 W_couple + 200 W_manip) * clearance_risk`

This keeps its relative influence approximately constant across phases. The
coefficient is intentionally capped at 5% of the corresponding task scale.
Task success remains the primary objective; reduce this coefficient if open- or
table-scene task progress is materially worse than the previous run.

## Review Metrics

Compare open and table scenes separately:

1. Task progress: `d_active_hand`, physical grasp, and success ratio.
2. Geometry: minimum clearance, unsafe ratio, and body collision ratio.
3. Posture response: actual and commanded root height and torso pitch.
4. Action health: posture boundary ratio, action retention, and all non-finite
   safety metrics.

Evidence of useful perception requires a systematic posture difference between
open and table scenes together with better table clearance or task progress. A
lower geometry loss alone is not sufficient because the policy could simply stay
away from the task.

## JD Run

- tmux: `scene_geometry_4k`
- task: `HBC-Isaac-G1-Dex1-SceneAware-Geometry-Squashed-HierDrc-v0`
- environments: 4096
- seed: 42
- iterations: 8000
- run name: `geometry_clearance_from_scratch_4k`
- console log: `/home/zju/code/hbc_lab/scene_geometry_4k.log`
