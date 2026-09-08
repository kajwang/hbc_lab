# GraspRef Pose Guidance Controlled Ablation

## Goal

Test whether GraspGen 6-DoF references improve grasp execution over an equal-dimension object-center control, without letting strict orientation acceptance suppress early learning or the object-mass curriculum.

## Design

The shared Dex1 couple reward keeps its existing structure. When pose guidance is enabled, only its `near` term becomes:

```text
position_fine = 1 - tanh(position_error / position_scale)
orientation_near_gate = 1 - tanh(position_error / orientation_gate_scale)
orientation_fine = 1 - clamp(orientation_error, 0, pi) / pi
pose_guidance = 0.65 * position_fine
              + 0.35 * orientation_near_gate * orientation_fine
```

`position_scale=0.15 m` preserves fine position guidance. `orientation_gate_scale=0.50 m` gives orientation a useful gradient before contact. The orientation error remains symmetric to a 180-degree jaw swap. Once a physical grasp is established, the pose-guidance score is smoothly released toward one so an imperfect proposal cannot fight a stable real contact.

The physical-grasp signal is the EMA of two-pad contact multiplied by gripper closure. It drives `c_grasp`, `c_couple`, `W_manip`, success, and the object-mass curriculum. Position and orientation quality remain diagnostics and soft shaping only; they never gate DRC progress. Gripper closure is rewarded when either the contact target is near or a real inner-pad contact already exists, and orientation error never blocks closure.

## Scope

- GraspRef training and play configurations enable 6-DoF guidance.
- The control task uses the same scene, objects, BPS terms, pose observation size, policy architecture, and reward implementation. It replaces the selected reference with object-center position plus identity orientation and sets pose guidance to position-only.
- Ordinary PnP, Cart, Door, Drawer, and Box Carry retain their current reward exactly.
- Existing pad contact and pinch terms remain. Pose gates are removed from physical grasp progress and closure acceptance.

## Experiment

Both experiments use BPS, 4096 environments, seed 42, the same low-level checkpoint, and a 50,000 iteration budget. Both start from randomly initialized high-level policies.

- GraspGen arm: `HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-v0`.
- Equal-dimension control: `HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-v0`.

This isolates the selected contact-pose reference while keeping shape observation and network size fixed.

## Diagnostics

Log position/orientation guidance scores, physical grasp, strict grasp, both close gates, reference-to-command orientation error, command-to-actual orientation error, and actual-to-reference orientation error. Per-shape grasp rate uses physical grasp for a fair comparison between arms.
