# Box Carry Opposing-Face Target Design

## Goal

Replace center-only box approach guidance with deterministic, object-relative targets on two opposing faces. This experiment tests whether explicit geometric contact regions remove the current top-press/side-contact local optimum without adding force-direction or wrist-orientation shaping.

## Target Geometry

The box dimensions are `0.30 x 0.25 x 0.20 m`. The target faces are fixed to the ends of the longest local axis, so their local offsets are:

```text
positive target: (+0.15, 0.0, 0.0)
negative target: (-0.15, 0.0, 0.0)
```

The world-frame target positions are recomputed from the current object pose every step. They therefore translate and rotate with the box.

## Hand Assignment

At reset, compute the total hand-to-target distance for both possible assignments. Assign the lower-cost pairing to the left and right hands and store that assignment for the entire episode. The assignment must not switch after reset, even if the box rotates, to prevent discontinuous goals.

## Policy Inputs

Append the assigned left and right face-target positions to the task observation in the robot root frame. The existing object center, object goal, hand-center positions, and effector mask remain available. Both actor and critic receive the face targets because they can be obtained from the estimated object pose at deployment.

This adds six scalar inputs to the high-level observation. Existing BoxCarry high-level checkpoints are therefore incompatible. The frozen low-level checkpoint and its observation interface are unchanged.

## Progress And Rewards

Replace each hand's distance to the object center with its distance to the assigned face target:

```text
d_left = ||left_hand_center - left_face_target||
d_right = ||right_hand_center - right_face_target||
d_active = max(d_left, d_right)
```

These distances drive the existing approach reward and DRC weights. Existing semantic support contacts and `bimanual_support_contact` remain unchanged. This version adds no force-opposition, contact-normal, wrist-orientation, lift, or new gripper-close reward.

## Visualization And Diagnostics

Show both assigned face targets during training and Play. Log the left and right face-target errors so the geometric behavior can be separated from contact quality.

## Verification

Add focused tests for:

- longest-axis face-center calculation under object translation and yaw;
- minimum-distance hand assignment;
- assignment stability while the object subsequently rotates;
- observation shape and root-frame target ordering;
- progress distances using face targets rather than the object center.

Run the focused static/unit tests and an Isaac Lab smoke reset/step when the simulator is available.
