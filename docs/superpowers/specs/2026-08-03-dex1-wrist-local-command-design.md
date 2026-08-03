# Dex1 Wrist-Local Hand-Center Command Design

## Goal

Replace the Dex1 low-level hand-center pose sampler with a kinematically meaningful pipeline that:

- makes the sampled hand workspace follow commanded root height and torso pitch;
- samples a comfortable hand-base position before applying the gripper-center offset;
- derives orientation from the physical three-joint wrist chain;
- emits the same final hand-center pose format already consumed by the low-level policy and HRC task;
- uses one shared transform contract in low-level training and hierarchical deployment.

## Command Contract

The external command remains a seven-dimensional hand-center pose in the posture-aware shoulder anchor frame:

```text
[hand_center_x, hand_center_y, hand_center_z, qw, qx, qy, qz]
```

Internally, each command is generated in three stages:

1. Sample the `hand_base_link` origin in shoulder-anchored spherical coordinates.
2. Sample the physical wrist joint tuple `(roll, pitch, yaw)` and evaluate the wrist-chain orientation, including the fixed wrist-yaw-to-hand-base rotation.
3. Transform `HAND_CENTER_OFFSET` by the sampled hand-base orientation to obtain the final hand-center position and orientation.

The policy therefore continues tracking the contact-relevant hand center, while the sampler operates on a physically interpretable wrist/flange representation.

## Position Sampling

The anchor pose follows commanded posture in both training and HRC:

```text
anchor_orientation = root_yaw * torso_pitch_command
anchor_height = environment_origin_z + root_height_command
```

The shoulder offset is transformed by this anchor. This is the same convention currently used by the HRC low-level observation builder and will become the single shared implementation.

Because the sampled point moves from hand center to hand base, the spherical radius ranges shrink by 0.10 m:

```text
initial wrist radius: [0.20, 0.38] m
limit wrist radius:   [0.12, 0.58] m
```

After wrist orientation is sampled, the final command position is:

```text
p_hand_center = p_hand_base + R_hand_base * HAND_CENTER_OFFSET
```

## Orientation Sampling

The sampled variables represent the physical wrist chain rather than Euler rotations in the final hand-base frame:

```text
wrist roll limit:  +/-100 degrees
wrist pitch limit: +/-80 degrees
wrist yaw limit:   +/-80 degrees
```

The initial curriculum ranges remain comfortable and expand toward these limits. Wrist orientation is composed in joint-chain order and then multiplied by the fixed palm transform from the Dex1 asset. The resulting hand-base orientation is also the hand-center orientation because `HAND_CENTER_OFFSET` has no rotational component.

The implementation will use explicit wrist-joint naming internally. Existing configuration field names may be retained only where required for compatibility, with their semantics documented and tested.

## Shared Transform

Low-level command generation and HRC observation construction must call the same pure transform helpers for:

- posture-aware shoulder anchor pose;
- wrist-chain orientation composition;
- hand-base-to-hand-center conversion.

Debug visualization must display the final hand-center command produced by these helpers.

## Compatibility

This changes both the command distribution and the observation semantics of the frozen low-level policy. Existing exported low-level checkpoints are not expected to remain valid for the new task. HRC checkpoints using the old low-level policy are also not directly comparable after swapping in the retrained policy.

The low-level observation tensor shape should remain unchanged unless implementation evidence requires otherwise. Keeping the shape stable does not imply behavioral checkpoint compatibility.

## Verification

Static and pure-tensor tests will cover:

- posture anchor translation and rotation under root-height and torso-pitch commands;
- zero wrist angles reproducing the nominal hand-base orientation;
- each wrist axis rotating around the intended physical joint axis;
- hand-center position changing by the rotated fixed offset;
- configured angular limits equal to +/-100, +/-80, and +/-80 degrees;
- low-level training and HRC using the shared transform implementation;
- final command shape remaining seven-dimensional per hand.

A simulator smoke test will then verify that command visualization follows posture and that isolated right-wrist pitch/roll/yaw commands produce the expected hand motion before launching full training.
