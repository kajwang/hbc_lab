# GraspGenX Reference Visualization Design

## Goal

Run GraspGenX once on every object already registered by the G1 Dex1 multishape PnP task, filter the predictions into a compact and diverse grasp library, and provide a local IsaacLab viewer so the candidate poses can be inspected before they are used for policy training.

## Scope

- Use all 27 entries in `ALL_SHAPE_NAMES`, including the three held-out shapes.
- Generate candidates at each asset's nominal authored scale. Runtime uniform size randomization scales the object-relative grasp translation by the same scalar; it does not rerun inference.
- Use the released GraspGenX checkpoint with a custom Dex1 sweep-volume descriptor. The released
  `unitree_g1` descriptor is the seven-joint Dex3 hand and is not geometrically compatible with
  the two-finger Dex1 gripper. Do not train a new grasp model in this iteration.
- Keep BPS as a geometry observation. GraspGenX supplies the selected local grasp reference; it does not replace BPS.
- Do not start HRC training until the filtered candidates have been visually reviewed.

## Candidate Data Contract

The committed runtime artifact is a compressed NPZ file containing fixed-size arrays:

- `shape_names: (S,)`
- `grasp_positions_o: (S, K, 3)`
- `grasp_quaternions_o: (S, K, 4)`, scalar-first quaternion
- `grasp_scores: (S, K)`
- `grasp_valid: (S, K)`
- `grasp_modes: (S, K)`, where 0 is top, 1 is side, and 2 is oblique

All poses are expressed in the object root frame used by the IsaacLab asset. `K=16` is the first-version budget. Objects with fewer than 16 accepted candidates keep invalid padded rows rather than duplicating candidates.

## Generation And Filtering

GraspGenX runs on the complete nominal-scale object mesh and returns object-frame SE(3) poses and scores. Filtering is deterministic:

1. Reject non-finite transforms and invalid rotations.
2. Reject candidates whose hand/gripper proxy intersects the object support plane.
3. Reject candidates outside a conservative object-relative distance bound derived from the mesh bounds and Dex1 hand depth.
4. Classify approach direction as top, side, or oblique.
5. Apply score-ordered SE(3) non-maximum suppression using translation and rotation thresholds.
6. Keep a mode-balanced set when available, then fill remaining slots by score.

The raw model output is retained outside the runtime NPZ so filtering thresholds can be changed without rerunning the network.

## Visualization

Add a dedicated play-only IsaacLab task with one environment per shape. Each environment shows:

- the real object at its normal support pose and scale;
- up to 16 filtered grasp frames;
- color by grasp mode and brightness/size by score;
- only the support platform as additional task visualization.

The viewer is independent of an RL checkpoint. It cycles candidate subsets if rendering all frames becomes cluttered. A launch configuration opens all 27 assets together for direct inspection.

## Runtime Integration

At reset, the environment selects one valid object-local candidate and stores its index. Every control step computes

`T_world_grasp = T_world_object @ T_object_grasp`.

This live world pose is written into `ContactLabel.target_region` and `ContactLabel.target_orientation` for the active hand. It therefore replaces the previous object-center contact target. `object_target_pos_w` remains the transport/place goal and is not replaced.

The actor receives the selected target through the existing task observation path rather than through a second parallel grasp-reference observation. Position uses the existing target-region slots. Orientation is represented as rotation-6D in the same task term, because a 6-DoF grasp cannot be executed from position alone. This intentionally changes the policy input and requires a new checkpoint or an explicit checkpoint-expansion migration.

## Verification

- Pure tensor tests cover NPZ validation, quaternion conversion, SE(3) NMS, mode balancing, and object-to-world pose composition.
- Static configuration tests cover the new task registration and launch entry.
- A 27-environment local smoke test verifies all assets load and every displayed frame is finite.
- The candidate library is accepted only after visual review confirms that the gripper closing plane and approach axis match the real Dex1 convention.
