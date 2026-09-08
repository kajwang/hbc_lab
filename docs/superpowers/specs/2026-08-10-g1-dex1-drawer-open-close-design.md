# G1 Dex1 Drawer Open-Close Design

## Goal

Add a hierarchical G1 Dex1 task that grasps one randomly selected drawer handle with the right hand, opens the drawer by 0.30 m, and then closes it fully. The task must use the same contact-label and sparse pose-keyframe interface as the latest door task so the actor does not receive drawer-specific joint state or reward terms.

## Asset And Reset

- Load NVIDIA's two-drawer Sektion cabinet asset used by Robot Lab and Isaac Lab.
- Spawn the cabinet in front of the robot and reset both drawer joints to zero.
- Select the top or bottom drawer independently for every environment with equal probability.
- Keep the right hand active with `effector_mask=(0, 1)` and `contact_mode=GRASP`.
- Track both handle frames internally, but expose only the selected handle through the contact label and task observation.

## Contact Label

`target_region` and `target_orientation` follow the selected handle's live world pose. This preserves the existing approach/couple DRC behavior while ensuring contact with the non-selected handle cannot satisfy the selected target's distance gate.

The contact sensors filter both drawer handles because Isaac Lab contact filters are configured statically. Per-environment target selection is enforced by the selected handle pose and the active right-hand distance gate.

## Sparse Motion

Each environment has two pose keyframes:

1. Move the selected handle 0.30 m along the drawer's outward prismatic direction.
2. Return the selected handle to its closed reset pose.

The outward direction is the cabinet root frame's local positive X axis, which is also the axis used by the stock handle offset. It is rotated into world space with the live cabinet root orientation. A two-environment runtime check confirmed that a positive 0.30 m drawer-joint displacement moves both handles by 0.30 m along this direction.

Position error is active in all Cartesian axes. Rotation error is masked out because the drawer translates without rotating. Keyframes advance only after the selected handle remains within tolerance for several consecutive high-level steps.

## Observation And Reward

The policy receives the same generic motion observation as the latest door task: current object pose, current keyframe target pose, hand-center positions, effector mask, pose masks, phase, validity, command state, previous action, and ten-frame actor history.

The DRC reward keeps the shared approach and grasp/couple terms. Manipulation uses only decrease in masked pose error plus a keyframe completion bonus. It does not expose or reward drawer joint position, opening distance, handle alignment, or a task-specific stage scalar.

Success requires the second keyframe to be stably completed while the right-hand grasp confidence remains above the common threshold.

## Validation And Deployment Gate

- Static tests verify registration, asset setup, random top/bottom selection, right-hand contact label, two-keyframe order, generic observation, and absence of drawer-joint reward shaping.
- A one-environment headless smoke run verifies asset loading, joint/body/frame names, outward-axis sign, and finite observations/rewards.
- `.vscode/launch.json` provides a 16-environment non-headless training launch with debug markers enabled.
- Code may be synchronized to JD after validation, but the 4096-environment tmux training run must not start until the user reviews the implementation and visually checks the local 16-environment run.
