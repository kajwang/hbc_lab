# Unified Door Manipulation Progress

## Goal

Keep the door task compatible with the shared contact-label interface used by
PnP, box carrying, and future articulated-object tasks. The high-level policy
must infer the required motion from the live interaction target and the object
goal instead of receiving door-specific articulation state or reward shaping.

## Policy Interface

- Keep the shared task observation: live contact target region, object target
  position, hand-center positions, and effector mask.
- Remove normalized handle and hinge joint angles from both actor and critic
  observations.
- Keep handle and hinge angles internal to the simulator for latch dynamics,
  task success, and diagnostics only.

## Reward

Use the same spatial manipulation progress as the original Go2 PnP task:

`progress = clamp(initial_distance - current_distance, 0) / initial_distance`

where `initial_distance` is from the initial handle position to the fixed goal
position and `current_distance` is from the live handle position to that goal.
The manipulation reward is:

`0.7 * progress + 0.3 * c_couple`

No handle-angle reward or handle-angle gate contributes to policy reward.

## Door Physics

- The handle angle threshold still releases the simulated latch.
- The hinge angle and grasp confidence still define task success.
- Restore the handle actuator stiffness to the original Go2 value of `1.0`.
- Retain articulation metrics in TensorBoard for diagnosis.

## Verification

Static regression tests will verify the shared observation and reward
interfaces. Pure tensor tests will verify normalized spatial progress. An
Isaac Lab smoke test on `jc_5090` will verify environment construction and
finite stepping.
