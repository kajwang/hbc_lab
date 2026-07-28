# G1 Dex1 Door Opening and Cart Pushing Design

## Goal

Add two G1 Dex1 hierarchical interaction tasks to `hbc_lab`:

- Open a latched articulated door with the right hand.
- Push an articulated cart to a sampled goal while grasping its handle with both
  hands.

Both tasks reuse the existing 19-dimensional high-level command policy and the
frozen G1 Dex1 whole-body hand-center tracking policy. They must not alter the
existing PnP or box-carry behavior.

## Architecture

Create two task-local subclasses of `G1Dex1HierDrcEnv`:

- `G1Dex1DoorOpenEnv`
- `G1Dex1CartPushEnv`

Each task owns its scene, reset logic, contact progress, rewards, observations,
and success checks. Shared low-level policy loading, high-level action decoding,
command history, whole-body control, gripper control, and common diagnostics
remain in the existing G1 Dex1 hierarchy.

Do not introduce a broad articulated-interaction base class in this version.
The door and cart have different state and contact semantics, and the next
articulated task should first validate which pieces are genuinely reusable.

## Local Assets

Copy only these self-contained USD files from `robot_lab`:

- `door_0_bot.usd`
- `instance_turn_R_10kg.usd`

Store them under `hbc_lab/assets/models/articulated/`. Both USD files have no
external USD dependencies, so the source asset directories and textures are not
needed.

Define their IsaacLab configurations in
`hbc_lab/assets/articulated_objects.py`.

### Door

The door articulation contains:

- `joint_1`: door hinge, range 0 to 86.4 degrees.
- `joint_2`: handle hinge, range 0 to 86.4 degrees.
- `link_2`: handle link.

Retain the source actuator settings and the simulated latch:

- The hinge is locked by a spring-damper effort while the handle angle is below
  0.5 rad.
- The latch also releases if the hinge has already moved beyond 0.2 rad, which
  prevents the artificial lock from pulling an already opened door closed.
- Latch stiffness is 5000 and damping is 10.

Expose frame-transform targets for the live handle center and the handle goal.

### Cart

The cart is a 10 kg articulated asset with four passive wheel joints and two
passive steering joints. Retain the source passive actuator parameters and the
asset scale of 0.8.

Expose one handle frame. The task derives two contact targets by applying
opposite local offsets of 0.18 m along the handle's lateral axis.

The first version keeps the cart at its nominal total mass. It has no mass
curriculum. Later domain randomization must scale the masses and inertias of all
cart links together rather than changing only the articulation root.

## Shared Interface

The high-level action remains 19-dimensional:

- 3 base velocity commands.
- 2 posture commands.
- 6 left hand-center pose increments plus one left grip command.
- 6 right hand-center pose increments plus one right grip command.

The actor retains the current 10-step flattened observation history and current
high-level command state. Simulator-only contact forces stay out of the actor.

The common `ContactLabel` remains unchanged:

- `effector_mask`
- `target_region`
- `contact_mode`

The contact mode selects task reward semantics inside the environment and is not
added as an actor observation.

## Door Task

### Contact Label

- `effector_mask = (0, 1)`
- `contact_mode = GRASP`
- Right target region: current door-handle center.
- Left target region: zeroed by the effector mask.

The left hand remains available to the whole-body controller for balance, but
left-hand contact with the door is penalized as inactive-effector contact.

### Actor Observation

Reuse the common goal, hand-center, target-region, effector-mask, grip, command
state, last action, angular velocity, and gravity observations.

Append two normalized articulated-state values:

- Handle angle divided by the 0.5 rad latch threshold.
- Door hinge angle divided by the 60-degree success target.

These values represent perceptually estimable object state, not simulator-only
contact state. A real deployment must obtain them from door and handle pose
estimation.

### Contact and DRC

Only the right Dex1 inner pad links `right_hand_Link1_3` and
`right_hand_Link2_3` contribute to grasp progress. Their contact sensors filter
contacts against `door/link_2`.

Compute the right-hand grasp confidence using the existing Dex1 two-pad contact,
grip, distance gate, and EMA implementation. Then compute:

```text
W_app    = (1 - c_couple) * (1 - exp(-5 d_hand))
W_couple = (1 - c_couple) * exp(-5 d_hand)
W_manip  = c_couple
```

### Manipulation Reward

Let:

```text
p_handle = clamp(abs(handle_angle) / 0.5, 0, 1)
g_handle = smooth gate derived from p_handle
p_hinge  = clamp(hinge_angle / radians(60), 0, 1)
```

Use:

```text
R_manip = 0.3 + 0.25 * p_handle + 0.75 * g_handle * p_hinge
```

The constant keeps a stable grasp valuable at the manipulation boundary. The
hinge term remains closed until the handle has been turned far enough to release
the latch.

Approach and coupling rewards follow the established G1 Dex1 PnP recipe for the
active right hand.

### Reset and Success

Reset the door 2.0 m in front of the environment origin with small lateral
variation, fixed orientation, and both articulation joints at zero. Reset the
robot near its standard origin with moderate XY/yaw variation.

Success requires:

- Hinge angle at least 60 degrees.
- Right-hand grasp confidence above the configured threshold.
- Both conditions sustained for 25 high-level steps.

The first version does not randomize door scale or dynamics.

## Cart Task

### Contact Label

- `effector_mask = (1, 1)`
- `contact_mode = GRASP`
- Left and right target regions: live handle-frame positions at opposite local
  lateral offsets of 0.18 m.

### Actor Observation

Reuse the common observation without privileged cart joint state. The two live
target points encode handle position and yaw, while the 10-step history exposes
cart motion. The object goal is the desired handle-midpoint position.

### Contact and DRC

Each hand independently computes Dex1 two-pad grasp confidence against the cart
handle. Let these values be `c_left` and `c_right`.

Use:

```text
c_couple = sqrt(clamp(c_left * c_right, 0, 1))
d_hand   = 0.5 * (d_left + d_right)
```

The geometric mean requires both hands while retaining a smoother signal than a
hard minimum. DRC weights use the same formula as the door task.

### Manipulation Reward

Measure planar progress of the handle midpoint toward its sampled goal:

```text
transport_progress =
    clamp(initial_goal_distance - current_goal_distance, 0)
    / (initial_goal_distance + epsilon)

R_manip = 0.3 + transport_progress
```

Because this term is multiplied by `W_manip = c_couple`, losing either grasp
turns off the transportation reward.

### Reset and Success

Reset the cart 1.5 to 2.5 m in front of the robot with small lateral variation
and a fixed nominal orientation. Sample a goal in the cart's initial local
frame:

- Forward displacement: 2 to 4 m.
- Lateral displacement: -0.5 to 0.5 m.

Success requires:

- Planar handle-midpoint distance to goal below 0.25 m.
- Bimanual grasp confidence above the configured threshold.
- Both conditions sustained for 25 high-level steps.

The first version uses the nominal 10 kg cart without a mass curriculum.

## Environment Configuration

Both tasks:

- Use flat terrain.
- Disable terrain curriculum.
- Use the existing high-level and low-level decimation.
- Require an explicit frozen low-level policy path for training and play.
- Provide Train and Play Gym registrations.
- Provide RSL-RL PPO runner configurations using the current G1 Dex1 hierarchy
  as the baseline.
- Provide VS Code launch entries for train and play.
- Visualize the live contact targets, task goal, and articulated frames.

## Diagnostics

Door logs include:

- Right-hand distance, contact, grasp, and DRC weights.
- Handle angle and normalized handle progress.
- Hinge angle and normalized opening progress.
- Latch-release ratio.
- Inactive left-hand door contact.
- Goal success count.

Cart logs include:

- Per-hand distance, pad contact, grip, and grasp confidence.
- Bimanual geometric-mean coupling confidence.
- Handle-midpoint goal distance and transportation progress.
- Cart root linear speed.
- Goal success count.

## Validation

Add focused pure-PyTorch tests for:

- Door handle and hinge progress gates.
- Bimanual geometric-mean coupling.
- Cart local target and local-goal transforms.
- Task asset paths and configured joint names.
- Gym registration and launch configuration strings where they can be checked
  without booting Isaac Sim.

Run a small IsaacLab smoke test for each task:

- Instantiate a few environments with the low-level policy disabled only for
  initialization testing.
- Confirm the expected articulation joints and frames resolve.
- Confirm actor and critic observations are finite.
- Confirm the high-level action dimension is 19.
- Step once with zero actions.
