# Contact-Conditioned Motion Regularization Design

## Goal

Improve the execution quality of all G1 Dex1 hierarchical interaction tasks without adding a residual policy or relying on human motion demonstrations. The shared high-level policy must keep its task authority, while its commands and the resulting robot motion become bounded, smooth, purposeful, and safer for eventual sim-to-real deployment.

The first implementation targets PnP and is structured as shared infrastructure for Box Carry, Cart Push, Door Open, and Drawer Open-Close. It must not use task names, discrete phases, DRC weights, privileged contact force, or object dynamics to decide when motion constraints apply.

## Architecture Boundary

The architecture remains two-level:

1. A trainable high-level PPO policy runs at 10 Hz and emits locomotion, posture, bilateral hand-center, and gripper commands.
2. The existing pretrained HBC policy remains frozen and tracks those commands at 50 Hz.

No command adapter, residual action head, or second PPO policy is introduced. Future force or compliance compensation is explicitly outside this design and may later use a faster low-level residual with a distinct interface.

## High-Level Action Interface

Keep a fixed 19-dimensional action so single-hand and bimanual tasks share one policy contract:

| Indices | Meaning |
| --- | --- |
| 0:3 | Base linear X/Y and yaw velocity command |
| 3:5 | Root-height and torso-pitch command velocity |
| 5:11 | Left hand-center linear and angular command velocity |
| 11 | Left gripper command |
| 12:18 | Right hand-center linear and angular command velocity |
| 18 | Right gripper command |

Rename wrist command state fields to hand-center command fields throughout the current Dex1 hierarchy. Their frame is the posture-conditioned shoulder-anchor frame used by the frozen HBC interface.

Interpret all incremental outputs as physical rates and integrate them with the actual high-level time step:

```text
command[t+1] = command[t] + command_rate[t] * high_level_dt
```

Initial physical limits are:

- Hand-center linear speed: 0.30 m/s.
- Hand-center angular speed: 1.00 rad/s.
- Root-height speed: 0.12 m/s.
- Torso-pitch speed: 0.40 rad/s.
- Base linear acceleration: 1.50 m/s^2.
- Base yaw acceleration: 1.50 rad/s^2.

Physical-rate control does not itself prevent command drift. It makes speed limits frequency-independent and supplies meaningful quantities for the motion penalties below. Constant nonzero command velocity is penalized directly; command acceleration is also penalized. The previous raw-action-difference penalty is insufficient because a constant incremental action can move a command indefinitely at zero smoothness cost.

Do not add an XYZ hard workspace clamp. Retain only finite-value sanitation. Use soft feasibility and tracking penalties so the high-level policy learns to reposition the body instead of relying on clipping.

## Deployable High-Level Observation

The actor keeps the existing ten-frame history and receives only deployable quantities. Add a shared execution-state block used by every task:

- Base linear velocity estimate in the root frame.
- Upper-body joint positions for the waist and both arms; history provides their motion trend.
- Current left and right hand-center position and 6D orientation in the root frame.
- Left and right hand-center position and orientation tracking errors.
- Root-height and torso-pitch tracking errors.
- Current decoded command rates.

These quantities are available from the state estimator, joint encoders, robot kinematics, and the policy's own commands. Contact forces, DRC weights, object mass, joint efforts, and simulator-only state remain critic-only or reward-only.

Upper-body joint state is required because hand-center pose alone does not identify elbow and waist configuration. Without it, the actor cannot distinguish a comfortable arm pose from a singular or distorted pose that reaches the same endpoint.

## Continuous Reachability Gates

For hand `i`, let `m_i` be its binary `effector_mask`. Define two distances from the live `ContactLabel.target_region`:

```text
d_reach_i = norm(target_region_i - shoulder_anchor_i)
d_hand_i  = norm(target_region_i - hand_center_i)
```

The release gate is based on shoulder-to-target distance, not hand-to-target distance:

```text
g_reach_i = sigmoid((r_release - d_reach_i) / release_width)
```

Using hand distance for release would create a deadlock: the hand would need to approach before its comfort constraint is relaxed. Shoulder distance instead indicates when locomotion and posture have brought the target into the HBC workspace.

Derive `r_release` from the reliable outer radius of the low-level stick-figure command distribution and begin releasing 5-10 cm before the boundary. Use a smooth transition width rather than a discrete threshold. `d_hand_i` remains available for approach/contact progress and diagnostics, but does not control the initial release from the walking posture.

The per-hand comfort weight is:

```text
w_comfort_i = (1 - m_i) + m_i * (1 - g_reach_i)
```

Consequently:

- An inactive hand is constrained for the full episode.
- An active hand stays near a comfortable pose while its target is outside reach.
- Its comfort constraint weakens continuously as the target enters the reachable workspace.
- Bimanual tasks apply the same computation independently to both hands.

No hand is hard-locked. The effector mask changes reward weights, not action dimensionality or controller topology.

## Shared Motion Regularization

Implement the following shared loss definitions independently of task phase and DRC weights. Their gates depend only on `effector_mask`, live target geometry, robot state, and command state.

### Always-On Safety And Execution

- Hand-center position tracking error with a 4-5 cm dead zone.
- Hand-center orientation tracking error with an approximately 0.20 rad dead zone.
- Root-height and torso-pitch tracking error.
- Root roll and excessive pelvis tilt.
- Upper-body joint-limit proximity.
- Command acceleration and command jerk.
- A low nonzero command-speed and arm-motion cost.
- Soft violation of the HBC command distribution's reliable workspace.

These constraints remain active near contact. Interaction freedom must not disable tracking, joint safety, or smoothness.

### Distance-Conditioned Arm Prior

For each hand, weight the following by `w_comfort_i`:

- Hand-center command deviation from the posture-conditioned comfortable pose.
- Actual arm joint deviation from its comfortable configuration.
- Hand-center command speed.
- Actual arm joint speed.

Use a nonzero floor on active-hand speed and acceleration costs after release. This prevents motion inside the valid workspace from becoming free. A movement must improve task return enough to justify its motion cost.

The first version does not enforce a straight-line hand path. A directional path prior would conflict with obstacle avoidance and valid curved approaches. Minimum motion, command acceleration, and jerk provide a weaker analytic prior without prescribing a task-specific trajectory.

### Distance-Conditioned Upper-Body Prior

Aggregate active-effector reachability as:

```text
g_body = max_i(m_i * g_reach_i)
w_walk = lambda_near + (1 - lambda_near) * (1 - g_body)
```

Start with `lambda_near = 0.20`. Weight these walking-posture terms by `w_walk`:

- Root-height command deviation from the nominal walking height.
- Torso-pitch command deviation from upright.
- Waist roll/yaw deviation from a comfortable pose.
- Torso roll/yaw relative to the root.

Keep root-tilt safety and commanded-posture tracking always active. Far from the target, the policy must primarily navigate while keeping its arms and upper body composed. Near a reachable target, it may squat, bend, or rotate the waist, but gross leaning and tracking failure remain costly.

## Reward Scaling

Task rewards remain responsible for approach, contact, grasp/support, manipulation progress, and success. Motion regularization must not be hidden inside task-specific DRC functions.

Normalize each motion loss by a physical tolerance before weighting it, for example:

```text
position_loss = square(relu(position_error - 0.05) / 0.10)
orientation_loss = square(relu(orientation_error - 0.20) / 0.50)
```

Use the DRC weights only for numerical scale normalization:

```text
stage_scale = W_app * approach_scale + W_couple * couple_scale + W_manip * manip_scale
quality_loss = clamp(sum_k(lambda_k * normalized_loss_k), 0, 2)
R_quality = -0.10 * stop_gradient(stage_scale) * quality_loss
```

This gives an initial regularization budget near 10% of the active stage scale and caps it near 20%. DRC does not turn individual constraints on or off and does not affect `g_reach`, `w_comfort`, or `w_walk`. The same normalized losses and coefficients apply to every task.

Track each raw loss and weighted contribution separately. Do not tune from only their sum.

## Training Recipe

Validate on PnP first because it contains locomotion, single active-hand approach, grasp, and transport while providing an always-inactive opposite hand.

1. Freeze the current stick-figure hand-center HBC checkpoint.
2. Train the PnP high-level policy from scratch with the revised 19-dimensional physical-rate interface and shared observations/rewards.
3. Use 4096 environments and the existing PPO backbone initially; do not attribute behavior changes to a larger model in the first experiment.
4. Train for 50,000 iterations on JD in a named tmux session.
5. Compare against the latest successful PnP baseline using task and execution-quality metrics.

After PnP validates the design, apply the same shared implementation to Box Carry or Cart Push without changing coefficients. This second bimanual task checks that the priors do not depend on an inactive hand.

## Evaluation Criteria

The first experiment succeeds only if task behavior and motion quality both improve:

- PnP grasp and transport success remains within 5% of the baseline or improves.
- Active hand-center position and orientation errors decrease, especially their 95th percentiles.
- Inactive hand command deviation, hand speed, and arm joint speed decrease substantially.
- Root roll, pelvis tilt, torso roll/yaw, and waist distortion decrease during the far-field approach.
- Command linear/angular speed, acceleration, jerk, and action saturation decrease.
- The active hand and torso still gain enough freedom near the object to complete grasp and transport.
- Workspace violations and nonfinite safety metrics do not increase.

This design aims for restrained, purposeful, contact-conditioned motion. It does not claim human motion imitation; reproducing human arm swing or stylistic motion still requires a demonstrated or learned motion prior.
