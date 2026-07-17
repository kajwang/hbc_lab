# HBC Lab Idea Plan

This document records the original research idea behind `hbc_lab`, so future
implementation steps stay aligned with the paper direction instead of drifting
into a narrow locomotion benchmark.

## Core Motivation

The project is inspired by SUGAR and GRAIL. Both start from the observation that
humanoid whole-body interaction data is expensive when collected through
teleoperation or detailed human-object interaction reconstruction. Their shared
direction is to use scalable reinforcement learning in simulation to expand the
interaction data distribution.

The central question for this project is:

What makes humanoid whole-body interaction harder than fixed-base tabletop robot
manipulation?

The working answer is:

1. The robot must find an executable operating position, so navigation planning
   and approach pose selection are part of the manipulation problem.
2. During each manipulation phase, the robot must maintain whole-body locomotion
   stability, terrain traversal, collision avoidance, and contact reachability.
3. Viewpoint selection matters because self-occlusion, object occlusion, and
   task-relevant visibility affect execution.
4. Complex contact requires force compensation and residual stabilization beyond
   pure kinematic tracking.

## Data Abstraction

The project intentionally avoids high-cost HOI reconstruction and human-to-robot
retargeting as the primary supervision source.

From human-object interaction videos, the pipeline should extract sparse,
embodiment-agnostic task guidance:

1. Contact labels from VLM parsing:
   - Object category and basic attributes.
   - Task phase segmentation.
   - Which body parts contact the object in each phase.
   - Which contacts are optional, persistent, or transient.
2. Sparse object pose guidance from a video foundation model pipeline:
   - Initial and final object pose.
   - Optional keyframe object poses.
   - Optional initial pose plus motion trend.
   - Later extension to articulated object state, such as door angle, drawer
     joint displacement, cart handle trajectory, or object support state.

The VLM should act as a task-level brain. It should not reason about the concrete
robot morphology, joint motion, or exact hand-object contact execution. Those
details belong to the embodiment-specific low-level control and RL policy.

## Simulation and RL Pipeline

Given sparse video-derived guidance, simulation generates scalable interaction
experience:

1. Build a task in IsaacLab from sparse object pose and phase/contact labels.
2. Train policies with scale RL to reproduce object interaction in simulation.
3. Use curriculum to expand the distribution:
   - Object pose randomization.
   - Object physical property randomization.
   - Object shape and geometry randomization.
   - Terrain complexity.
   - Contact difficulty.
   - Partial observability and viewpoint variation.
4. Use the learned task-general policy as the bridge between high-level
   language/video intent and low-level embodiment control.

## Policy Architecture

The target policy structure follows the spirit of SUGAR:

1. High-level low-frequency command policy:
   - Coarse locomotion command.
   - End-effector or contact body target command.
   - Object pose or articulated state subgoal.
   - Phase/contact mode command.
2. Low-level whole-body controller policy:
   - Tracks base velocity, body pose, end-effector pose, and contact mode
     commands.
   - Preserves locomotion stability.
   - Avoids collisions.
   - Handles terrain traversal.
   - Produces joint position targets or residuals compatible with WBC-style
     control.

The first implementation milestone is a stable G1 locomotion baseline. The next
milestone is to extend this from pure base velocity tracking into whole-body
command tracking without breaking the official Unitree velocity recipe.

## Observation and Representation Choices

The planned observation stack should grow in stages:

1. Base locomotion state:
   - Base angular velocity.
   - Projected gravity.
   - Base velocity command.
   - Joint positions and velocities.
   - Last actions.
2. Terrain representation:
   - Height map from ray-casting.
   - Later option: a Gallant-style terrain representation if it improves rough
     traversal and obstacle crossing.
3. Whole-body command observations:
   - End-effector pose commands.
   - Contact body commands.
   - Phase/contact labels.
4. Object representation:
   - Object pose subgoal.
   - Sparse object trajectory keyframes.
   - Object shape encoding following the role played by GRAIL-style shape
     conditioning.
   - Articulated object state when needed.
5. Residual force compensation:
   - A residual module can compensate interaction forces or contact errors.
   - The residual should stabilize WBC rather than replace the base locomotion
     and tracking policy.

Current high-level observation decision:

- The deployable actor observation uses a 10-step flattened history, covering
  about one second at the current 10 Hz high-level control rate.
- The actor also observes the accumulated high-level command state directly,
  including base, posture, hand-center position, and hand orientation commands,
  so incremental commands do not create hidden state across long episodes.
- The critic keeps instantaneous privileged DRC observations; simulator-only
  contact quantities must not be added to the actor without a real-robot
  sensing counterpart.
- Follow-up if manipulation remains unstable: diagnose learned PPO standard
  deviation and action saturation per action group and separately for active
  and inactive hands. The current Gaussian policy has one global learned
  standard deviation per action dimension, so inactive-hand and weakly
  supervised orientation actions can inflate aggregate noise statistics.
- For bimanual ground-box transport, coupling uses horizontal opposition,
  horizontal radial balance, and hand-height agreement without assigning fixed
  object faces. Manipulation first rewards lift, then smoothly opens horizontal
  transport progress as the object reaches the target lift height.

## Generalization Goals

The method should train across common whole-body interaction tasks through a
unified sparse object pose interface. This is intended to improve over a narrow
GRAIL-style object interaction setting by supporting:

1. Rigid object manipulation.
2. Articulated objects:
   - Door opening.
   - Drawer pushing and pulling.
   - Cart pushing.
3. More complex object interaction:
   - Objects under a table versus on open ground.
   - Adaptive contact posture selection for the same object in different
     spatial contexts.
4. Optional foot interaction:
   - Foot-assisted manipulation.
   - Soccer-style kicking.
   - Carrying or guiding objects while locomoting.
5. Heterogeneous embodiment:
   - Humanoid.
   - Quadruped plus arm.
   - Potentially other mobile manipulators with the same sparse high-level
     guidance abstraction.

## Demo Targets

The final demo direction should include:

1. Same box, different context:
   - One box under a table.
   - One box directly on the ground.
   - The policy adapts whole-body contact posture and approach behavior.
2. Articulated object interaction:
   - Open door.
   - Push or pull drawer.
   - Push cart.
3. Optional foot interaction:
   - Include feet as possible contact effectors.
   - Kick a soccer ball or perform simple foot-assisted manipulation.

## Current Engineering Roadmap

1. Create an independent `hbc_lab` repository.
2. Migrate the official Unitree G1 29DoF velocity locomotion recipe locally.
3. Train and validate stable G1 locomotion with the official curriculum.
4. Extend G1 locomotion into whole-body command tracking:
   - Keep official velocity task intact.
   - Add a new task id for whole-body tracking.
   - Add terrain height map observation.
   - Add low-frequency wrist/end-effector pose commands.
   - Add dormant tracking rewards with zero initial weight.
5. Turn on tracking rewards through curriculum after the base locomotion policy
   is stable.
6. Add object pose command and simple rigid-object interaction.
7. Add articulated object state and contact phase labels.
8. Add shape encoding and residual force compensation.
9. Connect a VLM/video pipeline to generate sparse task guidance.

## Current Repository State

`hbc_lab` currently starts from the Unitree official G1 29DoF velocity task. The
goal is to preserve that baseline as the reproducibility anchor while adding
new task ids for increasingly general whole-body control and interaction.
