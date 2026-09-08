# Scene-Aware Action Saturation Experiment

Date: 2026-09-01

## Question

Does replacing the unbounded Gaussian plus hard action clipping with a correctly
squashed Gaussian preserve task learning while allowing the environment scan to
change root-height, torso-pitch, and hand commands before they saturate?

## Controlled Pair

- Host: `jd_5090`
- Seed: `42`
- Environments per run: `4096`
- Training horizon: `8000` iterations
- Low-level policy: `g1_dex1_handcenter_stickfigure_m49999.pt`
- Environment, observations, DRC rewards, motion-quality rewards, and domain
  randomization are identical.
- Baseline task: `HBC-Isaac-G1-Dex1-SceneAware-HierDrc-v0`
- Treatment task: `HBC-Isaac-G1-Dex1-SceneAware-Squashed-HierDrc-v0`
- Treatment changes only the high-level action interface. PPO samples, stores,
  and evaluates a latent Gaussian action `u`; the simulator receives
  `tanh(u)`. This avoids inverse-tanh and transformed-density Jacobian terms in
  PPO while preserving bounded physical commands and bounded exported policy
  outputs.
- Log standard deviation starts at `0.30` and is projected to `[0.08, 0.40]`.
  PPO uses a fixed learning rate of `5e-4`, small actor-output initialization,
  and stops before an optimizer step if any parameter or gradient is non-finite.

## Evaluation Gates

Compare matched checkpoints at 2k, 5k, and 8k iterations.

1. Task completion is primary. The treatment must not reduce open-scene or
   constrained-scene grasp/success performance in exchange for smoother motion.
2. `HL/action_boundary_095_ratio` and
   `Policy/posture_boundary_095_ratio` should remain below 15%.
3. `Policy/action_retention_ratio` should remain above 0.50. A low value means
   observation changes are again being erased by tanh saturation.
4. Inspect open/table `d_active_hand`, physical grasp, body collision, hand
   tracking error, root height, and torso pitch at matched iterations.
5. Geometry-conditioned posture shaping is allowed only after the bounded policy
   passes the task-completion gate. Any later motion term that materially reduces
   task success is rejected or weakened; motion quality is secondary to task
   completion.

## Numerical Failure And Fix

- The first transformed-Gaussian 4096-env run failed during PPO update 2418.
  Iteration 2417 was finite, while the model and Adam state in checkpoints 2500
  and 3000 were entirely NaN. Environment-side sanitization converted the NaN
  actions to zero, so the process continued and initially hid the policy failure.
- The likely trigger was the inverse-tanh log-probability path near bounded
  actions, amplified by adaptive learning-rate growth and a large value loss.
- A 25-iteration smoke test resumed clean checkpoint `model_2400.pt` without its
  optimizer. The latent-action implementation completed through iteration 2424;
  all model and optimizer tensors in `model_2424.pt` were explicitly checked and
  were finite. The environment-action boundary ratio remained around 1-2%.

## JD Sessions

- Active formal run: `scene_latent_squashed_4k`
- Completed smoke test: `scene_latent_resume2400`
- Stopped failed run: `scene_sat_squashed_4k`
- Stopped early diagnostics: `scene_sat_baseline` and `scene_sat_squashed`

The active run starts from scratch with seed 42 and writes to
`scene_latent_squashed_4k.log`. The smoke-test log is
`scene_latent_resume2400.log`. Failed and early diagnostic logs are retained as
action-saturation and numerical-stability references.
