# Stage 1 V8.1 — duty floor under the Spot recipe (2026-07-28 KST)

**First run in this project to pass all eight gait gates.**

## What changed from V8

Exactly one reward term was added. The fourteen published Spot weights, the
action scale, the control rates, the network and the PPO settings are inherited
from V8 unchanged.

```
duty_floor   weight -3.00   func v5_12_mdp.abandoned_foot_penalty
             contact_floor 0.35, threshold 5.0 N, window 1.0 s
```

### Why

V8 solved gait *phase* and left duty factor completely unconstrained. Its
contact fractions at 8000 iterations were FL 0.161 / FR 0.789 / RL 0.799 /
RR 0.162 — a textbook trot rhythm executed as a two-legged drag.

Two terms look like they should have prevented it and neither does:

* `GaitReward` compares only *relative* timing (one foot's air time against
  another's contact time). An 84/16 split satisfies all six of its pairwise
  comparisons exactly as well as 50/50 does.
* `air_time_reward` scores `max(air_time, contact_time)`, so it pays the same
  for a foot that stays up a long time as for one that stays down a long time —
  it rewards *both* extremes and penalises only the balanced middle.

Nothing in the fourteen terms mentions how long a foot must be loaded, so the
84/16 split was an optimum of the objective as written, not a training artifact.

The weight was sized against V8's own measured per-second reward rates
(TensorBoard, iteration 7999): positive total +17.10, largest existing penalty
1.27 (`action_smoothness`). At V8's measured 0.161 duty the deficit is 0.540, so
-3.0 yields **1.62/s** — the largest penalty in the table at the behaviour it
targets, bounded at 3.0, and identically zero once every foot clears the floor.

## Result — checkpoint sweep

| iter | gates | min contact | height | survival | speed | FL / FR / RL / RR | FL-FR | RL-RR |
|-----:|------:|------------:|-------:|---------:|------:|-------------------|------:|------:|
| 2000 | **8/8** | **0.454** | **0.375** | **1.000** | 0.679 | 0.48 / 0.45 / 0.47 / 0.46 | -0.19 | -0.44 |
| 3400 | 7/8 | 0.433 | 0.364 | 0.938 | 0.664 | 0.51 / 0.43 / 0.45 / 0.51 | -0.50 | -0.64 |
| 4000 | 8/8 | 0.429 | 0.355 | 0.969 | 0.657 | 0.54 / 0.43 / 0.43 / 0.55 | -0.61 | -0.66 |
| 6000 | 7/8 | 0.357 | 0.330 | 0.938 | 0.635 | 0.64 / 0.37 / 0.36 / 0.63 | -0.66 | -0.70 |
| 7999 | 8/8 | 0.354 | 0.316 | 0.969 | 0.599 | 0.67 / 0.36 / 0.35 / 0.66 | -0.73 | -0.72 |

Against V8 at the same 8000 iterations: min contact 0.161 -> 0.354,
height 0.302 -> 0.316, survival 0.922 -> 0.969, and all four phase correlations
improved (FL-FR -0.486 -> -0.733, RL-RR -0.229 -> -0.717, FL-RR +0.708 ->
+0.872, FR-RL +0.895 -> +0.854).

### The sweep's finding: longer training is worse

Duty balance and stance height decline **monotonically** with training while
phase sharpness improves monotonically. The policy keeps pushing toward the
asymmetric crouch; the floor stops it at 0.35 rather than removing the pressure.

**`model_2000.pt` is the best checkpoint** — 8/8 gates, near-perfect 50/50 duty,
stance 0.375 m (above the calibrated 0.3616 m), and 1.000 survival. It is kept
here alongside the final one.

This repeats V8's pattern, where 1600 iterations beat 8000. Two runs now show
the same over-training signature on this robot.

## Contents

```
checkpoint/model_2000.pt      best -- 8/8 gates, 50/50 duty, survival 1.000
checkpoint/model_7999.pt      final -- 8/8 gates, duty pinned at the 0.35 floor
reward_weights.json           15 weights + the added term's parameters
network_architecture.md       policy/critic shape, observation layout, PPO settings
results/                      per-checkpoint gate JSONs and the run status
config/                       v8.1 + v8 + v9 sources and their contract tests
videos/                       model_2000 side, model_7999 side + rear quarter
09_참고_vs_v8_vs_v8_1.png      reference / V8 / V8.1 gait strips
10_v8_vs_v8_1_자세확대.png      posture close-up, V8 vs V8.1 at 8000
11_v8_vs_v8_1_8000_vs_v8_1_2000.png   posture close-up including the best checkpoint
```

## Verification

327 contract tests pass. `probe_v8_1.py` proves the new term is live rather than
assuming it: with all four feet loaded the penalty is exactly 0.0; with one foot
abandoned it is -2.41 mean / -3.0 min, firing on 84% of environments (the rest
are commanded to stand and are correctly exempt).

## Known issues

* **Over-training.** Duty balance and height decay monotonically after ~2000
  iterations. The next isolated change is to stop at 2000-3000, or to address
  the pressure at its source: replace `air_time_reward`'s `max(air, contact)`
  with a term that pays for the balanced middle instead of both extremes.
* **Speed overshoot.** 0.599-0.679 m/s against a commanded 0.5. Not gated, but
  the fastest checkpoints are also the least stable.
* The sweep in the first version of `queue_v8_1_pilot.sh` asked for
  `model_1999/3999/5999`, which rsl_rl never writes (it saves on multiples of
  `save_interval = 100`). Every one was skipped silently and the ranking held
  only the final checkpoint. Fixed to 2000/4000/6000, missing checkpoints are now
  reported loudly, and a contract test pins the multiple-of-save_interval rule.
