# Stage 1 V10.1 / V10.2 / V10.3 — the axis fix, the reward term it switched off, and the spawn that was underground

2026-08-05 KST. Robot: `robot_v2`, 45.451 kg, flat ground.
Predecessor report: `../stage1_v10_robot_v2_2026-08-04_KST/REPORT.md`.

---

## 1. V10.1 — the axis was fixed and the duty floor went dark

V10-flat commanded base x. The robot's fore-aft axis is base **+y**: the front foot
pair sits 0.88 m ahead of the rear pair in y, and the head points +y. V10.1 changed
only the command band.

| channel | V10-flat | V10.1 |
|---|---:|---:|
| `lin_vel_x` | (−0.9, −0.3) | (0.0, 0.0) |
| `lin_vel_y` | (0.0, 0.0) | (+0.3, +0.9) |

Nothing else moved: same robot, same fifteen reward weights, same termination set,
same action scale, same control rates, same PPO settings.

### Result: 7 of 8 gates, at both scored checkpoints

3000 iterations, 4096 environments, 39 minutes.

| checkpoint | speed m/s (cmd 0.500) | survival | FL | FR | RL | RR | min foot contact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model_2999` | 0.4417 | 0.9844 | 0.7911 | 0.2481 | 0.2495 | 0.7878 | **0.2481** |
| `model_2000` | 0.4489 | 1.0000 | 0.7761 | 0.2651 | 0.2466 | 0.7817 | **0.2466** |

Both fail only `min_foot_contact >= 0.30`. Training-log mean episode length at
iteration 2995 was 972.21 / 1000. *(That figure comes from the training log, not
from the checkpoint evaluations above — it is not a gate metric.)*

HFE phase correlations:

| pair | `model_2999` | `model_2000` |
|---|---:|---:|
| FL-FR | −0.0912 | −0.3324 |
| RL-RR | −0.0927 | −0.0332 |
| **FL-RR** | **+0.7253** | **+0.7265** |
| **FR-RL** | **+0.4537** | **+0.4740** |
| FL-RL | −0.1761 | −0.0806 |
| FR-RR | −0.1841 | −0.2993 |

**The axis change worked.** V10-flat's crab walk had an FL-RR diagonal correlation of
−0.020, failing a `>= 0` bar. It is now +0.7253, both same-end pairs are out of phase,
and both diagonals are in phase. The phase structure is a diagonal trot.

### The defect

The duty split is 79% / 25%. One diagonal carries the robot for four fifths of the
cycle; the other spends three quarters of it in the air.

The duty floor exists precisely to prevent this — `v5_12_mdp.abandoned_foot_penalty`,
weight −3.0, `contact_floor` 0.35, window 1.0 s. It reported:

```
Episode_Reward/duty_floor: 0.0000
```

at every logged iteration. Across the complete V10.1 training log only **two** distinct
lines contain `duty_floor`: the weight-table row and that zero. The same grep over
V10-flat's log returns **565** distinct values.

### Root cause

`v5_12_mdp._moving_mask` decided "is the robot being asked to travel?" from a single
column:

```python
command = smooth_mdp.guide_velocity_command(env, command_name)[:, 0]
return torch.abs(command) >= moving_threshold
```

`guide_velocity_command` applies `guide_frame.base_to_guide_observation`, which is
`_rotate_planar_180` — it negates x and y and **does not swap them**, so column 0
remains the x channel. V10.1 pins `lin_vel_x` to exactly zero. The mask was therefore
False for every environment at every step, and `worst_foot_contact_deficit` multiplies
its deficit by that mask. The term stayed registered at −3.0 and returned zero for
3000 iterations.

**The decisive evidence is the code path itself**: the mask read column 0, V10.1 pins
that column to exactly zero, and the deficit is multiplied by the mask. That is
sufficient on its own. The following are corroboration, and they are *not* mutually
independent — the first two both rest on the same observation that the logged reward
was zero:

1. **Log distribution.** Two distinct `duty_floor` lines across V10.1's whole log
   against 565 in V10-flat's.
   *(An earlier draft of this report argued that a live term "could not have been zero
   at iteration 0". That is false: `_duty` initialises to 0.5, above the 0.35 floor, so
   a zero penalty early in an episode is entirely legitimate. The argument has been
   withdrawn.)*
2. **Magnitude.** At a sustained contact fraction of 0.248 the raw deficit would be
   `(0.35 − 0.248) / 0.35 = 0.291`, far above the 0.0000 print resolution. This
   substitutes the post-training gate duty for the training-time one-second EMA, so it
   is indicative rather than exact.
3. **Signature.** The contact split resembles V8's pre-duty-floor result:

   | run | duty floor | FL | FR | RL | RR |
   |---|---|---:|---:|---:|---:|
   | V8 | absent | 0.161 | 0.789 | 0.799 | 0.162 |
   | V10.1 | masked | 0.791 | 0.248 | 0.249 | 0.788 |

   Same signature, opposite diagonal. This is an analogous phenotype on a different
   robot and a different objective history, so it is suggestive, not proof of this
   code path.

This is the **second** hidden "forward is base x" assumption to cost a run. The first
put the command on the wrong axis and produced the crab walk; this one disabled the
term that would have shaped the gait once the axis was corrected.

The gates did expose the *consequences* of both — V10-flat's final checkpoint failed
`diagonal_FL_RR_in_phase` at −0.0202, and V10.1 failed `min_foot_contact`. What no gate
did was point at the cause, because a masked-off penalty does not report itself as
absent; it reports as satisfied. (V10-flat passed 8/8 at iterations 1000 and 2000 and
only failed at 2999, which is why the crab walk was first noticed on video rather than
in a gate.)

---

## 2. Three ways this was allowed through, and a fourth bug found while reviewing

1. **A single-axis mask documented as direction-agnostic.**
   `tests/test_v8_1_duty_floor_contract.py` asserted
   `"torch.abs(command) >= moving_threshold" in mdp_source` under a comment reading
   *"The mask uses the command magnitude, so it is direction-agnostic."* It was
   sign-agnostic on one axis. The assertion would have passed forever.
   **Fix:** pin the planar norm, and test motion on either axis arithmetically.

2. **A probe that declared a check it no longer performed.**
   `probe_v8_1.py` drives the term's internal duty estimate to an abandoned-foot value
   (`term._duty[:] = 1.0; term._duty[:, 0] = 0.0`) and requires the penalty to fire.
   `probe_v10_flat.py` and `probe_v10_1_flat.py` kept only the lookup line — `duty_col`
   and `term` assigned and never used — while both docstrings still claimed the check
   was present. **That check fails on V10.1's configuration, so the run would not have
   launched.**
   **Fix:** restored in `probe_v10_2_flat.py`, and pinned by a contract test.

3. **A red test suite that was not run.**
   `tests/test_v8_recipe_contract.py::test_v8_video_and_evaluator_use_the_same_direction_convention`
   pinned the exact one-line spelling
   `guide_forward_speed_range_to_base((args_cli.speed, args_cli.speed))`. Adding
   `--forward-axis` wrapped that call across three lines in `evaluate_v8.py`, so the
   test went red *during the V10.1 work* and stayed red through the V10.1 launch.
   **Fix:** compare whitespace-stripped source, and run the full suite before launch.

4. **`--forward-axis` reached the camera but not the command.**
   `play_v8_video.py` branched on the flag for the camera eye/target and the head-marker
   offset, but set `cmd.ranges.lin_vel_x = guide_forward_speed_range_to_base(...)` and
   `lin_vel_y = (0.0, 0.0)` unconditionally. The V10.1 videos were filmed from the
   correct side while the robot was commanded along an axis its policy had only ever
   seen pinned to zero. **Gate numbers are unaffected** — `evaluate_v8.py` branches
   correctly and its output records `"forward_axis": "+y"` — but the footage did not
   show what it claimed to.
   **Fix:** branch the video command exactly as the evaluator does. Both videos were
   re-rendered; `videos/v10_1_original_wrong_axis/` retains the originals as evidence.

---

## 3. V10.2 — the fix

```python
command = smooth_mdp.guide_velocity_command(env, command_name)
planar_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
return planar_speed >= moving_threshold
```

**Why this is safe in shared code.** Every configuration older than V10.1 pins the
lateral band to exactly zero — `smooth_env_cfg.py` line 192 for the V5 line,
`v8_env_cfg.py` line 151 for the V8 line — so there `norm((x, 0)) == |x|` and the mask
is numerically the one under which every earlier result was measured. It changes
behaviour on V10.1's command and nowhere else.

**Blast radius.** `_moving_mask` gates two terms: `abandoned_foot_penalty` (registered
as `duty_floor`) and `front_rear_pair_synchrony`. Only the former is in the V10 line's
active table, which is the 14 Spot terms plus `duty_floor`.

**V10.2's environment config is V10.1's with no changes at all** — no `__post_init__`.
The delta is entirely in the shared term. A second simultaneous change would make any
change in the duty split unattributable.

### Verification, in the order performed

| stage | result |
|---|---|
| Old implementation temporarily restored | `test_moving_mask_reads_the_planar_magnitude_not_one_axis`, `test_moving_mask_counts_motion_on_either_planar_axis`, and `test_duty_floor_exempts_a_robot_commanded_to_stand_still` all **failed**, then the fix was restored. The second is arithmetic rather than source-matching: it builds a 4-row command tensor and requires True when motion is carried by y. |
| Full suite | **405 passed** |
| `probe_v10_2_flat.py`, in-simulator, pre-training | `moving_mask_fraction` **1.0** (V10.1's code gives 0.0); duty penalty with all four feet loaded **0.0**; with one foot abandoned mean **−2.7099**, min **−3.0**, fraction of environments firing **0.9219**; `support_fraction` 0.952; `tilt_deg_mean` 1.36; `fore_aft_axis_measured` y; `cmd_lin_vel_x` [0.0, 0.0]; `cmd_lin_vel_y` [0.3, 0.9]; zero failures |
| Training, iteration 187 | `Episode_Reward/duty_floor` **−0.0016**. V10.1 was exactly 0.0000 at every iteration. |

The restored probe also checks two things V8.1's did not: it reads `_moving_mask`
directly rather than inferring it through the reward, and it fails if the forward
command band ever reaches the moving threshold — a band that did would exempt the term
as completely as the wrong axis did.

### V10.2 results — 8/8 at every scored checkpoint

3000 iterations, 4096 environments, 41 minutes. Seed 42, identical to V10.1: both
inherit it from `v8_ppo_cfg.py` line 21 by deep copy, and neither train script
overrides it. The two runs therefore differ in exactly one thing, `_moving_mask`.

| | V10.1 | **V10.2** | gate |
|---|---:|---:|---|
| speed m/s (cmd 0.500) | 0.4417 | **0.4743** | ≥ 0.300 |
| survival (64 envs, 8 s) | 63/64 | **64/64** | ≥ 0.95 |
| FL contact | 0.7911 | **0.4950** | |
| FR contact | 0.2481 | **0.5155** | |
| RL contact | 0.2495 | **0.5191** | |
| RR contact | 0.7878 | **0.5023** | |
| **min foot contact** | **0.2481** ✗ | **0.4950** ✓ | ≥ 0.30 |
| min foot swing | 0.2089 | 0.4809 | ≥ 0.20 |
| gates passed | 7 / 8 | **8 / 8** | |

The duty split went from 79 / 25 to 49.5 / 51.5 / 51.9 / 50.2 — within 2 points of even
on all four feet, and it is already there at iteration 1000 (0.4999) and stays there.
The training log carries the same signal: **123** distinct `duty_floor` values against
V10.1's **2**.

**Scope of the causal claim.** This is one run per arm at a single shared seed. What is
established is that the mask was mathematically false under V10.1's command, that the
fix is the only training-relevant difference between the two runs, and that for seed 42
the duty split moved from 79/25 to ~50/50. "The fix causes an even duty split" as a
seed-independent result is not established by one paired sample.

**"8/8" is a threshold result, not a certificate of a clean trot.** The out-of-phase
gates accept correlations up to **+0.30** and the diagonal gates accept anything ≥ 0,
so a weakly in-phase rear pair passes. See the next section.

### Where V10.2 is worse than V10.1, stated plainly

Phase correlations moved in both directions:

| pair | V10.1 | V10.2 | gate | note |
|---|---:|---:|---|---|
| FL-FR | −0.0912 | **−0.4144** | ≤ 0.30 | improved |
| RL-RR | −0.0927 | **+0.1786** | ≤ 0.30 | **degraded — now weakly in phase** |
| FL-RR | +0.7253 | +0.7104 | ≥ 0.0 | unchanged |
| FR-RL | +0.4537 | **+0.2071** | ≥ 0.0 | **degraded** |
| FL-RL | −0.1761 | −0.1248 | ≤ 0 expected | slightly worse |
| FR-RR | −0.1841 | −0.1427 | ≤ 0 expected | slightly worse |

`RL-RR` at +0.1786 passes only because the bar is ≤ 0.30. A clean trot wants the rear
pair clearly out of phase, and it is not. `FR-RL` more than halved.

**And within V10.2 the phase degrades monotonically while the duty stays flat:**

| checkpoint | min contact | RL-RR | FR-RL | FL-RR | FL-FR |
|---|---:|---:|---:|---:|---:|
| `model_1000` | 0.4999 | **−0.1906** | **+0.3495** | +0.5417 | **−0.5480** |
| `model_2000` | 0.5002 | +0.1229 | +0.2160 | +0.5592 | −0.4985 |
| `model_2999` | 0.4950 | +0.1786 | +0.2071 | **+0.7104** | −0.4144 |

`model_1000` is the better trot on three of the four pairs, and it is where the rear
pair is genuinely out of phase. **The final checkpoint is not the best one.** An earlier
draft of this section claimed the gait "is correct from early on and does not drift" —
that is true of the duty split and false of the phase, and it has been corrected.

How much of this is real degradation and how much is measurement noise is **not
established**. The evaluator pools every environment and timestep into a single Pearson
correlation of HFE joint velocity, reports no per-environment distribution or
confidence interval, and always evaluates at seed 42. HFE-velocity correlation is also
a proxy for footfall phase, not a direct measurement of it. Settling this needs the
existing checkpoints re-scored under several evaluator seeds with per-environment
contact traces retained — not more interpretation of the archived aggregate.

Training-time falls also rose while gate survival improved:

| | V10.1 | V10.2 |
|---|---:|---:|
| `Episode_Termination/base_tilt` | 0.0409 | **0.0578** |
| mean episode length | 972.21 | **930.05** |
| `Episode_Reward/gait` | 9.2829 | 8.1435 |
| `Episode_Reward/air_time` | 2.2741 | 2.7414 |
| gate survival | 0.9844 | **1.0000** |

These are not measuring the same thing. Training uses 20-second episodes with
randomised speed and yaw, 10% standing commands, observation corruption, randomised
reset pose and velocity, pushes, and randomised friction, restitution and base mass.
The gate is 8 seconds at a fixed 0.5 m/s with zero yaw, no standing environments, no
corruption, no pushes and no reset perturbation. The gate survival difference is also
**63/64 against 64/64** — a single robot. And the training termination figures are
final-log statistics, not whole-run failure rates.

> **[corrected by V10.3]** This section originally said the duty floor was "associated
> with worse training-rollout robustness". **It was not the duty floor.** V10.3 kept
> the duty floor completely unchanged, fixed only the spawn height, and training
> terminations fell from 0.0578 to **0.0012** — a 48× reduction — with mean episode
> length going to 1000.00/1000. The robustness cost belonged to the underground spawn,
> not to the reward term. See section 5.

**A methodological caveat that applies to every gate number above.** `evaluate_v8.py`
begins sampling at step 0 with no warmup discard, so anything that happens during the
reset transient is folded into the contact fractions, correlations, speed and height.
Section 4 raises the possibility of a large depenetration transient at reset; if that
is confirmed, these gate metrics are not pure steady-state gait measurements and the
evaluator needs a warmup skip.

**Visual comparison.** `figures/` holds three filmstrips taken with an identical crop
and frame spacing: V10-flat (the crab walk, robot facing the camera, head marker on
its flank), V10.1 (fore-aft travel, legs nearly straight — a stilted stride), and
V10.2 (fore-aft travel with visibly flexed knees and a more compact stride). The
progression the numbers describe is visible in the footage.

---

## 4. Confirmed — the spawn height constant has the wrong sign

**Status: confirmed by direct trace.** `spawn_trace_probe.py`, 64 environments, zero
action, reading `root_pos_w[:, 2]` and ground contact force from the spawn state onward:

| step | root z | ground support |
|---:|---:|---:|
| before any step | **−0.3700** | — |
| 1 | −0.3651 | **2.432** |
| 2 | −0.3534 | 2.093 |
| 3 | −0.3368 | 1.708 |
| 5 | −0.2958 | 1.060 |
| 10 | −0.1938 | 0.874 |
| 20 | −0.0018 | 0.979 |
| 50 | +0.3848 | 0.771 |
| 100 | +0.3215 | 0.938 |
| 200 | +0.4328 | 0.949 |
| 400 | **+0.4451** | 0.949 |

Three things are settled by this:

1. **`init_state.pos.z` is exactly what `root_pos_w[:, 2]` reports.** The robot appears
   at −0.3700, the configured value, with no hidden frame offset. The predecessor
   report's reasoning that `init_state.pos` is "a frame value on an asset whose frames
   are offset from its geometry" — the argument that justified `SETTLED_ROOT_FRAME_Z_M
   = -0.4192` — does not hold for the *root* frame.
2. **The robot starts 0.81 m underground.** Step 1 registers **2.43× body weight** of
   contact force, the signature of a depenetration impulse, and it rises 0.8151 m over
   roughly 200 steps before settling.
3. **`FEET_ABOVE_ROOT_FRAME_M = 0.42` has the wrong sign.** The feet sit ≈0.445 m
   *below* the root frame. Every spawn height in the project derived from
   `clearance − 0.42` is therefore low by about `2 × 0.44`.

An earlier draft of this section reasoned to the right conclusion from the wrong
evidence and overstated it. Those corrections are kept below, marked, because the
reasoning errors matter as much as the result.

`guide_robot_v2_cfg.py` declares, with a comment stating both were measured:

```python
SETTLED_ROOT_FRAME_Z_M  = -0.4192   # root frame z at rest
FEET_ABOVE_ROOT_FRAME_M = 0.42      # feet sit above the root frame
SPAWN_ROOT_FRAME_Z_M    = clearance - FEET_ABOVE_ROOT_FRAME_M
```

Direct measurements of `robot.data.root_pos_w[:, 2]` after the robot is dropped and
left to settle, with the ground carrying its weight:

| source | root frame z | support |
|---|---:|---:|
| `probe_v10_1_flat.py`, 400 steps, 64 envs, seed 42 | **+0.4357** | 0.936 |
| `probe_v10_2_flat.py`, 400 steps, 64 envs, seed 42 | **+0.4455** | 0.952 |
| `SETTLED_ROOT_FRAME_Z_M` | −0.4192 | — |

**[corrected]** These two probe numbers are **one measurement, not two** — the same
method, environment count, ground plane and seed, differing only in which config object
was instantiated. They corroborate each other's arithmetic, not each other's validity.

**[corrected]** An earlier draft also cited the MJCF `stand` keyframe base z (+0.4640)
as a third, independent source. That was wrong, and it contradicts a correction made in
the predecessor report (`../stage1_v10_robot_v2_2026-08-04_KST/REPORT.md`, step 6),
which explicitly warns that the MuJoCo keyframe value is *not* the same frame as
IsaacLab's `init_state.pos`. It has been withdrawn from the evidence.

What survives is this: a direct measurement of the quantity `init_state.pos.z` controls
reads **+0.44** when the feet are on the ground, and the constant says **−0.4192**. That
disagreement does not depend on the MJCF at all. It does not, however, establish that
the root frame is "ordinary" or geometry-aligned — only that it sits above the ground
by that amount in this pose.

If so, the spawn arithmetic is inverted and places the robot deep underground:

| configuration | spawn root z | resulting foot position |
|---|---:|---|
| flat (`0.05 − 0.42`) | −0.37 | 0.81 m **below** the floor |
| terrain (`0.32 − 0.42`) | −0.10 | 0.54 m **below** the tile origin |
| corrected (`clearance + 0.44`) | +0.49 / +0.76 | as intended |

A flat `GroundPlane` hides this: the solver ejects the robot and it settles correctly,
which is why every flat gate passed with the defect present.

**Consequence for the gate numbers in this report.** Depenetration takes roughly 200
steps (4 s) under zero action, and `evaluate_v8.py` samples for 8 s from step 0 with no
warmup discard. A trained policy may recover faster than a passive one — **that has not
been measured** — but until it is, every contact fraction, correlation and speed in
sections 1 and 3 should be read as possibly including a reset transient. The fix is a
warmup skip in the evaluator, and re-scoring the existing checkpoints.

**[corrected] — what the earlier draft got wrong about terrain.** It claimed "the
correct spawn height was never tried, because the arithmetic had the sign wrong". That
is **withdrawn**: the predecessor report records a **+0.6 m** terrain spawn that made
contact and still flipped to **165.6°**, and the hypothesised corrected terrain spawn is
+0.76 — close enough that the claim is not supportable as written.

There is a plausible mechanism that would reconcile the two, and it is **untested**: on
generated terrain each tile's origin sits at its own local surface, so a raised box tile
puts its surface *above* the origin. With the feet 0.44 m below the root, a +0.6 root
spawn places the feet only 0.16 m above the tile origin — which on a raised tile can
still be inside the geometry, producing the same depenetration impulse and the same
flip. Whether that is what happened is a matter for a terrain-side trace, not for
assertion here.

What the sign error does explain without qualification is the other terrain failure
mode: spawning the frame at −0.389 made the robot "fall straight through the mesh from
underneath", which puts the feet 0.83 m below the tile origin.

**Done in V10.3** (section 5): the constants were re-derived from measurement
(`FEET_BELOW_ROOT_FRAME_M = 0.445`, `SPAWN_ROOT_FRAME_Z_M = clearance + 0.445`) and
`--warmup-seconds` was added to the evaluator. Terrain remains untried under the
corrected spawn.

Two checks are written and queued for the next free GPU window:

- `spawn_height_check.py` — decisive. Drops the robot at the configured height and at
  `settled + clearance`, recording root z and ground-support fraction at steps
  1/5/10/25/50/100/200/400. Deliberately does **not** read the suspect constants.
- `usd_foot_offset_check.py` — independent. Compares the world bounds of the trunk
  collider against the four foot colliders, using bounding boxes rather than frames,
  since displaced frames are the known defect of this asset.

Per-body frames *are* displaced — the probe measures the front foot pair 0.24 m above
the rear pair in z on a robot standing level — so the predecessor report's frame finding
holds for `body_pos_w`. What is new is that the **root** frame appears to be normal, and
that the spawn constant derived from it is not.

---

## 5. V10.3 — the spawn fix, and what it recovered

One change from V10.2: `SPAWN_ROOT_FRAME_Z_M = clearance + FEET_BELOW_ROOT_FRAME_M`,
flat spawn −0.37 → **+0.495**. Same seed, same objective, empty subclass, 3000
iterations.

**The probe confirms the mechanism before training starts:**

| | V10.2 | **V10.3** |
|---|---:|---:|
| configured spawn z | −0.37 | **+0.495** |
| root z before any step | −0.3700 | **+0.4950** |
| ground support on step 1 | **2.432 × body weight** | **0.0** |

The robot now begins 5 cm above the floor in free fall, touching nothing.

**Gates — 8/8 at all three checkpoints, both runs:**

| | speed | surv | min contact | RL-RR | FR-RL | FL-RR | FL-FR |
|---|---:|---:|---:|---:|---:|---:|---:|
| V10.2 @1000 | 0.4803 | 1.0000 | 0.4999 | −0.1906 | +0.3495 | +0.5417 | −0.5480 |
| V10.2 @2000 | 0.4891 | 1.0000 | 0.5002 | +0.1229 | +0.2160 | +0.5592 | −0.4985 |
| V10.2 @2999 | 0.4743 | 1.0000 | 0.4950 | +0.1786 | +0.2071 | +0.7104 | −0.4144 |
| V10.3 @1000 | 0.4649 | 1.0000 | **0.3638** | +0.0318 | +0.8186 | +0.2583 | −0.7980 |
| V10.3 @2000 | 0.4654 | 1.0000 | 0.4968 | +0.0952 | +0.6687 | +0.5773 | −0.7914 |
| **V10.3 @2999** | 0.4768 | 1.0000 | 0.4949 | +0.1628 | **+0.7124** | +0.5547 | **−0.6341** |

V10.3 @2999 contact fractions: FL 0.4952 / FR 0.4991 / RL 0.5059 / RR 0.4949 — all four
within 1% of even.

**What improved.** The weak diagonal recovered: `FR-RL` +0.2071 → **+0.7124**, and the
front pair went from −0.4144 to **−0.6341**. `FL-RR` moved the other way (+0.7104 →
+0.5547), so this is a redistribution across pairs rather than a uniform gain, but three
of four pairs are better and the previously-marginal one is no longer marginal.

**Training robustness, and a correction to section 3.**

| | V10.1 | V10.2 | **V10.3** |
|---|---:|---:|---:|
| `Episode_Termination/base_tilt` | 0.0409 | 0.0578 | **0.0012** |
| mean episode length | 972.21 | 930.05 | **1000.00** |

Section 3 originally attributed V10.2's worse training robustness to the duty floor.
V10.3 leaves the duty floor untouched, changes only the spawn, and the final logged
base-tilt share drops **~48×**.

That attribution was wrong, but the replacement must be stated carefully, because the
four-cell experiment is incomplete:

| | duty floor masked | duty floor live |
|---|---|---|
| **spawn underground** | V10.1 — 0.0409 | V10.2 — 0.0578 |
| **spawn corrected** | *never run* | V10.3 — **0.0012** |

With the spawn underground, switching the duty floor on *did* raise falls (0.0409 →
0.0578). With the duty floor on, fixing the spawn removed nearly all of them. The
missing cell means the two cannot be fully separated. What is established at seed 42:
**with the spawn corrected, the duty floor no longer costs measurable robustness.**
Note also that these are final-logging-window shares, not whole-run fall counts.

**Video (corroboration, not the primary evidence).** `videos/v10_3/` and
`figures/four_stage_comparison.png`. V10.2's recording opens with roughly eight frames
of empty floor before the trunk emerges through it; V10.3's shows the robot **fully
visible above the floor from frame 2** — not "standing", since the probe measures zero
contact force on step 1 and it is briefly in free fall. The camera also aims relative to
root z, so V10.2's underground reset moved the camera as well as the robot. The direct
evidence is the pre-step root trace, not the footage. `figures/v10_2_first_frames.png` and
`figures/v10_3_first_frames.png` are the same extraction on both.

### What V10.3 did not fix

`RL-RR` is still weakly **positive** (+0.1628) — the rear pair is not clearly out of
phase, and passes only because the gate bar is ≤ 0.30. It also still drifts upward
within the run (+0.0318 → +0.0952 → +0.1628), the same direction V10.2 showed. The
spawn fix did not touch this. It is the clearest remaining gait weakness and the
obvious subject for V10.4 — with the standing caveat that the metric is a pooled HFE
velocity correlation, a proxy for footfall phase rather than a direct measurement.

`min contact` at `model_1000` is **0.3638**, well below the ~0.50 the other five
checkpoints show. It clears the 0.30 bar but it is the weakest number in the table.

### Scope, again

One run per arm at one seed. What is established: the spawn was measurably wrong, the
probe now proves it is right before training, and at seed 42 the corrected spawn
coincided with better phase on three of four pairs and a 48× drop in training falls.
Seed-independence is not established by one paired sample.

---

## 6. Contents

```
REPORT.md
checkpoint/    v10_2_model_{1000,2000,2999}.pt   8/8 gates, duty floor live
               v10_3_model_{1000,2000,2999}.pt   8/8 gates, spawn corrected  <- current best
config/        V10.1 / V10.2 / V10.3 configs, probes, queues, the corrected
               v5_12_mdp.py and guide_robot_v2_cfg.py, 5 test files, and
               v5_12_mdp_moving_mask.patch (the before/after of the mask fix)
results/       gate JSONs for every scored checkpoint, sweeps, probe reports,
               spawn_trace_probe.json, env/agent yaml, pytest output,
               duty_floor log evidence
videos/
  v10_1_original_wrong_axis/  filmed with the command on the wrong axis (evidence)
  v10_1_axis_fixed/           re-rendered after the play_v8_video fix
  v10_2/                      duty floor live; opens with the robot underground
  v10_3/                      spawn corrected; robot stands from frame 2
figures/       four_stage_comparison.png  <- the one-picture summary
               per-version filmstrips, first-frame extractions
```

### Recommended checkpoint

**`checkpoint/v10_3_model_2000.pt`.** An earlier draft recommended `model_2999` and
called it "the cleanest diagonal phase of any run"; that was wrong on both counts.
Compared metric by metric, `model_2000` is better on **four of six**:

| | `model_2000` | `model_2999` | better |
|---|---:|---:|---|
| `RL-RR` (want ≤ 0) | **+0.0952** | +0.1628 | 2000 |
| `FL-FR` (want ≤ 0) | **−0.7914** | −0.6341 | 2000 |
| `FL-RR` (want ≥ 0) | **+0.5773** | +0.5547 | 2000 |
| min foot contact | **0.4968** | 0.4949 | 2000 |
| `FR-RL` (want ≥ 0) | +0.6687 | **+0.7124** | 2999 |
| speed (cmd 0.500) | 0.4654 | **0.4768** | 2999 |

`model_2999` wins only `FR-RL` and speed. `model_1000` has the single highest `FR-RL`
(+0.8186) but a clearly split duty — FL 0.3638 / FR 0.6307 / RL 0.6336 / RR 0.3657 —
so it is not a candidate despite passing every gate.

Both `model_2000` and `model_2999` are kept in `checkpoint/`.
