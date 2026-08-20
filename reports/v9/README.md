# V9 line findings — terrain (2026-07-28)

Not a result backup: none of these runs produced a usable policy. This is the
evidence record for three diagnoses, kept because each one cost hours to obtain
and each fix is now inherited by later versions.

## V9 — terminated on contacts that terrain makes unavoidable

Stopped at iteration 4142 of 8000.

| | V8.1 (flat) | V9 (terrain) |
|---|---|---|
| body_contact ends | 17.5% | **82.5%** |
| mean episode length | 845 / 1000 | **93 / 1000** |
| terrain_levels | — | 0.212 → **0.180** over 4142 iters |

Cause: V8's termination set `base_link\|.*_hip\|.*_thigh\|.*_calf` at 1.0 N,
inherited unchanged. On flat ground any non-foot contact is a fall. On stairs a
calf grazing a step edge is ordinary locomotion. IsaacLab's own rough-terrain
config terminates on `base` only.

Consequence beyond low reward: `terrain_levels_vel` promotes on distance walked,
and nothing living 1.86 s ever qualified — the robot would have finished all
8000 iterations without seeing a stair.

**Fix (V9.1): `base_link|.*_hip`.** Threshold left at 1.0 N.

## V9.1 — survival solved, curriculum still went backwards

Stopped at iteration 2270 of 8000.

| | V9 @4142 | V9.1 @2270 |
|---|---|---|
| body_contact ends | 82.5% | **0.05%** |
| mean episode length | 93 / 1000 | **1000 / 1000** |
| gait reward | 0.814 | **7.43** (V8.1 on flat: 7.71) |
| base_linear_velocity | 0.311 | **3.23** (V8.1 on flat: 3.40) |

Terrain quality equal to flat, every episode complete. But the curriculum peaked
at 1.584 (iter 500) and fell to 0.716 by 2000.

Cause: `terrain_levels_vel` scores straight-line displacement
(`move_up` at `size[0]/2` = 4.0 m), while V8's command sampled a yaw rate of
±0.6 rad/s and held it for 10 s. A constant speed with a constant yaw rate traces
a circle of radius `v/w`; at v=0.6, w=0.6 the maximum displacement is 2.0 m.
Promotion was arithmetically impossible. Stock IsaacLab avoids this with
`heading_command=True`, which V8 correctly disabled for a flat guide task.

**Fix (V9.2): yaw band ±0.2.** `heading_command` left off — a guide dog must
still turn on command. Chosen from `chord = (2v/w)·|sin(wT/2)|`: at ±0.2 the fast
half of the speed band clears 4.0 m (8.18 m at v=0.9); at ±0.6 nothing does.

## V9.2 — curriculum promoted, then the robot stopped moving

Completed 5000 iterations. Gate failed.

Curriculum climbed as calculated — 0.987 → **2.111 @1479** — then collapsed:

    2000 → 1.417   2500 → 0.797   3000 → 0.401   4999 → 0.265

| iteration | 2000 | 3000 | 4000 | 4999 |
|---|---|---|---|---|
| speed m/s (cmd 0.500) | 0.156 | 0.085 | 0.091 | 0.085 |
| min contact | 0.678 | 0.690 | 0.643 | 0.664 |
| height m (calib 0.3616) | 0.282 | 0.259 | 0.276 | 0.283 |
| survival | 1.000 | 1.000 | 1.000 | 1.000 |
| stairs climb success | 0.000 | 0.000 | 0.000 | 0.000 |

`13_v9_2_stairs_시간경과.png` shows frames 20 / 200 / 400 — eight seconds apart,
the robot still in the pit at the foot of the stairs, splayed low, never touching
a step. It survives 100% of episodes by not attempting anything.

Cause: `base_linear_velocity_reward` is `exp(-error/std)` with `std = 1.0`. A
stationary robot under a 0.6 m/s command scores `exp(-0.6) = 0.549`, which at
weight 5.0 is **2.74/s** — the measured value was 2.99. Perfect tracking scores
5.00, so the entire incentive to move is ~2.0/s, cheaper to decline than to earn
on rough terrain. Meanwhile `GaitReward` scores contact *timing*, which a robot
cycling its legs on the spot satisfies: 6.44 of 10 with zero displacement.

Nothing in the fifteen terms mentions displacement — the same structural error as
V8's missing duty factor: the objective describes a pattern and omits the
quantity.

**Fix (V9.3): `std` 1.0 → 0.5.** Standing drops 2.74 → 1.50/s, moving becomes
worth 3.50/s, and the gradient at the operating error *rises* (0.549 → 0.602).
Sharper widths keep improving the payoff gap but flatten the reward far from
target, which stops teaching a policy that starts at rest.

## Tooling lessons, each now pinned by a regression test

* **A threshold at one iteration cannot see a trend.** V9.1's early check read
  0.9065 against a 0.5 bar and passed — while falling from a 1.584 peak. The
  checker now reads the series and fails on decline from peak.
* **Iteration 0 is not curriculum behaviour.** `init_at_random_ep_len=True`
  truncates most first episodes; V9.1 and V9.2 both logged exactly 0.987 there.
  Counting it as the peak would fail healthy runs. Excluded via a 250-iteration
  warmup.
* **Stopping at the first verdict misses late collapse.** V9.2's check passed at
  1501 and the collapse happened over the next 3500 iterations unwatched. The
  check now runs for the whole job and writes an ALERT file; replayed on V9.2's
  own series it fires at ~2500.
* **`error_vel_xy` is not a speed proxy on terrain.** V8.1 flat walking 0.394,
  V9.1 terrain walking 1.073, V9.2 terrain stationary 1.151 — the two terrain
  runs differ by 0.08 with opposite behaviour. `terrain_levels` is computed from
  displacement by definition and is the right signal.

## Contents

```
videos/    model_2000 and model_4999, rough + stairs, side view
12_v9_2_rough_시간경과.png     frames 20/120/220, rough
13_v9_2_stairs_시간경과.png    frames 20/200/400, stairs — the decisive evidence
```

Working policy to date remains `stage1_v8_1_duty_floor_2026-07-28_KST`
(flat ground, all eight gait gates passed, best checkpoint `model_2000.pt`).
