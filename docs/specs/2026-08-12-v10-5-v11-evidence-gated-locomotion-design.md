# V10.5 / V11 Evidence-Gated Locomotion Design

## Goal

Continue the `robot_v2` Stage-1 teacher line without repeating completed work:

1. replace pooled HFE phase guesses with multi-seed, per-environment contact evidence;
2. improve the flat gait only if that evidence confirms the weak rear pair;
3. carry the best policy into a terrain-aware V11 model with the corrected spawn;
4. measure rough traversal and ascending-stair success directly.

## Existing baseline

- V10.3 flat `model_2000` is the recommended flat checkpoint. It passes the old
  eight gates, but pooled HFE `RL-RR` is weakly positive at `+0.0952`.
- A later remote-only V10.4 blind-terrain run already exists. Its final checkpoint
  passes the old gates with speed `0.622 m/s`, evaluation survival `0.953`, and
  pooled HFE `RL-RR = -0.504`, while its final training window reports
  `base_tilt = 0.286`. It did not use a scanner or run a stairs-specific gate.
- Both source trees are unversioned directories. New work must be isolated in a
  new remote directory and archived locally with checksums rather than mutating
  the historical run.

## Architecture

### Phase A: trustworthy gait evidence

Add a pure-Torch metric module used by the simulator evaluator. It consumes a
`[time, environment, 4]` binary contact tensor and returns:

- contact correlation for all six foot pairs per environment;
- same-pair transition coincidence for FL/FR and RL/RR;
- valid-environment fraction when a foot has zero variance;
- median, 10th percentile and 90th percentile across environments.

The evaluator accepts an explicit seed and a warm-up interval. A driver evaluates
V10.3 checkpoints 1000/2000/2999 and V10.4 terrain checkpoints 1000/2000/2999 at
seeds 42-46, preserving each raw JSON and writing one aggregate JSON.

The rear weakness is confirmed when at least four of five seeds have either
`RL-RR` contact-correlation median above `-0.20` or transition-coincidence above
`0.05`, while at least 90% of environments have valid correlations. The threshold
is intentionally stricter than the historical HFE `<= +0.30` gate.

### Phase B: V10.5 flat

If Phase A confirms the weakness, V10.5 changes one training-relevant item from
V10.3: a bounded contact-state overlap penalty on the two same-end pairs. While a
robot is commanded to move, it returns the average of `FL == FR` and `RL == RR`;
standing commands are masked. The existing duty-floor prevents the degenerate
one-foot-always-up solution. Weight starts at `-1.0`, below the `-3.0` duty floor
and far below the `+10.0` gait reward.

V10.5 trains from scratch for 3000 iterations at seed 42 so its causal comparison
with V10.3 remains clean. Checkpoints 1000, 2000 and 2999 are evaluated at five
seeds. A V10.5 checkpoint is accepted only when it preserves speed >= 0.45 m/s,
survival >= 0.95, every foot's contact and swing fraction >= 0.35, and the new
rear-contact phase criterion. If Phase A does not confirm the weakness, no phase
reward is invented; the best multi-seed V10.3 checkpoint becomes the flat source.

### Phase C: V11 terrain-aware transfer

V11 uses generated rough terrain, corrected `+y` forward commands, corrected
terrain spawn arithmetic, yaw range `[-0.2, +0.2]`, and the existing frame-free
`median(hit_z) - hit_z` 187-point height scan. A probe rejects:

- first-step support above 0.5 body weight;
- any non-finite scan value;
- more than 1% scan saturation at either clip;
- absent terrain curriculum;
- an observation width other than 235;
- a command axis other than base `+y`.

The 48-input flat policy is transferred by expanding `actor.0.weight` and
`critic.0.weight` from `[512, 48]` to `[512, 235]`. Existing 48 columns are copied
exactly and the new 187 columns are zero. All other parameters are copied, the
Adam moments and checkpoint iteration are preserved, and the two widened Adam
moment tensors receive the same zero columns as the network weights. The PPO
algorithm's in-memory adaptive learning-rate field is explicitly restored from
the loaded optimizer group. On a zero scan, the converted network must match the
original actor and critic outputs within `1e-6` before training.

This exact-resume requirement was established empirically. The first bootstrap
reset Adam and iteration as initially designed; its inference policy passed with
100% survival, but `model_0` fell to 0% survival after one PPO update and the
training fall fraction reached 99% by update 5. Preserving optimizer moments,
iteration 2000 and learning rate `1.1390625e-4` kept survival at 100% after a
five-update smoke run.

Continuation runs also restore the terrain curriculum operationally. IsaacLab
does not store per-environment terrain rows in the model checkpoint, so a plain
resume restarted at level 0 (measured mean 0.766 after ten updates) even though
the prior run ended at 5.17. Continued V11 training therefore accepts an exact
initial terrain row, validates it against the generator, rewrites environment
origins and resets before PPO collection. The level-5 smoke started at 4.985.

V11 trains a 3-iteration smoke test and then a 4000-iteration pilot, scoring
1000/2000/3000/3999. A progress monitor fails loudly if the terrain level drops
by more than 1.0 from its post-warmup peak or stays below 1.0 after iteration
1500. Selection ranks rough survival first, then forward progress, terrain level,
contact phase, and speed.

### Phase D: direct terrain gates

Rough evaluation uses several seeds and records survival, forward displacement,
speed, contact phase and terrain level. Stairs evaluation freezes the curriculum
on ascending stairs from 0.05 to 0.20 m. Success requires:

- rough survival >= 0.90 and mean forward speed >= 0.30 m/s;
- stair survival >= 0.80;
- at least 50% climb success overall;
- at least 50% success on a step height >= 0.15 m.

If scanner V11 fails but the blind V10.4 baseline passes the identical terrain
gate, the result is reported as a scanner-transfer regression rather than hidden
by another simultaneous reward change.

## Error handling and evidence retention

Every stage writes a status file before it starts and on exit. Probes and smoke
tests stop a training launch. Missing checkpoints, invalid JSON, non-finite
metrics and incomplete seed sets are failures, not skipped rows. Raw evaluator
JSON, aggregate JSON, logs, selected checkpoints, config files and SHA256 sums
are copied back to a dated local archive.

## Testing

- Pure tensor tests cover contact correlations, constant-foot invalidity,
  transition coincidence and quantiles.
- Contract tests pin the single V10.5 reward delta, the V11 observation width,
  +y scanner geometry, corrected spawn, curriculum and checkpoint expansion.
- Simulator probes cover live reward behavior, physical spawn, scan quality and
  zero-scan checkpoint equivalence.
- A 3-iteration smoke run precedes each full training run.
