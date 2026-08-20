# V12 Depth ViT Distillation Training Design

## Scope and success definition

Train the prepared `TinyDepthViT` on synchronized IsaacLab depth and privileged
V11 teacher targets, select a checkpoint on a held-out simulator seed, and run
the resulting camera policy in closed loop. The immutable V11
`model_4999.pt` remains the teacher and safety baseline.

Training is successful only if all tensors and gradients remain finite, the
selected checkpoint improves both scan and action validation errors over the
untrained initialization, and closed-loop results are reported without
representing simulator performance as hardware validation.

## Alternatives and selected approach

Three approaches were considered:

1. End-to-end camera PPO has the cleanest final interface but discards the
   strong V11 supervision and couples perception, control, and reward failures.
2. Direct behavior cloning predicts 12 actions from depth and proprioception,
   but removes the interpretable 187-value terrain boundary and is vulnerable
   to state-distribution drift.
3. Privileged-perception distillation predicts the V11 scan, then passes it
   through the frozen V11 actor. This preserves the tested control network,
   supports scan-level diagnostics, and can add a bounded DAgger pass if the
   first closed-loop rollout drifts.

Use approach 3. A single student-state DAgger collection is allowed only when
the initial teacher-rollout dataset improves offline metrics but fails the
closed-loop gate. PPO fine-tuning is outside this run.

## Camera and environment contract

Use IsaacLab `TiledCameraCfg` with `distance_to_image_plane` only:

- parent prim: `{ENV_REGEX_NS}/Robot/base/base/front_depth`
- image: float32 `84×84×1`, clipped to 0.1–5.0 m and normalized to `[0,1]`
- pinhole focal length: 24 mm; horizontal aperture: 20.955 mm
- mount position in the root body frame: `(0.0, 0.30, 0.10)` m
- ROS-frame quaternion `(w,x,y,z) = (0.573576, -0.819152, 0.0, 0.0)`,
  which points along robot base `+y` and 20 degrees downward
- camera update period equals the 50 Hz policy step

The first live probe must show finite nonconstant depth, visible terrain rather
than only the robot body or sky, the exact `[N,84,84,1]` sensor shape, and exact
manual-actor equivalence to the RSL-RL inference policy.

Use 128 parallel V11 rough-terrain environments. Disable observation corruption
so the recorded proprioception and scan targets are deterministic functions of
the current state; keep physics/material/mass randomization active. Distribute
environments across all eight terrain difficulty rows and available terrain
types after construction. Teacher actions drive the initial collection.

## Dataset contract

Every synchronized sample contains normalized float16 `depth [1,84,84]`,
float16 `proprioception [48]`, float16 clipped `teacher_scan [187]`, and float16
deterministic `teacher_action [12]`. Samples are captured before applying that
action, after a 50-step warm-up, every four control steps. Store atomic shards
of 2,048 samples plus a JSON manifest containing the split, seed, checkpoint
SHA-256, camera contract, tensor shapes, dtypes, sample count, and per-field
finite/range statistics.

- train: seed 42, 65,536 samples
- validation: seed 142, 8,192 samples, separate simulator launch
- optional DAgger: seed 242, 32,768 student-state samples

Incomplete final shards are valid. Loader validation rejects missing fields,
wrong trailing shapes, non-finite stored values, out-of-range normalized depth,
or manifest/sample-count mismatches.

## Frozen teacher actor

Reconstruct the exact V11 actor from checkpoint keys `actor.0`, `actor.2`,
`actor.4`, and `actor.6`: `235→512→256→128→12` with ELU between linear layers.
All actor parameters have `requires_grad=False`, while gradients continue through
the actor input into the ViT. Before collection or training, compare its output
against RSL-RL `get_inference_policy` and require maximum absolute error at most
`1e-6` on a live observation batch.

## Optimization and model selection

Initialize from the archived deterministic seed-42 Tiny ViT state. Train only
the 579,387 ViT parameters with AdamW, learning rate `3e-4`, weight decay
`1e-4`, batch size 128, gradient-norm clipping at 1.0, and seed 42. Use at most
20 epochs with patience 5 on validation total loss.

The objective remains:

`smooth_l1(predicted_scan, teacher_scan) + 0.25 * mse(frozen_actor(proprio + predicted_scan), teacher_action)`

Record train and validation scan MAE/RMSE, action MAE/RMSE, component losses,
learning rate, gradient norm, epoch duration, and checkpoint SHA-256. Save
`best.pt` only when validation total loss improves; retain `last.pt` for audit.
The selected checkpoint must improve both validation scan MAE and action MAE
against the untrained initialization and contain no non-finite state.

## Closed-loop evaluation and bounded recovery

Replace only the final 187 values of the live observation with the ViT prediction
and execute the frozen actor. Compare against teacher scan/action online without
using either as student input. Evaluate rough terrain level 4 and 15.714 cm
ascending and descending stairs on seeds 42–44.

Report survival, forward speed, horizontal progress, scan MAE, action MAE, and
scenario traversal. The operational target is survival at least 0.80 and speed
at least 0.25 m/s on rough terrain, plus at least one successful seed in each
stair direction. Failure is reported honestly. If offline metrics improved but
closed-loop drift is the cause, collect the one optional student-state dataset,
resume from `best.pt` for at most 10 additional epochs on the combined data, and
repeat evaluation. Do not add another recovery mechanism in this run.

## Verification and artifacts

All pure tensor, dataset, actor-loading, metric, and checkpoint-selection logic
is developed test-first. Run the full static suite, the live camera probe,
dataset validation, fresh training, checkpoint reload, and closed-loop evaluator.
Archive source, manifests, compact metric reports, logs, and selected/last
checkpoints locally. The large depth shards may remain on the remote training
host, but their manifest and per-shard SHA-256 hashes must be archived. Regenerate
and verify the release `SHA256SUMS` file. Remove temporary credential helpers
after all remote work.
