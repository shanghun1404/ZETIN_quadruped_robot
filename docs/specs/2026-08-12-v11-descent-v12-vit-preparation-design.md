# V11 Descent Video and V12 ViT Preparation Design

## Scope

Produce an evidence-backed video of the selected V11 `model_4999.pt` descending
15.714 cm stairs, then prepare the Stage 2 vision model boundary without launching
a long vision-training run. The V11 checkpoint remains immutable.

## Current-system constraint

The released V11 teacher is a plain Actor-Critic MLP. It concatenates 48
proprioceptive values with 187 raw height-scan values; it does not expose the
64-dimensional terrain latent assumed by the older Stage 2 document. Retrofitting
that latent would invalidate the trained first actor layer and require another
teacher training cycle.

## Descending-stair recording

The existing V11 recorder gains two explicit scenarios:

- `stairs_up`: inverted-pyramid stairs, preserving the current ascent behavior.
- `stairs_down`: pyramid stairs, whose origin is on the upper platform.

Both scenarios use level 5 from the existing 5–20 cm, eight-row generator, which
is 0.157142857 m per step. A descending rollout passes only when the robot remains
alive, travels at least 2.1 m, and drops at least two physical step heights after
reaching the stair approach boundary. Probe-only seeded rollouts select a passing
sample before camera rendering. The final MP4 must decode completely and show the
approach, descent and exit.

## V12 student architecture

Use privileged-perception distillation rather than direct camera-to-action
training:

```text
84×84 metric depth
  -> finite/clipped normalization
  -> Conv2d patch embedding (7×7 stride 7, 128 channels)
  -> 144 patch tokens + one CLS token
  -> 4 pre-norm Transformer blocks, 4 heads, MLP width 256
  -> LayerNorm + Linear(128, 187)
  -> predicted V11 height scan

48 proprioception + predicted 187 scan
  -> frozen V11 Actor [512, 256, 128]
  -> 12 action means
```

The first preparation milestone does not change PPO or the V11 checkpoint. It
provides a dependency-free PyTorch Tiny ViT, deterministic depth preprocessing,
the exact 48/187/235/12 tensor contract, and losses for scan and action
distillation. The Transformer uses built-in PyTorch only; `timm` and
`torchvision` are not required.

## Data and losses

Each future collection sample contains:

- `depth`: float32 `[1, 84, 84]`, invalid values replaced by the far clip and
  normalized to `[0, 1]` over `[0.1 m, 5.0 m]`.
- `proprioception`: float32 `[48]` in the exact V11 order.
- `teacher_scan`: float32 `[187]`, after the V11 scan clipping contract.
- `teacher_action_mean`: float32 `[12]` from frozen `model_4999.pt`.

Preparation tests enforce tensor shapes, finite depth handling, patch-token
count, observation reconstruction order, finite gradients and zero loss when
student targets equal teacher targets. The initial training objective is
`smooth_l1(predicted_scan, teacher_scan) + 0.25 * mse(student_action, teacher_action)`.

## Gates and deferred work

This milestone is complete when the descending video passes telemetry and media
checks, all repository tests pass, and a V12 synthetic forward/backward smoke run
produces finite loss and gradients. Real camera mounting/orientation, IsaacLab
depth-pair collection, dataset generation, and ViT optimization remain the next
execution phase; no claim of a trained vision policy is made here.
