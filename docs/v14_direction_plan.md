# V14 Direction Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Steps use checkbox syntax. User authorized autonomous V14 fixes; do not add approval gates or commits.

**Goal:** Correct stair measurement, diagnose direction loss, and train/evaluate a bounded V14 direction-recovery candidate on HPC.
**Architecture:** Optional fixed evaluation protocol plus trajectory summary; body-frame velocity command carries heading/cross-track feedback; V14 fine-tuning uses tighter yaw tracking.
**Tech Stack:** Python, NumPy/PyTorch, IsaacLab, RSL-RL, Slurm.
**Spec:** docs/v14_direction_design.md

## Global Constraints

- All training/simulation on HPC ws1 with qos=admin; GPU 1/job, maximum 2 concurrent.
- Keep original USD, legacy results, 0.99 torque guard, frozen V12 depth student and 235-dimensional observations.
- No automatic commits. Worktree v14-leg-collision already exists.

## Review Focus

- Direction sign for +y-forward and angles wrapping around pi: task 2 pure tensor test.
- Reset after termination contaminating trajectory: task 1 masked-series test.
- Body bounce falsely counted as ascent: task 1 use ground height test.
- Random terrain row mislabeled fixed difficulty or overlapping envs: task 1 HPC configured geometry audit.
- Heading reference reset every command resample, or standing spin: task 2 reset/standing tests and HPC smoke.

### Task 1: Fixed terrain protocol and directed trajectory metrics

**Files:** config/v14_path_metrics.py, config/v14_eval_protocol.py, config/evaluate_v13_torque.py, config/tests/test_v14_path_metrics.py, config/launch_v14_direction_eval.sbatch.
**Interfaces:** summary from positions/quaternions/ground/valid/alive arrays; evaluator flags `--protocol fixed_v2`, `--trajectory-output`, later `--heading-control`.
- [x] Write and run failing tests for actual forward ascent, sideways escape, body bounce, post-reset teleport and descent.
- [x] Implement NumPy summaries and fixed stair configuration; record final ground and path metrics.
- [x] Run pure tests; HPC baseline geometry audit and V14 5700 fixed_v2 evaluation.

### Task 2: Direction recovery command and adaptation configuration

**Files:** config/v14_heading_math.py, config/v14_heading_command.py, config/v14_leg_env_cfg.py, config/train_v14_leg_collision.py, config/tests/test_v14_heading.py.
**Interfaces:** `lane_yaw_rate(yaw, target_yaw, cross_track)` and `V14LaneVelocityCommandCfg`; CLI `--direction-focus`.
- [x] Write/run failing feedback sign, wrap, saturation, stationary and episode reference tests on HPC CPU.
- [x] Implement reset reference and command feedback; configure yaw reward std=.25, weight=7.5 and unique columns.
- [x] Smoke train and eval with heading feedback; verify 235 observations, guard, manifests and checkpoint.

### Task 3: HPC pilot and comparison

**Files:** config/launch_v14_direction_train.sbatch, reports/v14/direction/.
**Interfaces:** model checkpoint and fixed_v2 JSON/NPZ from tasks 1–2.
- [x] Run initial 300-update pilot; compare checkpoints on fixed stairs and flat against unchanged actor with same feedback.
- [x] Evaluate selected candidate on a second seed; record limits and retain only supported improvements.
- [x] Fresh code review and targeted fixes, archive exact jobs/configuration/results and report status.

## Outcome

Implemented and verified fixed geometry/trajectory evaluation and optional heading feedback.
Strong300, its early checkpoint, and softer100 adaptation all failed the stair gate and were rejected.
Feedback-only was compared with the unchanged actor on seeds42/43: survival49/64 vs48/64,
directed stair success0/64 in both, with worse lateral drift and capsule overlap despite better heading.
No replacement actor selected. Overall V14 gait/collision avoidance remains unresolved.
