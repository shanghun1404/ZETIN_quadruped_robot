# V14 Residual Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. User approved autonomous implementation; no extra approval gates or commits.

**Goal:** Add a bounded residual policy, verify zero-correction equivalence, and train/evaluate a frozen-base HPC pilot.
**Architecture:** Tensor geometry and stateful observation wrapper; existing depth policy produces frozen base actions; reward manager scores pre-reset clearance/corrections. Existing evaluator remains the measurement authority.
**Tech Stack:** PyTorch, RSL-RL, IsaacLab, Slurm, NumPy.
**Spec:** docs/v14_residual_design.md

## Global Constraints

- All training/simulation on HPC ws1, qos=admin,1GPU/job,max2 concurrent. Pure tests may run HPC CPU.
- Existing v14-leg-collision worktree, no commits, original USD/checkpoints preserved.
- Frozen235-input base and V12 student; max correction.05rad, scale.2, LPF25Hz,guard.99.
- No new external dependencies. Keep legacy and direction evaluation defaults.

## Review Focus

- Reset/double-read gap history must not invent approach velocities: task1 asynchronous reset test.
- RSL PPO must store sampled raw residual, while base observes actual combined action: task1 wrapper real-step test.
- Exploration cannot exceed joint correction bounds; zero correction must reproduce baseline: tasks1/3.
- Clearance reward must use pre-reset poses and physical units, not post-reset teleports: task2 integration and task1 geometry reference test.
- Wrong base/student/USD or wrong residual model must fail rather than silently evaluate: task2 binding test.

### Task 1: Residual math and observation/action wrapper

**Files:** config/v14_residual_math.py, config/v14_residual_policy.py, config/tests/test_v14_residual.py.
**Interfaces:** capsule_gaps(positions,quaternions)->[N,24]; bounded_delta(raw,max_rad); ResidualPolicyWrapper(depth_env,frozen_actor,max_rad=.05), get_observations/step/reset plus delegated VecEnv attributes.
- [x] Tests RED: geometry versus existing NumPy including parallel/coincident pairs; zero/bounded composition; frozen-base action history; reset and double reads.
- [x] Implement pure tensor geometry/history/composition and307-input wrapper. Expected HPC CPU tests pass.

### Task 2: Reward/configuration/training/evaluation integration

**Files:** config/v14_residual_env_cfg.py, config/train_v14_residual.py, config/evaluate_v13_torque.py, config/v14_residual_policy.py, config/launch_v14_residual.sbatch.
**Interfaces:** extra RewardTerms read current correction on base env; CLI mode zero/train/eval, sidecar manifest validates frozen inputs; evaluator residual checkpoint flag optional.
- [x] Binding failure and zero-init tests RED, implement checkpoint config validation and freeze verification.
- [x] Set unchanged yaw reward, residual PPO128×128/LR1e-4/std.05/entropy0; save zero checkpoint without training.
- [x] Extend evaluator with optional residual wrapper/checkpoint and trace bound evidence; preserve all existing metrics.
- [x] Relevant pure tests, syntax checks, fresh whole-change review and one Important fix pass.

### Task 3: HPC proof and bounded pilot

**Files:** reports/v14/residual/, docs/v14_residual_plan.md.
**Interfaces:** zero checkpoint → fixed_v2 evaluation → comparison gate → training → candidate evaluation.
- [x] Generate zero checkpoint on HPC; evaluate and compare baseline first-episode traces/torque/survival exactly or explain any numerical tolerance before proceeding.
- [x] Train300updates only after zero gate; frozen actor/student tensor hashes and max correction must hold.
- [x] Evaluate useful checkpoints on stairs/flat; extra seed for a candidate that passes preliminary gates. Archive jobs/manifests/reports and outcome.

## Outcome

Zero residual exactly reproduced fixed baseline.300-update training preserved frozen tensors and bounds.
Early model100 retained as a research candidate: stair survival49/64->55/64 on seeds42/43,
deep overlap.2469%->.2082%, flat32/32 and zero measured torque over-limit. Directed passes0/64,
any-overlap and lateral drift increased, forward progress decreased2.1%. No final V14 replacement adopted.
