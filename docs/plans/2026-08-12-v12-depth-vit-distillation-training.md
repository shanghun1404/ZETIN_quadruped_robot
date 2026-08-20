# V12 Depth ViT Distillation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate the prepared Tiny ViT on synchronized IsaacLab depth and immutable V11 teacher targets.

**Architecture:** A tiled depth camera supplies normalized 84×84 frames. The Tiny ViT predicts the V11 187-value terrain scan, which is concatenated after unchanged 48-value proprioception and passed through an exact frozen reconstruction of the V11 actor. Sharded offline distillation is followed by closed-loop evaluation and at most one bounded DAgger recovery pass.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab 2.3, RSL-RL, pytest, RTX 5070 CUDA, JSON and PyTorch tensor shards.

## Global Constraints

- Never modify or resume `final/checkpoint/model_4999.pt`.
- Camera input is one normalized `84×84` depth frame over 0.1–5.0 m.
- Preserve the literal V11 `48 + 187 = 235` observation order.
- Train only Tiny ViT parameters; the `235→512→256→128→12` teacher actor is frozen.
- Use train seed 42, validation seed 142, and optional DAgger seed 242.
- Limit initial training to 20 epochs and optional recovery to 10 epochs.
- Do not claim hardware performance from simulator training.
- The source tree is not a Git repository; checkpoint source and artifacts with SHA-256 instead of commits.

---

### Task 1: Dataset and frozen-actor tensor contracts

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/v12_dataset.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/v12_teacher_actor.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_dataset.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_teacher_actor.py`

**Interfaces:**
- Produces: `validate_sample_batch(batch)`, `load_dataset_shards(path)`, `FrozenV11Actor`, and `load_frozen_v11_actor(checkpoint)`.
- Consumes: shard dictionaries with `depth`, `proprioception`, `teacher_scan`, and `teacher_action`.

- [x] **Step 1: Write failing dataset tests**

Require literal trailing shapes `[1,84,84]`, `[48]`, `[187]`, and `[12]`, equal
batch lengths, finite values, depth in `[0,1]`, manifest sample-count agreement,
deterministic shard ordering, and concatenation across two small real `.pt`
shards in a pytest temporary directory.

- [x] **Step 2: Run dataset tests and confirm RED**

Run `PYTHONPATH=. pytest -q tests/test_v12_dataset.py`; expect
`ModuleNotFoundError: v12_dataset`.

- [x] **Step 3: Implement dataset validation and loading**

Use a fixed schema mapping field names to trailing shapes. Load `manifest.json`,
sort its explicit shard list, verify each shard SHA when present, validate each
batch, concatenate each field, and reject a final count different from
`manifest["sample_count"]`.

- [x] **Step 4: Write failing frozen-actor tests**

Build a synthetic checkpoint with exact RSL-RL keys, require output shape
`[B,12]`, compare against a manually evaluated ELU stack, require all actor
parameters frozen, require gradient propagation to a `[B,235]` input, and reject
missing or wrong-shaped weights.

- [x] **Step 5: Run actor tests and confirm RED**

Run `PYTHONPATH=. pytest -q tests/test_v12_teacher_actor.py`; expect
`ModuleNotFoundError: v12_teacher_actor`.

- [x] **Step 6: Implement the exact actor and verify GREEN**

Create four linear layers of sizes `235→512→256→128→12`, load checkpoint
`actor.0/2/4/6` weights and biases exactly, apply ELU after the first three,
freeze parameters, and run both focused test files to zero failures.

### Task 2: Tiled camera and synchronized collector

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/v12_camera.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/collect_v12_dataset.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_camera_contract.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_collection_contract.py`

**Interfaces:**
- Produces: `make_v12_depth_camera_cfg(step_dt)`, `camera_depth_nchw(sensor)`, and CLI dataset collection.
- Consumes: V11 environment, immutable teacher checkpoint, split/seed/sample arguments.

- [x] **Step 1: Write and run RED camera contract tests**

Assert the source-facing constants `84`, `0.1`, `5.0`, mount position
`(0.0,0.30,0.10)`, ROS quaternion `(0.573576,-0.819152,0,0)`, and output type
`distance_to_image_plane`. Run the focused tests and require the missing-module
or missing-interface failure before implementation.

- [x] **Step 2: Implement camera conversion**

Construct `TiledCameraCfg` under `{ENV_REGEX_NS}/Robot/base/base/front_depth`,
read `[N,84,84,1]`, move the channel first, normalize with `normalize_depth`,
and reject all other shapes.

- [x] **Step 3: Write and run RED collection contract tests**

Require CLI literals for 128 environments, 50 warm-up steps, stride 4,
2,048-sample shards, teacher/student rollout choices, pre-action sampling, atomic
temporary-shard rename, manifest statistics, and teacher equivalence tolerance
`1e-6`.

- [x] **Step 4: Implement the collector**

Launch cameras, distribute terrain levels with `arange(num_envs) % 8`, disable
observation corruption and curriculum, load both RSL-RL policy and manual frozen
actor, assert equivalence, warm up, record synchronized pre-action tensors, and
flush float16 shards atomically with SHA-256 entries in `manifest.json`. Teacher
mode applies teacher actions. Student mode applies the supplied ViT through the
frozen actor while still recording teacher targets.

- [x] **Step 5: Run live one-shard camera probe**

Collect 2,048 train-probe samples with 32 environments and require exact shapes,
finite nonconstant depth, depth values spanning at least 0.05 normalized units,
manual actor error at most `1e-6`, valid manifest hashes, and a saved depth
preview tensor.

### Task 3: Deterministic distillation trainer

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/v12_training.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/train_v12_distillation.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_training.py`

**Interfaces:**
- Produces: `distillation_metrics`, `checkpoint_is_finite`, `EarlyStopper`, and training CLI.
- Consumes: validated train/validation datasets, initial ViT state, V11 checkpoint.

- [x] **Step 1: Write failing training tests**

Require exact-zero metrics for exact targets, known MAE/RMSE values for small
tensors, rejection of non-finite checkpoints, patience-5 early stopping only
after five non-improving epochs, and a real optimizer step that decreases loss
on a repeated eight-sample synthetic dataset.

- [x] **Step 2: Run focused tests and confirm RED**

Run `PYTHONPATH=. pytest -q tests/test_v12_training.py`; expect a missing module.

- [x] **Step 3: Implement metrics and training loop**

Load validated data to CPU, transfer shuffled batches to CUDA, use AdamW
`lr=3e-4, weight_decay=1e-4`, clip gradient norm at 1.0, and evaluate without
gradients. Save `best.pt`, `last.pt`, JSONL epoch metrics, and `training_report.json`
with initial/best metrics and SHA-256 values. Only lower validation total replaces
the best checkpoint.

- [x] **Step 4: Run focused tests and full static suite**

Require all new training tests and the complete remote pytest suite to pass.

### Task 4: Full teacher-rollout collection and initial training

**Files:**
- Create at runtime: `/home/kang/quadruped_training_smooth_v11/datasets/v12/train/`
- Create at runtime: `/home/kang/quadruped_training_smooth_v11/datasets/v12/validation/`
- Create at runtime: `/home/kang/quadruped_training_smooth_v11/runs/v12_distillation_initial/`

**Interfaces:**
- Produces: 65,536 train samples, 8,192 validation samples, and selected `best.pt`.

- [x] **Step 1: Collect the validation split**

Run the collector with seed 142, 128 environments, 8,192 samples, teacher mode,
headless cameras enabled, and capture the complete log. Validate every shard and
manifest entry before proceeding.

- [x] **Step 2: Collect the training split**

Run the collector with seed 42, 128 environments, 65,536 samples, teacher mode,
and the same camera/physics contract. Validate every shard and manifest entry.

- [x] **Step 3: Train the initial student**

Run at most 20 epochs with batch size 128 and patience 5. Require finite state,
successful checkpoint reload, and improvement in both validation scan MAE and
action MAE over initialization.

### Task 5: Closed-loop camera-policy evaluator

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/v12_eval_metrics.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/evaluate_v12_vision.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_eval_metrics.py`

**Interfaces:**
- Produces: rough/stairs JSON per seed with survival, speed, progress, traversal, scan MAE, and action MAE.
- Consumes: selected ViT checkpoint, frozen teacher actor, live camera and privileged observations used only as labels.

- [x] **Step 1: Write and run RED metric tests**

Require correct alive masking, online error accumulation, rough gates
`survival>=0.80` and `speed>=0.25`, and stair traversal using the existing
physical ascent/descent helpers. Confirm expected missing-interface failures.

- [x] **Step 2: Implement metrics and evaluator**

For every step, infer scan from camera, concatenate only live proprioception,
apply the frozen actor, retain privileged scan/action solely for error metrics,
and step the environment with student actions. Support rough level 4,
stairs-up level 5, and stairs-down level 5 with 32 camera environments.

- [x] **Step 3: Evaluate seeds 42–44**

Run all three scenarios for three seeds. Aggregate without dropping failed or
terminated environments and write `initial_closed_loop_summary.json`.

### Task 6: Conditional one-pass DAgger recovery

**Files:**
- Create at runtime only if required: `/home/kang/quadruped_training_smooth_v11/datasets/v12/dagger/`
- Create at runtime only if required: `/home/kang/quadruped_training_smooth_v11/runs/v12_distillation_dagger/`

**Interfaces:**
- Consumes: failed initial closed-loop summary and initial `best.pt`.
- Produces: at most one retrained checkpoint and repeated closed-loop summary.

- [x] **Step 1: Apply the conditional gate**

Skip this task when rough survival/speed and at least one seed per stair direction
meet the operational targets. Otherwise require improved offline metrics before
classifying the failure as distribution drift.

- [x] **Step 2: Collect and train once when gated**

Collect 32,768 samples with seed 242 under student actions, combine their shard
paths with the original training manifest, resume the selected ViT for at most
10 epochs, and evaluate exactly the same nine closed-loop cases. Do not repeat.

### Task 7: Verification and archive

**Files:**
- Modify: `stage1_v10_5_v11_2026-08-12_KST/REPORT.md`
- Modify: `stage1_v10_5_v11_2026-08-12_KST/SHA256SUMS`
- Create at runtime: `stage1_v10_5_v11_2026-08-12_KST/final/vit_stage2_trained/`

**Interfaces:**
- Produces: reproducible local training handoff and explicit limitation report.

- [x] **Step 1: Fresh verification**

Run the full pytest suite, dataset manifest verification, selected checkpoint
reload and finite-state scan, and all available closed-loop result validation.
Verify the V11 teacher SHA remains
`53a6a5002f4cafb87889d777fcc567157245c786f82376db08962c9227acc2b7`.

- [x] **Step 2: Archive compact artifacts**

Copy source, train/validation manifests, per-shard hashes, logs, epoch reports,
selected/last checkpoints, and evaluation JSON into `final/vit_stage2_trained`.
Do not copy the approximately 1 GB depth tensors locally unless needed to resolve
a verification failure.

- [x] **Step 3: Report and checksum**

Document dataset counts, actual epochs, initial/best errors, closed-loop results,
whether DAgger ran, and all failures or limitations. Regenerate and verify every
`SHA256SUMS` entry, confirm remote/local source parity, and remove the temporary
SSH credential helper.
