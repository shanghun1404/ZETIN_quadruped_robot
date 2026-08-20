# V11 Descent and V12 ViT Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a verified 15.714 cm descending-stair rollout and prepare a tested Tiny ViT that reconstructs the V11 teacher's 187-value scan interface.

**Architecture:** Extend the existing V11 stair recorder with an explicit pyramid-stair descent scenario and symmetric physical traversal telemetry. Add a standalone PyTorch Tiny ViT that converts normalized `84×84` depth into a 187-value pseudo-scan, preserving the frozen V11 Actor's 235-value input boundary.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab 2.3, Gymnasium RecordVideo, pytest, ffmpeg/ffprobe.

## Global Constraints

- Do not mutate or resume `model_4999.pt`.
- Use level 5, exactly `0.15714285714285717 m`, for the descent video.
- Keep V11 ascent behavior backward compatible.
- Use only built-in PyTorch modules for the ViT preparation.
- Do not launch long ViT training or claim a trained camera policy.

---

### Task 1: Physical descending-stair contract

**Files:**
- Modify: `stage1_v10_5_v11_2026-08-12_KST/config/v9_env_cfg.py`
- Modify: `stage1_v10_5_v11_2026-08-12_KST/config/v11_video_geometry.py`
- Modify: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v11_video_geometry.py`

**Interfaces:**
- Produces: `STAIRS_DOWN_EVAL_TERRAIN_CFG` and `stair_descent_passed(...) -> bool`.

- [x] **Step 1: Write the failing descent tests**

Add literal checks that survival, `2.1 m` progress and at least `0.314 m` descent
are all required, and that the configured physical level remains 0.157142857 m.

- [x] **Step 2: Run the focused test and confirm RED**

Run `PYTHONPATH=. pytest -q tests/test_v11_video_geometry.py` remotely. Expect an
import failure for `stair_descent_passed`.

- [x] **Step 3: Implement the minimum geometry and terrain config**

Implement `stair_descent_passed` with the same survival/progress boundary as
ascent and `max_descent_after_approach_m >= 2 * step_height_m`. Configure the
descending generator with `MeshPyramidStairsTerrainCfg` and the existing size,
row, column, step-range and platform constants.

- [x] **Step 4: Run the focused test and confirm GREEN**

Run the same test and require zero failures.

### Task 2: Descending video scenario

**Files:**
- Modify: `stage1_v10_5_v11_2026-08-12_KST/config/play_v11_stairs_video.py`

**Interfaces:**
- Consumes: both stair terrain configs and physical traversal helpers.
- Produces: `--scenario stairs_up|stairs_down`, MP4 and telemetry JSON.

- [x] **Step 1: Add scenario selection and symmetric telemetry**

Use the inverted generator for ascent and pyramid generator for descent. Track
maximum positive height for ascent and maximum drop for descent only after 2.1 m
approach. Include `scenario`, `traversed`, and the scenario-specific vertical
metric in JSON and the artifact prefix.

- [x] **Step 2: Probe deterministic seeds without rendering**

Run level-5, 600-step, probe-only rollouts until one has `traversed=true`. Keep
all physics randomization identical to evaluation.

- [x] **Step 3: Render and validate the selected seed**

Render the selected seed from the side, copy MP4/JSON/log locally, decode the
whole file with ffmpeg, inspect a four-frame contact sheet, and assert telemetry.

### Task 3: Tiny ViT pseudo-scan student

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/v12_vision_student.py`
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/tests/test_v12_vision_student.py`

**Interfaces:**
- Produces: `normalize_depth`, `TinyDepthViT`, `reconstruct_teacher_observation`, and `distillation_loss`.

- [x] **Step 1: Write failing behavior tests**

Test invalid-depth replacement and literal normalization endpoints; require
`TinyDepthViT(depth)` shape `[batch,187]`; require reconstructed observation
shape `[batch,235]` with proprio first; require zero losses for exact targets;
and backpropagate a non-zero synthetic loss to finite ViT gradients.

- [x] **Step 2: Run the focused test and confirm RED**

Run `PYTHONPATH=. pytest -q tests/test_v12_vision_student.py` remotely. Expect a
module import failure.

- [x] **Step 3: Implement the minimum student**

Use `Conv2d(1,128,7,7)`, one learnable CLS token, one learnable 145-token
positional embedding, four `TransformerEncoderLayer(128,4,256,batch_first=True,
norm_first=True)` blocks, and `LayerNorm(128) -> Linear(128,187)`. Validate every
public tensor boundary and use Smooth-L1 scan loss plus `0.25` action MSE.

- [x] **Step 4: Run focused and full tests**

Require the focused tests and the full remote suite to pass with no failures.

### Task 4: Reproducible readiness artifact

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/config/smoke_v12_vision_student.py`
- Create at runtime: `stage1_v10_5_v11_2026-08-12_KST/final/vit_stage2/`

**Interfaces:**
- Consumes: the Tiny ViT public API.
- Produces: initial state dict and JSON with architecture, parameter count,
  tensor shapes, loss and gradient-finiteness results.

- [x] **Step 1: Run a deterministic synthetic forward/backward smoke test**

Use seed 42, batch size 4, depth `[4,1,84,84]`, proprio `[4,48]`, scan
`[4,187]`, and action `[4,12]`. Require finite outputs, finite loss and finite
non-zero gradients.

- [x] **Step 2: Archive and checksum**

Copy the state dict and report into `final/vit_stage2`, update the main report
without claiming training, regenerate `SHA256SUMS`, and verify every entry.
