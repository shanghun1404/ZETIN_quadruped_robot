# V10.5 / V11 Evidence-Gated Locomotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a multi-seed-validated flat gait model and a scanner-aware rough/stairs model for `robot_v2`.

**Architecture:** Add pure contact-phase metrics to the existing evaluator, gate a single bounded V10.5 reward change on the measured result, then expand the selected 48-input policy to a 235-input V11 policy whose new scan columns start at zero. Run physical probes, smoke tests, full pilots, multi-checkpoint gates and a stairs-only evaluation on the remote RTX 5070.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab/Isaac Sim 5.1, Gymnasium, rsl_rl PPO, pytest, Bash, SSH/rsync.

## Global Constraints

- Preserve all historical remote runs; work in `/home/kang/quadruped_training_smooth_v11`.
- Preserve base `+y` as the `robot_v2` forward axis.
- Preserve `SPAWN_ROOT_FRAME_Z_M = clearance + 0.445 m` and verify first-step support <= 0.5 body weight.
- Use seeds 42, 43, 44, 45 and 46 for evaluation.
- Do not launch a full run unless the full static suite, simulator probe and 3-iteration smoke test pass.
- Keep V10.5 to one reward-table change from V10.3.
- Do not silently skip missing checkpoints, seeds or invalid metric rows.
- The source folders are not Git repositories; retain SHA256 manifests at each archived milestone in place of commit steps.

---

### Task 1: Isolate and archive the current remote baseline

**Files:**
- Create: `/home/kang/quadruped_training_smooth_v11/` on the remote host
- Create: `stage1_v10_5_v11_2026-08-12_KST/baseline/`
- Create: `stage1_v10_5_v11_2026-08-12_KST/SHA256SUMS.baseline`

**Interfaces:**
- Consumes: remote V10.3 and V10.4 source, checkpoints and gate JSON
- Produces: isolated working tree and immutable local baseline evidence

- [ ] **Step 1: Copy code without the 3.2 GB historical logs**

  Use `rsync -a` on the remote host, excluding `logs*`, `run_control*`, caches and rendered media, then create empty `logs/rsl_rl` and `run_control` directories.

- [ ] **Step 2: Copy baseline evidence locally**

  Copy V10.3 checkpoints 1000/2000/2999, V10.4 checkpoints 1000/2000/2999, their gate JSON, status files and training summaries into the dated archive.

- [ ] **Step 3: Verify isolation**

  Run `diff -u` on the source files needed by V10.3/V10.4 and ensure no training process references the new directory.

- [ ] **Step 4: Record checksums**

  Run `sha256sum` over copied sources, checkpoints and JSON and store the sorted manifest.

### Task 2: Add pure per-environment contact-phase metrics

**Files:**
- Create: `gait_phase_metrics.py`
- Create: `tests/test_gait_phase_metrics.py`

**Interfaces:**
- Consumes: `torch.Tensor` contacts shaped `[T, N, 4]`
- Produces: `contact_phase_metrics(contacts) -> dict[str, object]`

- [ ] **Step 1: Write failing tensor tests**

```python
def test_perfect_trot_has_negative_same_pair_and_positive_diagonals():
    a = torch.tensor([0, 0, 1, 1] * 20, dtype=torch.float32)
    b = 1.0 - a
    contacts = torch.stack((a, b, b, a), dim=-1)[:, None, :]
    result = contact_phase_metrics(contacts)
    assert result["pairs"]["FL-FR"]["median"] == pytest.approx(-1.0)
    assert result["pairs"]["RL-RR"]["median"] == pytest.approx(-1.0)
    assert result["pairs"]["FL-RR"]["median"] == pytest.approx(1.0)
    assert result["pairs"]["FR-RL"]["median"] == pytest.approx(1.0)

def test_constant_foot_is_invalid_not_a_fake_zero_correlation():
    contacts = torch.zeros((40, 3, 4))
    result = contact_phase_metrics(contacts)
    assert result["pairs"]["RL-RR"]["valid_fraction"] == 0.0

def test_synchronous_transitions_are_counted():
    contacts = torch.tensor([[[0, 0, 1, 1]], [[1, 1, 0, 0]], [[0, 0, 1, 1]]], dtype=torch.float32)
    result = contact_phase_metrics(contacts)
    assert result["transition_coincidence"]["front"] == pytest.approx(1.0)
    assert result["transition_coincidence"]["rear"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests and confirm import failure**

  Run: `pytest -q tests/test_gait_phase_metrics.py`
  Expected: FAIL because `gait_phase_metrics` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

```python
FEET = ("FL", "FR", "RL", "RR")
PAIRS = (("FL", "FR"), ("RL", "RR"), ("FL", "RR"), ("FR", "RL"), ("FL", "RL"), ("FR", "RR"))

def contact_phase_metrics(contacts: torch.Tensor) -> dict[str, object]:
    if contacts.ndim != 3 or contacts.shape[-1] != 4:
        raise ValueError("contacts must have shape [time, environments, 4]")
    values = contacts.to(dtype=torch.float32)
    centered = values - values.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=0)
    index = {name: i for i, name in enumerate(FEET)}
    pairs = {}
    for left, right in PAIRS:
        li, ri = index[left], index[right]
        denom = norms[:, li] * norms[:, ri]
        corr = (centered[:, :, li] * centered[:, :, ri]).sum(dim=0) / denom.clamp(min=1e-8)
        corr = torch.where(denom > 1e-8, corr, torch.full_like(corr, torch.nan))
        finite = corr[torch.isfinite(corr)]
        pairs[f"{left}-{right}"] = {
            "median": None if finite.numel() == 0 else float(finite.median()),
            "p10": None if finite.numel() == 0 else float(torch.quantile(finite, 0.10)),
            "p90": None if finite.numel() == 0 else float(torch.quantile(finite, 0.90)),
            "valid_fraction": float(finite.numel() / corr.numel()),
        }
    before, after = values[:-1].bool(), values[1:].bool()
    landing = ~before & after
    lifting = before & ~after

    def coincidence(i: int, j: int) -> float:
        same_event = (landing[:, :, i] & landing[:, :, j]) | (lifting[:, :, i] & lifting[:, :, j])
        any_event = landing[:, :, i] | landing[:, :, j] | lifting[:, :, i] | lifting[:, :, j]
        return float(same_event.sum() / any_event.sum().clamp(min=1))

    return {
        "pairs": pairs,
        "transition_coincidence": {
            "front": coincidence(0, 1),
            "rear": coincidence(2, 3),
        },
    }
```

  Implement finite-value `median`, `p10`, `p90`, `valid_fraction`, and exact
  state-transition coincidence for front and rear pairs.

- [ ] **Step 4: Run tests**

  Run: `pytest -q tests/test_gait_phase_metrics.py`
  Expected: PASS.

- [ ] **Step 5: Record checksums**

  Append the new files to `SHA256SUMS.implementation`.

### Task 3: Extend the evaluator and aggregate five seeds

**Files:**
- Modify: `evaluate_v8.py`
- Create: `evaluation_cli.py`
- Create: `aggregate_multiseed_gates.py`
- Create: `queue_v10_5_baseline_audit.sh`
- Create: `tests/test_multiseed_evaluator_contract.py`

**Interfaces:**
- Consumes: checkpoint, `--seed`, `--warmup-seconds`, raw contact tensor
- Produces: one JSON per checkpoint/seed plus `baseline_multiseed_summary.json`

- [ ] **Step 1: Write failing behavior tests**

```python
def test_evaluator_uses_requested_seed():
    parser = argparse.ArgumentParser()
    add_metric_arguments(parser)
    args = parser.parse_args(["--seed", "46", "--warmup-seconds", "1.5"])
    assert args.seed == 46
    assert args.warmup_seconds == 1.5

def test_phase_payload_contains_direct_contact_phase():
    contacts = torch.tensor([[[1, 0, 0, 1]], [[0, 1, 1, 0]]], dtype=torch.float32)
    payload = build_contact_phase_payload(contacts)
    assert payload["pairs"]["RL-RR"]["median"] == pytest.approx(-1.0)
```

- [ ] **Step 2: Run the contract test and verify failure**

  Run: `pytest -q tests/test_multiseed_evaluator_contract.py`

- [ ] **Step 3: Add seed and direct metrics**

  Implement the reusable pure parser helper with `--seed` default 42. Make the
  simulator entry point set `env_cfg.seed`, call `contact_phase_metrics(contact_t)`,
  store it under `phase_metrics.contact_phase`, and retain all historical fields.
  The remote baseline audit is the integration test for the simulator entry point.

- [ ] **Step 4: Implement strict aggregation**

  `aggregate_multiseed_gates.py` must reject duplicate/missing seeds, differing
  checkpoint identities, invalid JSON, or contact valid-fraction below 0.90. It
  writes per-seed rows, median across seeds and `rear_weakness_confirmed`, which
  is true only when at least four seeds exceed either design threshold.

- [ ] **Step 5: Run unit and full static suites**

  Run: `pytest -q tests/test_gait_phase_metrics.py tests/test_multiseed_evaluator_contract.py`
  Run: `pytest -q`
  Expected: all pass.

- [ ] **Step 6: Run the remote baseline audit**

  Evaluate V10.3 and V10.4 checkpoints 1000/2000/2999 for seeds 42-46 with
  `--warmup-seconds 1.0`. Preserve all 30 raw JSON files and aggregate them.

- [ ] **Step 7: Record the decision**

  Write `results/v10_5_baseline_decision.json` containing the measured decision,
  selected baseline checkpoint, thresholds and exact source JSON paths.

### Task 4: Implement and probe the single V10.5 reward change

**Files:**
- Create: `v10_5_math.py`
- Create: `v10_5_mdp.py`
- Create: `v10_5_flat_env_cfg.py`
- Create: `v10_5_flat_ppo_cfg.py`
- Create: `train_v10_5_flat.py`
- Create: `probe_v10_5_flat.py`
- Create: `queue_v10_5_flat_pilot.sh`
- Create: `tests/test_v10_5_phase_contract.py`

**Interfaces:**
- Consumes: four contact flags and planar moving mask
- Produces: bounded `[N]` `same_pair_contact_overlap(contacts, moving)` penalty

- [ ] **Step 1: Write failing math tests**

```python
def test_alternating_pairs_cost_zero():
    contacts = torch.tensor([[1, 0, 0, 1]], dtype=torch.float32)
    assert same_pair_contact_overlap(contacts, torch.tensor([True])).item() == 0.0

def test_synchronized_pairs_cost_one():
    contacts = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    assert same_pair_contact_overlap(contacts, torch.tensor([True])).item() == 1.0

def test_standing_command_masks_penalty():
    contacts = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)
    assert same_pair_contact_overlap(contacts, torch.tensor([False])).item() == 0.0
```

- [ ] **Step 2: Verify the tests fail, implement, and verify they pass**

  The implementation validates four columns, converts contacts to bool, averages
  equality of `(FL, FR)` and `(RL, RR)`, and multiplies by the moving mask.

- [ ] **Step 3: Add the environment delta**

  Subclass `GuideV2FlatSpawnFixedEnvCfg`; add exactly one reward named
  `same_pair_overlap`, weight `-1.0`, threshold `5.0 N`, and the existing
  axis-independent moving mask. Do not alter any other reward or environment
  setting.

- [ ] **Step 4: Add live configuration contract checks**

  In `probe_v10_5_flat.py`, instantiate the real IsaacLab environment and assert
  the reward dictionary equals V10.3 plus one key, the weight magnitude is below
  the duty floor and gait reward, the observation remains 48, and PPO behavior
  differs only in experiment/run names.

- [ ] **Step 5: Run full static tests**

  Run: `pytest -q`
  Expected: all pass.

- [ ] **Step 6: Run simulator probe**

  The probe forces alternating, synchronized and standing cases through the live
  reward term; verifies spawn, axis, mass, active rewards and observation width;
  exits nonzero on any mismatch.

- [ ] **Step 7: Run the 3-iteration smoke test**

  Run V10.5 with 4096 environments and confirm finite reward and a saved model.

### Task 5: Train and select V10.5

**Files:**
- Create: `run_control/v10_5_flat_*`
- Create: `results/v10_5_multiseed_summary.json`

**Interfaces:**
- Consumes: V10.5 config at seed 42
- Produces: accepted checkpoint or explicit rejection with evidence

- [ ] **Step 1: Launch 3000 iterations**

  Run from scratch at 4096 environments, seed 42. Save every 100 iterations.

- [ ] **Step 2: Score checkpoints 1000/2000/2999 at seeds 42-46**

  Use 1.0 s warmup and preserve raw JSON.

- [ ] **Step 3: Apply the acceptance gates**

  Require speed >= 0.45 m/s, survival >= 0.95, contact and swing >= 0.35 on every
  foot, valid contact phase >= 0.90 and rear contact median <= -0.20 in at least
  four seeds. Rank accepted checkpoints by worst-seed survival, worst-seed phase,
  speed and duty margin.

- [ ] **Step 4: Render selected and final checkpoints**

  Render side and rear-quarter videos with base `+y` commands.

- [ ] **Step 5: Archive evidence and checksums**

  Copy source, logs, JSON, selected checkpoint and videos to the local dated
  archive and update the manifest.

### Task 6: Add scanner-aware V11 and exact checkpoint expansion

**Files:**
- Create: `v11_terrain_env_cfg.py`
- Create: `v11_terrain_ppo_cfg.py`
- Create: `expand_v11_checkpoint.py`
- Create: `train_v11_terrain.py`
- Create: `probe_v11_terrain.py`
- Create: `tests/test_expand_v11_checkpoint.py`

**Interfaces:**
- Consumes: selected `[512,48]` flat checkpoint
- Produces: `[512,235]` bootstrap checkpoint with zero scan columns

- [ ] **Step 1: Write failing expansion tests**

```python
def test_expansion_preserves_old_columns_and_zeros_scan_columns(tmp_path):
    old = torch.arange(512 * 48, dtype=torch.float32).reshape(512, 48)
    expanded = expand_input_weight(old, new_width=235)
    assert torch.equal(expanded[:, :48], old)
    assert torch.count_nonzero(expanded[:, 48:]) == 0
```

- [x] **Step 2: Implement checkpoint expansion**

  Expand only `actor.0.weight` and `critic.0.weight`; copy every other tensor
  byte-for-byte; widen the matching Adam moments with zero columns; preserve the
  optimizer learning rate and iteration for exact resume; and record source
  SHA256 and widths in `infos.bootstrap`. The reset-state version was rejected
  after its first PPO update changed a 100%-survival policy into a 0%-survival
  policy.

- [ ] **Step 3: Add V11 terrain configuration**

  Subclass the selected flat environment. Enable `ROUGH_TERRAINS_CFG`, restore
  `terrain_levels_vel`, set yaw to `[-0.2, 0.2]`, set terrain root spawn to
  `0.32 + 0.445 = 0.765 m`, attach a +y-forward 17x11 ray grid, and expose
  `v10_mdp.relative_height_scan` clipped to `[-1,1]`.

- [ ] **Step 4: Add V11 live contract checks**

  In `probe_v11_terrain.py`, instantiate the real environment and check observation
  width 235, scanner count 187, +y offset, frame-free relief values, corrected
  spawn, live curriculum and source reward weights.

- [ ] **Step 5: Run static tests**

  Run: `pytest -q`
  Expected: all pass.

- [ ] **Step 6: Run the physical/scanner probe**

  Verify first-step support, scan finiteness/saturation/range, local relief on
  non-flat tiles, mass, axis, curriculum, active rewards and observation width.

- [ ] **Step 7: Verify zero-scan network equivalence**

  Feed identical 48-column tensors to the source model and the same tensors with
  187 appended zeros to the expanded model. Require actor mean and critic output
  max absolute error <= `1e-6`.

- [x] **Step 8: Run the bootstrap smoke test**

  Load the expanded checkpoint with optimizer state and the optimizer's adaptive
  learning rate restored. Confirm iteration continues from 2000, rewards are
  finite, and the five-update model retains 100% survival.

### Task 7: Train V11 and run rough/stairs gates

**Files:**
- Create: `queue_v11_terrain_pilot.sh`
- Create: `evaluate_v11_terrain.py`
- Create: `run_control/v11_*`
- Create: `results/v11_rough_multiseed.json`
- Create: `results/v11_stairs_multiseed.json`

**Interfaces:**
- Consumes: expanded selected flat checkpoint
- Produces: selected V11 checkpoint and direct rough/stairs evidence

- [ ] **Step 1: Launch the monitored 4000-iteration pilot**

  Score/save 1000/2000/3000/3999. Run the terrain trend monitor throughout and
  write an alert if the design thresholds fail.

- [ ] **Step 2: Run rough evaluation at seeds 42-46**

  Record survival, speed, displacement, terrain level, direct contact phase and
  HFE proxy. Require survival >= 0.90 and speed >= 0.30 m/s.

- [ ] **Step 3: Run stairs-only evaluation at seeds 42-46**

  Freeze the curriculum across eight ascending-stair levels spanning 0.05-0.20 m.
  Record per-level sample count, survival, height gain and climb success. Require
  survival >= 0.80, overall climb success >= 0.50 and >= 0.50 success at a step
  height >= 0.15 m.

- [ ] **Step 4: Compare against blind V10.4**

  Run the identical rough/stairs evaluator on V10.4. Attribute any regression to
  scanner transfer only when the blind baseline passes the same gate.

- [ ] **Step 5: Select and render the best checkpoint**

  Rank by stairs gate, rough survival, forward progress, terrain-level trend and
  direct contact phase. Render rough, side-stairs and rear-quarter-stairs videos.

- [ ] **Step 6: Archive final evidence**

  Copy configs, tests, logs, all gate JSON, selected checkpoint, videos and the
  final SHA256 manifest into `stage1_v10_5_v11_2026-08-12_KST/`.

### Task 8: Final verification and report

**Files:**
- Create: `stage1_v10_5_v11_2026-08-12_KST/REPORT.md`
- Create: `stage1_v10_5_v11_2026-08-12_KST/SHA256SUMS`

**Interfaces:**
- Consumes: all run evidence
- Produces: reproducible result report and recommended checkpoint

- [ ] **Step 1: Run the complete static suite from a clean process**

  Run: `pytest -q`

- [ ] **Step 2: Verify every status and JSON artifact**

  Confirm each required seed/checkpoint exists, every JSON parses, all reported
  checkpoints match their SHA256 and no status remains `running`.

- [ ] **Step 3: Write the evidence-led report**

  Separate facts from inference, include failed gates and regressions, compare
  V10.3/V10.4/V10.5/V11, and recommend a checkpoint only if its gates pass.

- [ ] **Step 4: Verify the archive manifest**

  Run `sha256sum -c SHA256SUMS` from the archive root; expected: every file OK.
