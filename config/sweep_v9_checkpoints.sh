#!/usr/bin/env bash
# Score V9 checkpoints across training, on both scenarios, and rank them.
#
# Why this is a separate script rather than a stage inside queue_v9_pilot.sh:
# the queue was already running when the sweep was requested, and bash reads a
# script incrementally from disk, so editing a running one corrupts execution.
# This waits for the pilot to reach a terminal stage and then sweeps, which also
# avoids contending with the queue's own gate and video stages for the GPU.
#
# What the V8.1 sweep found, and why V9 needs the same treatment: duty balance
# and stance height declined monotonically from iteration 2000 to 8000 while
# phase sharpness improved, so the final checkpoint was the *worst* balanced one
# that still passed. V8 showed the same shape (1600 beat 8000). Judging V9 on its
# final checkpoint alone would repeat that mistake a third time.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/kang/miniforge3/envs/env_isaaclab/bin/python"
CONTROL_DIR="$ROOT/run_control"
# Which version to sweep. V9 was stopped at iteration 4142 with its terrain
# curriculum pinned, so V9.1 is the default; pass VER=v9 to re-examine that run.
VER="${V9_SWEEP_VERSION:-v9_3}"
EXPERIMENT="${V9_SWEEP_EXPERIMENT:-guide_${VER}_terrain}"
STATUS_FILE="$CONTROL_DIR/${VER}_sweep.status"
SWEEP_JSON="$CONTROL_DIR/${VER}_pilot_sweep.json"
VIDEO_DIR="$CONTROL_DIR/${VER}_best_video"
SPEED="0.5"

# Must be multiples of save_interval (100). rsl_rl writes model_0, model_100,
# ... plus a final model_<iters-1>; model_1999 is never written, and the first
# V8.1 sweep silently scored nothing by asking for exactly that.
#
# Four points rather than V8.1's three: V8.1's quality peaked at 2000 and the
# next sample was 4000, so where the peak actually sits between them was never
# measured. V9 costs 2.29 s/iteration against V8.1's 0.63, which makes a wrong
# choice of checkpoint expensive to discover late, and each extra sample costs
# about four minutes against a five-hour run.
SWEEP_ITERS="${V9_SWEEP_ITERS:-2000 3000 4000 6000}"

mkdir -p "$CONTROL_DIR"
exec >>"$CONTROL_DIR/${VER}_sweep.log" 2>&1

write_status() { printf '%s\ntime=%s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S %Z')" >"$STATUS_FILE"; }

cd "$ROOT"

# ---------------------------------------------------------------------------
# Wait for the pilot to finish
# ---------------------------------------------------------------------------
write_status "stage=waiting_for_pilot"
WAITED=0
MAX_WAIT="${V9_SWEEP_MAX_WAIT:-28800}"   # 8 h, well past a normal V9 run
while true; do
    stage="$(head -1 "$CONTROL_DIR/${VER}_pilot.status" 2>/dev/null || true)"
    case "$stage" in
        stage=passed|stage=gate_failed|stage=*_failed|stage=checkpoint_missing) break;;
    esac
    if (( WAITED >= MAX_WAIT )); then
        write_status "stage=timed_out_waiting_for_pilot"
        printf 'last_seen_pilot_stage=%s\n' "$stage" >>"$STATUS_FILE"
        exit 1
    fi
    sleep 60
    WAITED=$((WAITED + 60))
done
printf 'pilot_stage=%s\n' "$stage" >>"$STATUS_FILE"

PILOT_DIR="$({ find "logs/rsl_rl/$EXPERIMENT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' || true; } | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -z "$PILOT_DIR" ]]; then
    write_status "stage=pilot_dir_missing"; exit 1
fi
printf 'pilot_dir=%s\n' "$PILOT_DIR" >>"$STATUS_FILE"

# ---------------------------------------------------------------------------
# Score each checkpoint on both scenarios
# ---------------------------------------------------------------------------
write_status "stage=sweeping"
SWEEP_MISSING=""
for it in $SWEEP_ITERS; do
    ckpt="$PILOT_DIR/model_${it}.pt"
    # Report what could not be scored. A silently empty sweep is indistinguishable
    # from a sweep that ran and found nothing wrong.
    if [[ ! -f "$ckpt" ]]; then
        SWEEP_MISSING="$SWEEP_MISSING $it"
        printf 'sweep_missing_checkpoint=%s\n' "$ckpt" >>"$STATUS_FILE"
        continue
    fi
    "$PYTHON" evaluate_v9.py --headless --env "$VER" --checkpoint "$ckpt" --speed "$SPEED" \
        --output "$CONTROL_DIR/${VER}_gate_${it}.json" \
        >"$CONTROL_DIR/${VER}_gate_${it}.log" 2>&1 || true
    "$PYTHON" evaluate_v9.py --headless --env "$VER" --checkpoint "$ckpt" --speed "$SPEED" \
        --scenario stairs_up --seconds 12 \
        --output "$CONTROL_DIR/${VER}_stairs_gate_${it}.json" \
        >"$CONTROL_DIR/${VER}_stairs_gate_${it}.log" 2>&1 || true
done
if [[ -n "$SWEEP_MISSING" ]]; then
    printf 'sweep_incomplete=%s\n' "$SWEEP_MISSING" >>"$STATUS_FILE"
    echo "[WARN] sweep skipped absent checkpoints:$SWEEP_MISSING (must be multiples of save_interval)"
fi

# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------
write_status "stage=ranking"
BEST_ITER="$("$PYTHON" - "$CONTROL_DIR" "$SWEEP_JSON" "$VER" <<'PY'
import glob, json, os, re, sys

control_dir, sweep_out, ver = sys.argv[1], sys.argv[2], sys.argv[3]


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def iteration_of(path):
    m = re.search(r"_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else None


rows = {}
for path in glob.glob(os.path.join(control_dir, f"{ver}_gate_*.json")) + [
    os.path.join(control_dir, f"{ver}_pilot_gate.json")
]:
    d = load(path)
    if d is None:
        continue
    it = iteration_of(path)
    if it is None:
        it = -1  # the final checkpoint, whose filename carries no iteration
    p = d.get("phase_metrics", {})
    m = d.get("metrics", {})
    rows[it] = {
        "iteration": it,
        "rough_passed": bool(d.get("passed")),
        "rough_gates_passed": sum(1 for v in d.get("gates", {}).values() if v),
        "min_foot_contact": p.get("min_foot_contact_fraction"),
        "survival": m.get("survival_fraction"),
        "speed": m.get("mean_forward_speed_m_s"),
        "height": m.get("mean_base_height_m"),
        "foot_contact_fraction": p.get("foot_contact_fraction"),
    }

for path in glob.glob(os.path.join(control_dir, f"{ver}_stairs_gate_*.json")) + [
    os.path.join(control_dir, f"{ver}_stairs_gate.json")
]:
    d = load(path)
    if d is None:
        continue
    it = iteration_of(path)
    if it is None:
        it = -1
    row = rows.setdefault(it, {"iteration": it})
    s = d.get("stairs", {})
    row.update(
        {
            "stairs_passed": bool(d.get("passed")),
            "stairs_gates_passed": sum(1 for v in d.get("gates", {}).values() if v),
            "stairs_climb_success": s.get("climb_success_fraction"),
            "hardest_level_cleared": s.get("hardest_level_cleared"),
            "stairs_survival": d.get("metrics", {}).get("survival_fraction"),
        }
    )

ranked = sorted(
    rows.values(),
    key=lambda r: (
        r.get("rough_gates_passed") or 0,
        r.get("stairs_gates_passed") or 0,
        r.get("min_foot_contact") or 0.0,
        r.get("stairs_climb_success") or 0.0,
    ),
    reverse=True,
)
json.dump({"ranked": ranked}, open(sweep_out, "w"), indent=2)
if ranked:
    print(ranked[0]["iteration"])
PY
)"
printf 'best_iteration=%s\nsweep_json=%s\n' "$BEST_ITER" "$SWEEP_JSON" >>"$STATUS_FILE"

# ---------------------------------------------------------------------------
# Record the winner, if it is not the final checkpoint the queue already filmed
# ---------------------------------------------------------------------------
if [[ -n "${BEST_ITER:-}" && "$BEST_ITER" != "-1" && -f "$PILOT_DIR/model_${BEST_ITER}.pt" ]]; then
    write_status "stage=recording_best"
    printf 'best_iteration=%s\n' "$BEST_ITER" >>"$STATUS_FILE"
    BEST="$PILOT_DIR/model_${BEST_ITER}.pt"
    "$PYTHON" play_v9_video.py --env "$VER" --checkpoint "$BEST" --view side --speed "$SPEED" \
        --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/${VER}_best_side.log" 2>&1 || true
    "$PYTHON" play_v9_video.py --env "$VER" --checkpoint "$BEST" --scenario stairs_up --view side --speed "$SPEED" \
        --video_length 500 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/${VER}_best_stairs.log" 2>&1 || true
fi

write_status "stage=done"
printf 'best_iteration=%s\nsweep_json=%s\nvideo_dir=%s\n' "$BEST_ITER" "$SWEEP_JSON" "$VIDEO_DIR" >>"$STATUS_FILE"
exit 0
