#!/usr/bin/env bash
# V10.4-terrain pilot: V8's Spot recipe plus a contact-duty floor.
#
# Differs from queue_v8_pilot.sh in one operational way: it gates *several*
# checkpoints, not just the last one. V8 finished 8000 iterations with a worse
# gait than it had at 1600, and that was only discovered after the fact because
# nothing along the way was measured. Here every sampled checkpoint is scored and
# the best one is reported alongside the final one, so training drift is visible
# in the run's own output.
#
# Evaluation reuses evaluate_v8.py / play_v8_video.py deliberately: V10.4-terrain changes
# the ground and the yaw band, so the observation and action spaces are identical and a
# separate evaluator would be a copy that could silently drift from V8's.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/kang/miniforge3/envs/env_isaaclab/bin/python"
CONTROL_DIR="$ROOT/run_control"
STATUS_FILE="$CONTROL_DIR/v10_4_terrain_pilot.status"
GATE_JSON="$CONTROL_DIR/v10_4_terrain_pilot_gate.json"
SWEEP_JSON="$CONTROL_DIR/v10_4_terrain_pilot_sweep.json"
VIDEO_DIR="$CONTROL_DIR/v10_4_terrain_pilot_video"
ITERS="${V10_4_TERRAIN_ITERS:-8000}"
SPEED="0.5"
# Checkpoints to score in addition to the final one, chosen to bracket the
# 1600-vs-8000 regression V8 showed.
#
# These MUST be multiples of the agent's save_interval (100). rsl_rl writes
# model_0, model_100, ... model_7900 plus a final model_<iters-1>; asking for
# model_1999 names a file that is never written. The first version of this
# script did exactly that and the sweep silently scored nothing, which is the
# same blindness it exists to remove -- hence the loud check below.
SWEEP_ITERS="${V10_4_TERRAIN_SWEEP_ITERS:-2000 4000 6000}"

mkdir -p "$CONTROL_DIR"
exec >>"$CONTROL_DIR/v10_4_terrain_pilot_queue.log" 2>&1

write_status() {
    printf '%s\ntime=%s\niters=%s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$ITERS" >"$STATUS_FILE"
}

cd "$ROOT"
write_status "stage=running_probe"
if ! "$PYTHON" probe_v10_4_terrain.py --headless --num_envs 64 >"$CONTROL_DIR/v10_4_terrain_probe.log" 2>&1; then
    write_status "stage=probe_failed"; exit 1
fi

write_status "stage=running_smoke"
if ! "$PYTHON" train_v10_4_terrain.py --headless --num_envs 4096 --max_iterations 3 >"$CONTROL_DIR/v10_4_terrain_smoke.log" 2>&1; then
    write_status "stage=smoke_failed"; exit 1
fi

write_status "stage=running_training"
if ! "$PYTHON" train_v10_4_terrain.py --headless --num_envs 4096 --max_iterations "$ITERS" >"$CONTROL_DIR/v10_4_terrain_train.log" 2>&1; then
    write_status "stage=training_failed"; exit 1
fi

PILOT_DIR="$({ find logs/rsl_rl/guide_v10_4_terrain_robot_v2 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' || true; } | sort -n | tail -n 1 | cut -d' ' -f2-)"
CHECKPOINT="$(find "$PILOT_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [[ -z "$PILOT_DIR" || -z "$CHECKPOINT" ]]; then
    write_status "stage=checkpoint_missing"; exit 1
fi
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"

write_status "stage=running_gate"
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"
"$PYTHON" evaluate_v8.py --headless --env v10_4_terrain --forward-axis +y --checkpoint "$CHECKPOINT" --output "$GATE_JSON" --speed "$SPEED" \
    >"$CONTROL_DIR/v10_4_terrain_gate.log" 2>&1 || true

write_status "stage=running_checkpoint_sweep"
SWEEP_ARGS=("$GATE_JSON")
SWEEP_MISSING=""
for it in $SWEEP_ITERS; do
    ckpt="$PILOT_DIR/model_${it}.pt"
    # A missing checkpoint must be reported, not skipped. Silent skipping makes
    # an empty sweep look exactly like a sweep that found no regression.
    if [[ ! -f "$ckpt" ]]; then
        SWEEP_MISSING="$SWEEP_MISSING $it"
        printf 'sweep_missing_checkpoint=%s\n' "$ckpt" >>"$STATUS_FILE"
        continue
    fi
    out="$CONTROL_DIR/v10_4_terrain_gate_${it}.json"
    "$PYTHON" evaluate_v8.py --headless --env v10_4_terrain --forward-axis +y --checkpoint "$ckpt" --output "$out" --speed "$SPEED" \
        >"$CONTROL_DIR/v10_4_terrain_gate_${it}.log" 2>&1 || true
    [[ -f "$out" ]] && SWEEP_ARGS+=("$out")
done
if [[ -n "$SWEEP_MISSING" ]]; then
    printf 'sweep_incomplete=%s\n' "$SWEEP_MISSING" >>"$STATUS_FILE"
    echo "[WARN] sweep skipped absent checkpoints:$SWEEP_MISSING (must be multiples of save_interval)"
fi

# Rank every scored checkpoint so a mid-training regression is reported, not
# discovered later from a video.
BEST_CKPT="$("$PYTHON" - "$PILOT_DIR" "$SWEEP_JSON" "${SWEEP_ARGS[@]}" <<'PY'
import json, os, re, sys
pilot_dir, sweep_out = sys.argv[1], sys.argv[2]
rows = []
for path in sys.argv[3:]:
    try:
        d = json.load(open(path))
    except Exception:
        continue
    m = re.search(r"_(\d+)\.json$", os.path.basename(path))
    it = int(m.group(1)) if m else -1
    p = d.get("phase_metrics", {})
    rows.append({
        "gate_json": path,
        "iteration": it,
        "passed": bool(d.get("passed")),
        "gates_passed": sum(1 for v in d.get("gates", {}).values() if v),
        "min_foot_contact": p.get("min_foot_contact_fraction"),
        "survival": d.get("metrics", {}).get("survival_fraction"),
        "speed": d.get("metrics", {}).get("mean_forward_speed_m_s"),
        "height": d.get("metrics", {}).get("mean_base_height_m"),
    })
rows.sort(key=lambda r: (r["passed"], r["gates_passed"], r["min_foot_contact"] or 0.0), reverse=True)
json.dump({"ranked": rows}, open(sweep_out, "w"), indent=2)
if rows and rows[0]["iteration"] >= 0:
    print(os.path.join(pilot_dir, f"model_{rows[0]['iteration']}.pt"))
PY
)"

write_status "stage=recording_side"
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"
"$PYTHON" play_v8_video.py --env v10_4_terrain --forward-axis +y --checkpoint "$CHECKPOINT" --view side --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v10_4_terrain_side_video.log" 2>&1 || true
"$PYTHON" play_v8_video.py --env v10_4_terrain --forward-axis +y --checkpoint "$CHECKPOINT" --view rear_quarter --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v10_4_terrain_rear_video.log" 2>&1 || true

# If some earlier checkpoint scored better than the final one, record it too so
# the comparison is on video rather than on numbers alone.
if [[ -n "${BEST_CKPT:-}" && -f "${BEST_CKPT:-}" && "$BEST_CKPT" != "$CHECKPOINT" ]]; then
    write_status "stage=recording_best"
    printf 'best_checkpoint=%s\n' "$BEST_CKPT" >>"$STATUS_FILE"
    "$PYTHON" play_v8_video.py --env v10_4_terrain --forward-axis +y --checkpoint "$BEST_CKPT" --view side --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR/best" >"$CONTROL_DIR/v10_4_terrain_best_side_video.log" 2>&1 || true
fi

if "$PYTHON" - "$GATE_JSON" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') is True else 1)
PY
then
    write_status "stage=passed"
    printf 'pilot_dir=%s\ncheckpoint=%s\ngate_json=%s\nsweep_json=%s\nvideo_dir=%s\n' "$PILOT_DIR" "$CHECKPOINT" "$GATE_JSON" "$SWEEP_JSON" "$VIDEO_DIR" >>"$STATUS_FILE"
    exit 0
fi
write_status "stage=gate_failed"
printf 'pilot_dir=%s\ncheckpoint=%s\ngate_json=%s\nsweep_json=%s\nvideo_dir=%s\n' "$PILOT_DIR" "$CHECKPOINT" "$GATE_JSON" "$SWEEP_JSON" "$VIDEO_DIR" >>"$STATUS_FILE"
exit 2
