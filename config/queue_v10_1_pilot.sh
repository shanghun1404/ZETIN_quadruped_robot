#!/usr/bin/env bash
# V10_1 pilot: V9's terrain task with a termination set terrain can survive.
#
# One operational addition over queue_v9_pilot.sh: an early curriculum check.
# V9 sat at terrain level 0.14-0.21 for 4142 iterations and that was only noticed
# because someone looked. V10_1's whole hypothesis is that the termination set was
# what pinned it, so the run reports whether the level has moved by iteration
# TERRAIN_LEVEL_CHECK_ITER instead of letting five hours pass first.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/kang/miniforge3/envs/env_isaaclab/bin/python"
CONTROL_DIR="$ROOT/run_control"
STATUS_FILE="$CONTROL_DIR/v10_1_pilot.status"
GATE_JSON="$CONTROL_DIR/v10_1_pilot_gate.json"
VIDEO_DIR="$CONTROL_DIR/v10_1_pilot_video"
TRAIN_LOG="$CONTROL_DIR/v10_1_train.log"
ITERS="${V10_1_ITERS:-8000}"
SPEED="0.5"

mkdir -p "$CONTROL_DIR"
exec >>"$CONTROL_DIR/v10_1_pilot_queue.log" 2>&1

write_status() {
    printf '%s\ntime=%s\niters=%s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$ITERS" >"$STATUS_FILE"
}

cd "$ROOT"
write_status "stage=running_probe"
if ! "$PYTHON" probe_v10_1.py --headless --num_envs 64 >"$CONTROL_DIR/v10_1_probe.log" 2>&1; then
    write_status "stage=probe_failed"; exit 1
fi

write_status "stage=running_smoke"
if ! "$PYTHON" train_v10_1.py --headless --num_envs 4096 --max_iterations 3 >"$CONTROL_DIR/v10_1_smoke.log" 2>&1; then
    write_status "stage=smoke_failed"; exit 1
fi

write_status "stage=running_training"
"$PYTHON" train_v10_1.py --headless --num_envs 4096 --max_iterations "$ITERS" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
printf 'train_pid=%s\n' "$TRAIN_PID" >>"$STATUS_FILE"

# ---------------------------------------------------------------------------
# Early curriculum check -- report the falsifiable prediction while it is cheap
# ---------------------------------------------------------------------------
#
# It does NOT stop at the first verdict. V9.2's check passed at iteration 1501
# with the curriculum at 2.10 and climbing, and the collapse to 0.265 happened
# over the next 3500 iterations with nobody looking. Running for the whole job
# turns that from a post-mortem into an alert at roughly iteration 2500.
(
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 120
        if "$PYTHON" check_terrain_progress.py --log "$TRAIN_LOG" \
            --output "$CONTROL_DIR/v10_1_terrain_progress.json" --quiet; then
            if ! grep -q '"passed": true' "$CONTROL_DIR/v10_1_terrain_progress.json"; then
                cp "$CONTROL_DIR/v10_1_terrain_progress.json" \
                   "$CONTROL_DIR/v10_1_terrain_ALERT.json"
            fi
        fi
    done
) &
CHECK_PID=$!

wait "$TRAIN_PID"
TRAIN_RC=$?
kill "$CHECK_PID" 2>/dev/null
if [[ "$TRAIN_RC" -ne 0 ]]; then
    write_status "stage=training_failed"; exit 1
fi

PILOT_DIR="$({ find logs/rsl_rl/guide_v10_1_robot_v2 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' || true; } | sort -n | tail -n 1 | cut -d' ' -f2-)"
CHECKPOINT="$(find "$PILOT_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [[ -z "$PILOT_DIR" || -z "$CHECKPOINT" ]]; then
    write_status "stage=checkpoint_missing"; exit 1
fi
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"

write_status "stage=running_gate"
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"
"$PYTHON" evaluate_v9.py --headless --env v10_1 --checkpoint "$CHECKPOINT" --output "$GATE_JSON" --speed "$SPEED" \
    >"$CONTROL_DIR/v10_1_gate.log" 2>&1 || true

write_status "stage=running_stairs_gate"
"$PYTHON" evaluate_v9.py --headless --env v10_1 --checkpoint "$CHECKPOINT" --scenario stairs_up --seconds 12 \
    --output "$CONTROL_DIR/v10_1_stairs_gate.json" --speed "$SPEED" \
    >"$CONTROL_DIR/v10_1_stairs_gate.log" 2>&1 || true

write_status "stage=recording_side"
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"
"$PYTHON" play_v9_video.py --env v10_1 --checkpoint "$CHECKPOINT" --view side --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v10_1_side_video.log" 2>&1 || true
"$PYTHON" play_v9_video.py --env v10_1 --checkpoint "$CHECKPOINT" --view rear_quarter --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v10_1_rear_video.log" 2>&1 || true

write_status "stage=recording_stairs"
"$PYTHON" play_v9_video.py --env v10_1 --checkpoint "$CHECKPOINT" --scenario stairs_up --view side --speed "$SPEED" --video_length 500 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v10_1_stairs_side_video.log" 2>&1 || true
"$PYTHON" play_v9_video.py --env v10_1 --checkpoint "$CHECKPOINT" --scenario stairs_up --view rear_quarter --speed "$SPEED" --video_length 500 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v10_1_stairs_rear_video.log" 2>&1 || true

if "$PYTHON" - "$GATE_JSON" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') is True else 1)
PY
then
    write_status "stage=passed"
    printf 'pilot_dir=%s\ncheckpoint=%s\ngate_json=%s\nvideo_dir=%s\n' "$PILOT_DIR" "$CHECKPOINT" "$GATE_JSON" "$VIDEO_DIR" >>"$STATUS_FILE"
    exit 0
fi
write_status "stage=gate_failed"
printf 'pilot_dir=%s\ncheckpoint=%s\ngate_json=%s\nvideo_dir=%s\n' "$PILOT_DIR" "$CHECKPOINT" "$GATE_JSON" "$VIDEO_DIR" >>"$STATUS_FILE"
exit 2
