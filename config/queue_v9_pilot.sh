#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/kang/miniforge3/envs/env_isaaclab/bin/python"
CONTROL_DIR="$ROOT/run_control"
STATUS_FILE="$CONTROL_DIR/v9_pilot.status"
GATE_JSON="$CONTROL_DIR/v9_pilot_gate.json"
VIDEO_DIR="$CONTROL_DIR/v9_pilot_video"
ITERS="${V9_ITERS:-8000}"
SPEED="0.5"

mkdir -p "$CONTROL_DIR"
exec >>"$CONTROL_DIR/v9_pilot_queue.log" 2>&1

write_status() {
    printf '%s\ntime=%s\niters=%s\n' "$1" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$ITERS" >"$STATUS_FILE"
}

cd "$ROOT"
write_status "stage=running_probe"
if ! "$PYTHON" probe_v9.py --headless --num_envs 64 >"$CONTROL_DIR/v9_probe.log" 2>&1; then
    write_status "stage=probe_failed"; exit 1
fi

write_status "stage=running_smoke"
if ! "$PYTHON" train_v9.py --headless --num_envs 4096 --max_iterations 3 >"$CONTROL_DIR/v9_smoke.log" 2>&1; then
    write_status "stage=smoke_failed"; exit 1
fi

write_status "stage=running_training"
if ! "$PYTHON" train_v9.py --headless --num_envs 4096 --max_iterations "$ITERS" >"$CONTROL_DIR/v9_train.log" 2>&1; then
    write_status "stage=training_failed"; exit 1
fi

PILOT_DIR="$({ find logs/rsl_rl/guide_v9_terrain -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' || true; } | sort -n | tail -n 1 | cut -d' ' -f2-)"
CHECKPOINT="$(find "$PILOT_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"
if [[ -z "$PILOT_DIR" || -z "$CHECKPOINT" ]]; then
    write_status "stage=checkpoint_missing"; exit 1
fi
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"

write_status "stage=running_gate"
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"
"$PYTHON" evaluate_v9.py --headless --checkpoint "$CHECKPOINT" --output "$GATE_JSON" --speed "$SPEED" \
    >"$CONTROL_DIR/v9_gate.log" 2>&1 || true

write_status "stage=running_stairs_gate"
"$PYTHON" evaluate_v9.py --headless --checkpoint "$CHECKPOINT" --scenario stairs_up --seconds 12 \
    --output "$CONTROL_DIR/v9_stairs_gate.json" --speed "$SPEED" \
    >"$CONTROL_DIR/v9_stairs_gate.log" 2>&1 || true

write_status "stage=recording_side"
printf 'pilot_dir=%s\ncheckpoint=%s\n' "$PILOT_DIR" "$CHECKPOINT" >>"$STATUS_FILE"
"$PYTHON" play_v9_video.py --checkpoint "$CHECKPOINT" --view side --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v9_side_video.log" 2>&1 || true
"$PYTHON" play_v9_video.py --checkpoint "$CHECKPOINT" --view rear_quarter --speed "$SPEED" --video_length 300 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v9_rear_video.log" 2>&1 || true

write_status "stage=recording_stairs"
"$PYTHON" play_v9_video.py --checkpoint "$CHECKPOINT" --scenario stairs_up --view side --speed "$SPEED" --video_length 500 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v9_stairs_side_video.log" 2>&1 || true
"$PYTHON" play_v9_video.py --checkpoint "$CHECKPOINT" --scenario stairs_up --view rear_quarter --speed "$SPEED" --video_length 500 --output_dir "$VIDEO_DIR" >"$CONTROL_DIR/v9_stairs_rear_video.log" 2>&1 || true

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
