#!/usr/bin/env bash
# Run the V8.1 pilot, then the V9 pilot, on the single GPU.
#
# V9 inherits V8.1's reward table, so V8.1 has to finish before V9 starts. V9
# runs regardless of V8.1's gate: a failed flat-ground gate is informative about
# the duty floor, but it does not make the terrain and scanner result worthless,
# and leaving the GPU idle overnight waiting for a human to read one number is
# more expensive than the run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_DIR="$ROOT/run_control"
CHAIN_STATUS="$CONTROL_DIR/v8_1_v9_chain.status"

mkdir -p "$CONTROL_DIR"
exec >>"$CONTROL_DIR/v8_1_v9_chain.log" 2>&1

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
note() { printf '%s\ntime=%s\n' "$1" "$(stamp)" >"$CHAIN_STATUS"; }

cd "$ROOT"

note "stage=v8_1_running"
bash queue_v8_1_pilot.sh
V8_1_RC=$?
printf 'v8_1_exit=%s\nv8_1_finished=%s\n' "$V8_1_RC" "$(stamp)" >>"$CHAIN_STATUS"

note "stage=v9_running"
printf 'v8_1_exit=%s\n' "$V8_1_RC" >>"$CHAIN_STATUS"
bash queue_v9_pilot.sh
V9_RC=$?

note "stage=done"
printf 'v8_1_exit=%s\nv9_exit=%s\nfinished=%s\n' "$V8_1_RC" "$V9_RC" "$(stamp)" >>"$CHAIN_STATUS"
exit 0
