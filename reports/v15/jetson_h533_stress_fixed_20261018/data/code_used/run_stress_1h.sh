#!/usr/bin/env bash
set -eo pipefail
taskdir=/home/scienceowl/robot_dog/h533_link_20261017
output="${1:-$taskdir/stress_1h_latency_fixed_20261018}"
if [ -e "$output/service_started_utc.txt" ]; then
    printf 'Refusing to overwrite existing stress evidence: %s\n' "$output" >&2
    exit 2
fi
mkdir -p "$output"
source /home/scienceowl/robot_dog_env.bash
export ROS_DOMAIN_ID=174 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$taskdir"
date -u +%FT%TZ > "$output/service_started_utc.txt"
sudo -n nvpmodel -q > "$output/power-mode.before.txt"
tegrastats --interval 5000 > "$output/tegrastats.log" 2>&1 &
telemetry_pid=$!
monitor_pid=
cleanup() {
    kill "$telemetry_pid" 2>/dev/null || true
    if [ -n "$monitor_pid" ]; then kill "$monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
python latency_fix/benchmark_with_stm.py ros --root "$ROBOT_DOG_ROOT" --output "$output" \
    --device cuda --seconds 3605 --scenario walk --continuous-wbc --continue-after-fault \
    --physical-gateway-config "$taskdir/h533_gateway_config.json" > "$output/run.log" 2>&1 &
workload_pid=$!
printf '%s\n' "$workload_pid" > "$output/workload.pid"
python stress_monitor.py --pid "$workload_pid" --output "$output" > "$output/monitor.log" 2>&1 &
monitor_pid=$!
set +e
wait "$workload_pid"
workload_return=$?
set -e
printf '%s\n' "$workload_return" > "$output/workload.returncode"
cleanup
./tools/openocd-st/src/openocd -s tools/openocd-st/tcl -f h533_openocd.cfg \
    -c 'init; halt; echo H533_DIAG; echo [read_memory 0x20000000 32 26]; resume; shutdown' \
    > "$output/mcu-final.log" 2>&1 || true
sudo -n nvpmodel -q > "$output/power-mode.after.txt"
date -u +%FT%TZ > "$output/service_finished_utc.txt"
python finalize_stress.py "$output" "$workload_return" > "$output/final-summary.log" 2>&1
