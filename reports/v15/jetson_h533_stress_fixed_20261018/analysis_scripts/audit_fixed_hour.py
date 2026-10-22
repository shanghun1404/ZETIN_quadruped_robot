"""Audit a completed corrected stress run while preserving all raw verdicts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run
def read(name):
    return json.loads((run / name).read_text())
def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(samples=len(a), mean_ms=float(a.mean()), p99_ms=float(np.percentile(a, 99)),
                max_ms=float(a.max()), above_20ms=int((a > 20).sum())) if len(a) else {}

request = read('stress_start.json')
controller = read('ros_controller_benchmark.json')
raw = read('stress_result.json')
cleanup = read('diagnostic_cleanup_result.json')
start = request['start_perf_ns']
warm = request['warmup_seconds']
seconds = request['seconds']
end = start + round(seconds * 1e9)
expected_commands = round((seconds - warm) * 200)
expected_plans = round((seconds - warm) * 50)
checks = {}
command_count = measured_commands = 0
all_observed_commands = 0
command_finite = True
command_grid = True
command_times = []
command_compute = []
plan_ages = []
with (run / 'ros_samples.csv').open() as stream:
    for row in csv.DictReader(stream):
        step = int(row['step'])
        command_grid &= step == command_count and int(row['stamp_ns']) == step * 5_000_000
        for key, value in row.items():
            if value and key not in ('mode', 'policy_tick'):
                command_finite &= math.isfinite(float(value))
        complete = bool(row['command_received_ns'] and row['compute_ms'])
        all_observed_commands += bool(row['command_received_ns'])
        if step >= warm * 200 and complete:
            measured_commands += 1
            command_times.append(int(row['command_received_ns']))
            command_compute.append(float(row['compute_ms']))
            plan_ages.append(float(row['plan_age_ms']))
        command_count += 1
checks['command_source_grid_and_finite'] = command_grid and command_finite
checks['all_sensor_records_present'] = command_count == round(seconds * 200)
checks['every_measured_command_present'] = measured_commands == expected_commands

wbc = read('wbc_result.json')
wbc_rows = 0
wbc_measured = []
wbc_valid = True
last_wbc_completion = None
with (run / 'wbc_samples.csv').open() as stream:
    for row in csv.DictReader(stream):
        step = int(row['step'])
        duration = float(row['compute_ms'])
        wbc_valid &= (step == wbc_rows and int(row['scheduled_ns']) == start + step * 20_000_000
                      and math.isfinite(duration) and duration > 0)
        last_wbc_completion = int(row['started_ns']) + round(duration * 1e6)
        if step >= warm * 50:
            wbc_measured.append(duration)
        wbc_rows += 1
checks['wbc_grid_and_finite'] = wbc_valid
checks['every_measured_wbc_present'] = len(wbc_measured) == expected_plans
checks['wbc_last_compute_in_window'] = bool(last_wbc_completion and last_wbc_completion <= end + 20_000_000)

plans = read('planner_samples.json')
plan_stamps = [row['stamp_ns'] for row in plans]
plan_arrivals = [row['received_ns'] for row in plans]
checks['every_measured_plan_present'] = plan_stamps == list(range(warm * 1_000_000_000, round(seconds * 1e9), 20_000_000))
checks['plan_values_finite'] = all(math.isfinite(row['compute_ms']) and row['compute_ms'] > 0 for row in plans)
trace_count = 0
trace_grid = True
trace_previous = None
max_completion_gap = 0
with (run / 'planner_trace.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line)
        trace_grid &= row['stamp_ns'] == trace_count * 20_000_000 and row['end_ns'] >= row['start_ns']
        if trace_previous is not None and row['stamp_ns'] >= warm * 1_000_000_000:
            max_completion_gap = max(max_completion_gap, (row['end_ns'] - trace_previous) / 1e6)
        trace_previous = row['end_ns']
        trace_count += 1
checks['planner_trace_complete_grid'] = trace_grid and trace_count == round(seconds * 50)

worker_count = 0
worker_valid = True
worker_ms = []
with (run / 'policy_worker_timing.csv').open() as stream:
    for row in csv.DictReader(stream):
        worker_count += 1
        begin, finish = int(row['start_ns']), int(row['end_ns'])
        worker_valid &= int(row['step']) == worker_count and finish > begin
        if start + warm * 1_000_000_000 <= begin < end:
            worker_ms.append((finish - begin) / 1e6)
checks['worker_record_sequence'] = worker_valid
checks['worker_records_match_requests'] = worker_count == 100 + trace_count

write_count = command_writes = 0
write_valid = True
with (run / 'gateway_writes.csv').open() as stream:
    for row in csv.DictReader(stream):
        write_valid &= (int(row['sequence']) == write_count and not row['error']
                        and int(row['ended_ns']) >= int(row['started_ns']))
        command_writes += int(row['kind']) == 3
        write_count += 1
metrics = read('gateway_trace_metrics.json')
checks['gateway_records_valid_complete'] = write_valid and write_count == metrics['write_calls']
checks['gateway_command_counts_agree'] = command_writes == all_observed_commands == metrics['command_write_calls']
checks['gateway_no_io_errors'] = all(metrics[k] == 0 for k in
    ('os_write_errors', 'partial_writes', 'logging_overflows', 'buffer_overflows'))

feedback = read('physical_feedback.json')
feedback_gaps = sum(max(0, (b['stamp_ns'] - a['stamp_ns']) // 5_000_000 - 1)
                    for a, b in zip(feedback, feedback[1:]))
measured_feedback = [r for r in feedback if start + warm * 1_000_000_000 <= r['received_ns'] < end]
statuses = read('hardware_statuses.json')
checks['never_armed'] = not any(r.get('armed', False) for r in statuses)
checks['no_mcu_status_fault'] = not any(r.get('fault_flags', 0) for r in statuses)
mcu = raw['mcu_diagnostics']
checks['mcu_accepted_every_command'] = bool(mcu and mcu['commands_received'] == mcu['commands_accepted'] == all_observed_commands)
checks['mcu_io_and_tick_and_motor_apply_zero'] = bool(mcu and all(mcu[k] == 0 for k in
    ('rx_overflows', 'uart_errors', 'tx_errors', 'missed_ticks', 'apply_calls')))
checks['no_feedback_source_tick_gaps'] = feedback_gaps == 0
checks['no_stream_faults'] = not controller['faults_during_stream']
checks['deferred_logging_cleanup_passed'] = cleanup['passed'] and controller['diagnostic_cleanup_passed']

gc_events = {}
for path in sorted(run.glob('gc_*.jsonl')):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    during = [r for r in events if 'start_ns' in r and start <= r['start_ns'] < end]
    gc_events[path.name] = dict(total_events=len(events), during_stream_events_ge_1ms=len(during),
        max_during_stream_ms=max((r['wall_ms'] for r in during), default=0))

pg = controller['physical_gateway']
original_gates = dict(
    command_count=controller['missing'] == 0,
    plan_count=controller['planner_samples_complete'],
    no_faults=not controller['faults_during_stream'],
    not_armed=not pg['any_armed'], no_mcu_fault=not pg['any_mcu_fault'],
    feedback_source_ticks=pg['missing_feedback_ticks'] == 0,
    feedback_frequency=abs(pg['raw_feedback_hz'] - 200) < 1,
    feedback_arrival_count=abs(pg['raw_feedback_samples'] - pg['expected_raw_feedback_samples']) <= 2,
    final_feedback_fresh=pg['last_feedback_before_end_ms'] is not None and pg['last_feedback_before_end_ms'] < 20,
    acknowledgements=controller['gateway_ack_fraction'] >= .99,
    diagnostic_cleanup=controller['diagnostic_cleanup_passed'])
checks['original_integration_verdict_matches_gates'] = all(original_gates.values()) == controller['integration_checks_passed']
duration_completed = bool(
    command_grid and command_count == round(seconds * 200)
    and plan_stamps and min(plan_stamps) <= (warm + .04) * 1e9
    and max(plan_stamps) >= (seconds - .04) * 1e9
    and controller['elapsed_stream_s'] >= seconds and checks['every_measured_wbc_present']
    and checks['wbc_last_compute_in_window'] and wbc_valid)

audited = dict(
    measured_duration_s=seconds - warm,
    execution_duration_completed=duration_completed,
    integration_passed=raw['passed'],
    original_controller_integration_passed=controller['integration_checks_passed'],
    compute_stability_checks_passed=raw['ai_coverage']['passed'] and checks['wbc_last_compute_in_window'] and checks['every_measured_wbc_present'],
    strict_every_wbc_compute_within_20ms=len(wbc_measured) == expected_plans and all(v <= 20 for v in wbc_measured),
    checks=checks, original_integration_gates=original_gates,
    failed_original_integration_gates=[key for key, value in original_gates.items() if not value],
    ai=dict(stats([r['compute_ms'] for r in plans]), expected_samples=expected_plans,
        first_source_s=min(plan_stamps) / 1e9 if plan_stamps else None,
        last_source_s=max(plan_stamps) / 1e9 if plan_stamps else None,
        max_completion_gap_ms=max_completion_gap,
        max_observed_plan_arrival_gap_ms=float(np.diff(sorted(plan_arrivals)).max() / 1e6) if len(plan_arrivals) > 1 else None),
    wbc=dict(wbc, raw_csv_rows=wbc_rows, independently_counted_measured_rows=len(wbc_measured),
             independently_recomputed=stats(wbc_measured),
             reconstructed_last_compute_vs_end_ms=(last_wbc_completion - end) / 1e6 if last_wbc_completion else None),
    commands=dict(source_records=command_count, observed_all=all_observed_commands,
        measured_complete=measured_commands, measured_expected=expected_commands,
        command_compute_ms=stats(command_compute),
        max_command_interval_ms=float(np.diff(command_times).max() / 1e6) if len(command_times) > 1 else None,
        max_plan_age_ms=max(plan_ages, default=None)),
    worker=dict(raw_csv_rows=worker_count, measured_stats=stats(worker_ms)),
    gateway=dict(raw_csv_rows=write_count, command_writes=command_writes, metrics=metrics),
    feedback=dict(total_rows=len(feedback), measured_rows=len(measured_feedback), source_tick_gaps=feedback_gaps,
        original_gateway_result=pg),
    gc_recording_threshold_ms=1.0, gc=gc_events,
    faults_during_stream=controller['faults_during_stream'],
    first_command_fault_trace=raw['first_command_fault_trace'],
    os=raw['os'], mcu_diagnostics=mcu,
    scope=raw['scope'],
    post_fault_scope=raw['post_fault_scope'],
    caveats=[
        'Sensors and MCU feedback are synthetic; motor outputs permanently disabled. G474/CAN/physical robot not tested.',
        'WBC passed in the raw result describes duration completion, not every 20ms compute deadline.',
        'After any latched guard fault, commands remain stopped; only synthetic AI/WBC load continues.',
        'Raw verdicts, strict feedback arrival count tolerance, and50ms freshness guards are preserved.'])
(run / 'audited_result.json').write_text(json.dumps(audited, indent=2) + '\n')
hashes = {p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'raw_sha256.json'}
(run / 'raw_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps(dict(execution_duration_completed=duration_completed, integration_passed=raw['passed'],
    original_gates=original_gates, checks=checks, ai=audited['ai'], wbc=audited['wbc']), indent=2))
