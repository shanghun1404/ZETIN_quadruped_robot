"""Independently audit completed logs without rewriting the original verdict."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

p = Path(sys.argv[1])
def load(name):
    return json.loads((p / name).read_text())
def csv_rows(name):
    with (p / name).open() as stream:
        return list(csv.DictReader(stream))
def summary(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(samples=len(a), mean_ms=float(a.mean()), p99_ms=float(np.percentile(a, 99)),
                max_ms=float(a.max()), above_20ms=int((a > 20).sum()))

result = load('ros_controller_benchmark.json')
start = load('stress_start.json')
ready = load('child_ready.json')
t0 = start['start_perf_ns']
t1 = t0 + round(start['seconds'] * 1e9)
warm_ns = start['warmup_seconds'] * 1_000_000_000
commands = csv_rows('ros_samples.csv')
wbc = csv_rows('wbc_samples.csv')
worker = csv_rows('policy_worker_timing.csv')
writes = csv_rows('gateway_writes.csv')
plans = load('planner_samples.json')
trace = [json.loads(line) for line in (p / 'planner_trace.jsonl').read_text().splitlines()]
feedback = load('physical_feedback.json')
statuses = load('hardware_statuses.json')
measured_commands = [r for r in commands if int(r['stamp_ns']) >= warm_ns]
measured_wbc = [r for r in wbc if int(r['step']) >= start['warmup_seconds'] * 50]
assert [int(r['step']) for r in commands] == list(range(round(start['seconds'] * 200)))
assert [int(r['step']) for r in wbc] == list(range(round(start['seconds'] * 50)))
assert [r['stamp_ns'] for r in trace] == list(range(0, round(start['seconds'] * 1e9), 20_000_000))
assert [r['stamp_ns'] for r in plans] == list(range(warm_ns, round(start['seconds'] * 1e9), 20_000_000))
assert [int(r['step']) for r in worker] == list(range(1, len(worker) + 1))
assert len(worker) == ready['pre_stream_policy_warmup_steps'] + len(trace)
assert all(int(r['end_ns']) > int(r['start_ns']) for r in worker)
assert all(int(r['ended_ns']) > int(r['started_ns']) and not r['error'] for r in writes)
for row in commands:
    for key, value in row.items():
        if value and key not in ('mode', 'policy_tick'):
            assert np.isfinite(float(value)), (key, value)
assert all(r['command_received_ns'] and r['compute_ms'] and r['mode'] == 'WALK'
           for r in measured_commands)
assert all(float(r['compute_ms']) > 0 for r in measured_wbc)
assert all(int(r['scheduled_ns']) == t0 + int(r['step']) * 20_000_000 for r in wbc)
assert all(b['stamp_ns'] - a['stamp_ns'] == 5_000_000 for a, b in zip(feedback, feedback[1:]))
assert all(not r.get('armed', False) and not r.get('fault_flags', 0) for r in statuses)
gc_stats = {}
gc_files = list(p.glob('gc_*.jsonl'))
assert len(gc_files) == 6
for f in gc_files:
    events = [json.loads(line) for line in f.read_text().splitlines()]
    during = [r for r in events if t0 <= r['start_ns'] < t1]
    gc_stats[f.name] = dict(total_events=len(events), during_stream_events=len(during),
                            max_during_stream_ms=max((r['wall_ms'] for r in during), default=0))

fields = ('magic version clock_hz boot_stage rx_bytes rx_overflows uart_errors tx_errors '
          'frames_received commands_received commands_accepted feedback_sent missed_ticks '
          'max_rx_depth session apply_calls ready_checks last_uart_error parse_calls '
          'parse_cycles_mod32 max_parse_cycles tx_cycles_mod32 max_tx_cycles consumed_bytes '
          'max_tick_lateness_ms icache_enabled').split()
line = next(line for line in (p / 'mcu-final.log').read_text().splitlines()
            if line.startswith('0x48353331 '))
mcu = dict(zip(fields, [int(v, 16) for v in line.split()], strict=True))
observed = sum(bool(r['command_received_ns']) for r in commands)
assert mcu['commands_received'] == mcu['commands_accepted'] == observed == 60997
assert sum(int(r['kind']) == 3 for r in writes) == observed
assert all(mcu[k] == 0 for k in ('rx_overflows', 'uart_errors', 'tx_errors', 'missed_ticks', 'apply_calls'))
assert load('gateway_trace_metrics.json')['buffer_overflows'] == 0

pg = result['physical_gateway']
checks = dict(commands_complete=result['missing'] == 0,
              plans_complete=result['planner_samples_complete'],
              no_stream_faults=not result['faults_during_stream'],
              never_armed=not pg['any_armed'], no_mcu_fault=not pg['any_mcu_fault'],
              no_feedback_tick_gaps=pg['missing_feedback_ticks'] == 0,
              feedback_hz_within_one=abs(pg['raw_feedback_hz'] - 200) < 1,
              feedback_arrival_count_within_two=abs(pg['raw_feedback_samples'] - pg['expected_raw_feedback_samples']) <= 2,
              feedback_end_fresh=pg['last_feedback_before_end_ms'] < 20,
              acknowledgements_complete=result['gateway_ack_fraction'] >= .99)
assert all(checks.values()) == result['integration_checks_passed']
stream_worker = [r for r in worker if t0 <= int(r['start_ns']) < t1]
measured_worker = [r for r in stream_worker if int(r['start_ns']) >= t0 + warm_ns]
audited = dict(
    raw_logs_complete=True, source_result_unchanged=True,
    integration_checks_passed=result['integration_checks_passed'],
    integration_checks=checks, failed_checks=[k for k, v in checks.items() if not v],
    strict_wbc_all_compute_below_20ms=all(float(r['compute_ms']) <= 20 for r in measured_wbc),
    measurement_seconds=start['seconds'] - start['warmup_seconds'],
    counts=dict(all_commands=observed, measured_commands=len(measured_commands),
                measured_plans=len(plans), measured_wbc=len(measured_wbc),
                worker_including_warmup=len(worker), gateway_writes=len(writes)),
    planner_ms=summary([r['compute_ms'] for r in plans]),
    policy_worker_ms=summary([(int(r['end_ns']) - int(r['start_ns'])) / 1e6 for r in measured_worker]),
    wbc_ms=summary([r['compute_ms'] for r in measured_wbc]),
    max_command_plan_age_ms=max(float(r['plan_age_ms']) for r in measured_commands),
    recorded_gc_threshold_ms=1.0, gc=gc_stats, mcu=mcu,
    note='Five-second warmup and 300-second measured integration. Synthetic sensors, motors disabled. '
         'The strict feedback arrival count check remains failed (59997 vs 60000); no source tick gaps. '
         'Four WBC computations exceed 20ms. No corrected one-hour test is claimed.')
(p / 'independent_audit.json').write_text(json.dumps(audited, indent=2) + '\n')
files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.iterdir())
         if f.is_file() and f.name != 'raw_sha256.json'}
(p / 'raw_sha256.json').write_text(json.dumps(files, indent=2) + '\n')
print(json.dumps(audited, indent=2))
