"""Audit original hour artifacts without changing their bytes or verdicts."""
import csv
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/'stress_1h_20261017_attempt2'
def read(name):return json.loads((run/name).read_text())
request=read('stress_start.json');raw=read('stress_result.json');controller=read('ros_controller_benchmark.json')
wbc=read('wbc_result.json');start=request['start_perf_ns'];end=start+round(request['seconds']*1e9)
count=0;measured=0;over=0;valid=True;last_end=None
with (run/'wbc_samples.csv').open() as stream:
    for row in csv.DictReader(stream):
        step=int(row['step']);duration=float(row['compute_ms']);scheduled=int(row['scheduled_ns'])
        valid &= step==count and scheduled==start+step*20_000_000 and math.isfinite(duration) and duration>0
        last_end=int(row['started_ns'])+round(duration*1e6)
        if step>=request['warmup_seconds']*50:measured+=1;over+=duration>20
        count+=1
wbc_audit=dict(csv_rows=count,measured_rows=measured,step_and_grid_and_duration_checks=valid,
    actual_last_compute_vs_end_ms=(last_end-end)/1e6,
    raw_report_vs_end_ms=wbc['last_completion_relative_to_end_ms'],
    post_compute_reporting_time_ms=wbc['last_completion_relative_to_end_ms']-(last_end-end)/1e6,
    raw_report_passed=wbc['passed'],above_20ms=over,
    completed=valid and count==180250 and measured==180000 and last_end<=end+20_000_000,
    explanation='Raw completion timestamp was captured after CSV close and array conversion. Actual last completion is reconstructed from the captured final start plus measured duration; raw files and their verdicts are preserved.')
(run/'wbc_timing_audit.json').write_text(json.dumps(wbc_audit,indent=2)+'\n')
plans=[json.loads(line) for line in (run/'planner_trace.jsonl').read_text().splitlines()]
plans=[r for r in plans if 5_000_000_000<=r['stamp_ns']<3_605_000_000_000]
before,after=max(zip(plans,plans[1:]),key=lambda pair:pair[1]['end_ns']-pair[0]['end_ns'])
producer_gc=[]
for path in run.glob('gc_ros_*.jsonl'):
    producer_gc.extend(json.loads(line) for line in path.read_text().splitlines())
overlap=[r for r in producer_gc if r['start_ns']<after['end_ns'] and r['end_ns']>before['end_ns']]
duration_complete=bool(request['seconds']-request['warmup_seconds']==3600
    and controller['published_total']==721000 and controller['elapsed_stream_s']>=3605
    and controller['planner_first_measured_source_s']<=5.04
    and controller['planner_last_measured_source_s']>=3604.96 and wbc_audit['completed'])
report=dict(execution_duration_completed=duration_complete,measured_duration_s=3600,
    integration_passed=raw['passed'],compute_stability_checks_passed=raw['ai_coverage']['passed'] and wbc_audit['completed'],
    ai=dict(controller['planner_compute_ms'],expected_samples=180000,
        samples=controller['planner_measured_samples'],missed_source_cycles=180000-controller['planner_measured_samples'],
        first_source_s=controller['planner_first_measured_source_s'],last_source_s=controller['planner_last_measured_source_s'],
        max_completion_gap_ms=(after['end_ns']-before['end_ns'])/1e6,
        max_source_gap_ms=raw['ai_coverage']['max_source_gap_ms'],
        max_observed_plan_arrival_gap_ms=raw['ai_coverage']['max_arrival_gap_ms'],
        longest_gap_overlapping_producer_gc=overlap,
        cuda_devices={k:controller['child'][k] for k in ('actor_parameter_device','student_parameter_device')}),
    wbc=dict(wbc,timing_audit=wbc_audit),first_command_fault_trace=raw['first_command_fault_trace'],
    os=raw['os'],mcu_diagnostics=raw['mcu_diagnostics'],original_checks=raw['checks'],
    scope=raw['scope'],post_fault_scope=raw['post_fault_scope'],gc_configuration=raw['gc_configuration'],
    caveats=['The full hour of compute load completed, but the original100ms gap stability criterion failed and integrated control failed.',
        'At source time10.555s (including5s warmup), the command pipeline latched a50ms plan freshness fault. Commands stopped; AI/WBC continued.',
        'Inputs and MCU feedback are synthetic. No actual robot motion, sensors or G474 motor/CAN behavior tested.',
        'Output arrays and lists are retained for logging; memory growth includes measurement storage.'],
    raw_sha256={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in
        ('stress_start.json','wbc_samples.csv','wbc_result.json','stress_result.json','ros_controller_benchmark.json','planner_trace.jsonl')})
(run/'audited_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(execution_duration_completed=duration_complete,integration_passed=report['integration_passed'],
    compute_stability_checks_passed=report['compute_stability_checks_passed'],wbc_audit=wbc_audit),indent=2))
