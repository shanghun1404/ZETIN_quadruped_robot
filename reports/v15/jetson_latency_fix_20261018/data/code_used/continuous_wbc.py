#!/usr/bin/env python3
"""Continuous 50 Hz production C++ CAD WBC load, without hardware interfaces."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from v151_ros2_control.cad_standing import CadStandingController
from gc_timing import watch_gc, prepare_heap, check_gc
from logging_buffers import NumericBuffer, LatestSnapshotWriter

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=3600.)
args = p.parse_args()
close_gc = watch_gc(args.output / 'gc_wbc.jsonl')
samples = NumericBuffer(round(args.seconds * 50) + 250,
    [('step', 'i8'), ('scheduled_ns', 'i8'), ('started_ns', 'i8'), ('compute_ms', 'f8')])
progress = LatestSnapshotWriter(args.output / 'wbc_progress.json')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
c = CadStandingController(args.root / 'model/quadruped_cad.xml', qp_backend='cpp')
q = c.nominal_joint_position.copy()
xyz = [0., 0., c.nominal_height];quat = [1., 0., 0., 0.]
c.initialize(q, xyz, quat)


def compute(step):
    # A small varying reference exercises full reference/FK/QP work each tick.
    offset = .01 + .005 * np.sin(step * .02)
    reference, goal = c.contact_consistent_reference(q, xyz, quat, c.nominal_height, np.ones(4),
        center_body=1., forward_offset_m=offset)
    torque, diag = c.compute(q, np.zeros(12), xyz, quat, np.zeros(3), np.zeros(3),
        joint_reference=reference, position_reference_world=goal, forward_offset_m=offset)
    if not diag['finite'] or not diag['feasible'] or not np.isfinite(torque).all():
        raise RuntimeError('C++ WBC returned a nonfinite or infeasible result')


for step in range(20):
    compute(step)
prepare_heap(args.output / 'gc_prepare_wbc.json')
(args.output / 'wbc_ready.json').write_text(json.dumps(dict(backend='cpp', hz=50)))
while not (args.output / 'stress_start.json').exists():
    time.sleep(.05)
request = json.loads((args.output / 'stress_start.json').read_text())
start = request['start_perf_ns'];count = round(request['seconds'] * 50)
completed = 0;max_compute_ms = 0.;recent_above = np.zeros(250, dtype=np.bool_)
recent_index = recent_above_count = 0
result = dict(backend='cpp', target_hz=50, requested_samples=count, passed=False)
try:
    if count > len(samples.data):
        raise OverflowError('requested WBC duration exceeds preallocated sample capacity')
    for step in range(count):
        deadline = start + step * 20_000_000
        delay = (deadline - time.perf_counter_ns()) / 1e9
        if delay > 0:
            time.sleep(delay)
        before = time.perf_counter_ns();compute(step)
        last_compute_completed_ns = time.perf_counter_ns()
        elapsed = (last_compute_completed_ns - before) / 1e6
        samples.append((step, deadline, before, elapsed));completed += 1
        if step >= request['warmup_seconds'] * 50:
            max_compute_ms = max(max_compute_ms, elapsed)
            recent_above_count -= int(recent_above[recent_index])
            recent_above[recent_index] = elapsed > 20
            recent_above_count += int(recent_above[recent_index])
            recent_index = (recent_index + 1) % 250
        check_gc();progress.check()
        if step % 250 == 0:
            progress.submit(dict(completed=completed, elapsed_s=(time.perf_counter_ns() - start) / 1e9,
                max_compute_ms=max_compute_ms, above_20ms=recent_above_count))
    measured = samples.data[request['warmup_seconds'] * 50:completed]
    a = measured['compute_ms']
    l = (measured['started_ns'] - measured['scheduled_ns']) / 1e6
    result.update(passed=last_compute_completed_ns <= start + int(request['seconds'] * 1e9) + 20_000_000,
        completed_samples=completed, measured_samples=len(a),
        last_completion_relative_to_end_ms=(last_compute_completed_ns - start - int(request['seconds'] * 1e9)) / 1e6,
        elapsed_s=(time.perf_counter_ns() - start) / 1e9, mean_ms=float(a.mean()),
        p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()), above_20ms=int((a > 20).sum()),
        scheduling_lateness_p99_ms=float(np.percentile(l, 99)), scheduling_lateness_max_ms=float(l.max()))
    # Stay alive until the publisher has sent its final 200 Hz sensor sample.
    remaining = (start + int(request['seconds'] * 1e9) + 500_000_000 - time.perf_counter_ns()) / 1e9
    if remaining > 0:
        time.sleep(remaining)
except BaseException as exc:
    result.update(passed=False, completed_samples=completed, error=f'{type(exc).__name__}: {exc}')
finally:
    try:
        progress.close()
    except BaseException as exc:
        result.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    try:
        close_gc()
    except BaseException as exc:
        result.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    samples.write_csv(args.output / 'wbc_samples.csv')
    (args.output / 'wbc_result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result['passed'] else 1)
