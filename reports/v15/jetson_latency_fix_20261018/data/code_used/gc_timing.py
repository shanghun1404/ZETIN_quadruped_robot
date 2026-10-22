"""Scratch-only bounded GC observation; automatic collection stays enabled."""
import atexit
import gc
import json
import os
import threading
import time
from logging_buffers import NumericBuffer

_watches = []


def check_gc():
    for trace in _watches:
        if trace.overflows:
            raise OverflowError('GC instrumentation buffer capacity exceeded')


def watch_gc(path):
    trace = NumericBuffer(100_000, [('pid', 'i8'), ('tid', 'i8'), ('generation', 'i1'),
        ('start_ns', 'i8'), ('end_ns', 'i8'), ('wall_ms', 'f8'),
        ('thread_cpu_ms', 'f8'), ('collected', 'i8')])
    _watches.append(trace)
    pending = {}
    pid = os.getpid()
    def callback(phase, info):
        generation = info['generation'];tid = threading.get_native_id()
        if phase == 'start':
            pending[(tid, generation)] = (time.perf_counter_ns(), time.thread_time_ns())
        else:
            before = pending.pop((tid, generation), None)
            if before is None:
                return
            wall = time.perf_counter_ns();cpu = time.thread_time_ns()
            if wall - before[0] >= 1_000_000:
                # Exceptions inside GC callbacks are ignored by Python. Remember
                # overflow and let regular control flow fail via check_gc().
                try:
                    trace.append((pid, tid, generation, before[0], wall,
                        (wall - before[0]) / 1e6, (cpu - before[1]) / 1e6, info['collected']))
                except OverflowError:
                    pass
    gc.callbacks.append(callback)
    closed=False
    def close():
        nonlocal closed
        if closed:return
        closed=True
        gc.callbacks.remove(callback)
        with path.open('w') as stream:
            for record in trace.data[:trace.count]:
                stream.write(json.dumps({name: record[name].item()
                    for name in trace.data.dtype.names}) + '\n')
            if trace.overflows:
                stream.write(json.dumps(dict(error='GC instrumentation buffer overflow',
                    dropped_records=trace.overflows)) + '\n')
        check_gc()
    atexit.register(close)
    return close


def prepare_heap(path):
    """Collect/freeze only the prepared heap; automatic GC stays enabled."""
    before = time.perf_counter_ns()
    collected = gc.collect()
    gc.freeze()
    path.write_text(json.dumps(dict(collected=collected,
        frozen_objects=gc.get_freeze_count(), gc_enabled=gc.isenabled(),
        prepare_ms=(time.perf_counter_ns() - before) / 1e6)) + '\n')
