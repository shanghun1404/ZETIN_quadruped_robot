"""Run the unchanged production policy server with bounded RAM-only timing."""
import os
import time
from pathlib import Path
from gc_timing import watch_gc, prepare_heap, check_gc
from logging_buffers import NumericBuffer


class ConnectionProbe:
    def __init__(self, connection):
        self.connection = connection
        self.results = 0
        self.directory = Path(os.environ['ROBOT_STRESS_OUTPUT'])
        self.trace = NumericBuffer(round(float(os.environ.get('ROBOT_STRESS_SECONDS', '3600')) * 50) + 2000,
            [('step', 'i8'), ('start_ns', 'i8'), ('end_ns', 'i8'),
             ('thread_cpu_ms', 'f8'), ('runqueue_wait_ms', 'f8')])
        self.schedstat = Path('/proc/self/schedstat')
        self.started = self.cpu = self.runqueue = 0
        self.closed = False

    def recv(self):
        request = self.connection.recv()
        self.started = time.perf_counter_ns()
        self.cpu = time.thread_time_ns()
        self.runqueue = int(self.schedstat.read_text().split()[1])
        return request

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            return self.connection.close()
        finally:
            self.trace.write_csv(self.directory / 'policy_worker_timing.csv')

    def send(self, value):
        if value[0] == 'result':
            self.results += 1
            self.trace.append((self.results, self.started, time.perf_counter_ns(),
                (time.thread_time_ns() - self.cpu) / 1e6,
                (int(self.schedstat.read_text().split()[1]) - self.runqueue) / 1e6))
            check_gc()
            if self.results == 100:
                prepare_heap(self.directory / 'gc_prepare_policy.json')
        self.connection.send(value)


def serve(connection, actor, student, device):
    from v151_ros2_control.policy_process import _serve
    directory = Path(os.environ['ROBOT_STRESS_OUTPUT'])
    close_gc = watch_gc(directory / 'gc_policy.jsonl')
    try:
        return _serve(ConnectionProbe(connection), actor, student, device)
    finally:
        # multiprocessing uses os._exit(), which bypasses ordinary atexit hooks.
        close_gc()
