"""Bounded scratch instrumentation; raw records are written only at shutdown."""
import csv
import json
from pathlib import Path
import threading

import numpy as np


class NumericBuffer:
    def __init__(self, capacity, fields):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError('positive buffer capacity required')
        self.data = np.empty(capacity, dtype=fields)
        if self.data.dtype.hasobject:
            raise TypeError('instrumentation buffers cannot retain Python objects')
        # Commit the allocation before streaming, including its physical pages.
        self.data[...] = np.zeros((), dtype=self.data.dtype)
        self.count = 0
        self.overflows = 0
        self.strings = [(i, self.data.dtype[name].itemsize // 4)
                        for i, name in enumerate(self.data.dtype.names)
                        if self.data.dtype[name].kind == 'U']

    def append(self, values):
        if self.count >= len(self.data):
            self.overflows += 1
            raise OverflowError('instrumentation buffer capacity exceeded')
        for index, limit in self.strings:
            if len(values[index]) > limit:
                self.overflows += 1
                raise OverflowError('instrumentation string capacity exceeded')
        self.data[self.count] = values
        self.count += 1

    def write_csv(self, destination, fields=None, convert=None):
        with Path(destination).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields or self.data.dtype.names)
            for record in self.data[:self.count]:
                values = [record[name].item() for name in self.data.dtype.names]
                writer.writerow(convert(values) if convert else values)


class LatestSnapshotWriter:
    """Coalesce disposable UI snapshots; never perform file I/O on submit()."""
    def __init__(self, destination, announce=None):
        self.destination = Path(destination)
        self.announce = announce
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.latest = None
        self.closing = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name='progress-writer', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'progress writer failed: {self.error}') from self.error

    def submit(self, snapshot):
        self.check()
        with self.lock:
            if self.closing:
                raise RuntimeError('progress writer is closed')
            self.latest = snapshot
            self.ready.set()

    def _run(self):
        try:
            while True:
                self.ready.wait()
                with self.lock:
                    snapshot = self.latest
                    self.latest = None
                    closing = self.closing
                    self.ready.clear()
                if snapshot is not None:
                    temporary = self.destination.with_suffix('.tmp')
                    temporary.write_text(json.dumps(snapshot, indent=2) + '\n')
                    temporary.replace(self.destination)
                    if self.announce is not None:
                        self.announce(snapshot)
                if closing:
                    return
        except BaseException as error:
            self.error = error

    def close(self):
        with self.lock:
            self.closing = True
            self.ready.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('progress writer did not stop within 10 seconds')
        self.check()
