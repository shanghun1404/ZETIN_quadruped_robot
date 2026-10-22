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


def _live_process_group(pgid):
    """Linux process-group members that can still execute (excluding zombies)."""
    members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().rsplit(') ',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):
            continue
        # Other read/parse failures propagate: they cannot prove group absence.
        if int(fields[2])==pgid and fields[0] not in ('Z','X','x'):
            members.append(int(entry.name))
    return members


def finalize_diagnostic_cleanup(output, children, artifacts, *, primary_error=None,
                                resources=(), timeout=60., benchmark_result=None):
    """Reap every child, audit deferred records, then decide benchmark success."""
    import os
    import signal
    import subprocess
    import time

    output = Path(output)
    failures = []
    audit = dict(passed=False, children={}, artifacts={}, errors=failures,
        primary_error=None if primary_error is None else f'{type(primary_error).__name__}: {primary_error}')
    for name, process in children.items():
        if process is None:
            continue
        state = dict(pid=process.pid, signal_sent=False, timed_out=False, killed=False,
            group_killed=False, exitcode=None)
        audit['children'][name] = state
        try:
            running=process.poll() is None
            if not running and name!='wbc':
                failures.append(f'{name}: exited before shutdown was requested')
            if running:
                if name == 'wbc':
                    try:
                        process.wait(timeout=min(timeout, 10.))
                    except subprocess.TimeoutExpired:
                        process.send_signal(signal.SIGINT);state['signal_sent'] = True
                else:
                    process.send_signal(signal.SIGINT);state['signal_sent'] = True
            stop_timeout=timeout+15. if name=='planner' and timeout>=60. else timeout
            try:
                state['exitcode'] = process.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                state['timed_out'] = True;state['killed'] = True
                failures.append(f'{name}: timed out while stopping or dumping diagnostics')
                # Popen(start_new_session=True) fixes our owned PGID at pid.
                # Resolve no leader PID here: it can disappear before its workers.
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        state['group_killed'] = True
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                state['exitcode'] = process.wait(timeout=5.)
            allowed = (0, -signal.SIGINT, 130) if name == 'gateway' and state['signal_sent'] else (0,)
            if state['exitcode'] not in allowed:
                failures.append(f'{name}: unexpected exit code {state["exitcode"]}')
        except BaseException as error:
            failures.append(f'{name}: cleanup {type(error).__name__}: {error}')
            # A failed signal/wait must not skip cleanup of this or later children.
            try:
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        state['group_killed'] = True
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                state['killed'] = True
                state['exitcode'] = process.wait(timeout=5.)
            except BaseException as last_error:
                failures.append(f'{name}: reap {type(last_error).__name__}: {last_error}')
        finally:
            # A normal spawn worker can leave a resource tracker briefly exiting,
            # or zombies awaiting init. Allow bounded grace and count only tasks
            # that can still execute; a vanished leader does not imply no worker.
            if process.pid is not None:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    failures.append(f'{name}: group probe {type(error).__name__}: {error}')
                else:
                    members=None
                    try:
                        grace_end=time.monotonic()+1.
                        while True:
                            members=_live_process_group(process.pid)
                            if not members or time.monotonic()>=grace_end:break
                            time.sleep(.01)
                    except (OSError,ValueError,IndexError) as error:
                        failures.append(f'{name}: group scan {type(error).__name__}: {error}')
                    if members or members is None:
                        if members:
                            failures.append(f'{name}: live descendants remained after leader exit: {members}')
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                            state['group_killed'] = True
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            failures.append(f'{name}: group kill {type(error).__name__}: {error}')

    for name, close in resources:
        try:
            close()
        except BaseException as error:
            failures.append(f'{name}: {type(error).__name__}: {error}')

    json_documents = {}
    for filename, spec in artifacts.items():
        state = dict(passed=False)
        audit['artifacts'][filename] = state
        try:
            path = output / filename
            if spec['kind'] == 'json':
                with path.open() as stream:
                    document = json.load(stream)
                json_documents[filename] = document
                for key, expected in spec.get('values', {}).items():
                    if document.get(key) != expected:
                        raise ValueError(f'{key}={document.get(key)!r}; expected {expected!r}')
                state['passed'] = True
                continue
            with path.open('rb') as stream:
                stream.seek(0, 2);size = stream.tell()
                if size:
                    stream.seek(-1, 2)
                    if stream.read(1) != b'\n':
                        raise ValueError('truncated final record (missing newline)')
            count = 0;filtered_count = 0
            filter_spec=spec.get('count_where')
            with path.open(newline='') as stream:
                if spec['kind'] == 'csv':
                    reader = csv.reader(stream, strict=True)
                    header = next(reader)
                    required = spec.get('fields', ())
                    if not set(required).issubset(header):
                        raise ValueError(f'missing CSV fields: {set(required) - set(header)}')
                    if len(header) != len(set(header)):
                        raise ValueError('duplicate CSV fields')
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f'truncated CSV row {count + 1}')
                        for index, key in enumerate(header):
                            if key.endswith('_ns') and row[index]:
                                int(row[index])
                        if filter_spec and row[header.index(filter_spec['field'])]==filter_spec['equals']:
                            filtered_count+=1
                        count += 1
                elif spec['kind'] == 'jsonl':
                    for line in stream:
                        value = json.loads(line)
                        if not isinstance(value, dict) or 'error' in value and value['error']:
                            raise ValueError('invalid or failed JSONL diagnostic record')
                        if not set(spec.get('fields', ())).issubset(value):
                            raise ValueError('missing JSONL fields')
                        count += 1
                else:
                    raise ValueError(f'unknown artifact kind {spec["kind"]}')
            expected_count = spec.get('count')
            if 'count_from' in spec:
                source, key = spec['count_from']
                if source not in json_documents:
                    json_documents[source] = json.loads((output / source).read_text())
                expected_count = json_documents[source][key]
            state['records'] = count
            if expected_count is not None and count != expected_count:
                raise ValueError(f'{count} records; expected {expected_count}')
            if filter_spec:
                source,key=filter_spec['count_from']
                if source not in json_documents:
                    json_documents[source]=json.loads((output/source).read_text())
                state['filtered_records']=filtered_count
                if filtered_count!=json_documents[source][key]:
                    raise ValueError(f'{filtered_count} filtered records; expected {json_documents[source][key]}')
            state['passed'] = True
        except BaseException as error:
            state['error'] = f'{type(error).__name__}: {error}'
            failures.append(f'{filename}: {state["error"]}')

    audit['passed'] = not failures
    report_path = output / 'ros_controller_benchmark.json'
    if benchmark_result is not None or report_path.exists():
        try:
            report = dict(benchmark_result) if benchmark_result is not None else json.loads(report_path.read_text())
            report['diagnostic_cleanup_passed'] = audit['passed']
            if failures or primary_error is not None:
                report['integration_checks_passed'] = False
            report_path.write_text(json.dumps(report, indent=2) + '\n')
        except BaseException as error:
            failures.append(f'benchmark report: {type(error).__name__}: {error}')
            audit['passed'] = False
    try:
        (output / 'diagnostic_cleanup_result.json').write_text(json.dumps(audit, indent=2) + '\n')
    except BaseException as error:
        if primary_error is None:raise
        if hasattr(primary_error,'add_note'):
            primary_error.add_note(f'Could not write cleanup result: {type(error).__name__}: {error}')
    if failures and primary_error is None:
        raise RuntimeError('diagnostic cleanup failed: ' + '; '.join(failures))
    return audit
