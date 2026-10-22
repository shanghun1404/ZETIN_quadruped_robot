"""Trace the unchanged gateway with bounded RAM records and shutdown-only I/O."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from v15_hardware_bridge.usb_gateway import UsbGateway
import serial.serialposix
from gc_timing import watch_gc, prepare_heap, check_gc
from logging_buffers import NumericBuffer
import rclpy

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('destination', type=Path)
parser.add_argument('--seconds', type=float, default=3600.)
args, ros_args = parser.parse_known_args()
destination = args.destination
original = UsbGateway._write
close_gc = watch_gc(destination.with_name('gc_gateway.jsonl'))
original_os_write = os.write
original_select = serial.serialposix.select.select
active_fd = None
# Startup/shutdown traffic is bounded separately from the requested stream.
writes = NumericBuffer(round((args.seconds + 180) * 200) + 1000,
    [('started_ns', 'i8'), ('ended_ns', 'i8'), ('write_ms', 'f8'), ('kind', 'i4'),
     ('sequence', 'i8'), ('payload_bytes', 'i8'), ('outq_before', 'i8'), ('outq_after', 'i8'), ('error', 'i4')])
event_fields = [('operation', 'i1'), ('start_ns', 'i8'), ('end_ns', 'i8'),
    ('requested', 'i8'), ('returned', 'i8'), ('errno', 'i4'), ('timeout_s', 'f8'), ('writable', '?')]
events = NumericBuffer(64, event_fields)
slow_events = NumericBuffer(640_000, event_fields)
slow_writes = NumericBuffer(10_000, [('sequence', 'i8'), ('start_ns', 'i8'), ('end_ns', 'i8'),
    ('error', 'i4'), ('event_offset', 'i8'), ('event_count', 'i8')])
errors = NumericBuffer(1024, [('message', 'U1024')])
metrics = dict(write_calls=0, command_write_calls=0, os_write_calls=0, select_calls=0, os_write_errors=0,
    partial_writes=0, slow_or_failed_writes=0, logging_overflows=0)


def traced_os_write(fd, data):
    if fd != active_fd:
        return original_os_write(fd, data)
    before = time.monotonic_ns();metrics['os_write_calls'] += 1
    try:
        count = original_os_write(fd, data)
    except OSError as error:
        metrics['os_write_errors'] += 1
        events.append((0, before, time.monotonic_ns(), len(data), -1, error.errno or 0, 0., False))
        raise
    if count != len(data):
        metrics['partial_writes'] += 1
    events.append((0, before, time.monotonic_ns(), len(data), count, 0, 0., False))
    return count


def traced_select(read, write, error, timeout=None):
    if active_fd is None or active_fd not in write:
        return original_select(read, write, error, timeout)
    before = time.monotonic_ns();metrics['select_calls'] += 1
    result = original_select(read, write, error, timeout)
    events.append((1, before, time.monotonic_ns(), 0, 0, 0,
        float('nan') if timeout is None else timeout, active_fd in result[1]))
    return result


def queue(node):
    try:
        return node.serial.out_waiting
    except Exception:
        return -1


def timed_write(node, kind, stamp, payload):
    global active_fd
    active_fd = node.serial.fileno();events.count = 0
    before = time.monotonic_ns();queued = queue(node);sequence = node.sequence;error_code = 0
    metrics['write_calls'] += 1
    if kind==3:metrics['command_write_calls'] += 1
    try:
        check_gc()
        return original(node, kind, stamp, payload)
    except BaseException as exc:
        errors.append((f'{type(exc).__name__}: {exc}',));error_code = errors.count
        raise
    finally:
        ended = time.monotonic_ns()
        active_fd = None
        try:
            writes.append((before, ended, (ended - before) / 1e6, kind, sequence, len(payload),
                queued, queue(node), error_code))
            if error_code or ended - before > 2_000_000:
                metrics['slow_or_failed_writes'] += 1
                offset = slow_events.count
                if offset + events.count > len(slow_events.data):
                    slow_events.overflows += 1
                    raise OverflowError('serial diagnostic event capacity exceeded')
                slow_events.data[offset:offset + events.count] = events.data[:events.count]
                slow_events.count += events.count
                slow_writes.append((sequence, before, ended, error_code, offset, events.count))
        except OverflowError:
            metrics['logging_overflows'] += 1
            raise


def error_text(code):
    return str(errors.data[code - 1]['message']) if code else ''


def dump():
    writes.write_csv(destination, convert=lambda row: [*row[:-1], error_text(row[-1])])
    with destination.with_name('serial_syscalls_slow.jsonl').open('w') as stream:
        for record in slow_writes.data[:slow_writes.count]:
            detail = []
            offset = int(record['event_offset']);count = int(record['event_count'])
            for event in slow_events.data[offset:offset + count]:
                value = dict(operation='os.write' if event['operation'] == 0 else 'select',
                    start_ns=int(event['start_ns']), end_ns=int(event['end_ns']))
                if event['operation'] == 0:
                    if event['errno']:
                        value['errno'] = int(event['errno'])
                    else:
                        value.update(requested=int(event['requested']), returned=int(event['returned']))
                else:
                    timeout = float(event['timeout_s'])
                    value.update(timeout_s=None if timeout != timeout else timeout, writable=bool(event['writable']))
                detail.append(value)
            stream.write(json.dumps(dict(sequence=int(record['sequence']), start_ns=int(record['start_ns']),
                end_ns=int(record['end_ns']), error=error_text(int(record['error'])), events=detail)) + '\n')
    metrics.update(recorded_writes=writes.count, recorded_slow_writes=slow_writes.count,
        recorded_slow_events=slow_events.count, buffer_overflows=sum(
            b.overflows for b in (writes, events, slow_events, slow_writes, errors)))
    destination.with_name('gateway_trace_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    if metrics['buffer_overflows'] or metrics['logging_overflows']:
        raise OverflowError('gateway instrumentation lost records; see gateway_trace_metrics.json')


os.write = traced_os_write
serial.serialposix.select.select = traced_select
UsbGateway._write = timed_write
try:
    rclpy.init(args=ros_args)
    node = UsbGateway()
    executor = rclpy.get_global_executor();executor.add_node(node)
    executor.spin_once(timeout_sec=0)
    prepare_heap(destination.with_name('gc_prepare_gateway.json'))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
finally:
    os.write = original_os_write
    serial.serialposix.select.select = original_select
    UsbGateway._write = original
    try:
        dump()
    finally:
        close_gc()
