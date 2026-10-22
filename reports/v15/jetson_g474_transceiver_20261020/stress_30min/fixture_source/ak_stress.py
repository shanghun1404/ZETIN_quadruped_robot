#!/usr/bin/env python3
"""Bounded-memory UART stress for the explicitly virtual AK motor fixture."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import secrets
import struct
import time
import traceback

import protocol as p

RATE = 200
PERIOD_NS = 5_000_000
CAPABILITY = 0x80000001
ZERO = (0.0,) * 12


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def round_away(value):
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def targets(index):
    return tuple(f32((0.2 + 0.4 * j / 11) * math.sin(
        (index / RATE) * (0.31 + j * 0.037) + 0.23 + j * 0.47)) for j in range(12))


def servo_raw(position):
    return round_away(f32(position) * 180.0 / math.pi * 10000.0)


def expected_angle(mode, joint, position):
    position = f32(position)
    if mode == "mit":
        limit = 12.5 if joint % 3 == 0 else 12.56
        raw = max(0, min(65535, int((position + limit) * 65535.0 / (2 * limit))))
        position = raw * (2 * limit) / 65535.0 - limit
        if joint % 3 == 0:
            return f32(position)
    elif mode == "servo":
        position = servo_raw(position) / 10000.0 * math.pi / 180.0
    else:
        raise ValueError("unknown mode")
    raw = round_away(position * 180.0 / math.pi * 10.0)
    return f32(raw * 0.1 * math.pi / 180.0)


class Histogram:
    """201 fixed bins: 0.25 ms resolution, with the final bin for overflow."""
    def __init__(self):
        self.bins = [0] * 201
        self.count = 0
        self.total = self.maximum = 0.0

    def add(self, value):
        value = max(0.0, value)
        self.bins[min(200, int(value * 4))] += 1
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)

    def report(self):
        result = dict(count=self.count, mean=self.total / self.count if self.count else 0,
                      maximum=self.maximum, bins=self.bins)
        for percentile in (50, 95, 99):
            cumulative = 0
            for index, count in enumerate(self.bins):
                cumulative += count
                if self.count and cumulative >= math.ceil(self.count * percentile / 100):
                    result[f"p{percentile}_upper"] = (index + 1) / 4 if index < 200 else self.maximum
                    break
        return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def phase_stats():
    return dict(commands=0, feedback=0, errors=0, source_sequence_gaps=0,
                source_stamp_gaps=0, source_duplicates=0, ack_duplicates=0, ack_skips=0,
                deadline_misses_over_5ms=0, fault_frames=0, armed_frames=0,
                max_target_error_rad=0.0, max_expected_error_rad=0.0,
                lateness_ms=Histogram(), ack_latency_ms=Histogram())


def run(args, serial_factory=None, clock=time.monotonic_ns, sleep=time.sleep):
    counts = [args.seconds * RATE, args.warmup_seconds * RATE]
    if (not all(math.isfinite(x) and x >= 0 and abs(x - round(x)) < 1e-7 for x in counts)
            or counts[0] <= 0):
        raise ValueError("durations must be finite multiples of 1/200 s; measurement must be positive")
    measured_count, warmup_count = map(round, counts)
    expected_count = measured_count + warmup_count
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    phases = {"warmup": phase_stats(), "measured": phase_stats()}
    parser, pending = p.StreamParser(), [None] * 4096
    session, serial = secrets.randbits(32) or 1, None
    start = measured_start = measured_end = None
    wall_times = dict(start_wall_utc=None, measured_start_wall_utc=None, measured_end_wall_utc=None)
    sent = ack = rx_bytes = tx_bytes = valid_bytes = foreign_frames = errors = 0
    last_sequence = last_stamp = last_rx = None
    max_rx_gap = 0
    startup_discard = 0
    capability, exception, last_error = None, None, None
    rx_log = tx_log = None

    def error(message, phase=None):
        nonlocal errors, last_error
        errors += 1
        last_error = message
        if phase is not None:
            phase["errors"] += 1

    def report(final=False):
        elapsed = (clock() - start) / 1e9 if start is not None else 0.0
        measured_elapsed = ((measured_end or clock()) - measured_start) / 1e9 if measured_start is not None else 0.0
        data = dict(mode=args.mode, port=args.port, seconds=args.seconds,
                    warmup_seconds=args.warmup_seconds, elapsed_seconds=elapsed,
                    measured_elapsed_seconds=max(0, measured_elapsed), expected_commands=expected_count,
                    expected_measured_commands=measured_count, commands=sent, final_ack=ack,
                    capability=capability, errors=errors, last_error=last_error, exception=exception,
                    serial_sent_bytes=tx_bytes, serial_received_bytes=rx_bytes,
                    rx_max_gap_ms=max_rx_gap / 1e6, foreign_frames=foreign_frames,
                    startup_discarded_bytes=startup_discard,
                    discarded_bytes_after_hello=max(0, rx_bytes - valid_bytes - parser.buffered_bytes - startup_discard),
                    parser_buffered_bytes=parser.buffered_bytes, pending_capacity=len(pending),
                    passed=bool(final and exception is None and errors == 0 and capability == CAPABILITY
                                and sent == expected_count and ack == expected_count
                                and phases["measured"]["commands"] == measured_count
                                and phases["measured"]["feedback"] > 0))
        data.update(wall_times)
        for name, phase in phases.items():
            data[name] = {key: value.report() if isinstance(value, Histogram) else value
                          for key, value in phase.items()}
        return data

    def send(kind, sequence, payload=b""):
        nonlocal tx_bytes
        raw = p.encode_frame(p.Frame(kind, session, sequence, clock(), payload))
        written = serial.write(raw)
        tx_bytes += written
        tx_log.write(raw[:written])
        if written != len(raw):
            raise OSError("partial UART write")

    def poll():
        nonlocal capability, ack, rx_bytes, valid_bytes, last_sequence, last_stamp
        nonlocal last_rx, max_rx_gap, foreign_frames
        raw = serial.read(min(4096, serial.in_waiting))
        if not raw:
            return
        rx_log.write(raw)
        rx_bytes += len(raw)
        now = clock()
        for frame in parser.feed(raw):
            valid_bytes += p.HEADER.size + len(frame.payload) + p.CRC.size
            if frame.session != session:
                foreign_frames += 1
                continue
            if frame.kind == p.HELLO_ACK:
                capability = struct.unpack("<I", frame.payload)[0]
                continue
            if frame.kind != p.FEEDBACK:
                error("unexpected frame kind")
                continue
            state = p.unpack_feedback(frame.payload)
            current_ack = state["ack_sequence"]
            phase = phases["measured" if current_ack > warmup_count else "warmup"]
            phase["feedback"] += 1
            if last_rx is not None:
                max_rx_gap = max(max_rx_gap, now - last_rx)
            last_rx = now
            if last_sequence is not None:
                delta = (frame.sequence - last_sequence) & 0xffffffff
                if delta == 0 or delta >= 0x80000000:
                    phase["source_duplicates"] += 1
                    error("duplicate or backwards feedback sequence", phase)
                elif delta > 1:
                    phase["source_sequence_gaps"] += delta - 1
                    error("missing feedback sequence", phase)
                if frame.stamp_ns <= last_stamp or (frame.stamp_ns - last_stamp) % PERIOD_NS:
                    error("invalid feedback source timestamp", phase)
                elif frame.stamp_ns - last_stamp > PERIOD_NS:
                    phase["source_stamp_gaps"] += (frame.stamp_ns - last_stamp) // PERIOD_NS - 1
                    error("missing feedback source timestamp", phase)
            last_sequence, last_stamp = frame.sequence, frame.stamp_ns
            phase["fault_frames"] += bool(state["fault_flags"])
            phase["armed_frames"] += bool(state["armed"])
            if state["fault_flags"] or state["armed"]:
                error(f"fault/armed feedback: {state['fault_flags']}/{state['armed']}", phase)
            if current_ack == 0:
                continue
            if current_ack > sent or current_ack < ack:
                error("invalid command acknowledgement", phase)
                continue
            if current_ack == ack:
                phase["ack_duplicates"] += 1
            else:
                phase["ack_skips"] += max(0, current_ack - ack - 1)
                entry = pending[current_ack % len(pending)]
                if entry is None or entry[0] != current_ack:
                    error("acknowledgement outside bounded timestamp window", phase)
                else:
                    phase["ack_latency_ms"].add((now - entry[1]) / 1e6)
            ack = current_ack
            target = targets(ack - 1)
            for joint, value in enumerate(state["position"]):
                expected_error = abs(value - expected_angle(args.mode, joint, target[joint]))
                target_error = abs(value - target[joint])
                phase["max_expected_error_rad"] = max(phase["max_expected_error_rad"], expected_error)
                phase["max_target_error_rad"] = max(phase["max_target_error_rad"], target_error)
                if expected_error > 0.000002 or target_error > 0.0014:
                    error(f"angle mismatch at ack={ack} joint={joint}", phase)
            if any(state["velocity"]) or any(state["effort"]):
                error("nonzero virtual velocity/effort", phase)

    def until(deadline):
        while clock() < deadline:
            poll()
            sleep(min(0.0005, max(0, deadline - clock()) / 1e9))
        poll()

    try:
        rx_log = (directory / "uart_rx.bin").open("wb")
        tx_log = (directory / "uart_tx.bin").open("wb")
        if serial_factory is None:
            import serial as serial_module
            serial_factory = serial_module.Serial
        serial = serial_factory(args.port, 921600, timeout=0, write_timeout=1, exclusive=True)
        if hasattr(serial, "reset_input_buffer"):
            serial.reset_input_buffer()
        send(p.HELLO, 0)
        hello_deadline = clock() + 2_000_000_000
        while capability is None and clock() < hello_deadline:
            poll()
            sleep(0.0005)
        if capability != CAPABILITY:
            raise RuntimeError(f"wrong/missing diagnostic HELLO capability: {capability!r}")
        startup_discard = rx_bytes - valid_bytes - parser.buffered_bytes
        start = clock()
        wall_times["start_wall_utc"] = datetime.now(timezone.utc).isoformat()
        next_progress = start
        for index in range(expected_count):
            until(start + index * PERIOD_NS)
            now = clock()
            if index == warmup_count:
                measured_start = now
                wall_times["measured_start_wall_utc"] = datetime.now(timezone.utc).isoformat()
            phase = phases["warmup" if index < warmup_count else "measured"]
            lateness = max(0, now - (start + index * PERIOD_NS))
            phase["lateness_ms"].add(lateness / 1e6)
            phase["deadline_misses_over_5ms"] += lateness > PERIOD_NS
            sequence = index + 1
            pending[sequence % len(pending)] = (sequence, now)
            send(p.COMMAND, sequence, p.pack_command(targets(index), ZERO, ZERO, ZERO, ZERO, enabled=False))
            sent = sequence
            phase["commands"] += 1
            poll()
            if now >= next_progress:
                atomic_json(directory / "progress.json", report())
                next_progress = now + 5_000_000_000
        until(max(start + expected_count * PERIOD_NS,
                  measured_start + round(args.seconds * 1e9)))
        measured_end = clock()
        wall_times["measured_end_wall_utc"] = datetime.now(timezone.utc).isoformat()
        until(clock() + 250_000_000)
        discarded = rx_bytes - valid_bytes - parser.buffered_bytes - startup_discard
        if discarded:
            error(f"parser discarded {discarded} UART bytes")
        if ack != expected_count:
            error(f"final ACK {ack} != expected {expected_count}")
    except BaseException as exc:
        exception = f"{type(exc).__name__}: {exc}"
        error(exception)
        raise
    finally:
        if serial is not None:
            try:
                send(p.STOP, sent + 1)
            except Exception as exc:
                error(f"STOP failed: {exc}")
            try:
                serial.close()
            except Exception as exc:
                error(f"serial close failed: {exc}")
        for log in (rx_log, tx_log):
            if log is not None:
                try:
                    log.close()
                except Exception as exc:
                    error(f"log close failed: {exc}")
        atomic_json(directory / "summary.json", report(final=True))
    return report(final=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--mode", choices=("mit", "servo"), required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--warmup-seconds", type=float, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
        print(json.dumps(summary, allow_nan=False))
        return 0 if summary["passed"] else 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
