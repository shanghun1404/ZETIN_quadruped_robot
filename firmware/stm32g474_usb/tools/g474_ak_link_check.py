#!/usr/bin/env python3
"""USB gate check for board/g474_ak_firmware.

Sends disabled commands at 200 Hz. With --gate-test it also sends ONE
enabled command whose positions, velocities, kp, kd and effort are all
zero; with the unverified config the MCU must reject it. Ends with STOP.
"""
import argparse
import json
from pathlib import Path
import secrets
import struct
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--port', required=True)
p.add_argument('--protocol-dir', type=Path, required=True, help='directory with protocol.py')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=5.0)
p.add_argument('--gate-test', action='store_true')
a = p.parse_args()
sys.path.insert(0, str(a.protocol_dir))
import serial  # noqa: E402
from protocol import Frame, StreamParser, encode_frame, pack_command, unpack_feedback  # noqa: E402
from protocol import HELLO, HELLO_ACK, COMMAND, FEEDBACK, STOP  # noqa: E402

ZERO = [0.0] * 12
session = secrets.randbelow(0xffffffff) + 1
parser = StreamParser()
feedback = []
result = dict(port=a.port, session=session, gate_test=a.gate_test)
with serial.Serial(a.port, 921600, timeout=0, write_timeout=.3, exclusive=True) as uart:
    def send(kind, sequence=0, payload=b''):
        packet = encode_frame(Frame(kind, session, sequence, time.monotonic_ns(), payload))
        assert uart.write(packet) == len(packet)

    def poll():
        frames = parser.feed(uart.read(max(1, uart.in_waiting)))
        for frame in frames:
            if frame.kind == FEEDBACK and frame.session == session:
                feedback.append(unpack_feedback(frame.payload))
        return frames
    uart.reset_input_buffer()
    capability = None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and capability is None:
        send(HELLO)
        until = time.monotonic() + .15
        while time.monotonic() < until and capability is None:
            for frame in poll():
                if frame.kind == HELLO_ACK and frame.session == session:
                    capability, = struct.unpack('<I', frame.payload)
            time.sleep(.0002)
    result['capability'] = capability
    assert capability == 1, f'expected full-command capability 1, got {capability}'
    disabled = pack_command(ZERO, ZERO, ZERO, ZERO, ZERO, enabled=False)
    count = round(a.seconds * 200)
    start = time.monotonic_ns()
    late = 0
    for index in range(count):
        due = start + index * 5_000_000
        while time.monotonic_ns() < due:
            poll()
            time.sleep(.0001)
        late += time.monotonic_ns() - due > 5_000_000
        send(COMMAND, index + 1, disabled)
        poll()
    sequence = count
    if a.gate_test:
        sequence += 1
        send(COMMAND, sequence, pack_command(ZERO, ZERO, ZERO, ZERO, ZERO, enabled=True))
    until = time.monotonic() + .3
    while time.monotonic() < until:
        poll()
        time.sleep(.0002)
    send(STOP, sequence + 1)
    time.sleep(.05)
    poll()
result.update(disabled_commands=count, schedule_late_over_5ms=late, feedback_frames=len(feedback),
              armed_feedback=sum(f['armed'] for f in feedback),
              fault_flags_seen=sorted({f['fault_flags'] for f in feedback}))
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
