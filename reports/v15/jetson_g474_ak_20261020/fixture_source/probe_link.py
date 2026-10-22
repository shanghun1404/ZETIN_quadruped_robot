"""Diagnostic G474 VCP only: never sends a motor-enable command."""
import argparse
import json
from pathlib import Path
import secrets
import struct
import time

import serial
from protocol import Frame, StreamParser, encode_frame, pack_command, unpack_feedback
from protocol import HELLO, HELLO_ACK, COMMAND, FEEDBACK, STOP

p = argparse.ArgumentParser()
p.add_argument('--port', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=0)
p.add_argument('--capture', action='store_true')
a = p.parse_args()
session = secrets.randbelow(0xffffffff) + 1
parser = StreamParser()
feedback = []
result = dict(port=a.port, enabled=False, passed=False, session=session)
with serial.Serial(a.port, 921600, timeout=0, write_timeout=.3, exclusive=True) as uart:
    def send(kind, sequence=0, payload=b''):
        packet = encode_frame(Frame(kind, session, sequence, time.monotonic_ns(), payload))
        assert uart.write(packet) == len(packet)
    def poll():
        frames = parser.feed(uart.read(max(1, uart.in_waiting)))
        for frame in frames:
            if frame.kind == FEEDBACK and frame.session == session:
                state = unpack_feedback(frame.payload)
                assert not state['armed'] and not state['fault_flags'], state
                feedback.append(dict(received_ns=time.monotonic_ns(), stamp_ns=frame.stamp_ns, **state))
        return frames
    uart.reset_input_buffer()
    deadline = time.monotonic() + 2
    capability = None
    while time.monotonic() < deadline and capability is None:
        send(HELLO)
        until = time.monotonic() + .15
        while time.monotonic() < until:
            for frame in poll():
                if frame.kind == HELLO_ACK and frame.session == session:
                    capability, = struct.unpack('<I', frame.payload)
            time.sleep(.0002)
    assert capability == 0x80000001, 'No diagnostic G474 HELLO_ACK'
    result['capabilities'] = hex(capability)
    # Distinct position tags for all12 joints; all gains/velocity/effort zero.
    q = [.01 * (i + 1) for i in range(12)]
    payload = pack_command(enabled=False, position=q, velocity=[0.]*12,
                           kp=[0.]*12, kd=[0.]*12, effort=[0.]*12)
    count = round(a.seconds * 200) if a.seconds else int(a.capture)
    start = time.monotonic_ns()
    for index in range(count):
        due = start + index * 5_000_000
        while time.monotonic_ns() < due:
            poll();time.sleep(.0001)
        send(COMMAND, index + 1, payload)
        poll()
    until = time.monotonic() + .2
    while time.monotonic() < until:
        poll();time.sleep(.0002)
    if count:
        assert feedback and max(r['ack_sequence'] for r in feedback) == count
    send(STOP, count + 1)
    result.update(passed=True, commands_sent=count, position_tags=q, feedback_samples=len(feedback),
                  final_ack=max((r['ack_sequence'] for r in feedback), default=0))
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
