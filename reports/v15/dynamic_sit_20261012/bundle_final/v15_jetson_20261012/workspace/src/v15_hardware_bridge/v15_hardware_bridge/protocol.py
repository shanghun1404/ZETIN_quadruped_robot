"""Version 1 USB wire protocol shared by the Jetson and STM32.

Joint arrays use output-shaft SI units and FL/FR/RL/RR × HAA/HFE/KFE
order. Framing is stateless: session/replay handling and watchdogs belong
to the endpoint controllers, after a frame has been parsed successfully.
"""

import binascii
from dataclasses import dataclass
import math
import operator
import struct
from typing import Iterable


JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR")
                    for joint in ("HAA", "HFE", "KFE"))
HELLO = 1
HELLO_ACK = 2
COMMAND = 3
FEEDBACK = 4
STOP = 5
VERSION = 1
MAGIC = b"QD"
MAX_PAYLOAD_SIZE = 256
HEADER = struct.Struct("<2sBBIIQH")
CRC = struct.Struct("<H")
COMMAND_PAYLOAD = struct.Struct("<I60f")
FEEDBACK_PAYLOAD = struct.Struct("<III36f")
PAYLOAD_LENGTHS = {HELLO: 0, HELLO_ACK: 4, COMMAND: 244, FEEDBACK: 156, STOP: 0}
MAX_FRAME_SIZE = HEADER.size + MAX_PAYLOAD_SIZE + CRC.size
EFFORT_LIMITS = (24.8, 48.0, 48.0) * 4


@dataclass(frozen=True)
class Frame:
    """One decoded USB frame; ``stamp_ns`` retains the sender's time."""

    kind: int
    session: int
    sequence: int
    stamp_ns: int
    payload: bytes


def crc16(data: bytes) -> int:
    """CRC16-CCITT-FALSE: polynomial 0x1021, init 0xffff, xorout 0."""
    return binascii.crc_hqx(data, 0xFFFF)


def _unsigned(value: int, bits: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer") from exc
    if not 0 <= result < (1 << bits):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return result


def _flag(value: bool, name: str) -> int:
    result = _unsigned(value, 32, name)
    if result not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return result


def _payload_bytes(payload: bytes, length: int) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("payload must be bytes")
    result = bytes(payload)
    if len(result) != length:
        raise ValueError(f"payload must contain exactly {length} bytes")
    return result


def encode_frame(frame: Frame) -> bytes:
    """Encode a frame after validating its metadata and declared type size."""
    kind = _unsigned(frame.kind, 8, "kind")
    if kind not in PAYLOAD_LENGTHS:
        raise ValueError("unknown frame kind")
    session = _unsigned(frame.session, 32, "session")
    if session == 0:
        raise ValueError("session must be nonzero")
    sequence = _unsigned(frame.sequence, 32, "sequence")
    stamp_ns = _unsigned(frame.stamp_ns, 64, "stamp_ns")
    payload = _payload_bytes(frame.payload, PAYLOAD_LENGTHS[kind])
    data = HEADER.pack(MAGIC, VERSION, kind, session, sequence, stamp_ns, len(payload)) + payload
    return data + CRC.pack(crc16(data))


class StreamParser:
    """Parse fragmented/coalesced frames with at most 280 buffered bytes.

    Invalid headers and CRCs advance by one byte so an embedded subsequent
    magic marker survives. Noise retains only a final ``Q`` for split magic.
    """

    def __init__(self):
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes) -> list[Frame]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("stream data must be bytes")
        view = memoryview(data).cast("B")
        frames = []
        offset = 0
        while offset < len(view):
            count = min(MAX_FRAME_SIZE - len(self._buffer), len(view) - offset)
            self._buffer.extend(view[offset:offset + count])
            offset += count
            self._drain(frames)
        return frames

    def _drain(self, frames: list[Frame]) -> None:
        while self._buffer:
            start = self._buffer.find(MAGIC)
            if start < 0:
                tail = self._buffer[-1:] if self._buffer[-1] == MAGIC[0] else b""
                self._buffer[:] = tail
                return
            if start:
                del self._buffer[:start]
            if len(self._buffer) < HEADER.size:
                return
            _, version, kind, session, sequence, stamp_ns, length = HEADER.unpack_from(self._buffer)
            if (version != VERSION or kind not in PAYLOAD_LENGTHS or session == 0
                    or length > MAX_PAYLOAD_SIZE or length != PAYLOAD_LENGTHS[kind]):
                del self._buffer[0]
                continue
            total = HEADER.size + length + CRC.size
            if len(self._buffer) < total:
                # A corrupt/truncated large frame must not hide a later
                # complete frame. Only a valid header AND CRC justify
                # discarding the prefix; payload magic alone is insufficient.
                later = self._complete_candidate_after_prefix()
                if later is not None:
                    del self._buffer[:later]
                    continue
                return
            expected = CRC.unpack_from(self._buffer, total - CRC.size)[0]
            if crc16(self._buffer[:total - CRC.size]) != expected:
                del self._buffer[0]
                continue
            payload = bytes(self._buffer[HEADER.size:total - CRC.size])
            frames.append(Frame(kind, session, sequence, stamp_ns, payload))
            del self._buffer[:total]

    def _complete_candidate_after_prefix(self) -> int | None:
        start = self._buffer.find(MAGIC, 1)
        while start >= 0 and len(self._buffer) - start >= HEADER.size:
            _, version, kind, session, _, _, length = HEADER.unpack_from(self._buffer, start)
            total = HEADER.size + length + CRC.size
            if (version == VERSION and session != 0
                    and length == PAYLOAD_LENGTHS.get(kind)
                    and len(self._buffer) - start >= total):
                end = start + total
                expected = CRC.unpack_from(self._buffer, end - CRC.size)[0]
                if crc16(self._buffer[start:end - CRC.size]) == expected:
                    return start
            start = self._buffer.find(MAGIC, start + 1)
        return None


def newer_sequence(a: int, b: int) -> bool:
    """True for a forward uint32 distance strictly smaller than half a cycle."""
    delta = (_unsigned(a, 32, "sequence") - _unsigned(b, 32, "sequence")) & 0xFFFFFFFF
    return 0 < delta < 0x80000000


def _vector(values: Iterable[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain 12 finite numbers") from exc
    if len(result) != 12 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain 12 finite numbers")
    return result


def _command_vectors(position, velocity, kp, kd, effort):
    vectors = {
        "position": _vector(position, "position"),
        "velocity": _vector(velocity, "velocity"),
        "kp": _vector(kp, "kp"),
        "kd": _vector(kd, "kd"),
        "effort": _vector(effort, "effort"),
    }
    for name, low, high in (("position", -12.5, 12.5), ("velocity", -50.0, 50.0),
                            ("kp", 0.0, 300.0), ("kd", 0.0, 8.0)):
        if any(value < low or value > high for value in vectors[name]):
            raise ValueError(f"{name} is outside its USB format limits")
    if any(abs(value) > limit for value, limit in zip(vectors["effort"], EFFORT_LIMITS)):
        raise ValueError("effort is outside its joint torque limits")
    return vectors


def pack_command(position, velocity, kp, kd, effort, enabled=False) -> bytes:
    """Encode all 12 position/velocity/gain/feedforward targets atomically."""
    vectors = _command_vectors(position, velocity, kp, kd, effort)
    values = tuple(value for vector in vectors.values() for value in vector)
    return COMMAND_PAYLOAD.pack(_flag(enabled, "enabled"), *values)


def unpack_command(payload: bytes) -> dict:
    """Decode and validate untrusted COMMAND values before they are used."""
    values = COMMAND_PAYLOAD.unpack(_payload_bytes(payload, COMMAND_PAYLOAD.size))
    enabled = bool(_flag(values[0], "enabled"))
    decoded = _command_vectors(*(values[start:start + 12] for start in (1, 13, 25, 37, 49)))
    decoded["enabled"] = enabled
    return decoded


def pack_feedback(position, velocity, effort, ack_sequence=0, fault_flags=0, armed=False) -> bytes:
    """Encode finite measured joint state with command acknowledgement."""
    values = (_vector(position, "position") + _vector(velocity, "velocity")
              + _vector(effort, "effort"))
    ack = _unsigned(ack_sequence, 32, "ack_sequence")
    fault = _unsigned(fault_flags, 32, "fault_flags")
    arm = _flag(armed, "armed")
    try:
        return FEEDBACK_PAYLOAD.pack(ack, fault, arm, *values)
    except (OverflowError, struct.error) as exc:
        raise ValueError("feedback values must fit IEEE754 float32") from exc


def unpack_feedback(payload: bytes) -> dict:
    """Decode measured values without applying target-command bounds to them."""
    values = FEEDBACK_PAYLOAD.unpack(_payload_bytes(payload, FEEDBACK_PAYLOAD.size))
    return {
        "ack_sequence": values[0],
        "fault_flags": values[1],
        "armed": bool(_flag(values[2], "armed")),
        "position": _vector(values[3:15], "position"),
        "velocity": _vector(values[15:27], "velocity"),
        "effort": _vector(values[27:39], "effort"),
    }
