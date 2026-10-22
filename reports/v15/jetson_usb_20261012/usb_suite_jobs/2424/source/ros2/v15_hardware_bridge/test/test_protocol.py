"""USB wire contracts, rejection paths, and real operating-system PTY tests."""

import binascii
import importlib
import os
from pathlib import Path
import select
import struct
import sys
import time
import tty

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
HEADER = struct.Struct("<2sBBIIQH")
HELLO_GOLDEN = bytes.fromhex("51440101040302014433221108070605040302010000ffd5")


@pytest.fixture
def protocol():
    class ProtocolUnderTest:
        def __getattr__(self, name):
            module_path = PACKAGE_ROOT / "v15_hardware_bridge" / "protocol.py"
            assert module_path.is_file(), "USB protocol implementation is missing"
            return getattr(importlib.import_module("v15_hardware_bridge.protocol"), name)

    return ProtocolUnderTest()


def wire(kind, payload=b"", *, session=1, sequence=7, stamp_ns=123, version=1,
         claimed_length=None):
    """Independent reference encoder using Python's standard CRC primitive."""
    length = len(payload) if claimed_length is None else claimed_length
    data = HEADER.pack(b"QD", version, kind, session, sequence, stamp_ns, length) + payload
    return data + struct.pack("<H", binascii.crc_hqx(data, 0xFFFF))


def command_vectors():
    return {
        "position": [0.25 * (i - 6) for i in range(12)],
        "velocity": [0.5 * i for i in range(12)],
        "kp": [10.0 + i for i in range(12)],
        "kd": [0.125 * i for i in range(12)],
        "effort": [(-1.0 if i % 2 else 1.0) * (i + 1) for i in range(12)],
    }


def test_crc_known_ccitt_false_vector(protocol):
    assert protocol.crc16(b"123456789") == 0x29B1
    assert protocol.crc16(b"") == 0xFFFF


def test_hello_frame_matches_literal_little_endian_vector(protocol):
    frame = protocol.Frame(1, 0x01020304, 0x11223344, 0x0102030405060708, b"")
    assert protocol.encode_frame(frame) == HELLO_GOLDEN
    assert protocol.StreamParser().feed(HELLO_GOLDEN) == [frame]


def test_command_arrays_have_canonical_joint_order_and_si_fields(protocol):
    assert protocol.JOINT_NAMES == (
        "FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE",
        "RL_HAA", "RL_HFE", "RL_KFE", "RR_HAA", "RR_HFE", "RR_KFE",
    )
    vectors = command_vectors()
    payload = protocol.pack_command(**vectors, enabled=True)
    assert len(payload) == 244
    assert struct.unpack_from("<I", payload, 0) == (1,)
    # Each literal offset represents one complete block of 12 float32 values.
    for offset, key in ((4, "position"), (52, "velocity"), (100, "kp"),
                        (148, "kd"), (196, "effort")):
        assert struct.unpack_from("<12f", payload, offset) == pytest.approx(vectors[key])
    decoded = protocol.unpack_command(payload)
    assert decoded["enabled"] is True
    for key in vectors:
        assert decoded[key] == pytest.approx(vectors[key])
    assert struct.unpack_from("<I", protocol.pack_command(**vectors), 0) == (0,)


def test_feedback_wire_has_ack_fault_armed_before_measurements(protocol):
    position = [0.5 * i for i in range(12)]
    velocity = [-0.25 * i for i in range(12)]
    effort = [1.0 + i for i in range(12)]
    payload = protocol.pack_feedback(position, velocity, effort,
                                     ack_sequence=0xFFFFFFFF, fault_flags=9, armed=True)
    assert len(payload) == 156
    assert struct.unpack_from("<III", payload) == (0xFFFFFFFF, 9, 1)
    for offset, expected in ((12, position), (60, velocity), (108, effort)):
        assert struct.unpack_from("<12f", payload, offset) == pytest.approx(expected)
    decoded = protocol.unpack_feedback(payload)
    assert decoded["ack_sequence"] == 0xFFFFFFFF
    assert decoded["fault_flags"] == 9
    assert decoded["armed"] is True
    assert decoded["position"] == pytest.approx(position)
    assert decoded["velocity"] == pytest.approx(velocity)
    assert decoded["effort"] == pytest.approx(effort)
    assert struct.unpack_from("<III", protocol.pack_feedback(position, velocity, effort)) == (0, 0, 0)


@pytest.mark.parametrize("field,index,value", [
    ("position", 0, 12.5001), ("position", 0, -12.5001),
    ("velocity", 1, 50.0001), ("velocity", 1, -50.0001),
    ("kp", 2, -0.01), ("kp", 2, 300.01),
    ("kd", 3, -0.01), ("kd", 3, 8.01),
    ("effort", 0, 24.8001), ("effort", 3, -24.8001),
    ("effort", 1, 48.001), ("effort", 2, -48.001),
    ("position", 6, float("nan")), ("velocity", 7, float("inf")),
    ("effort", 8, float("-inf")), ("kp", 4, float("nan")),
    ("kd", 5, float("inf")),
])
def test_pack_command_rejects_unsafe_values(protocol, field, index, value):
    vectors = command_vectors()
    vectors[field][index] = value
    with pytest.raises(ValueError):
        protocol.pack_command(**vectors, enabled=True)


@pytest.mark.parametrize("field,index,value", [
    ("position", 0, 12.5001), ("velocity", 0, -50.001),
    ("kp", 0, -1.0), ("kp", 0, 300.01),
    ("kd", 0, 8.01), ("effort", 0, 24.801),
    ("effort", 1, 48.001), ("position", 0, float("nan")),
])
def test_unpack_command_validates_untrusted_wire_values(protocol, field, index, value):
    vectors = command_vectors()
    vectors[field][index] = value
    values = [item for key in ("position", "velocity", "kp", "kd", "effort")
              for item in vectors[key]]
    payload = struct.pack("<I60f", 1, *values)
    with pytest.raises(ValueError):
        protocol.unpack_command(payload)


def test_command_accepts_inclusive_format_limits(protocol):
    payload = protocol.pack_command([12.5, -12.5] * 6, [50.0, -50.0] * 6,
                                    [300.0] * 12, [8.0] * 12,
                                    [24.8, 48.0, -48.0] * 4, enabled=True)
    assert protocol.unpack_command(payload)["effort"] == pytest.approx([24.8, 48.0, -48.0] * 4)


@pytest.mark.parametrize("length", [0, 11, 13])
def test_command_requires_twelve_values_per_field(protocol, length):
    vectors = command_vectors()
    vectors["velocity"] = [0.0] * length
    with pytest.raises(ValueError):
        protocol.pack_command(**vectors)


@pytest.mark.parametrize("decoder,length", [("unpack_command", 243), ("unpack_command", 245),
                                           ("unpack_feedback", 155), ("unpack_feedback", 157)])
def test_payload_decoders_reject_wrong_length(protocol, decoder, length):
    with pytest.raises(ValueError):
        getattr(protocol, decoder)(bytes(length))


@pytest.mark.parametrize("flag", [2, -1, "yes"])
def test_pack_flags_require_zero_or_one(protocol, flag):
    with pytest.raises(ValueError):
        protocol.pack_command(**command_vectors(), enabled=flag)
    with pytest.raises(ValueError):
        protocol.pack_feedback([0.] * 12, [0.] * 12, [0.] * 12, armed=flag)


def test_decode_flags_reject_non_boolean_uint32(protocol):
    with pytest.raises(ValueError):
        protocol.unpack_command(struct.pack("<I60f", 2, *([0.] * 60)))
    with pytest.raises(ValueError):
        protocol.unpack_feedback(struct.pack("<III36f", 0, 0, 2, *([0.] * 36)))


@pytest.mark.parametrize("index,value", [(0, float("nan")), (12, float("inf")), (24, float("-inf"))])
def test_feedback_rejects_nonfinite_measurements(protocol, index, value):
    values = [0.] * 36
    values[index] = value
    with pytest.raises(ValueError):
        protocol.unpack_feedback(struct.pack("<III36f", 0, 0, 0, *values))
    with pytest.raises(ValueError):
        protocol.pack_feedback(values[:12], values[12:24], values[24:])


@pytest.mark.parametrize("ack,fault", [(-1, 0), (0x100000000, 0), (0, -1), (0, 0x100000000)])
def test_feedback_metadata_stays_uint32(protocol, ack, fault):
    with pytest.raises(ValueError):
        protocol.pack_feedback([0.] * 12, [0.] * 12, [0.] * 12,
                              ack_sequence=ack, fault_flags=fault)


@pytest.mark.parametrize("a,b,expected", [
    (1, 0, True), (0, 0, False), (0, 0xFFFFFFFF, True),
    (0xFFFFFFFF, 0, False), (0x7FFFFFFF, 0, True),
    (0x80000000, 0, False), (0, 0x80000000, False),
    (15, 17, False), (0xFFFFFFFE, 0xFFFFFFFD, True),
])
def test_sequence_comparison_handles_replay_wrap_and_ambiguity(protocol, a, b, expected):
    assert protocol.newer_sequence(a, b) is expected


@pytest.mark.parametrize("a,b", [(-1, 0), (0, -1), (0x100000000, 0), (0, 0x100000000)])
def test_sequence_comparison_rejects_values_outside_uint32(protocol, a, b):
    with pytest.raises(ValueError):
        protocol.newer_sequence(a, b)


@pytest.mark.parametrize("kind,payload", [(1, b""), (2, struct.pack("<I", 1)),
                                        (3, bytes(244)), (4, bytes(156)), (5, b"")])
def test_parser_accepts_each_declared_frame_kind(protocol, kind, payload):
    result = protocol.StreamParser().feed(wire(kind, payload))
    assert result == [protocol.Frame(kind, 1, 7, 123, payload)]


def test_parser_recovers_split_magic_fragmented_and_coalesced_frames(protocol):
    parser = protocol.StreamParser()
    frame1 = protocol.Frame(1, 1, 7, 123, b"")
    frame2 = protocol.Frame(5, 1, 8, 128, b"")
    assert parser.feed(b"garbageQ") == []
    assert parser.feed(HELLO_GOLDEN[1:3]) == []
    # Discard the incomplete noisy prefix when the actual frame arrives.
    combined = wire(1) + wire(5, sequence=8, stamp_ns=128)
    collected = []
    for octet in combined:
        collected.extend(parser.feed(bytes([octet])))
    assert collected == [frame1, frame2]
    assert protocol.StreamParser().feed(combined) == [frame1, frame2]


@pytest.mark.parametrize("bad", [
    wire(1, version=2), wire(99), wire(1, session=0),
    wire(1, b"x"), wire(2), wire(3, bytes(244), claimed_length=243),
    wire(4, bytes(156), claimed_length=257), wire(5, b"x"),
    HELLO_GOLDEN[:-1] + bytes([HELLO_GOLDEN[-1] ^ 1]),
])
def test_parser_rejects_bad_header_length_session_crc_then_resynchronizes(protocol, bad):
    result = protocol.StreamParser().feed(bad + wire(5, sequence=99))
    assert result == [protocol.Frame(5, 1, 99, 123, b"")]


def test_parser_preserves_frames_coalesced_in_large_input_and_bounds_noise(protocol):
    parser = protocol.StreamParser()
    expected = [protocol.Frame(1, 1, i, 123, b"") for i in range(1000)]
    assert parser.feed(b"!" * 200000 + b"".join(wire(1, sequence=i) for i in range(1000))) == expected
    assert parser.buffered_bytes <= 280
    assert parser.feed(b"!" * 200000 + b"Q") == []
    assert parser.buffered_bytes == 1
    assert parser.feed(wire(1)[1:]) == [protocol.Frame(1, 1, 7, 123, b"")]


@pytest.mark.parametrize("fragmented", [False, True])
def test_parser_recovers_complete_ack_after_truncated_command(protocol, fragmented):
    parser = protocol.StreamParser()
    truncated = wire(3, bytes(244))[:HEADER.size]
    ack = wire(2, struct.pack("<I", 1), sequence=99)
    expected = protocol.Frame(2, 1, 99, 123, struct.pack("<I", 1))
    received = []
    chunks = [bytes([value]) for value in truncated + ack] if fragmented else [truncated + ack]
    for chunk in chunks:
        received.extend(parser.feed(chunk))
    assert received == [expected]
    assert parser.buffered_bytes == 0


def test_parser_keeps_fragmented_command_with_bad_crc_magic_in_payload(protocol):
    nested = bytearray(wire(2, struct.pack("<I", 1)))
    nested[-1] ^= 1
    payload = (b"prefix" + nested).ljust(244, b"\x00")
    outer = wire(3, payload)
    parser = protocol.StreamParser()
    assert parser.feed(outer[:-2]) == []
    assert parser.feed(outer[-2:]) == [protocol.Frame(3, 1, 7, 123, payload)]


@pytest.mark.parametrize("kind,session,sequence,stamp_ns,payload", [
    (99, 1, 0, 0, b""), (1, 0, 0, 0, b""), (1, -1, 0, 0, b""),
    (1, 0x100000000, 0, 0, b""), (1, 1, -1, 0, b""),
    (1, 1, 0x100000000, 0, b""), (1, 1, 0, -1, b""),
    (1, 1, 0, 0x10000000000000000, b""), (1, 1, 0, 0, b"x"),
    (3, 1, 0, 0, bytes(243)), (4, 1, 0, 0, bytes(157)),
])
def test_encoder_rejects_invalid_metadata_and_type_lengths(protocol, kind, session,
                                                           sequence, stamp_ns, payload):
    with pytest.raises(ValueError):
        protocol.encode_frame(protocol.Frame(kind, session, sequence, stamp_ns, payload))


def receive_frames(fd, parser, count):
    frames = []
    deadline = time.monotonic() + 2.0
    while len(frames) < count and time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], max(0., deadline - time.monotonic()))
        if readable:
            frames.extend(parser.feed(os.read(fd, 17)))
    assert len(frames) == count, "PTY did not return the expected complete USB frames"
    return frames


def test_real_pty_command_feedback_roundtrip_with_noise_and_partial_reads(protocol):
    master, slave = os.openpty()
    try:
        tty.setraw(slave)
        command = protocol.Frame(3, 0xABCDEF01, 0xFFFFFFFF, 5000000,
                                 protocol.pack_command(**command_vectors(), enabled=True))
        command_bytes = protocol.encode_frame(command)
        damaged = bytearray(command_bytes)
        damaged[-2] ^= 0x80
        os.write(slave, b"noise" + damaged)
        for start in range(0, len(command_bytes), 11):
            os.write(slave, command_bytes[start:start + 11])
        received = receive_frames(master, protocol.StreamParser(), 1)
        assert received == [command]
        assert protocol.unpack_command(received[0].payload)["effort"] == pytest.approx(
            [1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12.])

        feedback = protocol.Frame(4, command.session, 0, 10000000,
                                  protocol.pack_feedback([0.1] * 12, [0.2] * 12,
                                                         [0.3] * 12,
                                                         ack_sequence=command.sequence, armed=True))
        os.write(master, protocol.encode_frame(feedback))
        received = receive_frames(slave, protocol.StreamParser(), 1)
        assert received == [feedback]
        decoded = protocol.unpack_feedback(received[0].payload)
        assert decoded["ack_sequence"] == 0xFFFFFFFF
        assert decoded["armed"] is True
        assert decoded["position"] == pytest.approx([0.1] * 12)
    finally:
        os.close(master)
        os.close(slave)
