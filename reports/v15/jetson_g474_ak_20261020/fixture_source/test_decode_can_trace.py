"""Independent, host-only classic-CAN waveform fixtures (stdlib only)."""

import random
import unittest

from decode_can_trace import CANDecodeError, decode_frame


def frame_fields(identifier, payload, *, rtr=0, ide=0, reserved=0, dlc=None):
    """Build unstuffed fields; this does not call decoder code."""
    if dlc is None:
        dlc = len(payload)
    text = (
        "0" + format(identifier, "011b") + str(rtr) + str(ide)
        + str(reserved) + format(dlc, "04b")
        + "".join(format(byte, "08b") for byte in payload)
    )
    return [int(bit) for bit in text]


def extended_fields(identifier, payload, *, srr=1, ide=1, rtr=0, r1=0, r0=0,
                    dlc=None):
    """Literal CAN 2.0B layout; no parser offsets/helpers are reused."""
    if dlc is None:
        dlc = len(payload)
    identifier_bits = format(identifier, "029b")
    text = (
        "0" + identifier_bits[:11] + str(srr) + str(ide)
        + identifier_bits[11:] + str(rtr) + str(r1) + str(r0)
        + format(dlc, "04b")
        + "".join(format(byte, "08b") for byte in payload)
    )
    return [int(bit) for bit in text]


def polynomial_crc(bits):
    """Polynomial long division, independent of the decoder's CRC register."""
    dividend = int("".join(map(str, bits)), 2) << 15
    generator = 0xC599  # x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1
    while dividend.bit_length() >= generator.bit_length():
        dividend ^= generator << (dividend.bit_length() - generator.bit_length())
    return dividend


def stuff(bits):
    """Return transmitted bits and indexes of inserted complementary bits."""
    wire, indexes = [], []
    last, run = None, 0
    for bit in bits:
        wire.append(bit)
        run = run + 1 if bit == last else 1
        last = bit
        if run == 5:
            wire.append(1 - bit)
            indexes.append(len(wire) - 1)
            last, run = 1 - bit, 1
    return wire, indexes


def frame_wire(identifier, payload, *, crc_override=None, **kwargs):
    fields = frame_fields(identifier, payload, **kwargs)
    crc = polynomial_crc(fields) if crc_override is None else crc_override
    return stuff(fields + [int(bit) for bit in format(crc, "015b")])


def extended_wire(identifier, payload, **kwargs):
    fields = extended_fields(identifier, payload, **kwargs)
    return stuff(fields + [int(bit) for bit in format(polynomial_crc(fields), "015b")])


def waveform(wire, *, phase=0, jitter=0, tail=(1,), start=1293,
             clock_hz=160_000_000, bitrate=1_000_000):
    """Sample NRZ at ~0.25 us with optional independent timestamp jitter.

    SOF occurs at `start`; the capture begins with more than 7 idle bit times.
    `tail` starts with the CRC delimiter. No ACK or EOF is needed by default.
    """
    rng = random.Random(20261020)
    bit_cycles = clock_hz / bitrate
    bits = wire + list(tail)
    last = start + len(bits) * bit_cycles
    samples, cycle = [], float(phase)
    while cycle <= last:
        index = int((cycle - start) // bit_cycles)
        level = 1 if cycle < start else bits[min(index, len(bits) - 1)]
        samples.append((int(cycle), level))
        cycle += 40 + rng.randint(-jitter, jitter)
    return samples


class DecodeFrameTests(unittest.TestCase):
    def decode(self, wire, **kwargs):
        try:
            result = decode_frame(waveform(wire, **kwargs))
        except CANDecodeError as error:
            self.fail(f"A valid TX prefix was rejected: {error}")
        self.assertIsInstance(result, dict)
        return result

    def test_recovers_literal_ids_payloads_and_crc_without_ack_or_eof(self):
        # Catches wrong SOF/ID/DLC offsets, MSB order, or CRC polynomial/init.
        fixtures = [
            (1, "0102030405060708", 0x7FE1),
            (2, "ffffffffffffffff", 0x64A7),
            (3, "0000000000000000", 0x0976),
            (4, "80a55a00ff012345", 0x3682),
        ]
        for identifier, payload_hex, expected_crc in fixtures:
            with self.subTest(identifier=identifier):
                wire, inserted = frame_wire(identifier, bytes.fromhex(payload_hex))
                result = self.decode(wire)
                self.assertEqual(result["id"], identifier)
                self.assertEqual(result["dlc"], 8)
                self.assertEqual(result["payload"], bytes.fromhex(payload_hex))
                self.assertEqual(result["payload_hex"], payload_hex)
                self.assertEqual(result["crc"], expected_crc)
                self.assertIs(result["crc_valid"], True)
                self.assertEqual(result["stuffed_bit_count"], len(inserted))

    def test_standard_frames_are_explicitly_marked_as_not_extended(self):
        # Catches guessing the frame format from the numeric ID alone.
        wire, _ = frame_wire(0x401, bytes.fromhex("01020304"))
        result = self.decode(wire)
        self.assertIn("extended", result)
        self.assertIs(result["extended"], False)

    def test_extended_ak_ids_dlc4_dlc8_and_all_29_identifier_bits(self):
        # Catches 11-bit truncation, wrong EID join/DLC offsets, and CRC scope.
        fixtures = [
            (0x00000801, "8000800000000000", 0x33DC),
            (0x00000801, "01020304", 0x5A95),
            (0x00000401, "ffffffff", 0x1635),
            (0x00000401, "0102030405060708", 0x1040),
            (0x00002901, "12345678", 0x5644),
            (0x00002901, "7fffa55000112233", 0x2728),
            (0x1ABCDEFF, "01020304", 0x2E4D),
            (0x1FFFFFFF, "0000000000000000", 0x7DA3),
        ]
        for identifier, payload_hex, expected_crc in fixtures:
            with self.subTest(identifier=identifier, payload_hex=payload_hex):
                wire, inserted = extended_wire(identifier, bytes.fromhex(payload_hex))
                result = self.decode(wire)
                self.assertEqual(result["id"], identifier)
                self.assertIs(result["extended"], True)
                self.assertEqual(result["dlc"], len(payload_hex) // 2)
                self.assertEqual(result["payload"], bytes.fromhex(payload_hex))
                self.assertEqual(result["payload_hex"], payload_hex)
                self.assertEqual(result["crc"], expected_crc)
                self.assertIs(result["crc_valid"], True)
                self.assertEqual(result["stuffed_bit_count"], len(inserted))
                self.assertEqual(result["bit_count"], len(wire) + 1)

    def test_extended_frame_sampling_phase_and_jitter(self):
        # Catches a parser that depends on an aligned or fixed sample cadence.
        wire, _ = extended_wire(0x1ABCDEFF, bytes.fromhex("0102030405060708"))
        for phase in (0, 19, 37):
            with self.subTest(phase=phase):
                result = self.decode(wire, phase=phase, jitter=6)
                self.assertEqual(result["id"], 0x1ABCDEFF)
                self.assertEqual(result["payload_hex"], "0102030405060708")

    def test_corrupt_extended_base_or_extension_id_without_crc_update_is_rejected(self):
        # Re-stuffing isolates CRC protection over both parts of the EID.
        original = extended_fields(0x1ABCDEFF, bytes.fromhex("01020304"))
        for bit_index in (1, 11, 14, 31):
            with self.subTest(bit_index=bit_index):
                altered = original.copy()
                altered[bit_index] ^= 1
                wire, _ = stuff(altered + [int(bit) for bit in
                                          format(polynomial_crc(original), "015b")])
                with self.assertRaisesRegex(CANDecodeError, "CRC"):
                    decode_frame(waveform(wire))

    def test_invalid_extended_srr_ide_rtr_reserved_or_dlc_are_rejected(self):
        # Each malformed header has its own matching CRC and legal stuffing.
        for field in ({"srr": 0}, {"ide": 0}, {"rtr": 1},
                      {"r1": 1}, {"r0": 1}, {"dlc": 9}):
            with self.subTest(field=field):
                wire, _ = extended_wire(0x801, bytes.fromhex("01020304"), **field)
                with self.assertRaises(CANDecodeError):
                    decode_frame(waveform(wire))

    def test_extended_stuffing_violation_is_rejected(self):
        # Catches accidentally applying destuffing only to standard frames.
        wire, inserted = extended_wire(0x801, bytes(8))
        wire[inserted[0]] ^= 1
        with self.assertRaisesRegex(CANDecodeError, "[Ss]tuff"):
            decode_frame(waveform(wire))

    def test_sampling_phase_and_timestamp_jitter_do_not_change_message(self):
        # Catches assuming SOF aligns with a sample or an exact sample cadence.
        wire, _ = frame_wire(4, bytes.fromhex("80a55a00ff012345"))
        for phase in (0, 7, 19, 37):
            for jitter in (0, 3, 6):
                with self.subTest(phase=phase, jitter=jitter):
                    result = self.decode(wire, phase=phase, jitter=jitter)
                    self.assertEqual((result["id"], result["payload_hex"]),
                                     (4, "80a55a00ff012345"))

    def test_all_classic_payload_lengths_and_maximum_standard_id(self):
        # Catches treating DLC as bytes in a fixed eight-byte-only layout.
        for size in range(9):
            with self.subTest(size=size):
                wire, _ = frame_wire(0x7FF, bytes(range(size)))
                result = self.decode(wire)
                self.assertEqual(result["id"], 0x7FF)
                self.assertEqual(result["dlc"], size)
                self.assertEqual(result["payload"], bytes(range(size)))

    def test_ack_error_flag_after_crc_is_allowed_for_tx_only_capture(self):
        # Catches incorrectly requiring ACK/EOF after a valid transmitted prefix.
        wire, _ = frame_wire(1, bytes.fromhex("0102030405060708"))
        result = self.decode(wire, tail=(1, 1, 0, 0, 0, 0, 0, 0, 1, 1))
        self.assertEqual(result["payload_hex"], "0102030405060708")

    def test_stuff_bit_after_final_crc_bit_precedes_crc_delimiter(self):
        # Catches ending destuffing one bit too early at the CRC boundary.
        for identifier, expected_crc in ((9, 0x7C20), (23, 0x521F)):
            with self.subTest(identifier=identifier):
                wire, inserted = frame_wire(identifier, b"")
                self.assertEqual(inserted[-1], len(wire) - 1)
                result = self.decode(wire)
                self.assertEqual(result["crc"], expected_crc)
                self.assertEqual(result["bit_count"], len(wire) + 1)

    def test_payload_changed_without_updating_crc_is_rejected(self):
        # Re-stuff changed data, so rejection must come from CRC, not stuffing.
        original = frame_fields(4, bytes.fromhex("80a55a00ff012345"))
        altered = original.copy()
        altered[19 + 13] ^= 1
        wire, _ = stuff(altered + [int(b) for b in format(polynomial_crc(original), "015b")])
        with self.assertRaisesRegex(CANDecodeError, "CRC"):
            decode_frame(waveform(wire))

    def test_changed_crc_with_legal_stuffing_is_rejected(self):
        wire, _ = frame_wire(1, bytes.fromhex("0102030405060708"),
                             crc_override=0x7FE0)
        with self.assertRaisesRegex(CANDecodeError, "CRC"):
            decode_frame(waveform(wire))

    def test_six_equal_bits_in_stuffed_region_are_rejected(self):
        wire, inserted = frame_wire(3, bytes(8))
        wire[inserted[0]] ^= 1
        with self.assertRaisesRegex(CANDecodeError, "[Ss]tuff"):
            decode_frame(waveform(wire))

    def test_dominant_crc_delimiter_is_rejected(self):
        wire, _ = frame_wire(1, bytes.fromhex("0102030405060708"))
        with self.assertRaisesRegex(CANDecodeError, "delimiter"):
            decode_frame(waveform(wire, tail=(0, 1, 1)))

    def test_remote_reserved_large_dlc_and_malformed_ide_are_rejected(self):
        # Catches interpreting other frame formats as this classic-data contract.
        for field in ({"rtr": 1}, {"ide": 1}, {"reserved": 1}, {"dlc": 9}):
            with self.subTest(field=field):
                wire, _ = frame_wire(1, b"", **field)
                with self.assertRaises(CANDecodeError):
                    decode_frame(waveform(wire))

    def test_capture_truncated_during_crc_is_rejected(self):
        wire, _ = frame_wire(1, bytes.fromhex("0102030405060708"))
        with self.assertRaisesRegex(CANDecodeError, "[Tt]runcat"):
            decode_frame(waveform(wire)[:-30])

    def test_idle_or_capture_starting_low_cannot_supply_sof(self):
        # Catches guessing a SOF without an observed preceding idle interval.
        for samples in ([(i * 40, 1) for i in range(100)],
                        [(i * 40, 0) for i in range(100)]):
            with self.subTest(level=samples[0][1]):
                with self.assertRaisesRegex(CANDecodeError, "SOF"):
                    decode_frame(samples)

    def test_invalid_samples_and_timing_are_rejected(self):
        # Catches silent acceptance of wrapped clocks, invalid levels or rates.
        for samples in ([], [(0, 1)], [(40, 1), (0, 0)],
                        [(0, 1), (40, 2)], [(0, 1), (float("nan"), 0)]):
            with self.subTest(samples=samples):
                with self.assertRaises(CANDecodeError):
                    decode_frame(samples)
        for options in ({"bitrate": 0}, {"clock_hz": 0}, {"bitrate": -1}):
            with self.subTest(options=options):
                with self.assertRaises(CANDecodeError):
                    decode_frame([(0, 1), (40, 0)], **options)

    def test_missing_dense_samples_within_frame_are_rejected(self):
        # Catches deriving a CRC from intervals with insufficient observations.
        wire, _ = frame_wire(1, bytes.fromhex("0102030405060708"))
        samples = waveform(wire)
        samples = samples[:100] + samples[104:]
        with self.assertRaisesRegex(CANDecodeError, "[Ss]ample gap"):
            decode_frame(samples)


if __name__ == "__main__":
    unittest.main()
