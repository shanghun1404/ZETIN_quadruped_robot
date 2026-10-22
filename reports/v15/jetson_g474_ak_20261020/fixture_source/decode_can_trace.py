"""Decode a sampled MCU classic-CAN TX prefix (host-only; stdlib only).

This proves the encoded TX prefix through its CRC delimiter, not delivery on a
CAN bus. ACK, ACK delimiter and EOF are intentionally not required: a bare MCU
can emit an ACK-error flag immediately after the recessive ACK slot.

Protocol reference: Bosch CAN Specification 2.0, part B, sections 3.1.1 and 6
(original Bosch document mirrored at https://tech-tools.com/files/can2spec.pdf).
"""

from bisect import bisect_left
from math import isfinite


class CANDecodeError(ValueError):
    """A capture cannot prove a supported, CRC-correct CAN TX prefix."""


def decode_frame(samples, *, bitrate=1_000_000, clock_hz=160_000_000):
    """Return one CRC-correct standard or extended classic data-frame TX prefix.

    ``id`` contains the complete 11-bit or 29-bit identifier; ``extended`` is a
    boolean taken from IDE, including for extended IDs numerically below 0x800.

    ``samples`` is a sequence of ``(cycle, pin_high)`` pairs. Cycles must be
    finite and strictly increasing (unwrap DWT rollover before calling), and
    levels must be 0/1: low is dominant, high is recessive. The first SOF must
    follow at least three observed idle bit times. Samples must cover the CRC
    delimiter, with no gaps larger than half a nominal bit during the frame.

    Timing uses the observed SOF edge and nominal bit centres. This is for a
    dense, single-controller trace at a known clock and bitrate; it does not
    perform receiver clock resynchronization or automatic bitrate detection.
    Raises CANDecodeError for unsupported, malformed or insufficient captures.
    ``crc_valid`` describes the transmitted prefix only, never ACK/delivery.
    """
    if (not isfinite(bitrate) or not isfinite(clock_hz)
            or bitrate <= 0 or clock_hz <= 0):
        raise CANDecodeError("Clock and bitrate must be positive and finite")
    bit_cycles = clock_hz / bitrate
    samples = list(samples)
    if len(samples) < 2:
        raise CANDecodeError("At least two samples are required to observe SOF")
    cycles, levels = [], []
    for cycle, level in samples:
        if not isfinite(cycle) or (cycles and cycle <= cycles[-1]):
            raise CANDecodeError("Sample cycles must be finite and strictly increasing")
        if level not in (0, 1):
            raise CANDecodeError("Sample pin levels must be 0 or 1")
        cycles.append(cycle)
        levels.append(int(level))

    idle_start = cycles[0] if levels[0] else None
    sof_cycle = None
    sof_index = None
    for index in range(1, len(samples)):
        if levels[index] == 0 and levels[index - 1] == 1:
            if cycles[index - 1] - idle_start >= 3 * bit_cycles:
                sof_cycle = (cycles[index - 1] + cycles[index]) / 2
                sof_index = index
                break
            idle_start = None
        elif levels[index] == 1 and levels[index - 1] == 0:
            idle_start = cycles[index]
    if sof_cycle is None:
        raise CANDecodeError("SOF not found after at least three idle bit times")

    raw_count = 0
    stuffed_count = 0
    last_bit = None
    run = 0

    def read_raw():
        nonlocal raw_count
        centre = sof_cycle + (raw_count + 0.5) * bit_cycles
        index = bisect_left(cycles, centre)
        if index == len(cycles):
            raise CANDecodeError("Truncated capture before CRC delimiter")
        if index > 0 and cycles[index] - cycles[index - 1] > bit_cycles / 2:
            raise CANDecodeError("Sample gap exceeds half a bit")
        if index > 0 and centre - cycles[index - 1] < cycles[index] - centre:
            index -= 1
        raw_count += 1
        return levels[index]

    def skip_stuff():
        nonlocal run, last_bit, stuffed_count
        if run == 5:
            bit = read_raw()
            if bit == last_bit:
                raise CANDecodeError("Stuff error: sixth consecutive equal bit")
            stuffed_count += 1
            last_bit, run = bit, 1

    def read_unstuffed():
        nonlocal run, last_bit
        skip_stuff()
        bit = read_raw()
        run = run + 1 if bit == last_bit else 1
        last_bit = bit
        return bit

    def integer(bits):
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value

    # SOF, the common 11 identifier bits, RTR/SRR, and IDE select the layout.
    header = [read_unstuffed() for _ in range(14)]
    if header[0] != 0:
        raise CANDecodeError("SOF must be dominant")
    extended = bool(header[13])
    if extended:
        if header[12] != 1:
            raise CANDecodeError("Extended data-frame SRR must be recessive")
        # Remaining 18 EID bits, RTR, r1, r0, and four DLC bits.
        header.extend(read_unstuffed() for _ in range(25))
        if header[32:35] != [0, 0, 0]:
            raise CANDecodeError("Extended data-frame RTR/r1/r0 must be dominant")
        identifier = (integer(header[1:12]) << 18) | integer(header[14:32])
        dlc = integer(header[35:39])
    else:
        if header[12] != 0:
            raise CANDecodeError("Standard data-frame RTR must be dominant")
        header.extend(read_unstuffed() for _ in range(5))
        if header[14] != 0:
            raise CANDecodeError("Standard data-frame r0 must be dominant")
        identifier = integer(header[1:12])
        dlc = integer(header[15:19])
    if dlc > 8:
        raise CANDecodeError("Classic data-frame DLC exceeds eight bytes")
    data_bits = [read_unstuffed() for _ in range(8 * dlc)]
    received_crc = integer([read_unstuffed() for _ in range(15)])
    # A run of five ending on the final CRC bit still requires its stuff bit.
    skip_stuff()
    if read_raw() != 1:
        raise CANDecodeError("CRC delimiter must be recessive")

    end_cycle = sof_cycle + (raw_count - 0.5) * bit_cycles
    end_index = bisect_left(cycles, end_cycle)
    if any(cycles[i] - cycles[i - 1] > bit_cycles / 2
           for i in range(sof_index, end_index + 1)):
        raise CANDecodeError("Sample gap exceeds half a bit")

    computed_crc = 0
    for bit in header + data_bits:
        feedback = bit ^ ((computed_crc >> 14) & 1)
        computed_crc = (computed_crc << 1) & 0x7FFF
        if feedback:
            computed_crc ^= 0x4599
    if received_crc != computed_crc:
        raise CANDecodeError(
            f"CRC mismatch: captured 0x{received_crc:04x}, computed 0x{computed_crc:04x}"
        )
    payload = bytes(integer(data_bits[i:i + 8]) for i in range(0, len(data_bits), 8))
    return {
        "id": identifier,
        "extended": extended,
        "dlc": dlc,
        "payload": payload,
        "payload_hex": payload.hex(),
        "crc": received_crc,
        "crc_computed": computed_crc,
        "crc_valid": True,
        "sof_cycle": sof_cycle,
        "bit_count": raw_count,
        "stuffed_bit_count": stuffed_count,
    }
