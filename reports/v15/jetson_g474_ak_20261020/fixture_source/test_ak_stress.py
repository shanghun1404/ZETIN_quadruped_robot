import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

import protocol as p
import ak_stress as a


class ReferenceTests(unittest.TestCase):
    def test_servo_signed_ninety_degree_fixtures(self):
        self.assertEqual(a.servo_raw(math.pi / 2), 900000)
        self.assertEqual(a.servo_raw(-math.pi / 2), -900000)
        self.assertAlmostEqual(a.expected_angle("servo", 0, math.pi / 2), math.pi / 2, places=6)
        self.assertAlmostEqual(a.expected_angle("servo", 0, -math.pi / 2), -math.pi / 2, places=6)
        self.assertEqual(a.round_away(0.5), 1)
        self.assertEqual(a.round_away(-0.5), -1)

    def test_mit_models_and_all_twelve_angles(self):
        self.assertAlmostEqual(a.expected_angle("mit", 0, 0), -12.5 / 65535, places=9)
        self.assertEqual(a.expected_angle("mit", 1, 0), 0)
        for index in (0, 1, 100, 10000):
            target = a.targets(index)
            self.assertEqual(len(set(target)), 12)
            for mode in ("mit", "servo"):
                for joint, value in enumerate(target):
                    self.assertLessEqual(abs(a.expected_angle(mode, joint, value) - value), 0.0014)


class FakeClock:
    def __init__(self):
        self.ns = 1_000_000_000

    def now(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += max(1, int(seconds * 1e9))


class FakeSerial:
    def __init__(self, mode, corrupt=False):
        self.mode, self.corrupt = mode, corrupt
        self.buffer = bytearray()
        self.parser = p.StreamParser()
        self.feedback_sequence = 0
        self.commands = 0
        self.stop = False

    @property
    def in_waiting(self):
        return len(self.buffer)

    def write(self, raw):
        for frame in self.parser.feed(raw):
            if frame.kind == p.STOP:
                self.stop = True
            elif frame.kind == p.HELLO:
                self.buffer.extend(p.encode_frame(p.Frame(
                    p.HELLO_ACK, frame.session, 1, 0, (0x80000001).to_bytes(4, "little"))))
            elif frame.kind == p.COMMAND:
                self.commands += 1
                command = p.unpack_command(frame.payload)
                assert not command["enabled"]
                assert not any(command[k][j] for k in ("velocity", "kp", "kd", "effort") for j in range(12))
                measured = [a.expected_angle(self.mode, j, q) for j, q in enumerate(command["position"])]
                if self.corrupt:
                    measured[0] += 0.1
                self.feedback_sequence += 1
                payload = p.pack_feedback(measured, (0,) * 12, (0,) * 12, frame.sequence)
                self.buffer.extend(p.encode_frame(p.Frame(
                    p.FEEDBACK, frame.session, self.feedback_sequence,
                    self.feedback_sequence * 5_000_000, payload)))
        return len(raw)

    def read(self, size):
        raw = bytes(self.buffer[:size])
        del self.buffer[:size]
        return raw

    def close(self):
        pass


class StressTests(unittest.TestCase):
    def run_fake(self, mode, corrupt=False, seconds=0.05, warmup=0.01):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        args = argparse.Namespace(port="fake", mode=mode, seconds=seconds,
                                  warmup_seconds=warmup, output_dir=Path(directory.name))
        clock, serial = FakeClock(), FakeSerial(mode, corrupt)
        summary = a.run(args, serial_factory=lambda *args, **kwargs: serial,
                        clock=clock.now, sleep=clock.sleep)
        return summary, serial, Path(directory.name)

    def test_both_modes_send_exact_slots_and_separate_warmup(self):
        for mode in ("mit", "servo"):
            summary, serial, directory = self.run_fake(mode)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["measured"]["commands"], 10)
            self.assertEqual(summary["warmup"]["commands"], 2)
            self.assertEqual(serial.commands, 12)
            self.assertEqual(summary["final_ack"], 12)
            self.assertTrue(serial.stop)
            self.assertGreaterEqual(summary["measured_elapsed_seconds"], 0.05)
            self.assertEqual(json.loads((directory / "summary.json").read_text())["passed"], True)

    def test_angle_errors_cannot_pass_and_stop_is_sent(self):
        summary, serial, _ = self.run_fake("mit", corrupt=True)
        self.assertFalse(summary["passed"])
        self.assertGreater(summary["measured"]["errors"], 0)
        self.assertTrue(serial.stop)

    def test_statistics_storage_is_bounded(self):
        stats = a.Histogram()
        for index in range(100_000):
            stats.add(index / 1000)
        self.assertEqual(len(stats.bins), 201)
        self.assertEqual(stats.count, 100_000)
        self.assertLess(len(json.dumps(stats.report())), 3000)

    def test_startup_noise_is_baselined_but_later_noise_fails(self):
        for noisy_kind, passed in ((p.HELLO, True), (p.COMMAND, False)):
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            serial, clock = FakeSerial("servo"), FakeClock()
            original_write = serial.write

            def noisy_write(raw):
                if raw[3] == noisy_kind:
                    serial.buffer.extend(b"startup-noise")
                return original_write(raw)

            serial.write = noisy_write
            args = argparse.Namespace(port="fake", mode="servo", seconds=0.01,
                                      warmup_seconds=0, output_dir=Path(directory.name))
            summary = a.run(args, serial_factory=lambda *args, **kwargs: serial,
                            clock=clock.now, sleep=clock.sleep)
            self.assertEqual(summary["passed"], passed)
            self.assertEqual(summary["startup_discarded_bytes"], 13 if passed else 0)
            self.assertGreater(summary["discarded_bytes_after_hello"], -1 if passed else 0)

    def test_late_first_measured_slot_still_runs_full_duration(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        serial, clock = FakeSerial("mit"), FakeClock()
        injected = False

        def jittered_sleep(seconds):
            nonlocal injected
            clock.sleep(seconds)
            if not injected and clock.ns >= 1_010_500_000:
                clock.ns += 1_000_000
                injected = True

        args = argparse.Namespace(port="fake", mode="mit", seconds=0.05,
                                  warmup_seconds=0.01, output_dir=Path(directory.name))
        summary = a.run(args, serial_factory=lambda *args, **kwargs: serial,
                        clock=clock.now, sleep=jittered_sleep)
        self.assertGreaterEqual(summary["measured_elapsed_seconds"], 0.05)
        self.assertEqual(summary["measured"]["commands"], 10)
        for key in ("start_wall_utc", "measured_start_wall_utc", "measured_end_wall_utc"):
            self.assertTrue(summary[key].endswith("+00:00"))

    def test_close_exception_still_writes_failed_summary(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        serial, clock = FakeSerial("mit"), FakeClock()

        def bad_close():
            raise OSError("close failure")

        serial.close = bad_close
        args = argparse.Namespace(port="fake", mode="mit", seconds=0.01,
                                  warmup_seconds=0, output_dir=Path(directory.name))
        summary = a.run(args, serial_factory=lambda *args, **kwargs: serial,
                        clock=clock.now, sleep=clock.sleep)
        self.assertFalse(summary["passed"])
        self.assertTrue(serial.stop)
        self.assertFalse(json.loads((Path(directory.name) / "summary.json").read_text())["passed"])

    def test_exception_writes_failed_summary_and_attempts_stop(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        serial, clock = FakeSerial("mit"), FakeClock()
        original_write = serial.write

        def fail_command(raw):
            if raw[3] == p.COMMAND:
                raise OSError("injected write failure")
            return original_write(raw)

        serial.write = fail_command
        args = argparse.Namespace(port="fake", mode="mit", seconds=0.01,
                                  warmup_seconds=0, output_dir=Path(directory.name))
        with self.assertRaisesRegex(OSError, "injected"):
            a.run(args, serial_factory=lambda *args, **kwargs: serial,
                  clock=clock.now, sleep=clock.sleep)
        self.assertTrue(serial.stop)
        self.assertFalse(json.loads((Path(directory.name) / "summary.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
