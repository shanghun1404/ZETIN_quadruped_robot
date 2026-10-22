"""Exercise the real C process. Expected wire bytes use Python's stdlib only."""
import binascii
import os
import pathlib
import select
import struct
import subprocess
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / "build/qd_emulator"


def wire(kind, sequence, payload=b"", session=19):
    body = struct.pack("<2sBBIIQH", b"QD", 1, kind, session, sequence, 123, len(payload)) + payload
    return body + struct.pack("<H", binascii.crc_hqx(body, 0xffff))


def command(sequence=1, *, effort=0., position=0., kp=0., enabled=1, session=19):
    payload = struct.pack("<I60f", enabled, *([position]*12+[0.]*12+[kp]*12+[0.]*12+[effort]*12))
    return wire(3, sequence, payload, session)


def frames(raw):
    out = []
    while raw:
        assert len(raw) >= 24, raw.hex()
        magic, version, kind, session, seq, stamp, size = struct.unpack("<2sBBIIQH", raw[:22])
        frame, raw = raw[:24+size], raw[24+size:]
        assert magic == b"QD" and version == 1 and session != 0
        assert len(frame) == size+24
        assert struct.unpack("<H", frame[-2:])[0] == binascii.crc_hqx(frame[:-2], 0xffff)
        out.append((kind, session, seq, stamp, frame[22:-2]))
    return out


def feedback(frame):
    assert frame[0] == 4
    return struct.unpack("<III36f", frame[4])


class EmulatorTest(unittest.TestCase):
    def run_process(self, data, *args):
        p = subprocess.run([str(EXE), "--frame-driven", *args], input=data, capture_output=True, check=True)
        self.assertGreater(len(p.stdout), 0, "HELLO must produce a wire HELLO_ACK")
        return frames(p.stdout)

    def test_default_disabled_and_hello_reset(self):
        fs = self.run_process(wire(1, 7)+command()+wire(1, 8, session=20)+command(session=20, enabled=0))
        self.assertEqual(fs[0][:3], (2, 19, 7))
        self.assertEqual(fs[0][4], b"\x01\x00\x00\x00")
        self.assertEqual(feedback(fs[1])[:3], (0, 8, 0))
        self.assertEqual(feedback(fs[3])[:3], (1, 0, 0))

    def test_virtual_enable_stop_and_invalid_number(self):
        bad = bytearray(command(2))
        bad[26:30] = struct.pack("<f", float("nan"))
        bad[-2:] = struct.pack("<H", binascii.crc_hqx(bad[:-2], 0xffff))
        fs = self.run_process(wire(1, 7)+command()+bytes(bad)+command(3)+wire(5,4), "--virtual-enable")
        self.assertEqual(feedback(fs[1])[:3], (1, 0, 1))
        self.assertEqual(feedback(fs[2])[:3], (1, 2, 0))
        self.assertEqual(feedback(fs[3])[:3], (1, 2, 0))
        self.assertEqual(feedback(fs[4])[:3], (1, 2, 0))

    def test_injected_actual_measurement_drives_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp)/"measured.bin"
            path.write_bytes(struct.pack("<III36f", 0, 0, 0, *([2.]*12+[0.]*24)))
            fs = self.run_process(wire(1,0)+command(position=1.,kp=10.), "--virtual-enable", "--measurement", str(path))
            self.assertEqual(feedback(fs[1])[:3], (1, 0, 1))
            self.assertEqual(feedback(fs[1])[3], 2.)
            self.assertEqual(feedback(fs[1])[27], 0.)  # measured effort stays measured, not requested torque
            path.write_bytes(struct.pack("<III36f", 0, 0, 0, *([0.]*12+[25.]*12+[0.]*12)))
            fs = self.run_process(wire(1,0)+command(effort=12.28), "--virtual-enable", "--measurement", str(path))
            self.assertEqual(feedback(fs[1])[:3], (0, 4, 0))

    def test_real_time_watchdog_and_replay(self):
        p = subprocess.Popen([str(EXE),"--frame-driven","--virtual-enable"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        def read_exact(n):
            data = b""; deadline = time.monotonic()+2
            while len(data)<n:
                if not select.select([p.stdout],[],[],max(0,deadline-time.monotonic()))[0]:
                    self.fail("MCU emulator response timed out")
                part = os.read(p.stdout.fileno(),n-len(data))
                self.assertTrue(part)
                data += part
            return data
        try:
            p.stdin.write(wire(1,0)); p.stdin.flush(); self.assertEqual(frames(read_exact(28))[0][0],2)
            p.stdin.write(command()); p.stdin.flush(); self.assertEqual(feedback(frames(read_exact(180))[0])[:3],(1,0,1))
            time.sleep(.12)
            p.stdin.write(command()); p.stdin.flush()
            self.assertEqual(feedback(frames(read_exact(180))[0])[:3],(1,1,0))
            p.stdin.write(wire(1,7)+command()); p.stdin.flush()
            self.assertEqual(frames(read_exact(28))[0][0],2)
            self.assertEqual(feedback(frames(read_exact(180))[0])[:3],(1,1,0))
            p.stdin.close(); self.assertEqual(p.wait(timeout=2),0)
        finally:
            if p.poll() is None: p.kill(); p.wait()
            p.stdout.close(); p.stderr.close()

    def test_periodic_feedback_uses_grid_without_per_command_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp)/"measured.bin"
            path.write_bytes(struct.pack("<III36f",0,0,0,*([.5]*12+[-.25]*12+[2.]*12)))
            p = subprocess.Popen([str(EXE),"--measurement",str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
            def read_exact(n):
                data=b""; deadline=time.monotonic()+2
                while len(data)<n:
                    if not select.select([p.stdout],[],[],max(0,deadline-time.monotonic()))[0]:
                        self.fail("periodic feedback timed out")
                    part=os.read(p.stdout.fileno(),n-len(data)); self.assertTrue(part); data+=part
                return data
            try:
                p.stdin.write(wire(1,0)); self.assertEqual(frames(read_exact(28))[0][0],2)
                stamps=[]
                for sequence in range(1,7):
                    p.stdin.write(command(sequence,enabled=0))
                    f=frames(read_exact(180))[0]; stamps.append(f[3])
                    self.assertEqual(f[3]%5_000_000,0)
                    state=feedback(f)
                    self.assertEqual(state[3:15],(.5,)*12)
                    self.assertEqual(state[15:27],(-.25,)*12)
                    self.assertEqual(state[27:39],(2.,)*12)
                self.assertEqual([b-a for a,b in zip(stamps,stamps[1:])],[5_000_000]*5)
                p.stdin.close(); self.assertEqual(p.wait(timeout=2),0)
            finally:
                if p.poll() is None: p.kill(); p.wait()
                p.stdin.close(); p.stdout.close(); p.stderr.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
