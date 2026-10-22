"""Independent Python/C wire implementations must agree byte for byte."""
import ctypes as c
import pathlib
import random
import struct
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PACKAGE = REPO/"ros2/v15_hardware_bridge"
if not PACKAGE.is_dir():
    PACKAGE = REPO/"workspace/src/v15_hardware_bridge"
sys.path.insert(0, str(PACKAGE))
from v15_hardware_bridge import protocol as p

Float12 = c.c_float*12
ByteArray = c.c_uint8*p.MAX_FRAME_SIZE


class Frame(c.Structure):
    _fields_ = [("kind",c.c_uint8),("session",c.c_uint32),("sequence",c.c_uint32),
                ("stamp_ns",c.c_uint64),("length",c.c_uint16),("payload",c.c_uint8*256)]


class Command(c.Structure):
    _fields_ = [("enabled",c.c_uint32)]+[(name,Float12) for name in ("position","velocity","kp","kd","effort")]


class Measurement(c.Structure):
    _fields_ = [(name,Float12) for name in ("position","velocity","effort")]


class Feedback(c.Structure):
    _fields_ = [("ack",c.c_uint32),("fault",c.c_uint32),("armed",c.c_uint32),("measured",Measurement)]


class Parser(c.Structure):
    _fields_ = [("bytes",ByteArray),("used",c.c_size_t),("rejected",c.c_uint32),("emitted",c.c_uint32)]


CALLBACK = c.CFUNCTYPE(None,c.c_void_p,c.POINTER(Frame))
LIB = c.CDLL(str(ROOT/"build/libqd_protocol.so"))
LIB.qd_encode_frame.argtypes = [c.POINTER(Frame),c.c_void_p,c.c_size_t]
LIB.qd_encode_frame.restype = c.c_size_t
LIB.qd_decode_frame.argtypes = [c.c_void_p,c.c_size_t,c.POINTER(Frame)]
LIB.qd_decode_frame.restype = c.c_bool
LIB.qd_pack_command.argtypes = [c.POINTER(Command),c.c_void_p]
LIB.qd_pack_command.restype = c.c_bool
LIB.qd_unpack_command.argtypes = [c.c_void_p,c.c_size_t,c.POINTER(Command)]
LIB.qd_unpack_command.restype = c.c_bool
LIB.qd_pack_feedback.argtypes = [c.POINTER(Feedback),c.c_void_p]
LIB.qd_pack_feedback.restype = c.c_bool
LIB.qd_unpack_feedback.argtypes = [c.c_void_p,c.c_size_t,c.POINTER(Feedback)]
LIB.qd_unpack_feedback.restype = c.c_bool
LIB.qd_parser_init.argtypes = [c.POINTER(Parser)]
LIB.qd_parser_feed.argtypes = [c.POINTER(Parser),c.c_void_p,c.c_size_t,CALLBACK,c.c_void_p]


class CrossLanguageTest(unittest.TestCase):
    def test_all_frame_types_exact_wire(self):
        payloads = {p.HELLO:b"",p.HELLO_ACK:struct.pack("<I",1),p.STOP:b"",
                    p.COMMAND:p.pack_command([.125]*12,[-2.]*12,[3.]*12,[4.]*12,[5.]*12,True),
                    p.FEEDBACK:p.pack_feedback([-.125]*12,[51.]*12,[2.]*12,0xffffffff,16,True)}
        for kind,payload in payloads.items():
            with self.subTest(kind=kind):
                expected = p.encode_frame(p.Frame(kind,0x12345678,0xfffffffe,0x0102030405060708,payload))
                cf = Frame(kind,0x12345678,0xfffffffe,0x0102030405060708,len(payload))
                cf.payload[:len(payload)] = payload
                output = ByteArray()
                n = LIB.qd_encode_frame(c.byref(cf),output,len(output))
                self.assertEqual(bytes(output[:n]),expected)
                decoded = Frame()
                self.assertTrue(LIB.qd_decode_frame(expected,len(expected),c.byref(decoded)))
                self.assertEqual(decoded.stamp_ns,0x0102030405060708)
                self.assertEqual(bytes(decoded.payload[:decoded.length]),payload)

    def test_command_and_feedback_roundtrip_randomized(self):
        rng = random.Random(20261012)
        for _ in range(100):
            payload = p.pack_command([rng.uniform(-12.5,12.5) for _ in range(12)],
                                     [rng.uniform(-50,50) for _ in range(12)],
                                     [rng.uniform(0,300) for _ in range(12)],
                                     [rng.uniform(0,8) for _ in range(12)],
                                     [rng.uniform(-cap,cap) for cap in p.EFFORT_LIMITS],True)
            command = Command(); output = ByteArray()
            self.assertTrue(LIB.qd_unpack_command(payload,len(payload),c.byref(command)))
            self.assertTrue(LIB.qd_pack_command(c.byref(command),output))
            self.assertEqual(bytes(output[:244]),payload)
            decoded = p.unpack_command(bytes(output[:244]))
            self.assertEqual(tuple(command.position),decoded["position"])
            payload = p.pack_feedback([rng.uniform(-20,20) for _ in range(12)],
                                      [rng.uniform(-70,70) for _ in range(12)],
                                      [rng.uniform(-50,50) for _ in range(12)],0xfffffffe,9,True)
            fb = Feedback()
            self.assertTrue(LIB.qd_unpack_feedback(payload,len(payload),c.byref(fb)))
            self.assertTrue(LIB.qd_pack_feedback(c.byref(fb),output))
            self.assertEqual(bytes(output[:156]),payload)
            self.assertEqual(p.unpack_feedback(bytes(output[:156]))["fault_flags"],9)

    def test_python_to_c_stream_with_fragmentation_crc_noise(self):
        wires = [p.encode_frame(p.Frame(p.HELLO,29,7,88,b"")),
                 p.encode_frame(p.Frame(p.COMMAND,29,8,89,p.pack_command([0.]*12,[0.]*12,[0.]*12,[0.]*12,[0.]*12)))]
        broken = bytearray(wires[1]); broken[-1] ^= 0x80
        stream = b"noiseQ"+wires[0]+bytes(broken)+b"junk"+wires[1]
        parser = Parser(); LIB.qd_parser_init(c.byref(parser)); got = []
        @CALLBACK
        def receive(_user,frame):
            f = frame.contents
            got.append(p.Frame(f.kind,f.session,f.sequence,f.stamp_ns,bytes(f.payload[:f.length])))
        for start in range(0,len(stream),7):
            data = stream[start:start+7]
            LIB.qd_parser_feed(c.byref(parser),data,len(data),receive,None)
        self.assertEqual([p.encode_frame(f) for f in got],wires)
        self.assertEqual(parser.used,0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
