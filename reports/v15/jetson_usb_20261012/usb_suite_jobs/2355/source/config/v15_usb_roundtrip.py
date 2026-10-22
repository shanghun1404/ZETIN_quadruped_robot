"""Real PTY byte transfer to the native G474 C codec, independent of Isaac."""
import os
import pty
import subprocess
import time
import serial
from v15_hardware_bridge.protocol import StreamParser,encode_frame


class CodecRoundtrip:
    def __init__(self,executable):
        master,slave=pty.openpty()
        self.serial=serial.Serial(os.ttyname(slave),921600,timeout=.01,write_timeout=.1)
        os.close(slave)
        self.process=subprocess.Popen([str(executable)],stdin=master,stdout=master,stderr=subprocess.PIPE,close_fds=True)
        os.close(master)
        self.parser=StreamParser()
        self.stats=dict(frames=0,tx_bytes=0,rx_bytes=0,max_roundtrip_ms=0.,
                        transport='Linux PTY through native G474 C codec; actual USB/CAN/Jetson untested')

    def exchange(self,frame):
        data=encode_frame(frame);started=time.monotonic();deadline=started+2.
        if self.serial.write(data)!=len(data):raise RuntimeError('partial PTY write')
        self.stats['tx_bytes']+=len(data)
        while time.monotonic()<deadline:
            data=self.serial.read(max(1,min(4096,self.serial.in_waiting)))
            self.stats['rx_bytes']+=len(data)
            for decoded in self.parser.feed(data):
                if (decoded.kind,decoded.session,decoded.sequence,decoded.stamp_ns)!=(frame.kind,frame.session,frame.sequence,frame.stamp_ns):
                    raise ValueError('C codec response identity changed')
                self.stats['frames']+=1
                self.stats['max_roundtrip_ms']=max(self.stats['max_roundtrip_ms'],1000*(time.monotonic()-started))
                return decoded
            if self.process.poll() is not None:
                raise RuntimeError('C codec exited: '+self.process.stderr.read().decode(errors='replace'))
        raise TimeoutError('C codec PTY response missing')

    def close(self):
        self.serial.close();self.process.terminate()
        try:self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stderr.close()
