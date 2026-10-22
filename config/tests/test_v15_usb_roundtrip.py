"""Real PTY and compiled C codec must preserve the command used by the physics bridge."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'ros2/v15_hardware_bridge'))
sys.path.insert(0,str(ROOT/'config'))


def test_python_c_pty_command_and_measured_feedback(tmp_path):
    helper=ROOT/'config/v15_usb_codec_echo.c'
    assert helper.exists(),'native USB codec echo is missing'
    module=ROOT/'config/v15_usb_roundtrip.py'
    assert module.exists(),'PTY roundtrip is missing'
    from v15_usb_roundtrip import CodecRoundtrip
    from v15_hardware_bridge.protocol import Frame,COMMAND,FEEDBACK,pack_command,pack_feedback,unpack_command,unpack_feedback
    executable=tmp_path/'codec'
    subprocess.run(['gcc','-std=c11','-Wall','-Wextra','-Werror','-I'+str(ROOT/'firmware/stm32g474_usb/include'),str(helper),
                    str(ROOT/'firmware/stm32g474_usb/src/qd_protocol.c'),'-lm','-o',str(executable)],check=True)
    link=CodecRoundtrip(executable)
    try:
        fields=dict(position=[.1]*12,velocity=[0.]*12,kp=[150.]*12,kd=[4.]*12,effort=[.2]*12)
        decoded=unpack_command(link.exchange(Frame(COMMAND,123,0,0,pack_command(**fields))).payload)
        assert decoded['position']==pytest.approx([.1]*12)
        assert decoded['kp']==pytest.approx([150.]*12)
        measured=unpack_feedback(link.exchange(Frame(FEEDBACK,123,1,5_000_000,pack_feedback([.3]*12,[.5]*12,[2.]*12))).payload)
        assert measured['position']==pytest.approx([.3]*12)
        assert measured['velocity']==pytest.approx([.5]*12)
        assert link.stats['frames']==2
    finally:link.close()
