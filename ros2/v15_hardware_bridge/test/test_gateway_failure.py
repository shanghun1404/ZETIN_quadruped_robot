"""A failed emergency STOP must still publish the latched fault."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from v15_hardware_bridge.usb_gateway import UsbGateway


@pytest.mark.parametrize('stop_error',[RuntimeError('partial STOP'),ValueError('blocking fd')])
def test_stop_error_does_not_hide_latched_fault(stop_error):
    attempts=[];published=[]
    def stop(*args):
        attempts.append(args)
        raise stop_error
    node=SimpleNamespace(failed=False,session=SimpleNamespace(fault=False),_write=stop,
        diag_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda:SimpleNamespace(error=lambda _:None))
    UsbGateway.fail(node,'original transport failure')
    UsbGateway.fail(node,'repeated failure')
    assert node.failed and node.session.fault
    assert len(attempts)==1
    assert [json.loads(message.data) for message in published]==[
        dict(fault=True,reason='original transport failure',enabled=False)]
