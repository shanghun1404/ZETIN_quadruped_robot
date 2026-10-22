"""An old result handle cannot consume the following policy response."""
from types import SimpleNamespace
import numpy as np
import pytest
from v151_ros2_control.policy_process import _Result


def test_result_handle_can_only_join_its_own_response_once():
    reads=[]
    def receive(kind):
        reads.append(kind)
        return np.zeros(12,dtype=np.float32),np.ones(12,dtype=np.float32)
    owner=SimpleNamespace(_receive=receive,pending=True)
    result=_Result(owner);result.result()
    owner.pending=True
    with pytest.raises(RuntimeError,match='already joined'):result.result()
    assert reads==['result'] and owner.pending
