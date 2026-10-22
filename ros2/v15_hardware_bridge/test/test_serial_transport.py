"""Exercise real nonblocking descriptors and fail-closed partial writes."""
import importlib
import os
import pty
import signal
import tty
from unittest.mock import patch

import pytest


def write_once(fd, frame):
    # Import inside the test so a missing transport is an ordinary failing test.
    module=importlib.import_module('v15_hardware_bridge.serial_transport')
    return module.write_frame_once(fd,frame)


def test_complete_frame_reaches_real_pty_without_waiting_for_next_write():
    master,slave=pty.openpty()
    try:
        tty.setraw(slave);os.set_blocking(slave,False);os.set_blocking(master,False)
        frame=b'QD'+bytes(range(256))+b'0123456789'
        write_once(slave,frame)
        assert os.read(master,len(frame))==frame
    finally:
        os.close(master);os.close(slave)


def test_blocking_descriptor_is_rejected_before_writing():
    reader,writer=os.pipe()
    try:
        with pytest.raises(ValueError,match='nonblocking'):
            write_once(writer,b'QD')
    finally:
        os.close(reader);os.close(writer)


def test_full_nonblocking_pipe_fails_without_retry_or_busy_wait():
    reader,writer=os.pipe()
    old_handler=signal.getsignal(signal.SIGALRM)
    def timed_out(*_):raise AssertionError('nonblocking transport retried a full pipe')
    try:
        os.set_blocking(writer,False)
        while True:
            try:os.write(writer,b'x'*4096)
            except BlockingIOError:break
        signal.signal(signal.SIGALRM,timed_out);signal.alarm(1)
        with pytest.raises(BlockingIOError):write_once(writer,b'QD')
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        os.close(reader);os.close(writer)


def test_partial_write_is_fatal_and_never_retried():
    reader,writer=os.pipe()
    try:
        os.set_blocking(writer,False)
        with patch('os.write',return_value=3) as syscall:
            with pytest.raises(RuntimeError,match='partial'):
                write_once(writer,b'QD-frame')
            syscall.assert_called_once_with(writer,b'QD-frame')
    finally:
        os.close(reader);os.close(writer)
