"""One bounded POSIX write per frame; partial transmission is fatal."""
import os


def write_frame_once(fd, frame):
    """Queue an entire frame once, without waiting for the *next* write slot.

    The descriptor must already be nonblocking. Write errors propagate to the
    gateway's fault path. Never retry a possibly partial frame.
    """
    if os.get_blocking(fd):
        raise ValueError('serial descriptor must be nonblocking')
    count=os.write(fd,frame)
    if count!=len(frame):
        raise RuntimeError('partial USB write; command discarded')
