"""同步规划记录作用域；不持有设备或改变规划判据。"""
from contextlib import contextmanager
from contextvars import ContextVar

_RECORDING = ContextVar("motion_recording", default=None)


def current_motion_recording():
    return _RECORDING.get()


@contextmanager
def motion_recording(recorder):
    token = _RECORDING.set(recorder)
    try:
        yield recorder
    finally:
        _RECORDING.reset(token)
