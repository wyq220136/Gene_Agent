"""RealSense 短暂无帧不应直接终止采集，持续无帧应给出可定位的错误。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.perception_modules.camera.realsense_stream import RealSenseStream


class FakePipeline:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.stopped = False

    def start(self, config):
        return FakeProfile()

    def wait_for_frames(self, timeout_ms):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def stop(self):
        self.stopped = True


class FakeProfile:
    def get_stream(self, stream):
        return self

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return SimpleNamespace(fx=600, fy=601, ppx=1, ppy=1)

    def get_device(self):
        return self

    def first_depth_sensor(self):
        return self

    def get_depth_scale(self):
        return 0.001


def fake_rs(pipeline):
    class Config:
        def enable_device(self, serial):
            pass

        def enable_stream(self, *args):
            pass

    return SimpleNamespace(
        pipeline=lambda: pipeline,
        config=Config,
        align=lambda stream: SimpleNamespace(process=lambda frames: frames),
        stream=SimpleNamespace(depth="depth", color="color"),
        format=SimpleNamespace(z16="z16", bgr8="bgr8"),
    )


def test_startup_recovers_from_one_frame_timeout(monkeypatch) -> None:
    timeout = RuntimeError("Frame didn't arrive within 5000")
    pipeline = FakePipeline([timeout, object()])
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs(pipeline))

    stream = RealSenseStream(serial="337122070986", warmup_frames=1)

    assert pipeline.calls == 2
    assert not pipeline.stopped
    stream.close()
    assert pipeline.stopped


def test_startup_persistent_timeout_reports_profile_and_closes_pipeline(monkeypatch) -> None:
    pipeline = FakePipeline([RuntimeError("Frame didn't arrive within 5000")] * 8)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs(pipeline))

    with pytest.raises(RuntimeError, match="337122070986.*1280x720.*30"):
        RealSenseStream(serial="337122070986", warmup_frames=1)

    assert pipeline.stopped
    assert pipeline.calls < 8


def test_timeout_diagnostic_does_not_repeat_selected_low_bandwidth_profile(monkeypatch) -> None:
    pipeline = FakePipeline([RuntimeError("Frame didn't arrive within 5000")] * 8)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs(pipeline))

    with pytest.raises(RuntimeError) as error:
        RealSenseStream(serial="337122070986", width=640, height=480, fps=15, warmup_frames=1)

    message = str(error.value)
    assert "640x480@15" in message
    assert "可尝试 --width 640 --height 480 --stream-fps 15" not in message


def test_read_recovers_from_transient_timeout(monkeypatch) -> None:
    color = SimpleNamespace(
        get_data=lambda: np.array([[[10, 20, 30]]], dtype=np.uint8),
        get_frame_number=lambda: 7,
        get_timestamp=lambda: 123.0,
    )
    depth = SimpleNamespace(get_data=lambda: np.array([[1000]], dtype=np.uint16))
    frames = SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)
    pipeline = FakePipeline([
        object(), RuntimeError("Frame didn't arrive within 5000"), frames,
    ])
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs(pipeline))
    stream = RealSenseStream(serial="337122070986", warmup_frames=1)
    try:
        packet = stream.read()
    finally:
        stream.close()

    assert pipeline.calls == 3
    assert packet.rgb.tolist() == [[[30, 20, 10]]]
    assert packet.depth[0, 0] == pytest.approx(1.0)
