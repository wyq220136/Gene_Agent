"""单帧场景采集应保存可供场景生成器读取的 RGB-D 与相机内参。"""

import json

import numpy as np
import pytest
from PIL import Image

from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket
from tools.run.capture_realsense_frame import capture_frame


def test_capture_frame_writes_aligned_rgb_depth_and_camera_metadata(tmp_path):
    packet = FramePacket(
        rgb=np.full((2, 3, 3), 120, dtype=np.uint8),
        depth=np.full((2, 3), 0.75, dtype=np.float32),
        frame_number=7, timestamp_ms=1234.5, serial="camera-123",
        intrinsic_matrix=np.array([[500., 0., 1.], [0., 500., 1.], [0., 0., 1.]]),
    )

    paths = capture_frame(packet, tmp_path / "scene_capture")

    assert Image.open(paths.rgb).size == (3, 2)
    assert np.all(np.asarray(Image.open(paths.depth)) == 750)
    camera = json.loads(paths.camera.read_text())
    assert camera["serial"] == "camera-123"
    assert camera["depth_unit"] == "millimeter"
    assert camera["intrinsic_matrix"][0][0] == 500.


def test_capture_frame_refuses_to_replace_existing_capture(tmp_path):
    output = tmp_path / "scene_capture"
    output.mkdir()
    (output / "rgb.png").write_bytes(b"old frame")
    packet = FramePacket(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
        frame_number=1, timestamp_ms=1., serial="camera-123",
        intrinsic_matrix=np.eye(3),
    )

    with pytest.raises(FileExistsError, match="rgb.png"):
        capture_frame(packet, output)
    assert (output / "rgb.png").read_bytes() == b"old frame"
