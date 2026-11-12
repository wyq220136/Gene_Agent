"""Camera server 客户端接口。"""

from .camera_client import CameraClient
from .live_rgbd_protocol import FramePacket

__all__ = ["CameraClient", "FramePacket"]
