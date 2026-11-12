"""只读 Motion DebugBundle 的 Viser 显示适配。"""

from .app import MotionDebugApp
from .cameras import CameraPreset, camera_presets
from .capture import capture_motion_debug

__all__ = ["MotionDebugApp", "CameraPreset", "camera_presets", "capture_motion_debug"]
