"""反馈公共入口；按需加载 MuJoCo 功能，使 Motion 证据可独立使用。"""

from importlib import import_module

_MOTION_NAMES = {
    "CollisionMarker", "DebugBox", "MotionDebugAttempt", "MotionDebugBundle",
    "load_motion_debug_bundle", "save_motion_debug_bundle",
}
_STATE_NAMES = {"snapshot", "save_state", "restore_state"}
_RECORDER_NAMES = {"RunRecorder", "fingerprint_files"}


def __getattr__(name):
    """仅在调用对应功能时导入依赖，避免 Motion DebugBundle 强绑 MuJoCo。"""
    if name in _MOTION_NAMES:
        return getattr(import_module(".motion_debug", __name__), name)
    if name in _STATE_NAMES:
        return getattr(import_module(".state", __name__), name)
    if name in _RECORDER_NAMES:
        return getattr(import_module(".recorder", __name__), name)
    raise AttributeError(name)


__all__ = sorted(_MOTION_NAMES | _STATE_NAMES | _RECORDER_NAMES)
