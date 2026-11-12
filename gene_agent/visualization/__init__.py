"""可视化公共入口；MuJoCo 视图按需加载，不影响独立 Viser 工具。"""

from importlib import import_module

_MUJOCO_NAMES = {
    "COLLISION_PALETTE", "apply_collision_palette", "render_views", "open_viewer",
    "apply_scene_style", "camera_presets", "configure_camera",
}


def __getattr__(name):
    if name in _MUJOCO_NAMES:
        return getattr(import_module(".mujoco_views", __name__), name)
    raise AttributeError(name)


__all__ = sorted(_MUJOCO_NAMES)
