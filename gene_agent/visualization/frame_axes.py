"""向一次 MuJoCo renderer scene 追加选择性的带标签坐标轴。"""
from __future__ import annotations

import mujoco
import numpy as np


AXIS_COLORS = {
    "x": [1.0, 0.1, 0.1, 1.0],
    "y": [0.1, 0.9, 0.1, 1.0],
    "z": [0.1, 0.35, 1.0, 1.0],
}
ORIGIN_COLORS = {
    "WORLD": [1.0, 1.0, 1.0, 1.0],
    "TCP": [1.0, 0.2, 0.8, 1.0],
    "TARGET": [1.0, 0.9, 0.1, 1.0],
}


def _validated_frames(frames) -> dict[str, np.ndarray]:
    result = {}
    for label, value in frames.items():
        transform = np.asarray(value, dtype=float)
        if not label or transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("debug frame 必须有名称和有限 4x4 变换")
        if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
        ) or np.linalg.det(transform[:3, :3]) < 0:
            raise ValueError(f"debug frame 不是合法刚体变换: {label}")
        result[str(label)] = transform
    return result


def append_frame_axes(scene, frames, *, axis_length_m: float = .08) -> dict:
    """追加 X/Y/Z 箭头；名称与数值放在图像外的说明栏，避免遮住接触点。"""
    resolved = _validated_frames(frames)
    length = float(axis_length_m)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("frame axis length 必须为正有限值")
    needed = len(resolved) * 4
    if scene.ngeom + needed > scene.maxgeom:
        raise ValueError("renderer scene 没有足够 geom 容量显示坐标轴")
    for label, transform in resolved.items():
        origin = transform[:3, 3]
        for index, axis in enumerate(("x", "y", "z")):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),
                origin,
                np.eye(3).reshape(-1),
                np.asarray(AXIS_COLORS[axis], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                max(.0015, length * .025),
                origin,
                origin + transform[:3, index] * length,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
        marker = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            marker, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([.008, .008, .008]), origin, np.eye(3).reshape(-1),
            np.asarray(ORIGIN_COLORS.get(label, [1.0, 1.0, 1.0, 1.0]), dtype=np.float32),
        )
        marker.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
    return {
        "labels": list(resolved),
        "axis_length_m": length,
        "axis_colors": AXIS_COLORS,
        "origin_colors": {label: ORIGIN_COLORS.get(label, [1.0, 1.0, 1.0, 1.0])
                          for label in resolved},
    }


__all__ = ["AXIS_COLORS", "append_frame_axes"]
