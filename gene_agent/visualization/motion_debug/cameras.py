"""根据 DebugBundle 几何计算稳定、可复现的 Viser 相机。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gene_agent.feedback.motion_debug import MotionDebugBundle


@dataclass(frozen=True)
class CameraPreset:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_rad: float = 0.78
    near_m: float = 0.005
    far_m: float = 10.0

    def to_dict(self) -> dict[str, object]:
        return {
            "position": list(self.position),
            "look_at": list(self.look_at),
            "up_direction": list(self.up_direction),
            "fov_rad": self.fov_rad,
            "near_m": self.near_m,
            "far_m": self.far_m,
        }


def _box_corners(box) -> np.ndarray:
    half = np.asarray(box.size_xyz_m, dtype=float) / 2.0
    signs = np.array(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    local = signs * half
    return local @ box.pose_world[:3, :3].T + box.pose_world[:3, 3]


def _bounds(bundle: MotionDebugBundle) -> tuple[np.ndarray, float]:
    points: list[np.ndarray] = []
    for box in bundle.scene_boxes:
        points.extend(_box_corners(box))
    for attempt in bundle.attempts:
        points.append(attempt.request.target.matrix[:3, 3])
        points.extend(np.asarray(attempt.tcp_path_world_m, dtype=float))
    for marker in bundle.collision_markers:
        if marker.position_world_m is not None:
            points.append(np.asarray(marker.position_world_m, dtype=float))
    if not points:
        return np.zeros(3), 0.3
    stacked = np.asarray(points, dtype=float).reshape(-1, 3)
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    center = (minimum + maximum) / 2.0
    return center, max(float(np.max(maximum - minimum)), 0.2)


def _preset(focus: np.ndarray, offset: np.ndarray, extent: float) -> CameraPreset:
    position = focus + offset * extent
    return CameraPreset(
        position=tuple(float(value) for value in position),
        look_at=tuple(float(value) for value in focus),
        near_m=max(0.002, extent * 0.005),
        far_m=max(5.0, extent * 12.0),
    )


def camera_presets(bundle: MotionDebugBundle) -> dict[str, CameraPreset]:
    """返回全景、目标近景和碰撞近景；不读取 GUI 或 client 状态。"""
    center, extent = _bounds(bundle)
    target = (
        bundle.attempts[0].request.target.matrix[:3, 3].copy()
        if bundle.attempts
        else center.copy()
    )
    collision = next(
        (
            np.asarray(marker.position_world_m, dtype=float)
            for marker in bundle.collision_markers
            if marker.position_world_m is not None
        ),
        target,
    )
    return {
        "overview": _preset(center, np.array([1.45, -1.45, 1.05]), extent),
        "target": _preset(target, np.array([1.10, -1.10, 0.70]), max(0.30, extent * 0.55)),
        "collision": _preset(
            collision, np.array([1.00, -1.00, 0.60]), max(0.28, extent * 0.35)
        ),
    }


__all__ = ["CameraPreset", "camera_presets"]
