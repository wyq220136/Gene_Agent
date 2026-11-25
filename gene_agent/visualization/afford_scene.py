"""移植 AffordToolAct visualization/scene/viser_viewer.py 的只读场景控件。

保留原坐标轴、四层句柄和 GUI 开关机制；允许宿主提供符合实际几何的图层名称。
不导入 AffordToolAct 的设备、规划器或运行目录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation



def _wxyz(pose: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(pose, dtype=float)[:3, :3]
    scale = np.linalg.norm(rotation, axis=0)
    if np.any(scale <= 0):
        raise ValueError("显示 pose 的旋转部分不能包含零尺度")
    rotation = rotation / scale
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return (
        float(quaternion_xyzw[3]),
        float(quaternion_xyzw[0]),
        float(quaternion_xyzw[1]),
        float(quaternion_xyzw[2]),
    )


def create_scene_server(*, host: str = "127.0.0.1", port: int = 8120) -> Any:
    """创建 Viser server；导入 Viser 延迟到真正启动时。"""
    import viser

    return viser.ViserServer(host=host, port=port)


def add_world_origin_frame(
    server: Any,
    reference_from_world: np.ndarray,
    *,
    axes_length: float = 0.16,
) -> Any:
    """在当前 reference frame 中显示世界坐标系原点。"""
    pose = np.asarray(reference_from_world, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("reference_from_world 必须是有限的 4x4 矩阵")
    if axes_length <= 0:
        raise ValueError("axes_length 必须大于 0")
    return server.scene.add_frame(
        "/world_origin",
        position=pose[:3, 3],
        wxyz=_wxyz(pose),
        axes_length=float(axes_length),
        axes_radius=0.004,
    )


def add_coordinate_frames(
    server: Any,
    frames: Mapping[str, np.ndarray],
    *,
    axes_length: float = 0.16,
) -> list[Any]:
    """在当前 reference frame 中添加一组命名坐标系。"""
    handles = []
    for frame_name, frame_pose in frames.items():
        pose = np.asarray(frame_pose, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"坐标系 {frame_name!r} 的 pose 无效")
        handles.append(
            server.scene.add_frame(
                f"/frames/{frame_name}",
                position=pose[:3, 3],
                wxyz=_wxyz(pose),
                axes_length=float(axes_length),
                axes_radius=0.004,
            )
        )
    return handles


@dataclass
class SceneHandles:
    """保存图层句柄，供 GUI 开关控制。"""

    visual: list[Any]
    collision: list[Any]
    body_visual: list[Any]
    body_collision: list[Any]
    visual_by_side: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    @staticmethod
    def _set_visible(handles: Sequence[Any], visible: bool) -> None:
        for handle in handles:
            handle.visible = bool(visible)

    def add_layer_controls(self, server: Any, *, labels=None) -> None:
        """增加四个最必要的显示开关。"""
        labels = labels or ("visual geometry", "collision spheres", "body visual", "body collision")
        with server.gui.add_folder("scene layers · AffordToolAct"):
            visual = server.gui.add_checkbox(labels[0], initial_value=True)
            collision = server.gui.add_checkbox(
                labels[1], initial_value=True
            )
            body_visual = server.gui.add_checkbox(
                labels[2], initial_value=True
            )
            body_collision = server.gui.add_checkbox(
                labels[3], initial_value=True
            )
        visual.on_update(lambda _: self._set_visible(self.visual, visual.value))
        collision.on_update(
            lambda _: self._set_visible(self.collision, collision.value)
        )
        body_visual.on_update(
            lambda _: self._set_visible(self.body_visual, body_visual.value)
        )
        body_collision.on_update(
            lambda _: self._set_visible(
                self.body_collision,
                body_collision.value,
            )
        )

