"""实时 FoundationPose 的 viser 3D 可视化。"""

from __future__ import annotations

from collections import deque
from queue import Empty, SimpleQueue

import numpy as np


def depth_to_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    *,
    max_points: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把对齐 RGB-D 按针孔模型反投影，并确定性降采样。"""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    K = np.asarray(K, dtype=float)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必须是 HxWx3 uint8 数组")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("depth 必须与 rgb 高宽一致")
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K 必须是有效的 3x3 相机内参")
    if max_points <= 0:
        raise ValueError("max_points 必须大于 0")
    if max_depth <= 0:
        raise ValueError("max_depth 必须大于 0")

    valid = np.isfinite(depth) & (depth >= 0.001) & (depth <= max_depth)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    if len(rows) > max_points:
        selected = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows = rows[selected]
        cols = cols[selected]

    z = depth[rows, cols].astype(np.float32)
    x = ((cols.astype(np.float32) - K[0, 2]) * z / K[0, 0]).astype(np.float32)
    y = ((rows.astype(np.float32) - K[1, 2]) * z / K[1, 1]).astype(np.float32)
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    colors = np.ascontiguousarray(rgb[rows, cols])
    return points, colors


def matrix_to_wxyz_position(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 4x4 pose 转成 viser 使用的 wxyz 四元数和 position。"""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError("pose 必须是 4x4 矩阵")
    rotation = pose[:3, :3]
    trace = np.trace(rotation)

    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("pose 旋转矩阵无法转换为有效四元数")
    return quaternion / norm, pose[:3, 3].copy()


class LivePoseViser:
    """管理实时点云、静态 mesh、pose 节点和位置轨迹。"""

    def __init__(
        self,
        *,
        mesh,
        host: str = "0.0.0.0",
        port: int = 8080,
        max_points: int = 30000,
        max_depth: float = 3.0,
        trajectory_length: int = 100,
        server=None,
    ):
        owns_server = server is None
        if owns_server:
            try:
                import viser
            except ImportError as exc:
                raise RuntimeError(
                    "3D 可视化需要 viser，请在 FoundationPose 环境安装 viser"
                ) from exc
            server = viser.ViserServer(host=host, port=port)

        self.server = server
        self.max_points = int(max_points)
        self.max_depth = float(max_depth)
        self.trajectory = deque(maxlen=max(2, int(trajectory_length)))
        self.origin_position = np.zeros(3, dtype=float)
        bounds = getattr(mesh, "bounding_box", None)
        self.mesh_center_local = np.asarray(
            getattr(bounds, "centroid", np.zeros(3)),
            dtype=float,
        ).reshape(3)
        self.actions = SimpleQueue()
        self.point_cloud_handle = None
        self.trajectory_handle = None
        try:
            self.object_frame = self.server.scene.add_frame(
                "/object/pose",
                axes_length=0.08,
                axes_radius=0.003,
            )
            # mesh 挂在 pose 父节点下，后续只更新父节点变换。
            display_mesh = mesh.copy() if hasattr(mesh, "copy") else mesh
            if hasattr(display_mesh, "visual"):
                # TextureVisuals 会在 GLB 中保留原纹理，必须先显式转成
                # ColorVisuals，绿色顶点色才会真正进入浏览器材质。
                to_color = getattr(display_mesh.visual, "to_color", None)
                if callable(to_color):
                    display_mesh.visual = to_color()
                green = np.array([55, 205, 95, 255], dtype=np.uint8)
                display_mesh.visual.vertex_colors = np.tile(
                    green,
                    (len(display_mesh.vertices), 1),
                )
            self.server.scene.add_mesh_trimesh(
                "/object/pose/mesh",
                display_mesh,
                cast_shadow=True,
                receive_shadow=True,
            )
            self.camera_frame = self.server.scene.add_frame(
                "/camera",
                axes_length=0.12,
                axes_radius=0.004,
            )
            self.status = self.server.gui.add_markdown("等待 pose...")
            self.server.gui.add_button("停止").on_click(
                lambda _: self.actions.put("stop")
            )
        except Exception:
            if owns_server:
                self.server.stop()
            raise

    def set_origin_from_pose(self, pose) -> None:
        """把本次注册的 mesh 中心设为固定显示原点。"""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("pose 必须是 4x4 矩阵")
        center_h = np.append(self.mesh_center_local, 1.0)
        self.origin_position = (pose @ center_h)[:3]
        self.camera_frame.position = -self.origin_position
        self.trajectory.clear()
        if self.trajectory_handle is not None:
            self.trajectory_handle.remove()
            self.trajectory_handle = None
        if self.point_cloud_handle is not None:
            self.point_cloud_handle.remove()
            self.point_cloud_handle = None

    def update_pose(self, pose, *, frame_number: int, elapsed_ms: float) -> None:
        wxyz, position = matrix_to_wxyz_position(pose)
        self.object_frame.wxyz = wxyz
        display_position = position - self.origin_position
        self.object_frame.position = display_position
        self.trajectory.append(display_position.copy())
        self.status.content = (
            f"**状态：tracking**  \n帧号：`{frame_number}`  \n"
            f"耗时：`{elapsed_ms:.1f} ms`"
        )

        if len(self.trajectory) >= 2:
            points = np.asarray(self.trajectory, dtype=np.float32)
            segments = np.stack((points[:-1], points[1:]), axis=1)
            self.trajectory_handle = self.server.scene.add_line_segments(
                "/object/trajectory",
                points=segments,
                colors=(255, 165, 0),
                line_width=2.0,
            )

    def update_point_cloud(self, rgb, depth, K) -> bool:
        points, colors = depth_to_point_cloud(
            rgb,
            depth,
            K,
            max_points=self.max_points,
            max_depth=self.max_depth,
        )
        if len(points) == 0:
            return False
        points = points - self.origin_position.astype(np.float32)
        self.point_cloud_handle = self.server.scene.add_point_cloud(
            "/scene/rgbd",
            points=points,
            colors=colors,
            point_size=0.003,
            precision="float16",
        )
        return True

    def poll_actions(self) -> list[str]:
        """取出浏览器 GUI 线程产生的全部控制动作。"""
        actions = []
        while True:
            try:
                actions.append(self.actions.get_nowait())
            except Empty:
                return actions

    def close(self) -> None:
        stop = getattr(self.server, "stop", None)
        if callable(stop):
            stop()
