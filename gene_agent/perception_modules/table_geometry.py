"""从旧桌面知识中迁入的纯几何函数，供感知桌面提取使用。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class HorizontalPlane:
    """水平平面拟合结果。"""

    height: float
    normal: np.ndarray
    inlier_mask: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.height):
            raise ValueError("桌面高度必须有限")
        normal = np.asarray(self.normal, dtype=float)
        mask = np.asarray(self.inlier_mask, dtype=bool)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("平面法向必须是有限的长度为 3 的向量")
        if not np.allclose(normal, [0.0, 0.0, 1.0]):
            raise ValueError("当前 HorizontalPlane 只支持水平桌面 +Z 法向")
        if not mask.ndim == 1:
            raise ValueError("inlier_mask 必须是一维布尔数组")
        object.__setattr__(self, "normal", normal.copy())
        object.__setattr__(self, "inlier_mask", mask.copy())


@dataclass(frozen=True)
class HorizontalFreeSpace:
    """桌面上的空闲区域。

    所有多边形顶点都位于世界 XY 平面，约定按逆时针顺序排列。一个物体
    的 footprint 只有在完整落入某个 ``free_regions_xy`` 时才算可以放置；
    ``occupied_regions_xy`` 用于记录感知阶段检测到的障碍投影，当前不在
    本数据结构中自动做多边形布尔运算。
    """

    table_height: float
    boundary_xy: np.ndarray
    free_regions_xy: tuple[np.ndarray, ...]
    occupied_regions_xy: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.table_height):
            raise ValueError("table_height 必须有限")
        boundary = _validate_polygon(self.boundary_xy, "boundary_xy")
        free_regions = tuple(
            _validate_polygon(region, "free_regions_xy")
            for region in self.free_regions_xy
        )
        occupied_regions = tuple(
            _validate_polygon(region, "occupied_regions_xy")
            for region in self.occupied_regions_xy
        )
        if not free_regions:
            raise ValueError("至少需要一个 free_regions_xy")
        object.__setattr__(self, "boundary_xy", boundary)
        object.__setattr__(self, "free_regions_xy", free_regions)
        object.__setattr__(self, "occupied_regions_xy", occupied_regions)

    def contains_polygon(self, polygon_xy: np.ndarray) -> bool:
        """判断 polygon 的所有顶点是否完整位于一个空闲区域内。"""
        polygon = _validate_polygon(polygon_xy, "polygon_xy")
        return any(
            all(_point_in_polygon(point, region) for point in polygon)
            for region in self.free_regions_xy
        )


def fit_horizontal_table_plane(
    point_cloud_world: np.ndarray,
    *,
    distance_threshold: float = 0.005,
) -> HorizontalPlane:
    """用水平约束下的 RANSAC 拟合桌面高度。

    因为项目当前默认桌面水平，候选平面只需枚举 Z 高度。为避免大点云下
    O(n²) 性能问题，这里对 Z 值排序后用滑动窗口找最密区间，而非逐点全量扫描。
    """
    points = np.asarray(point_cloud_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point_cloud_world 必须是 N x 3 数组")
    if distance_threshold <= 0.0 or not np.isfinite(distance_threshold):
        raise ValueError("distance_threshold 必须是正数")
    finite_mask = np.all(np.isfinite(points), axis=1)
    finite_points = points[finite_mask]
    if len(finite_points) == 0:
        raise ValueError("point_cloud_world 中没有有限点")

    # 对 Z 值排序，滑动窗口找最密的 distance_threshold*2 宽度区间
    z_sorted = np.sort(finite_points[:, 2])
    window_half = float(distance_threshold)
    best_start = 0
    best_count = 0
    right = 0
    for left in range(len(z_sorted)):
        while right < len(z_sorted) and z_sorted[right] - z_sorted[left] <= window_half * 2:
            right += 1
        count = right - left
        if count > best_count:
            best_count = count
            best_start = left

    # 取最佳窗口内 Z 值的中位数作为桌面高度
    height = float(np.median(z_sorted[best_start : best_start + best_count]))

    # 生成全体点的 inlier mask
    best_mask = np.abs(finite_points[:, 2] - height) <= window_half
    full_mask = np.zeros(len(points), dtype=bool)
    finite_indices = np.flatnonzero(finite_mask)
    full_mask[finite_indices] = best_mask
    return HorizontalPlane(
        height=height,
        normal=np.array([0.0, 0.0, 1.0]),
        inlier_mask=full_mask,
    )


def free_space_from_table_points(
    point_cloud_world: np.ndarray,
    plane: HorizontalPlane,
    *,
    boundary_margin: float = 0.0,
) -> HorizontalFreeSpace:
    """从桌面内点生成一个初始桌面边界和空闲区域。

    这一步只描述桌面的可用边界；障碍物投影由感知上层另行提供到
    ``occupied_regions_xy``。初始 free region 是整个边界，适合没有检测到
    障碍物的桌面或作为后续多边形裁剪的输入。
    """
    points = np.asarray(point_cloud_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point_cloud_world 必须是 N x 3 数组")
    if len(plane.inlier_mask) != len(points):
        raise ValueError("plane.inlier_mask 与点云长度不一致")
    if boundary_margin < 0.0 or not np.isfinite(boundary_margin):
        raise ValueError("boundary_margin 不能为负且必须有限")
    table_points = points[plane.inlier_mask]
    if len(table_points) < 3:
        raise ValueError("至少需要三个桌面内点才能生成边界")
    minimum = table_points[:, :2].min(axis=0) - boundary_margin
    maximum = table_points[:, :2].max(axis=0) + boundary_margin
    boundary = np.array(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ]
    )
    return HorizontalFreeSpace(
        table_height=plane.height,
        boundary_xy=boundary,
        free_regions_xy=(boundary.copy(),),
    )




def _validate_polygon(polygon, name: str) -> np.ndarray:
    value = np.asarray(polygon, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 3:
        raise ValueError(f"{name} 必须是至少三个顶点的 N x 2 多边形")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 必须只包含有限数")
    return value.copy()


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """使用射线法判断点是否在多边形内，边界点视为在内部。"""
    x, y = point
    inside = False
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        if (start[1] > y) != (end[1] > y):
            intersection_x = (end[0] - start[0]) * (y - start[1]) / (
                end[1] - start[1]
            ) + start[0]
            if x < intersection_x:
                inside = not inside
    return inside


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    segment = end - start
    length = np.linalg.norm(segment)
    if length <= 1e-12:
        return np.linalg.norm(point - start) <= 1e-9
    other = point - start
    cross = abs(float(segment[0] * other[1] - segment[1] * other[0]))
    if cross > 1e-9 * length:
        return False
    dot = float(np.dot(point - start, point - end))
    return dot <= 1e-9


__all__ = ["HorizontalFreeSpace", "HorizontalPlane", "fit_horizontal_table_plane", "free_space_from_table_points"]
