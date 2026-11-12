"""桌面点云的最小感知适配层。

提供两个层次的接口：
1. ``perceive_horizontal_table`` —— 假设输入已经是桌面点云，直接拟合
2. ``extract_table_from_scene`` —— 从完整场景 RGB-D + tabletop mask 中提取桌面
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gene_agent.perception_modules.pose_est.legacy_geometry import validate_pose
from gene_agent.perception_modules.table_geometry import (
    HorizontalFreeSpace,
    HorizontalPlane,
    fit_horizontal_table_plane,
    free_space_from_table_points,
)


def perceive_horizontal_table(
    point_cloud_world: np.ndarray,
    *,
    distance_threshold: float = 0.005,
    boundary_margin: float = 0.0,
) -> tuple[HorizontalPlane, HorizontalFreeSpace]:
    """从世界坐标系点云中拟合水平桌面并生成初始空闲区域。

    这里的输入默认已经是桌面相关的点云。物体占据区域的语义分割和投影
    可以在后续感知模块中填入 ``HorizontalFreeSpace.occupied_regions_xy``
    并更新 free regions，本适配层只负责统一平面拟合入口和数据契约。
    """
    plane = fit_horizontal_table_plane(
        point_cloud_world,
        distance_threshold=distance_threshold,
    )
    free_space = free_space_from_table_points(
        point_cloud_world,
        plane,
        boundary_margin=boundary_margin,
    )
    return plane, free_space


def extract_table_from_scene(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    mask_path: str | Path,
    camera_path: str | Path,
    side: str,
    distance_threshold: float = 0.005,
    boundary_margin: float = 0.02,
    max_depth: float = 3.0,
    obstacle_height_min: float = 0.01,
    obstacle_height_max: float = 0.50,
) -> tuple[HorizontalPlane, HorizontalFreeSpace, np.ndarray, np.ndarray]:
    """从完整场景 RGB-D + tabletop mask 中提取桌面信息。

    **两步策略**：
    1. SAM mask 用于找到桌面大致高度（mask 不需要完整覆盖桌面）
    2. 用高度从**全场景**点云中恢复完整桌面区域，计算边界和空闲区域

    这样即使 SAM 只分割出桌面的一小部分，最终也能得到完整的桌面边界。

    Parameters
    ----------
    rgb_path : 场景 RGB 图像路径
    depth_path : 场景深度图像路径（PNG，millimeter uint16）
    mask_path : tabletop SAM 分割 mask 路径（PNG，仅用于高度参考）
    camera_path : camera_parameters_0.json 路径
    side : 机械臂侧，"left" 或 "right"
    distance_threshold : 平面拟合的距离阈值 (m)
    boundary_margin : 桌面边界外扩 margin (m)
    max_depth : 深度最大值 (m)
    obstacle_height_min : 障碍物高于桌面的最小高度 (m)
    obstacle_height_max : 障碍物高于桌面的最大高度 (m)

    Returns
    -------
    plane : HorizontalPlane 拟合结果
    free_space : HorizontalFreeSpace 空闲区域
    table_points_world : 全场景中属于桌面的点（用于可视化，任务坐标系）
    scene_points_world : 完整场景点云（任务坐标系）
    """
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 'left' 或 'right'")

    from gene_agent.perception_modules.pose_est.batch_est_pose import (
        load_camera_parameters,
        load_depth_image,
    )

    camera = load_camera_parameters(camera_path)
    key = f"{side}_extrinsic_matrix"
    if key not in camera:
        raise ValueError(f"camera 参数缺少 {key}: {camera_path}")
    task_from_camera = np.asarray(camera[key], dtype=float)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=float)

    depth = load_depth_image(
        depth_path,
        depth_unit=str(camera.get("depth_unit", "millimeter")),
    )

    # ---- 步骤 1: 用 SAM mask 找到桌面高度 ----
    with Image.open(mask_path) as img:
        mask = np.asarray(img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask.astype(bool)

    print("[tabletop] 步骤 1/3: 从 SAM mask 区域提取桌面高度参考...", flush=True)
    mask_points_camera = _depth_mask_to_points(
        depth, mask_bool, intrinsic, max_depth=max_depth, max_points=20000,
    )
    if len(mask_points_camera) == 0:
        raise ValueError(
            "tabletop mask 内没有有效深度点，请检查 mask 是否正确覆盖了桌面"
        )
    mask_points_world = _transform_points(mask_points_camera, task_from_camera)
    print(f"[tabletop] SAM mask 区域内点数: {len(mask_points_world)}", flush=True)

    # 只用 mask 区域拟合桌面高度（快）
    plane = fit_horizontal_table_plane(
        mask_points_world,
        distance_threshold=distance_threshold,
    )
    print(f"[tabletop] SAM 参考高度: {plane.height:.4f}m", flush=True)

    # ---- 步骤 2: 从全场景按高度恢复完整桌面 ----
    print("[tabletop] 步骤 2/3: 用高度从全场景恢复完整桌面...", flush=True)
    scene_points_camera = _depth_to_all_points(
        depth, intrinsic, max_depth=max_depth, max_points=100000,
    )
    scene_points_world = _transform_points(scene_points_camera, task_from_camera)
    print(f"[tabletop] 全场景点数: {len(scene_points_world)}", flush=True)

    # 从全场景中筛选高度接近桌面的所有点（即为完整桌面区域）
    height_tolerance = float(distance_threshold) * 3.0  # ~15mm
    table_mask = (
        np.abs(scene_points_world[:, 2] - plane.height) <= height_tolerance
    )
    table_points_world = scene_points_world[table_mask]
    print(
        f"[tabletop] 完整桌面点数: {len(table_points_world)} "
        f"(高度 {plane.height - height_tolerance:.3f}~{plane.height + height_tolerance:.3f}m)",
        flush=True,
    )

    if len(table_points_world) < 10:
        raise ValueError(
            f"全场景中高度接近 {plane.height:.3f}m 的点太少，"
            "SAM mask 高度参考可能不准确"
        )

    # 用完整桌面点云重新拟合（可选，提高精度）
    plane_full = fit_horizontal_table_plane(
        table_points_world,
        distance_threshold=distance_threshold,
    )
    print(f"[tabletop] 完整桌面拟合高度: {plane_full.height:.4f}m", flush=True)

    # ---- 步骤 3: 生成空闲区域 + 障碍物检测 ----
    print("[tabletop] 步骤 3/3: 生成空闲区域和障碍物检测...", flush=True)
    free_space = free_space_from_table_points(
        table_points_world,
        plane_full,
        boundary_margin=boundary_margin,
    )
    print(
        f"[tabletop] 桌面边界 XY: "
        f"[{free_space.boundary_xy[0]}, {free_space.boundary_xy[2]}]",
        flush=True,
    )

    occupied_polygons = _detect_occupied_regions(
        scene_points_world,
        plane_full,
        obstacle_height_min=obstacle_height_min,
        obstacle_height_max=obstacle_height_max,
        padding=0.01,
    )
    if occupied_polygons:
        free_space = HorizontalFreeSpace(
            table_height=free_space.table_height,
            boundary_xy=free_space.boundary_xy,
            free_regions_xy=free_space.free_regions_xy,
            occupied_regions_xy=tuple(occupied_polygons),
        )
        print(f"[tabletop] 检测到 {len(occupied_polygons)} 个障碍占据区域", flush=True)

    return plane_full, free_space, table_points_world, scene_points_world


def _depth_mask_to_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    *,
    max_depth: float = 3.0,
    max_points: int = 200000,
) -> np.ndarray:
    """将深度图中 mask 区域的像素反投影为相机坐标系 3D 点。"""
    rows, cols = np.nonzero(
        mask & np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    )
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(rows) > max_points:
        indices = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows = rows[indices]
        cols = cols[indices]
    z = depth[rows, cols]
    x = (cols - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.column_stack((x, y, z)).astype(np.float32)


def _depth_to_all_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    max_depth: float = 3.0,
    max_points: int = 100000,
) -> np.ndarray:
    """将深度图全部有效像素反投影为相机坐标系 3D 点（降采样）。"""
    mask = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    return _depth_mask_to_points(
        depth, mask, intrinsic, max_depth=max_depth, max_points=max_points
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """用 T_task_camera 把相机坐标系点云转到任务坐标系。"""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pose = validate_pose(transform, name="T_task_camera")
    return ((pose[:3, :3] @ points.T).T + pose[:3, 3]).astype(np.float32)


def _detect_occupied_regions(
    scene_points: np.ndarray,
    plane: HorizontalPlane,
    *,
    obstacle_height_min: float = 0.01,
    obstacle_height_max: float = 0.50,
    padding: float = 0.01,
) -> list[np.ndarray]:
    """检测桌面上方的障碍物，投影为 XY 多边形。

    简单的实现：取高于桌面的所有点，用 2D 直方图聚类，
    每个聚类生成一个轴对齐包围盒作为占据区域。
    """
    if len(scene_points) == 0:
        return []

    above_table = (
        (scene_points[:, 2] > plane.height + obstacle_height_min)
        & (scene_points[:, 2] < plane.height + obstacle_height_max)
        & np.isfinite(scene_points[:, 0])
        & np.isfinite(scene_points[:, 1])
    )
    obstacle_xy = scene_points[above_table, :2]
    if len(obstacle_xy) < 10:
        return []

    # 简单的基于网格的障碍物检测：在 XY 平面上做 2D 直方图
    # 如果某个格子的点数超过阈值，认为该格子被占据
    grid_size = 0.02  # 2cm grid
    if len(obstacle_xy) == 0:
        return []

    x_min, y_min = obstacle_xy.min(axis=0) - padding
    x_max, y_max = obstacle_xy.max(axis=0) + padding

    # 用连通区域方法：直接对 XY 点做简单的 bounding box
    # 简化版：把所有障碍物点合并为一个凸包或多边形
    # 这里使用简单的 min/max bounding box
    occupied_polygon = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ])
    return [occupied_polygon]


__all__ = [
    "extract_table_from_scene",
    "perceive_horizontal_table",
]
