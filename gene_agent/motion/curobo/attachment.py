"""沿用 AffordToolAct 的确定性 attached sphere 采样与坐标转换。

来源：``tamp/curobo/attachment_spheres.py``。这里只迁移视觉模拟
Pick 所需的局部网格采样与 link 坐标转换，不建立物理抓取判定。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def select_dominant_attachment_component(
    mesh: Any, *, min_area_ratio: float = 0.8,
) -> tuple[Any, bool]:
    """旧版规则：最大组件封闭且占绝大部分面积时去掉漂浮小组件。"""
    ratio = float(min_area_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("min_area_ratio 必须位于 (0, 1]")
    split = getattr(mesh, "split", None)
    if not callable(split):
        return mesh, False
    components = tuple(split(only_watertight=False))
    if len(components) <= 1:
        return mesh, False
    areas = np.asarray(
        [float(getattr(component, "area", 0.0)) for component in components],
        dtype=float,
    )
    if not np.all(np.isfinite(areas)) or np.any(areas < 0.0):
        return mesh, False
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        return mesh, False
    dominant_index = int(np.argmax(areas))
    dominant = components[dominant_index]
    if (bool(getattr(dominant, "is_watertight", False))
            and areas[dominant_index] / total_area >= ratio):
        return dominant, True
    return mesh, False


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """从字典序最小点起步的旧版最远点采样，保证球中心顺序可复现。"""
    values = np.asarray(points, dtype=float)
    requested = int(count)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points 必须是 [N, 3] 数组")
    if requested < 1:
        raise ValueError("count 必须是正整数")
    if len(values) < requested:
        raise ValueError("points 数量不足以完成采样")
    if not np.all(np.isfinite(values)):
        raise ValueError("points 必须是有限数组")
    first = int(np.lexsort((values[:, 2], values[:, 1], values[:, 0]))[0])
    selected = np.empty(requested, dtype=np.int64)
    selected[0] = first
    min_distance_sq = np.sum((values - values[first]) ** 2, axis=1)
    min_distance_sq[first] = -1.0
    for index in range(1, requested):
        next_index = int(np.argmax(min_distance_sq))
        selected[index] = next_index
        distance_sq = np.sum((values - values[next_index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected[: index + 1]] = -1.0
    return values[selected].copy()


def build_attachment_surface_spheres_from_mesh(
    mesh: Any, *, num_spheres: int, radius: float, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """复制旧版 Bowl/Plate 的 32 球表面代理规则，中心仍在物体局部 frame。"""
    from ..configuration import CollisionMode, resolve_collision_mode
    if resolve_collision_mode() is CollisionMode.ROBOT_ONLY:
        return np.empty((0, 3)), np.empty(0), False
    requested = int(num_spheres)
    sphere_radius = float(radius)
    if requested < 1 or sphere_radius <= 0.0:
        raise ValueError("num_spheres 必须为正，radius 必须大于 0")
    mesh, component_removed = select_dominant_attachment_component(mesh)
    import trimesh

    candidate_count = max(requested * 16, 256)
    try:
        candidates = trimesh.sample.sample_surface(
            mesh, candidate_count, seed=int(seed),
        )[0]
    except Exception:
        candidates = np.empty((0, 3), dtype=float)
    if len(candidates) < requested:
        vertices = np.asarray(
            getattr(mesh, "vertices", np.empty((0, 3))), dtype=float,
        )
        candidates = vertices if len(vertices) >= requested else candidates
    if len(candidates) < requested:
        raise ValueError("attached obstacle mesh 无法生成足够的表面采样点")
    centers = farthest_point_sample(candidates, requested)
    radii = np.full(requested, sphere_radius, dtype=float)
    return centers, radii, component_removed


def transform_object_centers_to_link(
    centers_object: np.ndarray, *,
    link_pose_planning: np.ndarray,
    object_pose_planning: np.ndarray,
) -> np.ndarray:
    """旧坐标链 ``T_link_object = inv(T_planning_link) @ T_planning_object``。"""
    centers = np.asarray(centers_object, dtype=float)
    link_pose = np.asarray(link_pose_planning, dtype=float)
    object_pose = np.asarray(object_pose_planning, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_object 必须是 [N, 3] 数组")
    for pose, name in ((link_pose, "link_pose_planning"),
                       (object_pose, "object_pose_planning")):
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} 必须是有限的 4x4 矩阵")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers_object 必须是有限数组")
    link_from_object = np.linalg.inv(link_pose) @ object_pose
    homogeneous = np.column_stack((centers, np.ones(len(centers))))
    return (link_from_object @ homogeneous.T).T[:, :3]


__all__ = [
    "build_attachment_surface_spheres_from_mesh",
    "farthest_point_sample", "select_dominant_attachment_component",
    "transform_object_centers_to_link",
]


def with_held_object(robot_config, *, tool_link, finger_links, spheres_link):
    """独立持物link便于限定工具接触；只允许与抓持指自接触，安装末端仍参与检查。"""
    from copy import deepcopy
    values=np.asarray(spheres_link,dtype=float)
    if values.ndim!=2 or values.shape[1]!=4 or not len(values) or not np.isfinite(values).all() or np.any(values[:,3]<=0):
        raise ValueError('持物球必须为有限[N,4]且半径为正')
    cfg=deepcopy(robot_config);kin=cfg['kinematics'];name='gene_held_tool'
    if name in kin['collision_spheres']:raise ValueError('重复持物link')
    kin.setdefault('extra_links',{})[name]=dict(link_name=name,parent_link_name=tool_link,
        joint_name=name+'_joint',joint_type='FIXED',fixed_transform=[0.,0.,0.,1.,0.,0.,0.])
    kin['collision_link_names'].append(name)
    kin['collision_spheres'][name]=[dict(center=row[:3].tolist(),radius=float(row[3])) for row in values]
    kin.setdefault('self_collision_ignore',{})[name]=list(dict.fromkeys(finger_links))
    return cfg
