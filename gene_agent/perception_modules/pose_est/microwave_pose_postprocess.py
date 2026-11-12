"""Microwave handle pose 的门板平面后处理。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .legacy_geometry import (
    camera_object_to_task_object,
    load_microwave_concepts,
    validate_pose,
)


_TASK1_FRONT_DIRECTION_TASK = np.array([-1.0, 0.0, 0.0])
# Task1 先验：handle/微波炉开口方向与全局 -X 的偏差不超过 20°。
_TASK1_FRONT_ALIGNMENT_TOLERANCE_DEG = 20.0
_TASK1_HALF_TURN_TOLERANCE_DEG = 10.0
# 初始门板点云筛选区域向门轴侧额外延长的距离。
_DOOR_REGION_HINGE_EXTENSION_M = 0.06


def _transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """使用齐次 pose 转换 Nx3 点。"""
    transform = validate_pose(pose, name="pose")
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("points 必须是 Nx3 数组")
    return (transform[:3, :3] @ value.T).T + transform[:3, 3]


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    """归一化有限三维向量。"""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须是有限三维向量")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} 不能是零向量")
    return vector / norm


def correct_pose_yaw_from_plane(
    microwave_pose_task: np.ndarray,
    door_pose_local: np.ndarray,
    plane_normal_task: np.ndarray,
) -> tuple[np.ndarray, float]:
    """计算并施加门板平面 yaw 修正，不改变 Microwave position。"""
    pose = validate_pose(microwave_pose_task, name="microwave_pose_task")
    door_pose = validate_pose(door_pose_local, name="door_pose_local")
    fitted_normal = _unit(plane_normal_task, name="plane_normal_task")
    expected_normal = _unit(
        pose[:3, :3] @ door_pose[:3, :3] @ np.array([0.0, 0.0, 1.0]),
        name="expected_door_normal",
    )
    # 平面法向有正负号，优先选择与当前 concept door 同向的一侧。
    if float(expected_normal @ fitted_normal) < 0.0:
        fitted_normal = -fitted_normal

    expected_xy = expected_normal[:2]
    fitted_xy = fitted_normal[:2]
    if np.linalg.norm(expected_xy) <= 1e-9 or np.linalg.norm(fitted_xy) <= 1e-9:
        raise ValueError("door 平面法向在全局 XY 平面的投影过小，无法估计 yaw")
    expected_xy = expected_xy / np.linalg.norm(expected_xy)
    fitted_xy = fitted_xy / np.linalg.norm(fitted_xy)
    cross_z = expected_xy[0] * fitted_xy[1] - expected_xy[1] * fitted_xy[0]
    dot = float(np.clip(expected_xy @ fitted_xy, -1.0, 1.0))
    yaw_rad = float(np.arctan2(cross_z, dot))

    corrected = pose.copy()
    corrected[:3, :3] = (
        Rotation.from_euler("z", yaw_rad, degrees=False).as_matrix()
        @ pose[:3, :3]
    )
    corrected[:3, 3] = pose[:3, 3]
    return validate_pose(corrected, name="corrected_microwave_pose"), float(
        np.rad2deg(yaw_rad)
    )


def apply_microwave_front_direction_prior(
    raw_pose_task: np.ndarray,
    candidate_pose_task: np.ndarray,
    cavity_pose_local: np.ndarray,
    *,
    yaw_correction_deg: float,
    alignment_tolerance_deg: float = _TASK1_FRONT_ALIGNMENT_TOLERANCE_DEG,
    half_turn_tolerance_deg: float = _TASK1_HALF_TURN_TOLERANCE_DEG,
) -> tuple[np.ndarray, dict]:
    """按 Task1 先验筛选门板平面 yaw 修正。

    Microwave concept 的 cavity 局部 ``+Y`` 是开口方向。Task1 假设初始
    状态下该方向指向当前 task frame 的全局 ``-Y``。普通门板平面修正如果
    让它偏离这个方向就被丢弃；接近 180° 的修正保留，用于处理 pose 的
    180° 朝向歧义。
    """
    raw_pose = validate_pose(raw_pose_task, name="raw_pose_task")
    candidate_pose = validate_pose(candidate_pose_task, name="candidate_pose_task")
    cavity_pose = validate_pose(cavity_pose_local, name="cavity_pose_local")
    yaw = float(yaw_correction_deg)
    alignment_tolerance = float(alignment_tolerance_deg)
    half_turn_tolerance = float(half_turn_tolerance_deg)
    if not np.isfinite(yaw):
        raise ValueError("yaw_correction_deg 必须是有限数")
    if (
        not np.isfinite(alignment_tolerance)
        or alignment_tolerance < 0.0
        or not np.isfinite(half_turn_tolerance)
        or half_turn_tolerance < 0.0
    ):
        raise ValueError("方向先验容差必须是非负有限数")

    opening_axis_local = np.array([0.0, 1.0, 0.0])
    raw_front = _unit(
        raw_pose[:3, :3] @ cavity_pose[:3, :3] @ opening_axis_local,
        name="raw_front_direction_task",
    )
    candidate_front = _unit(
        candidate_pose[:3, :3] @ cavity_pose[:3, :3] @ opening_axis_local,
        name="candidate_front_direction_task",
    )
    desired_front = _TASK1_FRONT_DIRECTION_TASK
    alignment_angle_deg = float(
        np.rad2deg(
            np.arccos(
                np.clip(float(candidate_front @ desired_front), -1.0, 1.0)
            )
        )
    )
    half_turn_exception = abs(abs(yaw) - 180.0) <= half_turn_tolerance
    aligned_with_prior = alignment_angle_deg <= alignment_tolerance
    if aligned_with_prior or half_turn_exception:
        reason = (
            "开口方向符合 Task1 先验"
            if aligned_with_prior
            else "保留接近 180° 的门板平面修正"
        )
        return candidate_pose, {
            "applied": True,
            "reason": reason,
            "front_direction_prior": True,
            "half_turn_exception": half_turn_exception,
            "raw_front_direction_task": raw_front.tolist(),
            "candidate_front_direction_task": candidate_front.tolist(),
            "desired_front_direction_task": desired_front.tolist(),
            "candidate_alignment_error_deg": alignment_angle_deg,
            "alignment_tolerance_deg": alignment_tolerance,
            "half_turn_tolerance_deg": half_turn_tolerance,
        }

    return raw_pose, {
        "applied": False,
        "reason": "门板平面修正使 Microwave 开口方向偏离 Task1 先验",
        "front_direction_prior": True,
        "half_turn_exception": False,
        "raw_front_direction_task": raw_front.tolist(),
        "candidate_front_direction_task": candidate_front.tolist(),
        "desired_front_direction_task": desired_front.tolist(),
        "candidate_alignment_error_deg": alignment_angle_deg,
        "alignment_tolerance_deg": alignment_tolerance,
        "half_turn_tolerance_deg": half_turn_tolerance,
    }


def select_door_region_points(
    points_task: np.ndarray,
    microwave_pose_task: np.ndarray,
    door,
    handle,
    *,
    region_margin_m: float = 0.04,
    plane_band_m: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """按 handle 在 door 上的 concept 关系筛选附近矩形区域点云。"""
    points = np.asarray(points_task, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_task 必须是 Nx3 数组")
    margin = float(region_margin_m)
    band = float(plane_band_m)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("region_margin_m 必须是非负有限数")
    if not np.isfinite(band) or band <= 0.0:
        raise ValueError("plane_band_m 必须是正有限数")

    microwave_pose = validate_pose(microwave_pose_task, name="microwave_pose_task")
    door_pose = validate_pose(door.pose, name="door.pose")
    handle_pose = validate_pose(handle.pose, name="handle.pose")
    door_points = _transform_points(np.linalg.inv(microwave_pose @ door_pose), points)

    # handle.local_vertices 先进入 Microwave 根坐标，再进入 door 局部坐标。
    handle_root_points = _transform_points(handle_pose, handle.local_vertices)
    handle_door_points = _transform_points(np.linalg.inv(door_pose), handle_root_points)
    half_size = 0.5 * np.asarray(door.size_xyz, dtype=float)
    handle_x_min = float(np.min(handle_door_points[:, 0]))
    handle_x_max = float(np.max(handle_door_points[:, 0]))
    handle_x_center = 0.5 * (handle_x_min + handle_x_max)
    if handle_x_center >= 0.0:
        # 把手在局部 +X 时，门轴位于局部 -X 侧。
        x_min = max(
            -half_size[0],
            handle_x_min - margin - _DOOR_REGION_HINGE_EXTENSION_M,
        )
        # 把手侧边界收至把手靠门板内部的一侧，避免筛入 handle 点云。
        x_max = min(half_size[0], handle_x_min)
    else:
        # 把手在局部 -X 时，门轴位于局部 +X 侧。
        # 把手侧边界收至把手靠门板内部的一侧，避免筛入 handle 点云。
        x_min = max(-half_size[0], handle_x_max)
        x_max = min(
            half_size[0],
            handle_x_max + margin + _DOOR_REGION_HINGE_EXTENSION_M,
        )
    y_min = max(-half_size[1], float(np.min(handle_door_points[:, 1])) - margin)
    y_max = min(half_size[1], float(np.max(handle_door_points[:, 1])) + margin)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("handle 附近矩形区域与 door 范围没有有效交集")

    selected = (
        (door_points[:, 0] >= x_min)
        & (door_points[:, 0] <= x_max)
        & (door_points[:, 1] >= y_min)
        & (door_points[:, 1] <= y_max)
        & (np.abs(door_points[:, 2]) <= band)
    )
    return selected, np.array([x_min, x_max, y_min, y_max], dtype=float)


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold_m: float = 0.006,
    iterations: int = 1500,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """用确定性 RANSAC 拟合平面，返回法向、偏置和内点掩码。"""
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3:
        raise ValueError("拟合平面至少需要 3 个 Nx3 点")
    threshold = float(distance_threshold_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("distance_threshold_m 必须是正有限数")
    count = int(iterations)
    if count <= 0:
        raise ValueError("iterations 必须是正整数")

    rng = np.random.default_rng(int(seed))
    best_mask = None
    best_count = -1
    for _ in range(count):
        sample = value[rng.choice(len(value), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        mask = np.abs(value @ normal + offset) <= threshold
        inlier_count = int(mask.sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best_mask = mask
    if best_mask is None or best_count < 3:
        raise RuntimeError("RANSAC 没有找到有效平面")

    inliers = value[best_mask]
    center = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = _unit(vh[-1], name="fitted_plane_normal")
    offset = -float(normal @ center)
    refined_mask = np.abs(value @ normal + offset) <= threshold
    return normal, offset, refined_mask


def _load_rgbd_points(
    rgb_path: str | Path,
    depth_path: str | Path,
    camera_parameters: dict,
    task_from_camera: np.ndarray,
    *,
    side: str,
    max_depth_m: float = 3.0,
    max_points: int = 80000,
) -> np.ndarray:
    """读取 RGB-D 并转换到当前 SDK base；RGB 仅用于尺寸校验。"""
    del side
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(depth_path) as image:
        depth = np.asarray(image).astype(np.float32)
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("RGB 和 depth 尺寸不一致")
    if str(camera_parameters.get("depth_unit", "millimeter")) == "millimeter":
        depth /= 1000.0
    elif str(camera_parameters.get("depth_unit", "millimeter")) != "meter":
        raise ValueError("不支持的 depth_unit")
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth_m))
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=float)
    if len(rows) > int(max_points):
        indexes = np.linspace(0, len(rows) - 1, int(max_points), dtype=np.int64)
        rows, cols = rows[indexes], cols[indexes]
    intrinsic = np.asarray(camera_parameters["intrinsic_matrix"], dtype=float)
    z = depth[rows, cols]
    camera_points = np.column_stack(
        (
            (cols - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        )
    )
    return _transform_points(task_from_camera, camera_points)


def correct_microwave_pose_with_door_plane(
    pose_camera: np.ndarray,
    *,
    category: str,
    camera_parameters: dict,
    task_from_camera: np.ndarray,
    rgb_path: str | Path,
    depth_path: str | Path,
    concept_path: str | Path,
    region_margin_m: float = 0.04,
    plane_band_m: float = 0.04,
    distance_threshold_m: float = 0.006,
    ransac_iterations: int = 1500,
    max_depth_m: float = 3.0,
    max_points: int = 80000,
    enforce_front_direction_prior: bool = False,
) -> tuple[np.ndarray, dict]:
    """用门板点云只修正 Microwave yaw，返回 camera 坐标系 pose。"""
    if str(category).lower() != "microwave":
        return validate_pose(pose_camera, name="pose_camera"), {
            "applied": False,
            "reason": "该类别不是 Microwave",
        }
    pose_camera = validate_pose(pose_camera, name="pose_camera")
    task_from_camera = validate_pose(task_from_camera, name="task_from_camera")
    pose_task = camera_object_to_task_object(task_from_camera, pose_camera)
    door, handle, cavity = load_microwave_concepts(concept_path)
    points_task = _load_rgbd_points(
        rgb_path,
        depth_path,
        camera_parameters,
        task_from_camera,
        side="task",
        max_depth_m=max_depth_m,
        max_points=max_points,
    )
    selected_mask, region_bounds = select_door_region_points(
        points_task,
        pose_task,
        door,
        handle,
        region_margin_m=region_margin_m,
        plane_band_m=plane_band_m,
    )
    selected_points = points_task[selected_mask]
    if len(selected_points) < 20:
        raise RuntimeError(
            f"handle 附近只筛到 {len(selected_points)} 个点，"
            "请增大 region_margin_m 或 plane_band_m"
        )
    plane_normal, plane_offset, inlier_mask = fit_plane_ransac(
        selected_points,
        distance_threshold_m=distance_threshold_m,
        iterations=ransac_iterations,
    )
    corrected_task, yaw_deg = correct_pose_yaw_from_plane(
        pose_task,
        door.pose,
        plane_normal,
    )
    front_prior = None
    if enforce_front_direction_prior:
        corrected_task, front_prior = apply_microwave_front_direction_prior(
            pose_task,
            corrected_task,
            cavity.pose,
            yaw_correction_deg=yaw_deg,
        )
    corrected_camera = np.linalg.inv(task_from_camera) @ corrected_task
    return validate_pose(corrected_camera, name="corrected_pose_camera"), {
        "applied": front_prior["applied"] if front_prior else True,
        "reason": (
            front_prior["reason"]
            if front_prior
            else "使用 handle 附近 door 点云平面修正全局 Z yaw"
        ),
        "yaw_correction_deg": float(yaw_deg),
        "selected_points": int(len(selected_points)),
        "plane_inliers": int(inlier_mask.sum()),
        "plane_normal_task": plane_normal.tolist(),
        "plane_offset_task": float(plane_offset),
        "region_bounds_door_local": region_bounds.tolist(),
        "position_unchanged": True,
        **({"front_direction_prior": front_prior} if front_prior else {}),
    }


__all__ = [
    "correct_microwave_pose_with_door_plane",
    "correct_pose_yaw_from_plane",
    "apply_microwave_front_direction_prior",
    "fit_plane_ransac",
    "select_door_region_points",
]
