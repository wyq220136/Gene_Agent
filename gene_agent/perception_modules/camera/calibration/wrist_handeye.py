"""固定棋盘格的 eye-in-hand 几何；这里不连接设备。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose


def camera_pose_from_flange(T_sdk_flange: np.ndarray, T_flange_camera: np.ndarray) -> np.ndarray:
    """运行时相机位姿：T_sdk_camera(q) = T_sdk_flange(q) @ T_flange_camera。"""
    return validate_pose(
        validate_pose(T_sdk_flange, field_name="T_sdk_flange")
        @ validate_pose(T_flange_camera, field_name="T_flange_camera"),
        field_name="T_sdk_camera",
    )


def check_stationary(
    joints_before_rad: Sequence[float], joints_after_rad: Sequence[float],
    *, max_change_deg: float = 0.2,
) -> float:
    """采帧前后关节读数都要稳定；返回最大关节变化（度）。"""
    before = np.asarray(joints_before_rad, dtype=float)
    after = np.asarray(joints_after_rad, dtype=float)
    if before.shape != (7,) or after.shape != (7,) or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("前后关节状态必须是七个有限弧度值")
    delta = float(np.rad2deg(np.max(np.abs(after - before))))
    if delta > max_change_deg:
        raise ValueError(f"采集期间机械臂移动：最大关节变化 {delta:.3f}° > {max_change_deg:.3f}°")
    return delta


def board_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern_size
    if cols < 3 or rows < 3 or not np.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("棋盘格内角点数至少 3x3，格宽必须为正的米值")
    grid = np.zeros((cols * rows, 3), dtype=np.float64)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    return grid


def detect_board_pose(rgb: np.ndarray, K: np.ndarray, pattern_size: tuple[int, int],
                      square_size_m: float, *, detector: str = "classic") -> dict | None:
    """棋盘格内角点原点在检测序列的 0 号点；返回 T_camera_board。

    `sb` 用更准确的角点定位；若传统检测也找到板，用其 0 号角点
    统一两种 OpenCV 检测器的 180° 角点顺序。旧手眼流程默认行为不变。
    """
    import cv2

    points = board_object_points(pattern_size, square_size_m)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("棋盘格检测需要 RGB uint8 图像")
    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("相机内参 K 必须是有限的 3x3 矩阵")
    if detector not in ("classic", "sb"):
        raise ValueError("棋盘格角点检测器必须是 classic 或 sb")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = False
    corners = None
    if detector == "sb":
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        if found:
            classic_found, classic_corners = cv2.findChessboardCorners(gray, pattern_size)
            if classic_found:
                first = classic_corners.reshape(-1, 2)[0]
                precise = corners.reshape(-1, 2)
                if np.linalg.norm(precise[-1] - first) < np.linalg.norm(precise[0] - first):
                    corners = corners[::-1].copy()
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            return None
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    # RealSense 已对齐深度至 RGB；PnP 使用 RGB 光学坐标系，当前相机协议未提供畸变系数。
    distortion = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(points, corners, matrix, distortion)
    if not ok:
        return None
    T_camera_board = np.eye(4)
    T_camera_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_camera_board[:3, 3] = tvec.reshape(3)
    camera_points = (T_camera_board[:3, :3] @ points.T).T + T_camera_board[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("PnP 得到的棋盘格位于相机后方，请检查角点方向和内参")
    projected = cv2.projectPoints(points, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - corners.reshape(-1, 2)) ** 2, axis=1))))
    return {"T_camera_board": validate_pose(T_camera_board),
            "corners": corners.reshape(-1, 2), "reprojection_rmse_px": error}


def _mean_pose(poses: Sequence[np.ndarray]) -> np.ndarray:
    mean = np.eye(4)
    mean[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    u, _, vt = np.linalg.svd(np.sum([pose[:3, :3] for pose in poses], axis=0))
    mean[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return mean


def _rotation_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _check_pose_diversity(flange_poses: Sequence[np.ndarray]) -> None:
    import cv2

    base = flange_poses[0][:3, :3]
    vectors = np.stack([
        cv2.Rodrigues(base.T @ pose[:3, :3])[0].reshape(3)
        for pose in flange_poses[1:]
    ])
    singular = np.linalg.svd(vectors, compute_uv=False)
    if singular[0] < math.radians(15) or singular[1] < math.radians(10):
        raise ValueError("机器人姿态旋转多样性不足：需要绕至少两个不同轴明显转动手腕")


def solve_handeye(samples: Sequence[dict], *, holdout_every: int = 5) -> dict:
    """输入多组 T_sdk_flange、T_camera_board，输出 T_flange_camera。

    每第 `holdout_every` 个样本只用于验证；棋盘格必须全程相对 SDK base 静止。
    """
    import cv2

    if len(samples) < 10 or holdout_every < 3:
        raise ValueError("至少需要 10 个不同姿态，holdout_every 至少为 3")
    flange = [validate_pose(np.asarray(item["T_sdk_flange"], dtype=float),
                            field_name="T_sdk_flange") for item in samples]
    camera_board = [validate_pose(np.asarray(item["T_camera_board"], dtype=float),
                                 field_name="T_camera_board") for item in samples]
    train = [i for i in range(len(samples)) if (i + 1) % holdout_every]
    holdout = [i for i in range(len(samples)) if (i + 1) % holdout_every == 0]
    if len(train) < 8 or not holdout:
        raise ValueError("训练样本至少 8 个，留出样本至少 1 个")
    _check_pose_diversity([flange[i] for i in train])
    R, t = cv2.calibrateHandEye(
        [flange[i][:3, :3] for i in train],
        [flange[i][:3, 3] for i in train],
        [camera_board[i][:3, :3] for i in train],
        [camera_board[i][:3, 3] for i in train],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    T_flange_camera = np.eye(4)
    T_flange_camera[:3, :3] = R
    T_flange_camera[:3, 3] = np.asarray(t).reshape(3)
    T_flange_camera = validate_pose(T_flange_camera, field_name="T_flange_camera")
    board_poses = [flange[i] @ T_flange_camera @ camera_board[i] for i in range(len(samples))]
    T_sdk_board = _mean_pose([board_poses[i] for i in train])
    residuals = []
    for i, pose in enumerate(board_poses):
        residuals.append({
            "index": i, "holdout": i in holdout,
            "translation_m": float(np.linalg.norm(pose[:3, 3] - T_sdk_board[:3, 3])),
            "rotation_deg": _rotation_deg(T_sdk_board[:3, :3].T @ pose[:3, :3]),
        })
    held = [residuals[i] for i in holdout]
    return {
        "T_flange_camera": T_flange_camera,
        "T_sdk_board_estimate": T_sdk_board,
        "train_count": len(train), "holdout_count": len(held),
        "holdout_translation_max_m": max(r["translation_m"] for r in held),
        "holdout_rotation_max_deg": max(r["rotation_deg"] for r in held),
        "residuals": residuals,
    }
