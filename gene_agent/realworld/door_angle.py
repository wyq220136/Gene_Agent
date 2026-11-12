"""复刻旧 Task1 的把手测角；不读取相机、不推断门轴。

门轴、关闭参考把手 pose 和当前把手 pose 必须都在同一个 SDK base。
旧版开门后 FoundationPose 跟踪的是把手 mesh；它的 pose 只供测角，
不能覆盖微波炉本体 pose。资产 frame 对齐由调用方单独校验。
"""

from __future__ import annotations

import math

import numpy as np

from gene_agent.knowledge.geometry import validate_pose


def estimate_door_angle_from_handle_pose(
    closed_handle_pose_sdk: np.ndarray,
    observed_handle_pose_sdk: np.ndarray,
    *, axis_point_sdk, axis_direction_sdk, clockwise: bool = True,
) -> float:
    """按 AffordToolAct 的把手圆周运动公式返回 [0, π] 门角。"""
    closed = validate_pose(closed_handle_pose_sdk, field_name="closed_handle_pose_sdk")
    observed = validate_pose(observed_handle_pose_sdk, field_name="observed_handle_pose_sdk")
    point = np.asarray(axis_point_sdk, dtype=float)
    direction = np.asarray(axis_direction_sdk, dtype=float)
    if (point.shape != (3,) or direction.shape != (3,)
            or not np.isfinite(point).all() or not np.isfinite(direction).all()):
        raise ValueError("门轴点与方向必须是有限三维向量")
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("门轴方向不能为零向量")
    direction /= length
    reference = closed[:3, 3] - point
    current = observed[:3, 3] - point
    reference -= direction * float(reference @ direction)
    current -= direction * float(current @ direction)
    reference_norm = float(np.linalg.norm(reference))
    current_norm = float(np.linalg.norm(current))
    if reference_norm <= 1e-9 or current_norm <= 1e-9:
        raise ValueError("把手原点不能位于门轴上")
    reference /= reference_norm
    current /= current_norm
    signed = float(np.arctan2(
        direction @ np.cross(reference, current), reference @ current,
    ))
    angle = -signed if clockwise else signed
    if angle < -1e-7 or angle > math.pi + 1e-7:
        raise ValueError(
            "观测把手姿态不符合当前门的单向开合范围: "
            f"{math.degrees(angle):.3f}°"
        )
    return float(np.clip(angle, 0., math.pi))


def estimate_post_handle_open_angle_reading(
    closed_handle_pose_sdk: np.ndarray,
    observed_handle_pose_sdk: np.ndarray,
    *, axis_point_sdk, axis_direction_sdk,
    observed_at_unix_ms: float, source: str,
) -> dict:
    """生成现场门角读数；规划角沿用旧 Task1 的 5° 网格。"""
    timestamp = float(observed_at_unix_ms)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("把手测角时间必须是有效 Unix 毫秒")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("把手测角须记录 FoundationPose 帧或人工来源")
    measured = estimate_door_angle_from_handle_pose(
        closed_handle_pose_sdk, observed_handle_pose_sdk,
        axis_point_sdk=axis_point_sdk, axis_direction_sdk=axis_direction_sdk,
        clockwise=True,
    )
    # 旧 `_snap_door_angle_to_planning_grid` 只接受开门后的 (0,180]°。
    if measured <= 0. or measured > math.pi + 1e-7:
        raise ValueError("感知得到的开门角必须位于 (0, 180] 度")
    degrees = max(5., min(180., round(math.degrees(measured) / 5.) * 5.))
    return {
        "angle_rad": math.radians(degrees),
        "measured_angle_rad": measured,
        "observed_at_unix_ms": timestamp,
        "source": source,
        "planning_grid_deg": 5.,
    }


def estimate_post_handle_open_angle_from_tracked_asset(
    closed_asset_pose_sdk: np.ndarray,
    tracked_asset_pose_sdk: np.ndarray,
    *, handle_pose_in_asset: np.ndarray,
    axis_point_sdk, axis_direction_sdk,
    observed_at_unix_ms: float, source: str,
) -> dict:
    """沿旧 Task1 调用链先乘 `handle.pose`，再测角并取五度网格。

    FoundationPose 跟踪把手 mesh 时返回的是该 mesh 原坐标系的位姿，
    而旧角度公式需要把手 concept 原点的位置。调用方必须核对传入的
    `handle_pose_in_asset` 与实际注册 mesh 使用同一个局部 frame。
    """
    closed = validate_pose(closed_asset_pose_sdk, field_name="closed_asset_pose_sdk")
    current = validate_pose(tracked_asset_pose_sdk, field_name="tracked_asset_pose_sdk")
    local = validate_pose(handle_pose_in_asset, field_name="handle_pose_in_asset")
    return estimate_post_handle_open_angle_reading(
        closed @ local, current @ local,
        axis_point_sdk=axis_point_sdk, axis_direction_sdk=axis_direction_sdk,
        observed_at_unix_ms=observed_at_unix_ms, source=source,
    )


__all__ = ["estimate_door_angle_from_handle_pose",
           "estimate_post_handle_open_angle_reading",
           "estimate_post_handle_open_angle_from_tracked_asset"]
