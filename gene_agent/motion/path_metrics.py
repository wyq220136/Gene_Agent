"""按真实 TCP FK 分别检查直线平移与原位旋转轨迹。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearPathMetrics:
    path_length_m: float
    max_lateral_deviation_m: float
    max_backward_step_m: float
    cumulative_backward_motion_m: float
    max_overshoot_m: float
    start_error_m: float
    goal_error_m: float
    actual_path_length_m: float = 0.0

    def to_dict(self) -> dict[str, float]:
        # 旧 path_length_m 实为起终点直距；保留其值，另明确记录实际路程。
        return {**vars(self), "direct_distance_m": self.path_length_m}

    def satisfies(self, *, lateral_m: float, backward_step_m: float,
                  cumulative_backward_m: float, overshoot_m: float,
                  endpoint_m: float) -> bool:
        return (self.max_lateral_deviation_m <= lateral_m
                and self.max_backward_step_m <= backward_step_m
                and self.cumulative_backward_motion_m <= cumulative_backward_m
                and self.max_overshoot_m <= overshoot_m
                and self.start_error_m <= endpoint_m
                and self.goal_error_m <= endpoint_m)


def measure_linear_tcp_path(tcp_positions_world_m, start_world_m, goal_world_m) -> LinearPathMetrics:
    """位置曲线可沿任意世界方向；回退和超出线段端点单独计量。"""
    points = np.asarray(tcp_positions_world_m, dtype=float)
    start = np.asarray(start_world_m, dtype=float)
    goal = np.asarray(goal_world_m, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or start.shape != (3,) or goal.shape != (3,)
            or not all(np.all(np.isfinite(v)) for v in (points, start, goal))):
        raise ValueError("TCP 路径须含至少两个有限的三维点，起终点也须为三维")
    displacement = goal - start
    length = float(np.linalg.norm(displacement))
    if length <= 1e-9:
        raise ValueError("TCP 直线规划的起终点须有非零位移")
    direction = displacement / length
    progress = (points - start) @ direction
    lateral = np.linalg.norm(points - start - progress[:, None] * direction, axis=1)
    backwards = np.maximum(0.0, -np.diff(progress))
    overshoot = np.maximum(np.maximum(0.0, -progress), np.maximum(0.0, progress - length))
    return LinearPathMetrics(
        length, float(np.max(lateral)), float(np.max(backwards, initial=0.0)),
        float(np.sum(backwards)), float(np.max(overshoot, initial=0.0)),
        float(np.linalg.norm(points[0] - start)), float(np.linalg.norm(points[-1] - goal)),
        float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))),
    )


@dataclass(frozen=True)
class PivotPathMetrics:
    """旋转阶段只约束 TCP 活动范围和总路程；小位移无单向性含义。"""

    path_length_m: float
    max_excursion_m: float
    start_error_m: float
    goal_error_m: float

    def to_dict(self) -> dict[str, float]:
        return vars(self).copy()

    def satisfies(self, *, max_excursion_m: float, max_path_length_m: float,
                  endpoint_m: float) -> bool:
        return (self.max_excursion_m <= max_excursion_m
                and self.path_length_m <= max_path_length_m
                and self.start_error_m <= endpoint_m
                and self.goal_error_m <= endpoint_m)


def measure_pivot_tcp_path(tcp_positions_world_m, start_world_m,
                           goal_world_m) -> PivotPathMetrics:
    points = np.asarray(tcp_positions_world_m, dtype=float)
    start = np.asarray(start_world_m, dtype=float)
    goal = np.asarray(goal_world_m, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or start.shape != (3,) or goal.shape != (3,)
            or not all(np.all(np.isfinite(v)) for v in (points, start, goal))):
        raise ValueError("旋转 TCP 路径须含至少两个有限三维点，起终点也须为三维")
    return PivotPathMetrics(
        path_length_m=float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))),
        max_excursion_m=float(np.max(np.linalg.norm(points - start, axis=1))),
        start_error_m=float(np.linalg.norm(points[0] - start)),
        goal_error_m=float(np.linalg.norm(points[-1] - goal)),
    )
