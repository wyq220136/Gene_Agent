"""显式旋转轴上的 TCP 几何路径，可用于旋钮和门。

门轴由调用者提供；Skill 决定推、拉、接触及运动规划。这里的路径仅是
期望几何位姿，不代表可达、无碰撞或已执行成功。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.knowledge.geometry import CylinderConcept


@dataclass(frozen=True, slots=True)
class AxisSpec:
    point: ArrayLike
    direction: ArrayLike
    frame: str
    source: str

    def __post_init__(self) -> None:
        if self.frame not in {"object", "world"}:
            raise ValueError("axis.frame 只支持 object 或 world")
        if not self.source:
            raise ValueError("axis.source 必须明确填写")
        point = np.asarray(self.point, dtype=float)
        direction = np.asarray(self.direction, dtype=float)
        if point.shape != (3,) or direction.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(direction).all():
            raise ValueError("旋转轴点和方向必须是有限三维向量")
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            raise ValueError("旋转轴方向不能为零")
        point = point.copy(); direction = direction / length
        point.setflags(write=False); direction.setflags(write=False)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class OperationWaypoint:
    angle_rad: float
    tcp_pose_world: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OperationPath:
    kind: str
    axis_point_world: NDArray[np.float64]
    axis_direction_world: NDArray[np.float64]
    axis_source: str
    start_angle_rad: float
    target_angle_rad: float
    waypoints: tuple[OperationWaypoint, ...]

    def to_dict(self) -> dict:
        """生成可存档的纯 JSON 数值反馈；所有 TCP pose 都显式标记 world frame。"""
        return {"kind": self.kind, "axis_point_world": self.axis_point_world.tolist(), "axis_direction_world": self.axis_direction_world.tolist(), "axis_source": self.axis_source, "start_angle_rad": self.start_angle_rad, "target_angle_rad": self.target_angle_rad, "waypoints": [{"angle_rad": item.angle_rad, "tcp_pose_world": item.tcp_pose_world.tolist()} for item in self.waypoints]}


def knob_axis_from_concept(concept: CylinderConcept, *, source: str) -> AxisSpec:
    """取旋钮 Concept 的局部 +Z 高度轴，经 part pose 转到 object frame。"""
    if not isinstance(concept, CylinderConcept):
        raise TypeError("旋钮轴要求 CylinderConcept")
    pose = concept.pose_in_object
    return AxisSpec(pose[:3, 3], pose[:3, 2], "object", source)


def generate_rotation_path(
    kind: str,
    start_tcp_pose_world: ArrayLike,
    axis: AxisSpec | None,
    *,
    target_angle_rad: float,
    step_rad: float,
    start_angle_rad: float = 0.0,
    object_pose_world: ArrayLike | None = None,
) -> OperationPath:
    """生成包含起点和精确终点的绕轴路径；角度是对应关节的绝对角。

    object frame 轴必须给出物体世界位姿。旋钮可从 Concept 显式构造轴，
    门的轴必须由任务数据明确给出，不能从其他 Concept 猜测。
    """
    if kind not in {"knob", "door"}:
        raise ValueError("kind 只支持 knob 或 door")
    if axis is None:
        raise ValueError(f"{kind} 缺少显式旋转轴")
    start = validate_pose(start_tcp_pose_world, field_name="start_tcp_pose_world")
    if not all(math.isfinite(float(v)) for v in (start_angle_rad, target_angle_rad, step_rad)) or step_rad <= 0:
        raise ValueError("角度必须有限，step_rad 必须大于零")
    if axis.frame == "object":
        if object_pose_world is None:
            raise ValueError("object frame 旋转轴需要 object_pose_world")
        world = validate_pose(object_pose_world, field_name="object_pose_world")
        point = world[:3, :3] @ axis.point + world[:3, 3]
        direction = world[:3, :3] @ axis.direction
    else:
        point, direction = axis.point, axis.direction
    delta = float(target_angle_rad - start_angle_rad)
    count = int(math.floor(abs(delta) / step_rad))
    sign = 1.0 if delta >= 0 else -1.0
    angles = [float(start_angle_rad + sign * step_rad * i) for i in range(count + 1)]
    if not math.isclose(angles[-1], target_angle_rad, abs_tol=1e-12):
        angles.append(float(target_angle_rad))
    else:
        angles[-1] = float(target_angle_rad)
    waypoints = []
    for angle in angles:
        rotation = Rotation.from_rotvec(direction * (angle - start_angle_rad)).as_matrix()
        pose = np.array(start, copy=True)
        pose[:3, :3] = rotation @ start[:3, :3]
        pose[:3, 3] = point + rotation @ (start[:3, 3] - point)
        pose.setflags(write=False)
        waypoints.append(OperationWaypoint(angle, pose))
    return OperationPath(kind, np.array(point, copy=True), np.array(direction, copy=True), axis.source, float(start_angle_rad), float(target_angle_rad), tuple(waypoints))
