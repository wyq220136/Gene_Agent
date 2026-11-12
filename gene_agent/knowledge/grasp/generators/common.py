"""多个 grasp generator 共用的局部候选与旋转工具。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class LocalGraspCandidate:
    """generator 输出的 part-local TCP；世界位姿只由统一入口生成。"""
    tcp_pose_part: NDArray[np.float64]
    closing_width_m: float
    parameters: Mapping[str, Any]
    parameter_indices: Mapping[str, int]


@dataclass(frozen=True)
class LocalGraspRejection:
    """单个离散参数组合的局部拒绝原因。

    generator 只负责说明该组合为何没有局部 pose；统一入口使用同一组
    ``parameter_indices`` 构造稳定 candidate ID，再转成公开 diagnostic。
    """

    code: str
    message: str
    context: Mapping[str, Any]
    parameter_indices: Mapping[str, int]

def rotation_xyz(rpy_rad: tuple[float, float, float]) -> NDArray[np.float64]:
    """固定父 frame XYZ RPY；列向量约定下矩阵为 ``Rz @ Ry @ Rx``。"""
    roll, pitch, yaw = rpy_rad
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx

def values(parameters: Mapping[str, Any], name: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(parameters.get(name, default))

__all__ = ["LocalGraspCandidate", "LocalGraspRejection", "rotation_xyz", "values"]
