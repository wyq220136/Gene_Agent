"""Knowledge 内统一使用的列向量齐次变换。

``T_A_B`` 表示把 B frame 中的列向量坐标变换到 A frame。连续变换始终写成
``T_A_C = T_A_B @ T_B_C``，避免生成器各自解释 pose 或重复应用物体位姿。
平移分量的单位为米。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

_ROTATION_ATOL = 1e-6
_HOMOGENEOUS_ATOL = 1e-9


def _readonly_copy(value: NDArray[np.float64]) -> NDArray[np.float64]:
    """返回独立的只读副本，避免调用方随后修改输入而改变几何事实。"""
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def validate_pose(
    value: ArrayLike, *, field_name: str = "pose"
) -> NDArray[np.float64]:
    """校验并复制一个 ``T_A_B`` 4x4 齐次位姿矩阵。

    旋转块必须属于 SO(3)：列正交且行列式为 1。错误信息携带调用方传入的字段名，
    这样从嵌套 Concept 或配置加载失败时可以直接定位数据来源。
    """
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是数值 4x4 位姿矩阵") from error

    if pose.shape != (4, 4):
        raise ValueError(f"{field_name} 必须是 4x4 位姿矩阵，实际 shape={pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{field_name} 必须全部为有限数值")
    if not np.allclose(
        pose[3], [0.0, 0.0, 0.0, 1.0], atol=_HOMOGENEOUS_ATOL, rtol=0.0
    ):
        raise ValueError(f"{field_name} 的齐次末行必须为 [0, 0, 0, 1]")

    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=_ROTATION_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{field_name} 的旋转块必须正交")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(
            f"{field_name} 的旋转块必须满足 det=1，实际 det={determinant:.9g}"
        )
    return _readonly_copy(pose)


def compose_poses(
    T_A_B: ArrayLike,
    T_B_C: ArrayLike,
    *,
    left_field_name: str = "T_A_B",
    right_field_name: str = "T_B_C",
) -> NDArray[np.float64]:
    """按列向量约定计算 ``T_A_C = T_A_B @ T_B_C``。"""
    left = validate_pose(T_A_B, field_name=left_field_name)
    right = validate_pose(T_B_C, field_name=right_field_name)
    return validate_pose(left @ right, field_name="T_A_C")


def transform_point(
    T_A_B: ArrayLike,
    point_B: ArrayLike,
    *,
    pose_field_name: str = "T_A_B",
    point_field_name: str = "point_B",
) -> NDArray[np.float64]:
    """把 B frame 中以米表示的三维点变换到 A frame。"""
    pose = validate_pose(T_A_B, field_name=pose_field_name)
    try:
        point = np.asarray(point_B, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{point_field_name} 必须是有限三维点") from error
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{point_field_name} 必须是有限三维点")
    result = pose[:3, :3] @ point + pose[:3, 3]
    return _readonly_copy(result)


def pose_delta(previous, current, *, field: str) -> tuple[float, float]:
    """返回两次观测的平移米数和旋转弧度，供阶段锚点容差复核。"""
    old = validate_pose(previous, field_name=f"原 {field}")
    new = validate_pose(current, field_name=f"当前 {field}")
    translation = float(np.linalg.norm(old[:3, 3] - new[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        old[:3, :3].T @ new[:3, :3],
    ).as_rotvec()))
    return translation, rotation
