"""只描述尺寸和 frame 的基础几何 Concept。

所有长度单位均为米。``pose_in_object`` 是 ``T_object_part``；圆柱和圆环的
``axis`` 在 part 自身 frame 中表达。这里不保存 grasp 偏好、机器人、renderer 或 planner
信息，后续生成器只消费这些几何事实。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transforms import validate_pose


def _positive_scalar(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是有限正数，单位为米") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} 必须是有限正数，单位为米")
    return number


def _positive_vector3(value: ArrayLike, field_name: str) -> NDArray[np.float64]:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是三个有限正数，单位为米") from error
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ValueError(f"{field_name} 必须是三个有限正数，单位为米")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positive_vector2(value: ArrayLike, field_name: str) -> NDArray[np.float64]:
    """复制两个有限正长度，供局部 X/Y 两个截面半径使用。"""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是两个有限正数，单位为米") from error
    if vector.shape != (2,) or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ValueError(f"{field_name} 必须是两个有限正数，单位为米")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _unit_axis(value: ArrayLike, field_name: str = "axis") -> NDArray[np.float64]:
    try:
        axis = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是有限三维非零向量") from error
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError(f"{field_name} 必须是有限三维非零向量")
    # 先按最大绝对分量缩放，避免直接平方在极大数时溢出、极小数时下溢。
    # 输入已确认有限，因此 scale 精确为零当且仅当三个分量全为零。
    scale = float(np.max(np.abs(axis)))
    if scale == 0.0:
        raise ValueError(f"{field_name} 必须是有限三维非零向量")
    scaled = axis / scale
    result = np.array(scaled / np.linalg.norm(scaled), dtype=float, copy=True)
    result_norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or not np.isclose(
        result_norm, 1.0, atol=8 * np.finfo(float).eps, rtol=0.0
    ):
        raise ValueError(f"{field_name} 归一化后必须为有限单位向量")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CuboidConcept:
    """长方体；``size_xyz_m`` 对应 part frame 的 X/Y/Z 全长。"""

    size_xyz_m: ArrayLike
    pose_in_object: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "size_xyz_m", _positive_vector3(self.size_xyz_m, "size_xyz_m")
        )
        object.__setattr__(
            self,
            "pose_in_object",
            validate_pose(self.pose_in_object, field_name="pose_in_object"),
        )


@dataclass(frozen=True, slots=True)
class CylinderConcept:
    """柱体；局部 X/Y 截面半径和高度单位为米。

    ``radius_xy_m`` 可表达椭圆截面；两个半径相等时是圆截面的特例。
    高度轴固定为 part 局部 +Z，整体朝向只由 ``pose_in_object`` 表达。
    """

    radius_xy_m: ArrayLike
    height_m: float
    pose_in_object: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius_xy_m",
            _positive_vector2(self.radius_xy_m, "radius_xy_m"),
        )
        object.__setattr__(self, "height_m", _positive_scalar(self.height_m, "height_m"))
        object.__setattr__(
            self,
            "pose_in_object",
            validate_pose(self.pose_in_object, field_name="pose_in_object"),
        )


@dataclass(frozen=True, slots=True)
class RingConcept:
    """有厚度的圆环或圆弧；尺寸单位为米，弧长角单位为弧度。

    ``angle_span_rad`` 表示从 part 局部方位角 0 开始存在的弧长。
    0 表示空弧，``2π`` 表示完整圆环；Concept 不保存独立起始角。
    """

    inner_radius_m: float
    outer_radius_m: float
    height_m: float
    axis: ArrayLike
    pose_in_object: ArrayLike
    angle_span_rad: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        try:
            inner = float(self.inner_radius_m)
            outer = float(self.outer_radius_m)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "outer_radius_m 与 inner_radius_m 必须满足 outer_radius_m > "
                "inner_radius_m > 0，单位为米"
            ) from error
        if not np.isfinite([inner, outer]).all() or not outer > inner > 0.0:
            raise ValueError(
                "outer_radius_m 与 inner_radius_m 必须满足 outer_radius_m > "
                "inner_radius_m > 0，单位为米"
            )
        object.__setattr__(self, "inner_radius_m", inner)
        object.__setattr__(self, "outer_radius_m", outer)
        object.__setattr__(self, "height_m", _positive_scalar(self.height_m, "height_m"))
        object.__setattr__(self, "axis", _unit_axis(self.axis))
        object.__setattr__(
            self,
            "pose_in_object",
            validate_pose(self.pose_in_object, field_name="pose_in_object"),
        )
        try:
            angle_span = float(self.angle_span_rad)
        except (TypeError, ValueError) as error:
            raise ValueError("angle_span_rad 必须是 0 到 2π 之间的有限弧度值") from error
        if not np.isfinite(angle_span) or not 0.0 <= angle_span <= 2.0 * np.pi:
            raise ValueError("angle_span_rad 必须是 0 到 2π 之间的有限弧度值")
        object.__setattr__(self, "angle_span_rad", angle_span)
