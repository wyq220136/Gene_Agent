"""当前开放式长方体容器需要的 typed geometry。"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from .basic import _positive_scalar, _positive_vector3, _unit_axis
from .transforms import validate_pose


@dataclass(frozen=True, slots=True)
class ContainerWall:
    """容器局部 frame 中的一个长方体墙体。"""

    name: str
    size_xyz_m: ArrayLike
    pose_in_container: ArrayLike

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name 必须是非空字符串")
        object.__setattr__(
            self, "size_xyz_m", _positive_vector3(self.size_xyz_m, "size_xyz_m")
        )
        object.__setattr__(
            self,
            "pose_in_container",
            validate_pose(self.pose_in_container, field_name="pose_in_container"),
        )


@dataclass(frozen=True, slots=True)
class ContainerConcept:
    """开放式容器的轮廓、轴、高度、墙体及 ``T_object_container``。

    当前资产的开口沿容器局部 +Y，内外轮廓均位于局部 XZ 平面；
    ``height_m`` 表示沿开口轴的总深度。保留墙体可精确表达后、上下和左右边界，
    但本类型不包含抓取、双臂或放置策略。
    """

    outer_contour_size_m: ArrayLike
    inner_contour_size_m: ArrayLike
    height_m: float
    opening_axis: ArrayLike
    pose_in_object: ArrayLike
    bounds_min_m: ArrayLike
    walls: tuple[ContainerWall, ...]

    def __post_init__(self) -> None:
        outer = _positive_vector3(
            [*self.outer_contour_size_m, self.height_m], "outer_contour_size_m"
        )[:2]
        inner = _positive_vector3(
            [*self.inner_contour_size_m, self.height_m], "inner_contour_size_m"
        )[:2]
        if not (inner < outer).all():
            raise ValueError("inner_contour_size_m 必须逐项小于 outer_contour_size_m")
        outer.setflags(write=False)
        inner.setflags(write=False)
        object.__setattr__(self, "outer_contour_size_m", outer)
        object.__setattr__(self, "inner_contour_size_m", inner)
        object.__setattr__(self, "height_m", _positive_scalar(self.height_m, "height_m"))
        object.__setattr__(self, "opening_axis", _unit_axis(self.opening_axis, "opening_axis"))
        object.__setattr__(
            self,
            "pose_in_object",
            validate_pose(self.pose_in_object, field_name="pose_in_object"),
        )
        object.__setattr__(
            self, "bounds_min_m", _finite_vector3(self.bounds_min_m, "bounds_min_m")
        )
        object.__setattr__(self, "walls", tuple(self.walls))
        if not self.walls or not all(isinstance(wall, ContainerWall) for wall in self.walls):
            raise ValueError("walls 必须是非空 ContainerWall 序列")


def _finite_vector3(value: ArrayLike, field_name: str):
    """校验可包含负数的三维米制坐标，并返回只读副本。"""
    import numpy as np

    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是三个有限数，单位为米") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{field_name} 必须是三个有限数，单位为米")
    result = result.copy()
    result.setflags(write=False)
    return result


__all__ = ["ContainerConcept", "ContainerWall"]
