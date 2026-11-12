"""由中间抓持段和两侧连接段组成的三段式几何。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .basic import _positive_scalar, _positive_vector3
from .transforms import validate_pose


@dataclass(frozen=True, slots=True)
class TrifoldSegment:
    """Trifold 局部 frame 中的一个长方体段，尺寸单位为米。"""

    size_xyz_m: ArrayLike
    pose_in_trifold: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "size_xyz_m", _positive_vector3(self.size_xyz_m, "size_xyz_m")
        )
        object.__setattr__(
            self,
            "pose_in_trifold",
            validate_pose(self.pose_in_trifold, field_name="pose_in_trifold"),
        )


@dataclass(frozen=True, slots=True)
class TrifoldConcept:
    """三段式结构及其相对 object frame 的位姿。

    两个连接段中心沿 trifold 局部 Y 轴对称分布，中间抓持段位于 Y=0；
    三段的局部 X/Y/Z 轴一致。连接段从局部 Z=0 向 +Z 延伸，抓持段接在
    连接段上方。该关系来自当前结构化资产，不包含具体物体名称或抓取偏好。
    """

    grip_size_xyz_m: ArrayLike
    connector_size_xyz_m: ArrayLike
    connector_center_separation_m: float
    pose_in_object: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grip_size_xyz_m",
            _positive_vector3(self.grip_size_xyz_m, "grip_size_xyz_m"),
        )
        object.__setattr__(
            self,
            "connector_size_xyz_m",
            _positive_vector3(self.connector_size_xyz_m, "connector_size_xyz_m"),
        )
        object.__setattr__(
            self,
            "connector_center_separation_m",
            _positive_scalar(
                self.connector_center_separation_m,
                "connector_center_separation_m",
            ),
        )
        object.__setattr__(
            self,
            "pose_in_object",
            validate_pose(self.pose_in_object, field_name="pose_in_object"),
        )

    @property
    def grip_segment(self) -> TrifoldSegment:
        """返回中段；其 pose 是 ``T_trifold_grip``。"""
        pose = np.eye(4)
        pose[2, 3] = self.connector_size_xyz_m[2] + self.grip_size_xyz_m[2] / 2.0
        return TrifoldSegment(self.grip_size_xyz_m, pose)

    @property
    def positive_connector(self) -> TrifoldSegment:
        """返回中心位于 trifold 局部 +Y 一侧的连接段。"""
        return self._connector(+1.0)

    @property
    def negative_connector(self) -> TrifoldSegment:
        """返回中心位于 trifold 局部 -Y 一侧的连接段。"""
        return self._connector(-1.0)

    def _connector(self, sign: float) -> TrifoldSegment:
        pose = np.eye(4)
        pose[1, 3] = sign * self.connector_center_separation_m / 2.0
        pose[2, 3] = self.connector_size_xyz_m[2] / 2.0
        return TrifoldSegment(self.connector_size_xyz_m, pose)


__all__ = ["TrifoldConcept", "TrifoldSegment"]
