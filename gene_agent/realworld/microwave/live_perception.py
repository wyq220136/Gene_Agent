"""从新 RGB-D 帧读门角；默认不测旋钮，不读手工状态 JSON。"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.perception import ObservedScene
from .perception_setup import BOWL, MICROWAVE, KNOB


def _axis_rotation(axis, angle: float, point) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    point = np.asarray(point, float)
    T[:3, 3] = point - T[:3, :3] @ point
    return T


def _signed_angle(reference, current, axis) -> float:
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    first = np.asarray(reference, float) - axis * np.dot(reference, axis)
    second = np.asarray(current, float) - axis * np.dot(current, axis)
    if min(np.linalg.norm(first), np.linalg.norm(second)) < 1e-8:
        raise ValueError("测角参考点或方向落在关节轴上")
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(np.arctan2(axis @ np.cross(first, second), first @ second))


class LiveArticulationReader:
    """保持已标定柜体参考，用新把手轨迹检查并更新状态。

    门角沿用 AffordToolAct 的把手绕门轴投影 atan2；这里的轴来自 URDF
    正关节方向，所以不再反号。柜体必须固定，残差检查不能代替独立柜体跟踪。
    """

    def __init__(self, setup: dict, *, include_knob: bool = False,
                 max_translation_m=.015, max_rotation_rad=.15):
        self.setup = setup
        self.assets = setup["assets"]
        self.include_knob = bool(include_knob)
        self.objects = {BOWL, MICROWAVE, KNOB} if self.include_knob else {BOWL, MICROWAVE}
        self.reference = {key: validate_pose(setup["closed_reference_poses_sdk"][key]).copy()
                          for key in self.objects}
        self.alignment = validate_pose(self.assets["alignment"])
        self.root = self.reference[MICROWAVE] @ self.alignment
        self.handle_local = self.alignment @ validate_pose(self.assets["T_asset_handle"])
        self.max_translation = float(max_translation_m)
        self.max_rotation = float(max_rotation_rad)
        self._knob_angle = 0.
        self._last_frame = -1
        self._last_stamp = -1.
        self._anchor = None

    def _joint(self, name):
        joint = self.assets["joints"][name]
        T = self.root @ validate_pose(joint["T_asset_joint"])
        axis = self.root[:3, :3] @ np.asarray(joint["axis_asset"], float)
        return T[:3, 3], axis / np.linalg.norm(axis), joint["limits_rad"]

    @staticmethod
    def _bounded(raw, limits, *, name):
        # 零位附近的跟踪抖动允许 2°；不接受另一侧的错误 pose 或越界大角度。
        low, high = limits
        if raw < low - math.radians(2.) or raw > high + math.radians(2.):
            raise ValueError(f"{name}实测角超出资产关节范围: {math.degrees(raw):.2f}°")
        return float(np.clip(raw, low, high))

    def __call__(self, observed: ObservedScene) -> dict:
        if (not observed.pose_frame_verified or set(observed.object_poses_sdk) != self.objects
                or observed.frame_number <= self._last_frame or observed.timestamp_ms < self._last_stamp):
            raise ValueError("实时测角需要本次启用对象的同一新帧，不能复用旧观测")
        if not np.allclose(observed.T_sdk_camera, self.setup["camera"]["T_sdk_camera"], atol=1e-8, rtol=0):
            raise ValueError("运行外参与感知参考不一致")
        current = {key: validate_pose(value) for key, value in observed.object_poses_sdk.items()}
        point, axis, limits = self._joint("door")
        closed_handle = self.reference[MICROWAVE] @ self.handle_local
        live_handle = current[MICROWAVE] @ self.handle_local
        raw_door = _signed_angle(closed_handle[:3, 3] - point, live_handle[:3, 3] - point, axis)
        door = self._bounded(raw_door, limits, name="门角")
        expected_handle_mesh = _axis_rotation(axis, door, point) @ self.reference[MICROWAVE]
        position_error = float(np.linalg.norm(expected_handle_mesh[:3, 3] - current[MICROWAVE][:3, 3]))
        rotation_error = float(Rotation.from_matrix(expected_handle_mesh[:3, :3].T @ current[MICROWAVE][:3, :3]).magnitude())
        if position_error > self.max_translation or rotation_error > self.max_rotation:
            raise ValueError("把手观测不符合固定门轴：柜体可能移动或 tracking 漂移，请重新准备感知")

        source = f"same_frame_foundationpose:{observed.frame_number}"
        readings = {"door": {"angle_rad": door, "observed_at_unix_ms": observed.timestamp_ms,
                             "source": source + ":handle_orbit", "raw_angle_rad": raw_door}}
        if self.include_knob:
            point_knob, axis_knob, knob_limits = self._joint("knob")
            reference_knob = self.reference[KNOB]
            # 使用最不平行于轴的 mesh 局部轴，避免指针/轴方向退化；不依赖相机轴约定。
            column = int(np.argmin(np.abs(reference_knob[:3, :3].T @ axis_knob)))
            wrapped = _signed_angle(reference_knob[:3, column], current[KNOB][:3, column], axis_knob)
            candidates = [wrapped + k * 2 * math.pi for k in (-1, 0, 1)]
            candidates = [v for v in candidates if knob_limits[0] - math.radians(2) <= v <= knob_limits[1] + math.radians(2)]
            if not candidates:
                raise ValueError("旋钮跟踪角超出资产范围，不能假定旋钮角为零")
            raw_knob = min(candidates, key=lambda value: abs(value - self._knob_angle))
            knob = self._bounded(raw_knob, knob_limits, name="旋钮角")
            expected_knob = _axis_rotation(axis_knob, knob, point_knob) @ reference_knob
            knob_position_error = float(np.linalg.norm(expected_knob[:3, 3] - current[KNOB][:3, 3]))
            knob_rotation_error = float(Rotation.from_matrix(expected_knob[:3, :3].T @ current[KNOB][:3, :3]).magnitude())
            if knob_position_error > self.max_translation or knob_rotation_error > self.max_rotation:
                raise ValueError("旋钮观测不符合固定柜体关节轴，请检查 mask、标定或 tracking")
            readings["knob"] = {"angle_rad": knob, "observed_at_unix_ms": observed.timestamp_ms,
                                "source": source + ":knob_orientation", "raw_angle_rad": raw_knob}
            self._knob_angle = knob
        # no_knob 只核对把手圆周，不能把未观测的旋钮写成新测量。
        self._anchor = {"T_world_scene_asset": self.root.tolist(),
                        "verified_at_unix_ms": observed.timestamp_ms,
                        "source": source + ":fixed_cabinet_articulation_consistency",
                        "checked_parts": ["door", "knob"] if self.include_knob else ["door"],
                        "assumption": ("柜体固定；把手圆周及旋钮固定轴一致性检查，不是独立柜体测量"
                                       if self.include_knob else
                                       "柜体固定；仅把手圆周一致性检查，不测旋钮，不是独立柜体测量")}
        self._last_frame, self._last_stamp = observed.frame_number, observed.timestamp_ms
        return readings

    def anchor(self, observed: ObservedScene) -> dict:
        if self._anchor is None or (observed.frame_number, observed.timestamp_ms) != (self._last_frame, self._last_stamp):
            raise ValueError("柜体参考尚未通过当前帧几何核对")
        return dict(self._anchor)


def collision_objects(observed: ObservedScene) -> ObservedScene:
    """旋钮 session 只供关节测量，碰撞仍由已有微波炉 URDF 生成，避免重复物体。"""
    return replace(observed, object_poses_sdk={k: v for k, v in observed.object_poses_sdk.items() if k != KNOB},
                   sources={k: v for k, v in observed.sources.items() if k != KNOB})
