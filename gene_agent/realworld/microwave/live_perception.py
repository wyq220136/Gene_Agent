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


class HandleOrbitError(ValueError):
    """保留被拒绝帧的几何证据；上层可在导出碰撞场景前落盘。"""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


class LiveArticulationReader:
    """保持已标定柜体参考，用新把手轨迹检查并更新状态。

    门角沿用 AffordToolAct 的把手绕门轴投影 atan2；这里的轴来自 URDF
    正关节方向，所以不再反号。圆轨道残差检查与测角使用同一个把手中心，
    不把 mesh 完整朝向误当作额外关节测量。除闭门接触的有界平移约束外假定柜体固定；这不替代独立柜体跟踪。
    """

    def __init__(self, setup: dict, *, include_knob: bool = False,
                 max_translation_m=.025, max_rotation_rad=.15):
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
        # 旋转上限仅用于依靠朝向测角的旋钮；门把手检查中心圆轨道。
        self.max_rotation = float(max_rotation_rad)
        self._knob_angle = 0.
        self._last_frame = -1
        self._last_stamp = -1.
        self._anchor = None
        self._initial_root = self.root.copy()
        self.closed_contact_pending = False
        self.max_closed_contact_translation_m = .03

    def canonical_observation(self, observed):
        config = self.setup.get("microwave_pose_disambiguation")
        if not config or MICROWAVE not in observed.object_poses_sdk:
            return observed, {}
        from .geometry_alignment import canonical_handle_pose
        pose, diagnostics = canonical_handle_pose(observed.object_poses_sdk[MICROWAVE], config)
        return replace(observed, object_poses_sdk={**observed.object_poses_sdk, MICROWAVE: pose}), diagnostics

    def _joint(self, name):
        joint = self.assets["joints"][name]
        T = self.root @ validate_pose(joint["T_asset_joint"])
        axis = self.root[:3, :3] @ np.asarray(joint["axis_asset"], float)
        return T[:3, 3], axis / np.linalg.norm(axis), joint["limits_rad"]

    @staticmethod
    def _bounded(raw, limits, *, name):
        # 传感测角在资产限位附近允许 5° 噪声；不接受另一侧的错误 pose 或越界大角度。
        low, high = limits
        if raw < low - math.radians(5.) or raw > high + math.radians(5.):
            raise ValueError(f"{name}实测角超出资产关节范围: {math.degrees(raw):.2f}°")
        return float(np.clip(raw, low, high))

    def __call__(self, observed: ObservedScene) -> dict:
        if (not observed.pose_frame_verified or set(observed.object_poses_sdk) != self.objects
                or observed.frame_number <= self._last_frame or observed.timestamp_ms < self._last_stamp):
            raise ValueError("实时测角需要本次启用对象的同一新帧，不能复用旧观测")
        if not np.allclose(observed.T_sdk_camera, self.setup["camera"]["T_sdk_camera"], atol=1e-8, rtol=0):
            raise ValueError("运行外参与感知参考不一致")
        observed, _ = self.canonical_observation(observed)
        current = {key: validate_pose(value) for key, value in observed.object_poses_sdk.items()}
        point, axis, limits = self._joint("door")
        closed_handle = self.reference[MICROWAVE] @ self.handle_local
        live_handle = current[MICROWAVE] @ self.handle_local
        raw_door = _signed_angle(closed_handle[:3, 3] - point, live_handle[:3, 3] - point, axis)
        orbit_angle = raw_door
        plane_angle = getattr(self, "door_plane_angle", None)
        # Plane is a second independent geometric cue, not a commanded 90deg prior.
        # Reject ambiguity rather than rotating a robot/cabinet to make it fit.
        if plane_angle is not None:
            if abs(plane_angle - raw_door) > math.radians(10.):
                raise HandleOrbitError("门板法向与把手绕轴测角相差超过10度", {
                    "check": "door_plane_vs_handle", "orbit_angle_rad": raw_door,
                    "plane_angle_rad": plane_angle, "frame_number": observed.frame_number})
            raw_door = .5 * (raw_door + plane_angle)
        try:
            door = self._bounded(raw_door, limits, name="门角")
        except ValueError as error:
            raise HandleOrbitError(str(error), {"check": "door_angle_bounds",
                "raw_angle_rad": raw_door, "orbit_angle_rad": orbit_angle,
                "plane_angle_rad": plane_angle, "frame_number": observed.frame_number,
                "limits_rad": list(limits), "sensor_padding_deg": 5.}) from error
        expected_handle_mesh = _axis_rotation(axis, door, point) @ self.reference[MICROWAVE]
        expected_handle = expected_handle_mesh @ self.handle_local
        # FoundationPose 返回 mesh 原坐标系 pose；它的原点不等于把手中心。
        # 测角只使用中心的圆周运动（与 AffordToolAct 一致）。朝向估计变化
        # 即使不改变中心，也会移动 mesh 原点，不能据此断言柜体移动。
        residual = live_handle[:3, 3] - expected_handle[:3, 3]
        position_error = float(np.linalg.norm(residual))
        mesh_position_error = float(np.linalg.norm(expected_handle_mesh[:3, 3] - current[MICROWAVE][:3, 3]))
        rotation_error = float(Rotation.from_matrix(expected_handle_mesh[:3, :3].T @ current[MICROWAVE][:3, :3]).magnitude())
        closed_vector = closed_handle[:3, 3] - point
        live_vector = live_handle[:3, 3] - point
        closed_radius = float(np.linalg.norm(closed_vector - axis * (closed_vector @ axis)))
        live_radius = float(np.linalg.norm(live_vector - axis * (live_vector @ axis)))
        contact_reanchor = None
        if self.closed_contact_pending:
            self.closed_contact_pending = False
            # 闭门且尚未拉门时，夹爪可能把机身推移；仅推导平移，不从噪声朝向估计机身旋转。
            delta = live_handle[:3, 3] - closed_handle[:3, 3]
            total_delta = self.root[:3, 3] + delta - self._initial_root[:3, 3]
            if (not self.include_knob and position_error > self.max_translation
                    and abs(raw_door) <= math.radians(5.)
                    and np.linalg.norm(delta) <= self.max_closed_contact_translation_m
                    and np.linalg.norm(total_delta) <= self.max_closed_contact_translation_m):
                contact_reanchor = {
                    "frame_number": observed.frame_number,
                    "translation_sdk_m": delta.tolist(),
                    "cumulative_translation_sdk_m": total_delta.tolist(),
                    "limit_m": self.max_closed_contact_translation_m,
                    "original_handle_center_error_m": position_error,
                    "raw_angle_before_reanchor_rad": raw_door,
                    "independent_cabinet_measurement": False,
                    "assumption": "闭门夹把手后的平移约束；柜体不转动；门仍接近零角",
                }
                self.root[:3, 3] += delta
                for pose in self.reference.values():
                    pose[:3, 3] += delta
                point, axis, limits = self._joint("door")
                closed_handle = self.reference[MICROWAVE] @ self.handle_local
                closed_vector = closed_handle[:3, 3] - point
                live_vector = live_handle[:3, 3] - point
                raw_door = _signed_angle(closed_vector, live_vector, axis)
                door = self._bounded(raw_door, limits, name="门角")
                expected_handle_mesh = _axis_rotation(axis, door, point) @ self.reference[MICROWAVE]
                expected_handle = expected_handle_mesh @ self.handle_local
                position_error = float(np.linalg.norm(live_handle[:3, 3] - expected_handle[:3, 3]))
                mesh_position_error = float(np.linalg.norm(expected_handle_mesh[:3, 3] - current[MICROWAVE][:3, 3]))
                closed_radius = float(np.linalg.norm(closed_vector - axis * (closed_vector @ axis)))
                live_radius = float(np.linalg.norm(live_vector - axis * (live_vector @ axis)))
        diagnostics = {
            "contact_reanchor": contact_reanchor,
            "status": "rejected" if position_error > self.max_translation else "accepted",
            "check": "handle_center_orbit", "frame_number": observed.frame_number,
            "observed_at_unix_ms": observed.timestamp_ms,
            "raw_angle_rad": raw_door, "angle_rad": door,
            "orbit_angle_rad": orbit_angle, "plane_angle_rad": plane_angle,
            "handle_center_error_m": position_error, "max_translation_m": self.max_translation,
            "radius_error_m": live_radius - closed_radius,
            "axial_error_m": float((live_vector - closed_vector) @ axis),
            "mesh_origin_error_m": mesh_position_error,
            "mesh_rotation_error_rad": rotation_error,
            "mesh_rotation_is_diagnostic_only": True,
            "axis_point_sdk": point.tolist(), "axis_direction_sdk": axis.tolist(),
            "T_mesh_handle": self.handle_local.tolist(),
            "reference_mesh_pose_sdk": self.reference[MICROWAVE].tolist(),
            "observed_mesh_pose_sdk": current[MICROWAVE].tolist(),
            "expected_handle_center_sdk": expected_handle[:3, 3].tolist(),
            "observed_handle_center_sdk": live_handle[:3, 3].tolist(),
        }
        if position_error > self.max_translation:
            raise HandleOrbitError(
                "把手中心偏离固定门轴圆轨道："
                f"中心误差 {position_error * 1000:.2f} mm > {self.max_translation * 1000:.2f} mm，"
                f"半径差 {diagnostics['radius_error_m'] * 1000:.2f} mm，"
                f"轴向差 {diagnostics['axial_error_m'] * 1000:.2f} mm，"
                f"门角 {math.degrees(raw_door):.2f}°，帧 {observed.frame_number}；"
                "柜体可能移动或 tracking 漂移，请核对关门参考、mask、标定和当前帧",
                diagnostics,
            )

        source = f"same_frame_foundationpose:{observed.frame_number}"
        readings = {"door": {"angle_rad": door, "observed_at_unix_ms": observed.timestamp_ms,
                             "source": source + (":door_plane_and_handle_orbit" if plane_angle is not None else ":handle_orbit"), "raw_angle_rad": raw_door,
                             "diagnostics": diagnostics}}
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
        if np.linalg.norm(self.root[:3, 3] - self._initial_root[:3, 3]) > 1e-9:
            # 机身被推移不意味着桌子/支撑箱也移动；保存其原始静态位置。
            support_local = np.eye(4)
            support = self.assets["support"]
            support_local[:3, 3] = [*support["center_xy_local_m"], -support["height_m"] / 2]
            self._anchor["stationary_support_pose_sdk"] = (self._initial_root @ support_local).tolist()
        if contact_reanchor is not None:
            self._anchor["contact_reanchor"] = contact_reanchor
            self._anchor["assumption"] = contact_reanchor["assumption"]
            self._anchor["source"] = source + ":closed_contact_translation_constraint"
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
