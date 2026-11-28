"""Task1 按操作刷新感知；固定机身与非操作对象沿用明确来源的状态。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import time

import numpy as np

from .live_perception import LiveArticulationReader
from .perception_setup import BOWL, MICROWAVE, KNOB

POLICY = "affordtoolact_operation_scoped"


def begin_observation_operation(observer, operation):
    """普通离线观察回调保持原接口；现场观察器显式切换操作范围。"""
    callback = getattr(observer, "begin_operation", None)
    if callback is not None:
        callback(operation)


def after_door_observation_phase(observer, phase):
    callback = getattr(observer, "after_door_phase", None)
    if callback is not None:
        callback(phase)


def forward_observation_hooks(source, target):
    for name in ("begin_operation", "after_door_phase", "before_interaction", "after_interaction", "set_door_final_phase", "accept_current_hold"):
        callback = getattr(source, name, None)
        if callback is not None:
            setattr(target, name, callback)


class KeyframeStateError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class OperationPerception:
    """真机关键帧感知、缓存和有来源的状态估计。

    SDK始终实读；可见关键帧视觉校验，炉内遮挡用已提交的夹持/释放关系。
    关闭参考的门板修正仅在运行副本中应用，不改写标定和原始参考文件。
    每个对象单独保留测量时间；估计和允许继续的开环检查均不冒充实测。
    """
    def __init__(self, setup, *, include_knob=False, sleep=None):
        self.reader = LiveArticulationReader(setup, include_knob=include_knob)
        from .door_plane import DoorPlaneObserver
        self.door_plane = DoorPlaneObserver(setup["assets"])
        self.reference_plane = {"applied": False, "reason": "reference_rgbd_unavailable"}
        reference_capture = self.door_plane.reference_capture(setup)
        if reference_capture is not None and self.door_plane.available:
            pose, self.reference_plane = self.door_plane.correct(
                self.reader.reference[MICROWAVE], reference_capture, setup["camera"]["T_sdk_camera"])
            if self.reference_plane["applied"]:
                self.reader.reference[MICROWAVE] = pose
                self.reader.root = pose @ self.reader.alignment
                self.reader._initial_root = self.reader.root.copy()
        self.include_knob = include_knob
        frame = setup.get("reference_frame", {"frame_number": 0})
        stamp = float(frame.get("captured_at_unix_ms", 0.))
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("关闭参考时间无效")
        self.poses = {key: value.copy() for key, value in self.reader.reference.items()}
        self.records = {key: {"frame_number": int(frame["frame_number"]),
                             "observed_at_unix_ms": stamp,
                             "source": "confirmed_closed_reference"}
                        for key in self.poses}
        self.angles = {"door": {"angle_rad": 0., "observed_at_unix_ms": stamp,
                               "source": "confirmed_closed_reference"}}
        if include_knob:
            self.angles["knob"] = {"angle_rad": 0., "observed_at_unix_ms": stamp,
                                  "source": "confirmed_zero_reference"}
        self.anchor = {"T_world_scene_asset": self.reader.root.tolist(),
                       "verified_at_unix_ms": stamp,
                       "source": "confirmed_closed_reference:fixed_cabinet"}
        self.operation = None
        self.sleep = time.sleep if sleep is None else sleep
        self.pending_settle_s = 0.
        self.last_settle_s = 0.
        self.pending = set()
        self.initialized = set()
        self.last_frame = -1
        self.last_stamp = -1.
        self.attachment_sdk = None
        self.last_tcp_sdk = None
        self.bowl_event = None
        self.bowl_mode = "measured"
        self.grasp_assumed = False
        self.released = False
        self.last_measurement_check = None
        self.estimated_door_angle = None
        self.door_phase = None
        self.door_final_observed = False
        self.plane_diagnostics = {}
        self.keyframe_counts = {}


    def begin_operation(self, operation):
        if operation == "plan_only":
            needed = {BOWL, MICROWAVE} | ({KNOB} if self.include_knob else set())
        elif operation in {"pick_bowl", "insert_bowl", "extract_bowl", "place_bowl_on_table",
                           "pick", "insert", "extract", "table", "load_bowl", "unload_bowl"}:
            needed = {BOWL}
        elif operation in {"open_door", "close_door", "door"}:
            needed = {MICROWAVE}
        elif operation in {"rotate_knob", "knob"} and self.include_knob:
            needed = {KNOB}
        elif operation in {"transition", "after_open_transition", "before_reopen_transition",
                           "before_close_first_transition", "before_close_second_transition"}:
            needed = set()
        else:
            raise ValueError(f"未定义感知时机的操作: {operation}")
        # 操作切换不清空尚未消费的交互事件；同次运行每个对象只初始化一次。
        self.pending.update(needed - self.initialized)
        if operation in {"open_door", "close_door", "door"} and operation != self.operation:
            self.pending.add(MICROWAVE)
            self.estimated_door_angle = None
            self.door_final_observed = False
        self.operation = operation
        self.reader.closed_contact_pending = False

    @staticmethod
    def _stage_name(stage):
        name = stage.rsplit("/", 1)[-1].split(":")[-1]
        prefix, sep, rest = name.partition("_")
        return rest if sep and prefix.isdigit() else name

    def before_interaction(self, stage, kind):
        # This hook runs before dispatch. Do not predict a successful command here.
        if (self.operation in {"open_door", "close_door", "door"}
                and kind == "grasp" and self._stage_name(stage) == "close"):
            self.reader.closed_contact_pending = True

    def after_interaction(self, stage, kind):
        """SDK completion hook; robot TCP is still sampled independently on consume."""
        name = self._stage_name(stage)
        op = self.operation
        if op in {"pick_bowl", "pick", "extract_bowl", "extract", "unload_bowl",
                  "insert_bowl", "insert", "load_bowl", "place_bowl_on_table", "table"}:
            if kind == "grasp":
                self.bowl_event = "attach"
            elif kind == "release":
                self.bowl_event = "detach"
            if kind == "motion" and (
                    (op in {"pick_bowl", "pick"} and name == "lift")
                    or (op in {"extract_bowl", "extract", "unload_bowl"} and name == "post_extract_2")
                    or (op in {"place_bowl_on_table", "table"} and name == "retreat")):
                self.pending.add(BOWL)
        elif op in {"rotate_knob", "knob"} and self.include_knob:
            if kind in {"grasp", "release", "last_joint"} or name == "grasp" or name.startswith("rotate_"):
                self.pending.add(KNOB)

    def after_door_phase(self, phase):
        # Intermediate motion points use the executed phase's assumption, never a
        # fabricated sensor reading. Final phase remeasures after settling; the
        # immediately adjacent post-contact reuses that same settled keyframe.
        if isinstance(phase, str):
            if "post_contact" in phase and not self.door_final_observed:
                self.pending.add(MICROWAVE)
                self.pending_settle_s = 2.0
            return
        self.door_phase = phase.name
        if phase.name.startswith(("pull_", "swing_", "push_")):
            self.estimated_door_angle = float(phase.door_angle_rad)
        if getattr(self, "final_door_phase_id", None) == phase.phase_id:
            self.pending.add(MICROWAVE)
            self.pending_settle_s = 2.0

    def set_door_final_phase(self, phase_id):
        if getattr(self, "final_door_phase_id", None) != phase_id:
            self.door_final_observed = False
        self.final_door_phase_id = phase_id

    def prepare_capture(self):
        if self.operation is None:
            raise ValueError("现场观察前必须设置当前操作")
        self.last_settle_s = self.pending_settle_s
        if self.pending_settle_s:
            # AffordToolAct Task1: 抬离后先等门稳定 2 秒，再采集用于重新测角的帧。
            self.sleep(self.pending_settle_s)
            self.pending_settle_s = 0.

    def selected_sessions(self, sessions):
        if self.operation is None:
            raise ValueError("现场观察前必须设置当前操作，不能默认刷新全部对象")
        selected = self.pending
        if not selected.issubset(sessions):
            raise ValueError("本操作所需的感知 session 缺失")
        return {key: session for key, session in sessions.items() if key in selected}

    def consume(self, observed, *, tcp_pose_sdk=None, capture_dir=None):
        if (not observed.pose_frame_verified or observed.frame_number <= self.last_frame
                or not math.isfinite(observed.timestamp_ms)
                or observed.timestamp_ms < 0 or observed.timestamp_ms < self.last_stamp):
            raise ValueError("操作感知需要带核对来源的新帧，不能复用旧观测")
        if not np.allclose(observed.T_sdk_camera, self.reader.setup["camera"]["T_sdk_camera"],
                           atol=1e-8, rtol=0):
            raise ValueError("运行外参与感知参考不一致")
        measured = set(observed.object_poses_sdk)
        expected = self.pending
        if measured != expected:
            raise ValueError("当前帧感知对象与操作范围不一致")
        observed, pose_correction = self.reader.canonical_observation(observed)
        self.reader.door_plane_angle = None
        if MICROWAVE in measured and capture_dir is not None:
            pose, self.plane_diagnostics = self.door_plane.correct(
                observed.object_poses_sdk[MICROWAVE], capture_dir, observed.T_sdk_camera)
            self.plane_diagnostics["reference"] = self.reference_plane
            if self.plane_diagnostics["applied"]:
                observed = replace(observed, object_poses_sdk={**observed.object_poses_sdk, MICROWAVE: pose})
                if self.reference_plane.get("applied"):
                    from .live_perception import _signed_angle
                    _, axis, _ = self.reader._joint("door")
                    ref_normal = np.asarray(self.reference_plane["normal_sdk"])
                    normal = np.asarray(self.plane_diagnostics["normal_sdk"])
                    self.reader.door_plane_angle = _signed_angle(ref_normal, normal, axis)
        tcp = None if tcp_pose_sdk is None else np.asarray(tcp_pose_sdk, dtype=float)
        if tcp is not None:
            from gene_agent.knowledge.geometry import validate_pose
            tcp = validate_pose(tcp, field_name="keyframe_sdk_tcp")
        if self.bowl_event == "attach":
            if tcp is None:
                raise ValueError("夹持估计需要当前 SDK 实测 TCP")
            self.attachment_sdk = np.linalg.inv(tcp) @ self.poses[BOWL]
            self.grasp_assumed, self.released = True, False
            self.bowl_mode = "sdk_attachment_estimate"
        elif self.bowl_event == "detach":
            if self.attachment_sdk is not None:
                if tcp is None:
                    raise ValueError("释放估计需要当前 SDK 实测 TCP")
                self.poses[BOWL] = tcp @ self.attachment_sdk
            self.attachment_sdk = None
            self.grasp_assumed, self.released = False, True
            self.bowl_mode = "released_pose_assumption"
        self.bowl_event = None
        predicted_ids = set()
        if self.attachment_sdk is not None:
            if tcp is None:
                raise ValueError("搬运估计需要当前 SDK 实测 TCP")
            expected_pose = tcp @ self.attachment_sdk
            if BOWL in measured:
                actual = observed.object_poses_sdk[BOWL].copy()
                from scipy.spatial.transform import Rotation
                relative = actual[:3, :3].T @ expected_pose[:3, :3]
                yaw = float(np.arctan2(relative[0, 2] - relative[2, 0], relative[0, 0] + relative[2, 2]))
                actual[:3, :3] = actual[:3, :3] @ Rotation.from_euler("y", yaw).as_matrix()
                observed = replace(observed, object_poses_sdk={**observed.object_poses_sdk, BOWL: actual})
                pose_correction["bowl_symmetry_yaw_rad"] = yaw
                translation = float(np.linalg.norm(actual[:3, 3] - expected_pose[:3, 3]))
                # Bowl_0 is rotationally symmetric about local +Y. Yaw about that
                # axis is unobservable and must not be mistaken for grip slippage.
                tilt = float(np.arccos(np.clip(actual[:3, 1] @ expected_pose[:3, 1], -1., 1.)))
                self.last_measurement_check = {
                    "frame_number": observed.frame_number, "translation_error_m": translation,
                    "axis_error_rad": tilt, "max_translation_m": .02,
                    "max_axis_error_rad": math.radians(20.),
                    "passed": translation <= .02 and tilt <= math.radians(20.)}
                if not self.last_measurement_check["passed"]:
                    raise KeyframeStateError("关键帧碗位姿偏离夹持估计", {**self.last_measurement_check, "expected_pose_sdk": expected_pose.tolist(), "observed_pose_sdk": actual.tolist()})
                self.attachment_sdk = np.linalg.inv(tcp) @ actual
            else:
                self.poses[BOWL] = expected_pose
                self.bowl_mode = "sdk_attachment_estimate"
                predicted_ids.add(BOWL)
        elif self.bowl_mode == "released_pose_assumption" and BOWL not in measured:
            predicted_ids.add(BOWL)
        if BOWL in measured:
            self.bowl_mode = "measured"
            if self.released:
                actual = observed.object_poses_sdk[BOWL]
                expected_pose = self.poses[BOWL]
                error = float(np.linalg.norm(actual[:3, 3] - expected_pose[:3, 3]))
                tilt = float(np.arccos(np.clip(actual[:3, 1] @ expected_pose[:3, 1], -1., 1.)))
                self.last_measurement_check = {"frame_number": observed.frame_number,
                    "released_pose_observed": True, "translation_error_m": error,
                    "axis_error_rad": tilt, "passed": error <= .02 and tilt <= math.radians(20.)}
                if not self.last_measurement_check["passed"]:
                    raise KeyframeStateError("释放后关键帧偏离已提交位置", {**self.last_measurement_check, "expected_pose_sdk": expected_pose.tolist(), "observed_pose_sdk": actual.tolist()})
        if self.estimated_door_angle is not None and MICROWAVE not in measured:
            from .live_perception import _axis_rotation
            point, axis, _ = self.reader._joint("door")
            self.poses[MICROWAVE] = (_axis_rotation(axis, self.estimated_door_angle, point)
                                    @ self.reader.reference[MICROWAVE])
            self.angles["door"] = {"angle_rad": self.estimated_door_angle,
                "observed_at_unix_ms": self.angles["door"]["observed_at_unix_ms"],
                "source": "executed_door_phase_assumption", "observed": False,
                "phase": self.door_phase, "physical_verification": False}
            predicted_ids.add(MICROWAVE)
        if MICROWAVE in measured:
            self.estimated_door_angle = None
            self.door_final_observed = bool(self.pending_settle_s == 0 and self.last_settle_s > 0)
        else:
            self.reader.closed_contact_pending = False

        poses = {**self.poses, **observed.object_poses_sdk}
        refreshed_angles = set()
        if measured & {MICROWAVE, KNOB}:
            full = replace(observed, object_poses_sdk=poses)
            readings = self.reader(full)
            for key, object_id in (("door", MICROWAVE), ("knob", KNOB)):
                if object_id in measured:
                    self.angles[key] = readings[key]
                    refreshed_angles.add(key)
            if MICROWAVE in measured:
                self.anchor = self.reader.anchor(full)
        records = deepcopy(self.records)
        for key in measured:
            records[key] = {"frame_number": observed.frame_number,
                            "observed_at_unix_ms": observed.timestamp_ms,
                            "source": observed.sources[key]}
        retained = set(poses) - measured - predicted_ids
        for key in measured:
            self.keyframe_counts[key] = self.keyframe_counts.get(key, 0) + 1
        estimates = {key: {"method": (self.bowl_mode if key == BOWL else "executed_door_phase_assumption"),
            "physical_verification": False, "measurement_origin": records[key],
            "sdk_frame_number": observed.frame_number,
            "sdk_tcp_pose": None if tcp is None else tcp.tolist()} for key in predicted_ids}
        refresh = {
            "policy": POLICY, "operation": self.operation, "refresh_mode": "interaction_events",
            "pose_frame_disambiguation": pose_correction,
            "settle_wait_s": self.last_settle_s,
            "refreshed_object_ids": sorted(measured),
            "retained_object_ids": sorted(retained),
            "object_observations": records,
            "estimated_object_ids": sorted(predicted_ids), "state_estimates": estimates,
            "keyframe_counts": deepcopy(self.keyframe_counts),
            "effective_sensor_tolerances": {"bowl_translation_m": .02, "bowl_axis_deg": 20.,
                "door_orbit_m": self.reader.max_translation, "door_bounds_padding_deg": 5.},
            "door_plane": deepcopy(self.plane_diagnostics),
            "schedule": "initial_and_visible_post_interaction_keyframes",
            "bowl_state_mode": self.bowl_mode,
            "bowl_measurement_check": deepcopy(self.last_measurement_check),
            "retained_angle_keys": sorted(set(self.angles) - refreshed_angles),
            "retained_cabinet_anchor": MICROWAVE not in measured,
            "assumption": "柜体除闭门接触有界平移外保持固定；对象在实际交互事件之间沿用状态；外部移动须重新启动观测",
        }
        result = replace(observed, object_poses_sdk=poses,
                         sources={**{k: v["source"] for k, v in records.items()}, **observed.sources},
                         perception_refresh=refresh)
        self.last_tcp_sdk = None if tcp is None else tcp.copy()
        self.poses = {k: np.array(v, copy=True) for k, v in poses.items()}
        self.records = records
        self.initialized.update(measured)
        self.pending.difference_update(measured)
        self.last_frame, self.last_stamp = observed.frame_number, observed.timestamp_ms
        return result, deepcopy(self.angles), deepcopy(self.anchor)


    def accept_current_hold(self, scene):
        if self.attachment_sdk is None:
            from gene_agent.realworld.scene import observed_tcp_pose
            source = scene["provenance"]
            world_sdk = np.asarray(source["T_world_sdk"])
            tcp = np.linalg.inv(world_sdk) @ observed_tcp_pose(scene)
            bowl = np.linalg.inv(world_sdk) @ np.asarray(source["aligned_object_poses_world"][BOWL])
            self.attachment_sdk = np.linalg.inv(tcp) @ bowl
            self.grasp_assumed, self.released = True, False

    def check_state(self, condition, scene, planned=None):
        from gene_agent.tasks.checks import CheckResult
        from gene_agent.realworld.checks import bind_check_to_scene
        source = scene["provenance"]
        refresh = source.get("perception_refresh") or {}
        measured = BOWL in refresh.get("refreshed_object_ids", ())
        evidence = {"policy": POLICY, "condition": condition,
            "authorized_open_loop": True, "physical_verification": False,
            "sdk_state_observed": bool(source.get("live_state")),
            "state_mode": refresh.get("bowl_state_mode"),
            "object_estimate": refresh.get("state_estimates", {}).get(BOWL),
            "last_visual_check": self.last_measurement_check}
        if condition in {"grasp", "hold", "pick", "extracted"}:
            known = self.grasp_assumed and self.attachment_sdk is not None
        elif condition in {"release", "place", "table"}:
            known = self.released and self.attachment_sdk is None
        elif condition == "path":
            # Extract's initial contact approach occurs BEFORE closing the gripper.
            # Its geometry may use the known released bowl pose without pretending
            # that a grasp already exists. No unplanned path is accepted here.
            known = (BOWL in self.initialized and isinstance(planned, dict)
                     and planned.get("success") is True)
            evidence["claim"] = "只继承已通过的路径几何检查；没有真实接触测量"
        else:
            known = False
        if not known:
            return CheckResult("unknown", "not_configured", evidence,
                               "没有本轮成功 SDK 交互事件，不能自动建立物体状态")
        # Lift/extracted require an actual visible keyframe. Failed/missing keyframes
        # propagate an error; they are not converted to an assumed successful lift.
        if condition in {"pick", "extracted"} and not measured:
            return CheckResult("unknown", "not_configured", evidence, "缺少可见关键帧")
        if measured and condition in {"pick", "extracted"}:
            evidence.update(physical_verification=True, authorized_open_loop=False)
            result = CheckResult("pass", "vision", evidence,
                                "关键帧碗位姿符合实测 SDK 夹持关系；不宣称接触传感器验证")
        elif measured and condition == "table":
            from .bowl_pose_check import check_bowl_pose_table
            table = np.asarray(source["table_pose_world"])
            bowl = np.asarray(source["aligned_object_poses_world"][BOWL])
            try:
                geometry = check_bowl_pose_table(bowl, table, source["accepted_table_thickness_m"], table_band_m=.02)
            except ValueError as error:
                return CheckResult("fail", "vision", evidence, str(error))
            evidence.update(physical_verification=True, geometry=geometry)
            result = CheckResult("pass", "vision", evidence, "释放后可见桌面关键帧通过")
        else:
            result = CheckResult("pass", "open_loop_object_state", evidence,
                                "按已完成 SDK 命令和对象状态假设继续；物理后置未视觉验证")
        return bind_check_to_scene(scene, result)

    def configure_checks(self, options):
        callbacks = {
            "pick": {"grasp_check": "grasp", "check_pick": "pick"},
            "insert": {"check_hold": "hold", "check_insert_path": "path",
                       "check_release": "release", "check_place": "place"},
            "extract": {"check_grasp": "grasp", "check_hold": "hold",
                        "check_contact_path": "path", "check_extracted": "extracted"},
            "table": {"check_hold": "hold", "check_place_path": "path",
                      "check_release": "release", "check_table": "table"}}
        for kind, mapping in callbacks.items():
            for key, condition in mapping.items():
                if kind in options:
                    options[kind].setdefault(key, lambda *args, c=condition: self.check_state(c, *args))
