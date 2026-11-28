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
    for name in ("begin_operation", "after_door_phase", "before_interaction"):
        callback = getattr(source, name, None)
        if callback is not None:
            setattr(target, name, callback)


class OperationPerception:
    """复用 AffordToolAct 的固定柜体/按操作刷新边界，不改写关闭参考。

    对象首次实测后，仅交互事件排队的对象重新测量；机器人状态仍每段实测。
    门操作初次进入时及实际门运动、
    post-contact 后刷新把手；普通接近、开爪和执行前复核不重复跟踪把手。
    被复用对象和关节保留原时间戳，不宣称为当前帧测量。
    """
    def __init__(self, setup, *, include_knob=False, sleep=None):
        self.reader = LiveArticulationReader(setup, include_knob=include_knob)
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
        self.operation = operation
        self.reader.closed_contact_pending = False

    def before_interaction(self, stage, kind):
        """通过发令前校验后排队；SDK 返回（包括部分失败）后才由观测消费。"""
        operation = self.operation
        name = stage.rsplit("/", 1)[-1].split(":")[-1]
        # 门 phase_id 形如 push_open/02_push_0；保留阶段真实名称。
        prefix, sep, remainder = name.partition("_")
        if sep and prefix.isdigit():
            name = remainder
        if operation in {"pick_bowl", "pick"}:
            changed = kind in {"grasp", "release"} or (
                kind == "motion" and name in {"approach", "lift"})
            if changed:
                self.pending.add(BOWL)
        elif operation in {"extract_bowl", "extract", "unload_bowl"}:
            if kind in {"grasp", "release"} or (
                    kind == "motion" and (name == "grasp" or name.startswith("post_extract"))):
                self.pending.add(BOWL)
        elif operation in {"insert_bowl", "insert", "place_bowl_on_table", "table", "load_bowl"}:
            if kind in {"grasp", "release", "release_settled"} or (
                    kind == "motion" and name in {"held_turn", "preplace", "insert", "place"}):
                self.pending.add(BOWL)
        elif operation in {"open_door", "close_door", "door"}:
            # 只有拉门前闭爪接触可建立闭门平移约束，普通观测/推拉中不重置参考。
            if kind == "grasp" and name == "close":
                self.reader.closed_contact_pending = True
            if kind in {"grasp", "release"} or (
                    kind == "motion" and name.startswith(("pull_", "swing_", "push_"))):
                self.pending.add(MICROWAVE)
        elif operation in {"rotate_knob", "knob"} and self.include_knob:
            if kind in {"grasp", "release", "last_joint"} or (
                    kind == "motion" and (name == "grasp" or name.startswith("rotate_"))):
                self.pending.add(KNOB)

    def after_door_phase(self, phase):
        # post-contact 包含卸载后门继续运动的稳定期。完成/规划失败本身不是交互。
        if phase.startswith(("pull_", "swing_", "push_")) or "post_contact" in phase:
            self.pending.add(MICROWAVE)
        if "post_contact" in phase:
            self.pending_settle_s = 2.0

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

    def consume(self, observed):
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
        retained = set(poses) - measured
        refresh = {
            "policy": POLICY, "operation": self.operation, "refresh_mode": "interaction_events",
            "pose_frame_disambiguation": pose_correction,
            "settle_wait_s": self.last_settle_s,
            "refreshed_object_ids": sorted(measured),
            "retained_object_ids": sorted(retained),
            "object_observations": records,
            "retained_angle_keys": sorted(set(self.angles) - refreshed_angles),
            "retained_cabinet_anchor": MICROWAVE not in measured,
            "assumption": "柜体除闭门接触有界平移外保持固定；对象在实际交互事件之间沿用状态；外部移动须重新启动观测",
        }
        result = replace(observed, object_poses_sdk=poses,
                         sources={**{k: v["source"] for k, v in records.items()}, **observed.sources},
                         perception_refresh=refresh)
        self.poses = {k: np.array(v, copy=True) for k, v in poses.items()}
        self.records = records
        self.initialized.update(measured)
        self.pending.difference_update(measured)
        self.last_frame, self.last_stamp = observed.frame_number, observed.timestamp_ms
        return result, deepcopy(self.angles), deepcopy(self.anchor)
