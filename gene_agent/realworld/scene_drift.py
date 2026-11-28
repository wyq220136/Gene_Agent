"""发真机命令前对照重新观测的规划条件。

同一物体的 FoundationPose 结果会有毫米级抖动，因此不能直接要求
两个碰撞快照 hash 相等。这里先要求资产、碰撞名单和来源一致，再用
调用方显式给出的误差界比较两臂、夹爪、门、旋钮和物体 pose。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.runtime import RealWorldRuntime
from gene_agent.tasks.checks import CheckResult


@dataclass(frozen=True)
class SceneDriftLimits:
    max_object_translation_m: float
    max_object_rotation_rad: float
    max_robot_joint_error_rad: float
    max_gripper_width_error_m: float
    max_articulated_angle_error_rad: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数")
            object.__setattr__(self, name, value)


def _pose_error(first, second, *, name: str) -> tuple[float, float]:
    left = validate_pose(first, field_name=f"{name} planned")
    right = validate_pose(second, field_name=f"{name} current")
    translation = float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        left[:3, :3].T @ right[:3, :3],
    ).as_rotvec()))
    return translation, rotation


def _collision_roster(scene: dict) -> tuple:
    fields = ("geom_name", "body_name", "mujoco_type", "contype", "conaffinity",
              "export_kind", "mesh_sha256", "contact_margin_m", "approximation")
    rows = []
    for record in scene["snapshot"]["records"]:
        row = tuple(record.get(name) for name in fields)
        rows.append((*row, tuple(record.get("geom_size", ()))))
    return tuple(sorted(rows, key=str))


def compare_scene_drift(
    planned: dict, current: dict, *, limits: SceneDriftLimits,
) -> CheckResult:
    """返回新的现场帧是否仍满足原规划条件；任一未知结构直接报错。"""
    if not isinstance(limits, SceneDriftLimits):
        raise TypeError("limits 必须是 SceneDriftLimits")
    before, after = planned["provenance"], current["provenance"]
    old_snapshot, new_snapshot = planned["snapshot"], current["snapshot"]
    for label, scene in (("原规划", planned), ("新观测", current)):
        source = scene["provenance"]
        if (source["physics_stepped"] is not False
                or source["pose_frame_verified"] is not True
                or source["collision_snapshot_sha256"] != scene["snapshot"]["snapshot_sha256"]):
            raise ValueError(f"{label}不是来源一致的真机规划快照")
    failures = []
    if int(after["source_frame_number"]) <= int(before["source_frame_number"]):
        failures.append("没有获得新相机帧")
    if float(after["source_timestamp_ms"]) < float(before["source_timestamp_ms"]):
        failures.append("新观测时间早于规划帧")
    for key, label in (("source_camera_id", "相机"), ("source_side", "相机 SDK base"),
                       ("scene_config_sha256", "场景配置"),
                       ("alignment_input_sha256", "坐标与 mesh 标定")):
        if before.get(key) != after.get(key):
            failures.append(f"{label}已变化")
    # “复用”必须指向同一份旧证据，不能只保持标签而替换锚点来源/时间。
    refresh = after.get("perception_refresh") or {}
    if refresh.get("retained_cabinet_anchor") is True:
        old_anchor = before.get("microwave_root_anchor")
        new_anchor = after.get("microwave_root_anchor")
        if not old_anchor or old_anchor != new_anchor:
            failures.append("复用的微波炉固定锚点或来源已变化")
    old_records = (before.get("perception_refresh") or {}).get("object_observations", {})
    new_records = refresh.get("object_observations", {})
    old_modes, new_modes = before.get("pose_source_modes", {}), after.get("pose_source_modes", {})
    for name in set(old_modes) | set(new_modes):
        if old_modes.get(name) == new_modes.get(name):
            continue
        # 首帧实测转成沿用同一证据是正常的事件边界；其他来源切换仍拒绝。
        retained_transition = (
            refresh.get("refresh_mode") == "interaction_events"
            and name in refresh.get("retained_object_ids", ())
            and old_modes.get(name) == "same_frame"
            and new_modes.get(name) == "retained_operation_state"
            and old_records.get(name) and old_records[name] == new_records.get(name)
        )
        estimated_transition = (
            refresh.get("policy") == "affordtoolact_operation_scoped"
            and name in refresh.get("estimated_object_ids", ())
            and new_modes.get(name) == "kinematic_state_estimate"
            and old_records.get(name) == new_records.get(name)
            and refresh.get("state_estimates", {}).get(name, {}).get("sdk_tcp_pose") is not None)
        if not retained_transition and not estimated_transition:
            failures.append(f"{name} 物体位姿来源已变化")
    for name in refresh.get("retained_object_ids", ()):
        if not old_records.get(name) or old_records.get(name) != new_records.get(name):
            failures.append(f"{name} 复用的原始观测来源已变化")
        old_pose = before["aligned_object_poses_world"].get(name)
        new_pose = after["aligned_object_poses_world"].get(name)
        if old_pose is None or new_pose is None or not np.array_equal(old_pose, new_pose):
            failures.append(f"{name} 复用的物体 pose 已变化")
    for key, label in (("active_group", "活动臂"), ("world_frame", "世界坐标系")):
        if old_snapshot[key] != new_snapshot[key]:
            failures.append(f"{label}已变化")
    if _collision_roster(planned) != _collision_roster(current):
        failures.append("碰撞资产或规划障碍名单已变化")
    for key in ("table_size_xy_m", "accepted_table_thickness_m",
                "accepted_support_size_xyz_m", "accepted_support_center_xy_local_m"):
        old = np.asarray(before[key], dtype=float)
        new = np.asarray(after[key], dtype=float)
        if old.shape != new.shape or not np.allclose(old, new, rtol=0., atol=1e-9):
            failures.append(f"{key} 规划代理尺寸已变化")

    old_objects = before["aligned_object_poses_world"]
    new_objects = after["aligned_object_poses_world"]
    if set(old_objects) != set(new_objects):
        failures.append("现场物体名单已变化")
    object_translation = {}
    object_rotation = {}
    for name in sorted(set(old_objects) & set(new_objects)):
        translation, rotation = _pose_error(old_objects[name], new_objects[name], name=name)
        object_translation[name] = translation
        object_rotation[name] = rotation
        if (translation > limits.max_object_translation_m
                or rotation > limits.max_object_rotation_rad):
            failures.append(f"{name} pose 已偏离规划值")
    table_translation, table_rotation = _pose_error(
        before["table_pose_world"], after["table_pose_world"], name="table",
    )
    if (table_translation > limits.max_object_translation_m
            or table_rotation > limits.max_object_rotation_rad):
        failures.append("桌面 pose 已变化")

    joints_before = before["joint_positions_by_group_rad"]
    joints_after = after["joint_positions_by_group_rad"]
    widths_before = before["gripper_widths_m"]
    widths_after = after["gripper_widths_m"]
    if set(joints_before) != set(joints_after) or set(widths_before) != set(widths_after):
        failures.append("机器人或夹爪组名单已变化")
    joint_errors = {}
    for group in sorted(set(joints_before) & set(joints_after)):
        old = np.asarray(joints_before[group], dtype=float)
        new = np.asarray(joints_after[group], dtype=float)
        if old.shape != new.shape or old.ndim != 1 or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError(f"{group} 新旧关节读数无效")
        error = float(np.max(np.abs(new - old)))
        joint_errors[group] = error
        if error > limits.max_robot_joint_error_rad:
            failures.append(f"{group} 臂关节已偏离规划状态")
    for group in sorted(set(widths_before) & set(widths_after)):
        if not all(math.isfinite(float(width)) for width in (widths_before[group], widths_after[group])):
            raise ValueError(f"{group} 夹爪读数无效")
    articulated_errors = {}
    for name in ("door_angle_rad", "knob_angle_rad"):
        part = "door" if name == "door_angle_rad" else "knob"
        old_unobserved = part == "knob" and before.get("live_state", {}).get(part, {}).get("observed") is False
        new_unobserved = part == "knob" and after.get("live_state", {}).get(part, {}).get("observed") is False
        if old_unobserved != new_unobserved:
            failures.append(f"{name} 感知启用范围已改变")
        if old_unobserved or new_unobserved:
            static_error = abs(float(after[name]) - float(before[name]))
            if not math.isfinite(static_error) or static_error > 1e-9:
                failures.append("未观测旋钮的静态碰撞代理已变化")
            continue  # 静态资产占位不构成观测，不能报告它通过了实测漂移检查。
        # 碰撞使用的可能是角度网格。若现场同时保留了原始实测角，
        # 必须两种角度都检查；否则同一网格内的真实门运动会被漏掉。
        grid_error = abs(float(after[name]) - float(before[name]))
        old_raw = before.get("live_state", {}).get(part, {}).get("measured_angle_rad")
        new_raw = after.get("live_state", {}).get(part, {}).get("measured_angle_rad")
        if (old_raw is None) != (new_raw is None):
            failures.append(f"{name} 原始实测角来源不一致")
        raw_error = (0. if old_raw is None or new_raw is None else
                     abs(float(new_raw) - float(old_raw)))
        error = max(grid_error, raw_error)
        if not math.isfinite(error):
            raise ValueError(f"{name} 读数无效")
        articulated_errors[name] = error
        if error > limits.max_articulated_angle_error_rad:
            failures.append(f"{name} 已偏离规划状态")
    evidence = {
        "planned_snapshot_sha256": old_snapshot["snapshot_sha256"],
        "current_snapshot_sha256": new_snapshot["snapshot_sha256"],
        "planned_frame_number": before["source_frame_number"],
        "current_frame_number": after["source_frame_number"],
        "object_translation_m": object_translation,
        "object_rotation_rad": object_rotation,
        "table_translation_m": table_translation,
        "table_rotation_rad": table_rotation,
        "joint_error_rad": joint_errors,
        "gripper_width_policy": "reuse_confirmed_endpoint",
        "articulated_angle_error_rad": articulated_errors,
    }
    return CheckResult(
        "fail" if failures else "pass", "realworld_reobserved_scene",
        evidence, "；".join(failures) if failures else "发令前新观测仍满足规划条件",
    )


def bind_live_scene_guard(
    observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], RealWorldRuntime], *, limits: SceneDriftLimits,
) -> tuple[Callable[[], dict], Callable[[str], RealWorldRuntime]]:
    """让现有同步子任务的两个注入接口共用一份规划现场。

    子任务仍按原顺序调用 ``observe_collision_scene()`` 和
    ``make_runtime(stage_hash)``。工厂只保存当前阶段的真实物理现场；
    阶段 hash 可以是允许接触/附着后的派生碰撞快照。执行前从原始观察器
    另取新帧，比较物理现场，同时 Runtime 仍核对真正规划阶段的 hash。
    这不会改变 Knowledge、运动请求或旧 Viser/MuJoCo 的行为。
    """
    if not isinstance(limits, SceneDriftLimits):
        raise TypeError("limits 必须是 SceneDriftLimits")
    last_scene: dict | None = None

    def observe(*, gripper_width_overrides=None) -> dict:
        nonlocal last_scene
        scene = (observe_collision_scene()
                 if gripper_width_overrides is None else
                 observe_collision_scene(gripper_width_overrides=gripper_width_overrides))
        if not isinstance(scene, dict):
            raise TypeError("现场观测必须返回 dict")
        # 后续规划可能派生/修改碰撞快照，基准必须固定为采集时的物理现场。
        last_scene = deepcopy(scene)
        return scene

    def make_guarded_runtime(stage_snapshot_sha256: str) -> RealWorldRuntime:
        if last_scene is None:
            raise RuntimeError("创建真机 Runtime 前必须先观测并保存本阶段现场")
        runtime = make_runtime(stage_snapshot_sha256)
        if not isinstance(runtime, RealWorldRuntime):
            raise TypeError("现场复核只能接入 RealWorldRuntime")
        if runtime.pre_execute_scene_check is not None:
            raise ValueError("Runtime 已有现场复核，不能悄悄覆盖")
        planned_scene = last_scene

        def recheck() -> CheckResult:
            current_scene = observe_collision_scene()
            result = compare_scene_drift(planned_scene, current_scene, limits=limits)
            provenance = current_scene["provenance"]
            group = runtime.robot.group
            live = provenance.get("live_state") or {}
            stamps = live.get("robot_sample_times", {}).get(group)
            sample = None if not stamps else {
                "group": group, **stamps,
                "joint_positions_rad": provenance["joint_positions_by_group_rad"][group],
                "width_m": provenance["gripper_widths_m"][group],
                "width_source": live.get("gripper_width_sources", {}).get(group)}
            return CheckResult(
                result.status, result.source,
                {**result.evidence,
                 "planned_stage_snapshot_sha256": stage_snapshot_sha256,
                 "robot_sample": sample},
                result.reason,
            )

        recheck.provides_robot_sample = True
        runtime.pre_execute_scene_check = recheck
        callback = getattr(observe_collision_scene, "before_interaction", None)
        if callback is not None:
            if runtime.before_interaction is not None and runtime.before_interaction != callback:
                raise ValueError("Runtime 已有交互事件回调，不能悄悄覆盖")
            runtime.before_interaction = callback
        completed = getattr(observe_collision_scene, "after_interaction", None)
        if completed is not None:
            runtime.after_interaction = completed
        return runtime

    # 包装后仍传递操作感知事件，避免发令前复核退回全对象感知。
    for name in ("begin_operation", "after_door_phase", "before_interaction", "after_interaction", "set_door_final_phase", "after_knob_phase", "accept_current_hold"):
        callback = getattr(observe_collision_scene, name, None)
        if callback is not None:
            setattr(observe, name, callback)
    return observe, make_guarded_runtime


__all__ = ["SceneDriftLimits", "compare_scene_drift", "bind_live_scene_guard"]
