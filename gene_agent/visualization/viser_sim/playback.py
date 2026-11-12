"""同步视觉模拟的状态时间轴；Viser 和录制器使用同一份结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping
from types import MappingProxyType

import numpy as np

from gene_agent.knowledge.geometry import validate_pose

from .model import VisualRun, VisualStage


_EVENT_DURATION_S = .001


@dataclass(frozen=True, eq=False)
class VisualState:
    """一个视觉时刻的完整操作状态；构造时复制数组以隔离外部修改。"""

    joint_names: tuple[str, ...]
    joint_positions_rad: np.ndarray
    gripper_width_m: float
    object_poses_world: dict[str, np.ndarray] = field(default_factory=dict)
    door_angle_rad: float | None = None
    attachments: tuple[str, ...] = ()
    object_joint_positions_rad: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        positions = np.asarray(self.joint_positions_rad, dtype=float).copy()
        if (not names or len(set(names)) != len(names) or positions.shape != (len(names),)
                or not np.isfinite(positions).all()):
            raise ValueError("初始机器人状态需要具名、有限且不重复的关节")
        width = float(self.gripper_width_m)
        if not math.isfinite(width) or width < 0:
            raise ValueError("夹爪开口必须是非负有限米数")
        poses = {str(name): validate_pose(pose, field_name=f"object_poses_world.{name}").copy()
                 for name, pose in self.object_poses_world.items()}
        attached = tuple(sorted(set(self.attachments)))
        if any(name not in poses for name in attached):
            raise ValueError("初始附着物体缺少世界位姿")
        if self.door_angle_rad is not None and not math.isfinite(self.door_angle_rad):
            raise ValueError("门角必须是有限弧度")
        articulation = {str(name): float(value) for name, value in self.object_joint_positions_rad.items()}
        if any(not math.isfinite(value) for value in articulation.values()):
            raise ValueError("场景关节状态必须为有限弧度")
        object.__setattr__(self, "object_joint_positions_rad", MappingProxyType(articulation))
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_positions_rad", positions)
        object.__setattr__(self, "gripper_width_m", width)
        object.__setattr__(self, "object_poses_world", poses)
        object.__setattr__(self, "attachments", attached)


@dataclass(frozen=True)
class VisualSample:
    time_s: float
    stage_id: str
    state: VisualState


class VisualPlayback:
    """按旧同步 Viser 的事件顺序，从初态重建任意时间的视觉状态。

    运动阶段只采样 planner 保存的关节轨迹。夹爪、attach 和门角等状态事件
    不生成机器人轨迹；附着物体在每个采样点使用 TCP 的正运动学跟随。
    每次 seek 都从初态重算，来回拖动滑块不会沿用上一次帧的物体位置。
    """

    def __init__(self, run: VisualRun, initial: VisualState,
                 tcp_pose: Callable[[tuple[str, ...], np.ndarray], np.ndarray]):
        self.run = run
        self.initial = initial
        self.tcp_pose = tcp_pose
        self._starts: dict[str, float] = {}
        self._durations: list[float] = []
        elapsed = 0.
        first_motion = True
        for stage in run.stages:
            if stage.stage_id in self._starts:
                raise ValueError(f"重复阶段 ID：{stage.stage_id}")
            if stage.trajectory is not None and stage.trajectory.joint_names != initial.joint_names:
                raise ValueError(f"{stage.stage_id} 的轨迹关节顺序与初态不同")
            if stage.trajectory is not None and first_motion:
                gap = float(np.max(np.abs(
                    stage.trajectory.positions_rad[0] - initial.joint_positions_rad
                )))
                if gap > 1e-3:
                    raise ValueError(
                        f"首段轨迹与场景初始关节状态不连续：{stage.stage_id}，"
                        f"最大跳变 {gap:.6f} rad"
                    )
                first_motion = False
            self._starts[stage.stage_id] = elapsed
            duration = (stage.trajectory.duration_s if stage.trajectory is not None
                        else stage.state_duration_s if stage.state_duration_s is not None else _EVENT_DURATION_S)
            self._durations.append(duration)
            elapsed += duration
        self.duration_s = elapsed

    def stage_start_s(self, stage_id: str) -> float:
        return self._starts[stage_id]

    def _tcp(self, joints: np.ndarray) -> np.ndarray:
        return validate_pose(self.tcp_pose(self.initial.joint_names, joints),
                             field_name="TCP 正运动学")

    def sample(self, time_s: float) -> VisualSample:
        requested = float(time_s)
        if not math.isfinite(requested):
            raise ValueError("时间必须是有限秒数")
        time_s = float(np.clip(requested, 0., self.duration_s))
        joints = self.initial.joint_positions_rad.copy()
        width = self.initial.gripper_width_m
        poses = {name: pose.copy() for name, pose in self.initial.object_poses_world.items()}
        door_angle = self.initial.door_angle_rad
        articulation = dict(self.initial.object_joint_positions_rad)
        tcp_from_object = {
            name: np.linalg.inv(self._tcp(joints)) @ poses[name]
            for name in self.initial.attachments
        }

        def follow_attached() -> None:
            if tcp_from_object:
                world_tcp = self._tcp(joints)
                for name, local in tcp_from_object.items():
                    poses[name] = world_tcp @ local

        def finish(stage: VisualStage) -> None:
            nonlocal width, door_angle
            if stage.object_pose_world is not None:
                assert stage.object_instance_id is not None
                poses[stage.object_instance_id] = stage.object_pose_world.copy()
            if stage.gripper_width_m is not None:
                width = stage.gripper_width_m
            elif stage.motion_kind == "gripper":
                raise ValueError(f"{stage.stage_id} 缺少夹爪开口数值")
            if stage.door_angle_rad is not None:
                door_angle = stage.door_angle_rad
            # articulation 是已有规划记录中的离散状态提交；不会臆造转动过程。
            articulation.update(stage.object_joint_positions_rad)
            after = set(stage.attach_after)
            for name in tuple(tcp_from_object):
                if name not in after:
                    del tcp_from_object[name]
            for name in after:
                if name not in tcp_from_object:
                    if name not in poses:
                        raise ValueError(f"{stage.stage_id} 附着未知物体：{name}")
                    tcp_from_object[name] = np.linalg.inv(self._tcp(joints)) @ poses[name]
            if stage.object_instance_id in tcp_from_object:
                tcp_from_object[stage.object_instance_id] = (
                    np.linalg.inv(self._tcp(joints)) @ poses[stage.object_instance_id]
                )

        def result(stage: VisualStage) -> VisualSample:
            state = VisualState(self.initial.joint_names, joints, width, poses,
                                door_angle, tuple(tcp_from_object), articulation)
            return VisualSample(time_s, stage.stage_id, state)

        for stage, duration in zip(self.run.stages, self._durations):
            start = self._starts[stage.stage_id]
            if set(stage.attach_before) != set(tcp_from_object):
                raise ValueError(f"{stage.stage_id} 起点 attachment 与上一阶段终点不一致")
            if not stage.success:
                # 失败阶段只有目标和诊断，没有到达目标的轨迹。画面必须停在
                # 最后一个成功状态，不能把门角或物体瞬移到失败的目标。
                return result(stage)
            trajectory = stage.trajectory
            if trajectory is not None:
                local = float(np.clip(time_s - start, 0., duration))
                coordinate = local / trajectory.sample_dt_s
                left = min(int(np.floor(coordinate)), len(trajectory.positions_rad) - 1)
                right = min(left + 1, len(trajectory.positions_rad) - 1)
                fraction = coordinate - left if right > left else 0.
                joints[:] = ((1. - fraction) * trajectory.positions_rad[left]
                             + fraction * trajectory.positions_rad[right])
                # 旧同步 Viser 在每个关节采样中一起更新门角与夹爪宽度。
                # 这里必须读取 planner 阶段保存的通道，不能等到目标点才跳变。
                if "door_angle_rad" in trajectory.channels:
                    values = trajectory.channels["door_angle_rad"]
                    door_angle = float((1. - fraction) * values[left] + fraction * values[right])
                if "gripper_width_m" in trajectory.channels:
                    values = trajectory.channels["gripper_width_m"]
                    width = float((1. - fraction) * values[left] + fraction * values[right])
                follow_attached()
                if time_s < start + duration:
                    return result(stage)
                finish(stage)
            else:
                # 旧门会话把开合夹爪作为有持续时间的状态段保存；仅关节不动，
                # 开口仍按原阶段的两个端点连续变化。
                if stage.state_duration_s is not None and stage.motion_kind == "gripper":
                    fraction = float(np.clip((time_s - start) / duration, 0., 1.))
                    if stage.gripper_width_m is not None:
                        width = ((1. - fraction) * width
                                 + fraction * stage.gripper_width_m)
                    if stage.door_angle_rad is not None and door_angle is not None:
                        door_angle = ((1. - fraction) * door_angle
                                      + fraction * stage.door_angle_rad)
                    if time_s >= start + duration:
                        finish(stage)
                elif not stage.state_update_at_end or time_s >= start + duration:
                    finish(stage)
                if time_s < start + duration:
                    return result(stage)
        return result(self.run.stages[-1])


__all__ = ["VisualState", "VisualSample", "VisualPlayback"]
