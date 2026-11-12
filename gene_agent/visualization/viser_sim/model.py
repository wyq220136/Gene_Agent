"""在线 Skill/Task 视觉调试器共用的最小数据模型。

这里不生成目标 pose，也不替 planner 补轨迹。Knowledge、Skill 与
motion planner 的结果先冻结为阶段记录，再交给 Viser 或视频录制器。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.visualization.skill_inspector.planned_timeline import (
    PlannedSegment, PlannedTimeline,
)


_MOTION_KINDS = frozenset({"plan", "linear_tcp", "joint"})
_STATE_KINDS = frozenset({"gripper", "attachment", "door_state", "object_state"})


def _pose(value: np.ndarray, *, name: str) -> np.ndarray:
    frozen = validate_pose(value, field_name=name).copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, eq=False)
class SceneEdits:
    """待应用到同一场景快照的物体、关节和桌面尺寸修改。"""

    object_poses_world: Mapping[str, np.ndarray] = field(default_factory=dict)
    object_joint_positions_rad: Mapping[str, float] = field(default_factory=dict)
    robot_joint_positions_rad: Mapping[str, float] = field(default_factory=dict)
    table_size_xy_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        poses = {
            str(name): _pose(value, name=f"object_poses_world.{name}")
            for name, value in self.object_poses_world.items()
        }
        object.__setattr__(self, "object_poses_world", MappingProxyType(poses))
        for field_name in ("object_joint_positions_rad", "robot_joint_positions_rad"):
            values = {
                str(name): float(value)
                for name, value in getattr(self, field_name).items()
            }
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError(f"{field_name} 必须全部为有限角度")
            object.__setattr__(self, field_name, MappingProxyType(values))
        if self.table_size_xy_m is not None:
            size = tuple(float(value) for value in self.table_size_xy_m)
            if len(size) != 2 or any(not math.isfinite(value) or value <= 0 for value in size):
                raise ValueError("桌面长宽必须是两个正的有限米数")
            object.__setattr__(self, "table_size_xy_m", size)


@dataclass(frozen=True, eq=False)
class VisualStage:
    """一个可单独查看的阶段；成功运动阶段必须携带 planner 轨迹。"""

    stage_id: str
    skill_id: str
    motion_kind: str
    success: bool
    trajectory: PlannedSegment | None = None
    target_tcp_world: np.ndarray | None = None
    object_instance_id: str | None = None
    object_pose_world: np.ndarray | None = None
    collision_enabled: tuple[str, ...] = ()
    collision_allowed: tuple[str, ...] = ()
    collision_snapshot_ref: str | None = None
    attach_before: tuple[str, ...] = ()
    attach_after: tuple[str, ...] = ()
    gripper_command: str | None = None
    gripper_width_m: float | None = None
    state_duration_s: float | None = None
    door_angle_rad: float | None = None
    planning_duration_s: float | None = None
    failure: dict | None = None
    diagnostics: dict = field(default_factory=dict)
    object_joint_positions_rad: Mapping[str, float] = field(default_factory=dict)
    state_update_at_end: bool = False

    def __post_init__(self) -> None:
        if not self.stage_id or not self.skill_id:
            raise ValueError("阶段和 Skill 名称不能为空")
        if self.motion_kind not in _MOTION_KINDS | _STATE_KINDS:
            raise ValueError(f"未知运动或状态类型：{self.motion_kind}")
        if self.motion_kind in _MOTION_KINDS and self.success and self.trajectory is None:
            raise ValueError("成功的运动阶段必须携带 planner 规划轨迹")
        if not self.success and self.trajectory is not None:
            raise ValueError("失败阶段不能播放规划轨迹")
        if not self.success and (not isinstance(self.failure, dict) or not self.failure):
            raise ValueError("失败阶段必须保留结构化失败诊断")
        if self.motion_kind in _STATE_KINDS and self.trajectory is not None:
            raise ValueError("状态事件不应携带伪造的关节轨迹")
        if self.trajectory is not None and (
            self.trajectory.stage_id != self.stage_id
            or self.trajectory.skill_id != self.skill_id
        ):
            raise ValueError("阶段标签与实际轨迹标签不一致")
        for name in ("target_tcp_world", "object_pose_world"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _pose(value, name=name))
        if self.object_pose_world is not None and not self.object_instance_id:
            # 旧阶段记录只有一个 attach 对象时可以无损补齐；多物体场景
            # 必须显式指明实例，不能靠 skill 名称猜物体。
            candidates = set(self.attach_before) | set(self.attach_after)
            if len(candidates) != 1:
                raise ValueError("物体目标位姿必须指明对象实例")
            object.__setattr__(self, "object_instance_id", candidates.pop())
        if self.gripper_width_m is not None:
            width = float(self.gripper_width_m)
            if not math.isfinite(width) or width < 0:
                raise ValueError("夹爪开口必须是非负有限米数")
            object.__setattr__(self, "gripper_width_m", width)
        if self.state_duration_s is not None:
            duration = float(self.state_duration_s)
            if self.motion_kind not in _STATE_KINDS or not math.isfinite(duration) or duration < 0:
                raise ValueError("状态阶段时长只适用于状态事件，且必须为非负有限秒数")
            object.__setattr__(self, "state_duration_s", duration)
        for name in ("collision_enabled", "collision_allowed", "attach_before", "attach_after"):
            values = tuple(dict.fromkeys(str(value) for value in getattr(self, name)))
            object.__setattr__(self, name, values)
        if self.door_angle_rad is not None:
            angle = float(self.door_angle_rad)
            if not math.isfinite(angle):
                raise ValueError("门角必须为有限弧度")
            object.__setattr__(self, "door_angle_rad", angle)
        if self.planning_duration_s is not None:
            duration = float(self.planning_duration_s)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("规划耗时必须为非负有限秒数")
            object.__setattr__(self, "planning_duration_s", duration)
        values = {str(name): float(value) for name, value in self.object_joint_positions_rad.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("场景关节状态必须为有限弧度")
        object.__setattr__(self, "object_joint_positions_rad", MappingProxyType(values))
        object.__setattr__(self, "failure", deepcopy(self.failure))
        object.__setattr__(self, "diagnostics", deepcopy(self.diagnostics))

    @property
    def attach_event(self) -> str | None:
        removed = set(self.attach_before) - set(self.attach_after)
        added = set(self.attach_after) - set(self.attach_before)
        events = [*(f"detach:{name}" for name in sorted(removed)),
                  *(f"attach:{name}" for name in sorted(added))]
        return ",".join(events) if events else None

    def manifest_row(self) -> dict:
        """输出与旧版阶段清单比较器共用的字段。"""
        return {
            "stage_id": self.stage_id,
            "skill_id": self.skill_id,
            "motion_kind": self.motion_kind,
            "success": self.success,
            "target_tcp_world": None if self.target_tcp_world is None else self.target_tcp_world.tolist(),
            "object_pose_world": None if self.object_pose_world is None else self.object_pose_world.tolist(),
            "object_instance_id": self.object_instance_id,
            "collision_enabled": list(self.collision_enabled),
            "collision_allowed": list(self.collision_allowed),
            "collision_snapshot_ref": self.collision_snapshot_ref,
            "attach_event": self.attach_event,
            "gripper_command": self.gripper_command,
            "gripper_width_m": self.gripper_width_m,
            "state_duration_s": self.state_duration_s,
            "state_update_at_end": self.state_update_at_end,
            "door_angle_rad": self.door_angle_rad,
            "object_joint_positions_rad": dict(self.object_joint_positions_rad),
            "planning_duration_s": self.planning_duration_s,
            "failure": deepcopy(self.failure),
            "diagnostics": deepcopy(self.diagnostics),
        }


@dataclass(frozen=True, eq=False)
class VisualRun:
    """一次场景 revision 的完整运行结果，保留失败前缀但不编造后续动作。"""

    selection_id: str
    scene_revision: int
    stages: tuple[VisualStage, ...]
    _timeline: PlannedTimeline | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.selection_id or self.scene_revision < 0:
            raise ValueError("选择名称和场景 revision 必须有效")
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("视觉运行至少需要一个阶段")
        if any(not stage.success for stage in stages[:-1]):
            raise ValueError("失败阶段之后不能继续声明已成功的阶段")
        segments = tuple(stage.trajectory for stage in stages if stage.trajectory is not None)
        object.__setattr__(self, "stages", stages)
        # PlannedTimeline 会验证相邻真实轨迹是否在同一个关节状态衔接。
        object.__setattr__(self, "_timeline", PlannedTimeline(segments) if segments else None)

    @property
    def success(self) -> bool:
        return all(stage.success for stage in self.stages)

    @property
    def timeline(self) -> PlannedTimeline | None:
        return self._timeline

    def manifest(self) -> dict:
        return {
            "selection_id": self.selection_id,
            "scene_revision": self.scene_revision,
            "success": self.success,
            "stages": [stage.manifest_row() for stage in self.stages],
        }


__all__ = ["SceneEdits", "VisualStage", "VisualRun"]
