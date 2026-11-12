"""门把手开关门 Skill 的纯阶段几何；门轴必须从任务配置显式传入。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from gene_agent.knowledge.geometry import validate_pose

import numpy as np

from gene_agent.knowledge.operation import AxisSpec, generate_rotation_path
from gene_agent.motion.model import MotionOperation


@dataclass(frozen=True)
class DoorStageTarget:
    name: str
    operation: MotionOperation | None
    tcp_pose_world: np.ndarray | None
    start_angle_rad: float
    end_angle_rad: float
    gripper_width_m: float


@dataclass(frozen=True)
class DoorSkillSettings:
    approach_distance_m: float
    target_angle_rad: float
    rotation_step_rad: float
    open_width_m: float
    close_width_m: float
    release_width_m: float

    def __post_init__(self) -> None:
        values = (self.approach_distance_m, self.target_angle_rad,
                  self.rotation_step_rad, self.open_width_m,
                  self.close_width_m, self.release_width_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("门 Skill 参数必须是有限数值")
        if self.approach_distance_m <= 0 or self.rotation_step_rad <= 0:
            raise ValueError("接近距离和绕轴步长必须为正")
        if self.open_width_m <= 0 or self.release_width_m <= 0 or self.close_width_m < 0:
            raise ValueError("夹爪开口必须为合法非负值")


class DoorSkill:
    """仅组织开门或关门的阶段目标，不耦合机器人型号与 planner。"""

    def targets(self, grasp_pose_world: np.ndarray, axis: AxisSpec,
                *, start_angle_rad: float,
                settings: DoorSkillSettings) -> tuple[DoorStageTarget, ...]:
        if axis.frame != "world":
            raise ValueError("门 Skill 要求已解析到 world 的显式门轴")
        path = generate_rotation_path(
            "door", grasp_pose_world, axis,
            start_angle_rad=start_angle_rad,
            target_angle_rad=settings.target_angle_rad,
            step_rad=settings.rotation_step_rad,
        )
        grasp = path.waypoints[0].tcp_pose_world
        pregrasp = np.array(grasp, copy=True)
        # 抓取接近段保持姿态，只沿 TCP 局部 -Z 退到净空位姿。
        pregrasp[:3, 3] -= settings.approach_distance_m * grasp[:3, 2]
        end = path.waypoints[-1].tcp_pose_world
        retreat = np.array(end, copy=True)
        retreat[:3, 3] -= settings.approach_distance_m * end[:3, 2]
        stages = [
            DoorStageTarget("open", None, None, start_angle_rad, start_angle_rad,
                            settings.open_width_m),
            DoorStageTarget("pregrasp", MotionOperation.PLAN, pregrasp,
                            start_angle_rad, start_angle_rad, settings.open_width_m),
            DoorStageTarget("approach", MotionOperation.LINEAR_TCP, grasp,
                            start_angle_rad, start_angle_rad, settings.open_width_m),
            DoorStageTarget("close", None, None, start_angle_rad, start_angle_rad,
                            settings.close_width_m),
        ]
        for previous, current in zip(path.waypoints, path.waypoints[1:]):
            degrees = round(math.degrees(abs(current.angle_rad - start_angle_rad)))
            stages.append(DoorStageTarget(
                f"swing_{degrees:02d}", MotionOperation.LINEAR_TCP,
                current.tcp_pose_world, previous.angle_rad, current.angle_rad,
                settings.close_width_m,
            ))
        stages.extend((
            DoorStageTarget("release", None, end,
                            settings.target_angle_rad, settings.target_angle_rad,
                            settings.release_width_m),
            DoorStageTarget("retreat", MotionOperation.LINEAR_TCP, retreat,
                            settings.target_angle_rad, settings.target_angle_rad,
                            settings.release_width_m),
        ))
        return tuple(stages)


__all__ = ["DoorSkill", "DoorSkillSettings", "DoorStageTarget"]


@dataclass(frozen=True)
class PushDoorStage:
    name: str
    operation: MotionOperation
    tcp_pose_world: np.ndarray
    target_angle_deg: float | None


class PushDoorSkill:
    """只表达原版 PushDoorPlan 的运动合同；碰撞许可由执行器逐阶段指定。"""

    def targets(
        self,
        precontact_pose_world: np.ndarray,
        push_poses_world: Sequence[np.ndarray],
        push_angles_deg: Sequence[float],
    ) -> tuple[PushDoorStage, ...]:
        poses = tuple(push_poses_world)
        angles = tuple(float(value) for value in push_angles_deg)
        if not poses or len(poses) != len(angles):
            raise ValueError("推门 pose 与门角数量必须相同且非空")
        if not np.isfinite(angles).all():
            raise ValueError("推门角度必须有限")
        if len(angles) > 1:
            differences = np.diff(angles)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError("推门角度必须严格单调")
        stages = [PushDoorStage(
            "precontact", MotionOperation.PLAN,
            validate_pose(precontact_pose_world, field_name="precontact_pose_world"),
            None,
        )]
        for index, (pose, angle) in enumerate(zip(poses, angles, strict=True), start=1):
            stages.append(PushDoorStage(
                f"push_{index:02d}", MotionOperation.LINEAR_TCP,
                validate_pose(pose, field_name=f"push_pose_{index}"), angle,
            ))
        return tuple(stages)


__all__ = ["PushDoorStage", "PushDoorSkill"]

__all__ = ["DoorSkill","DoorSkillSettings","DoorStageTarget","PushDoorSkill","PushDoorStage"]
