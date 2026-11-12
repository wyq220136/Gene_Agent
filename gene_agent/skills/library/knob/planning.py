"""旋钮 Skill 的纯阶段目标，保留旧版同步 Viser 的动作语义。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gene_agent.knowledge.operation import AxisSpec, generate_rotation_path
from gene_agent.motion.model import MotionOperation


_PREGRASP_ROTATIONS_DEG = (0.0, 5.0, -5.0, 10.0, -10.0)
_RETREAT_RETRY_SCALES = (0.9, 0.75, 0.5)


@dataclass(frozen=True)
class KnobStageTarget:
    """单个阶段的目标与执行策略；纯 Skill 不直接更改场景。"""

    name: str
    operation: MotionOperation | None
    tcp_pose_world: np.ndarray | None
    start_angle_rad: float
    end_angle_rad: float
    gripper_width_m: float
    # operation=None 还可表示等待或末关节转动；执行器须用 action 分流。
    action: str = "gripper_command"
    collision_scope: str = "none"
    gripper_is_grasping: bool = False
    target_collision_enabled: bool = False
    attach_object: bool = False
    joint_delta_rad: float | None = None
    visual_angle_commit: bool = False
    settle_time_s: float = 0.0
    pregrasp_candidates: tuple[np.ndarray, ...] = ()
    retry_non_terminal_scales: tuple[float, ...] = ()
    non_linear_fallback: bool = False


@dataclass(frozen=True)
class KnobSkillSettings:
    approach_distance_m: float
    target_angle_rad: float
    rotation_step_rad: float
    open_width_m: float
    # grasp 候选的几何闭合宽度；旧 Viser 的 grasp() 实际闭爪状态是 0。
    close_width_m: float
    release_width_m: float
    tip_rotations_x_rad: tuple[float, ...] = (0.0,)
    retreat_distance_m: float = 0.03
    grasp_settle_time_s: float = 1.5
    release_settle_time_s: float = 1.5
    retreat_retry_non_terminal_scales: tuple[float, ...] = _RETREAT_RETRY_SCALES

    def __post_init__(self) -> None:
        values = (
            self.approach_distance_m, self.target_angle_rad, self.rotation_step_rad,
            self.open_width_m, self.close_width_m, self.release_width_m,
            self.retreat_distance_m, self.grasp_settle_time_s, self.release_settle_time_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("旋钮 Skill 参数必须是有限数值")
        if self.approach_distance_m <= 0 or self.rotation_step_rad <= 0:
            raise ValueError("接近距离和旋转步长必须为正")
        if self.retreat_distance_m <= 0:
            raise ValueError("撤离距离必须为正")
        if self.open_width_m <= 0 or self.release_width_m <= 0 or self.close_width_m < 0:
            raise ValueError("夹爪开口必须为合法非负值")
        if self.grasp_settle_time_s < 0 or self.release_settle_time_s < 0:
            raise ValueError("夹爪等待时间不能为负")
        tips = tuple(float(value) for value in self.tip_rotations_x_rad)
        if not tips or any(not math.isfinite(value) for value in tips):
            raise ValueError("有效 grasp profile 的局部 X 倾角不能为空且必须有限")
        object.__setattr__(self, "tip_rotations_x_rad", tips)
        retries = tuple(float(value) for value in self.retreat_retry_non_terminal_scales)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in retries):
            raise ValueError("撤离重试比例必须位于 [0, 1]")
        object.__setattr__(self, "retreat_retry_non_terminal_scales", retries)

    @property
    def use_last_joint_rotation(self) -> bool:
        """有效 profile 的全部局部 X 倾角为零时，只转一次末关节。"""
        return bool(np.all(np.isclose(self.tip_rotations_x_rad, 0.0)))


class RotateKnobSkill:
    """生成预抓取、接触、绕轴旋转与释放撤离的有序目标。"""

    @staticmethod
    def pregrasp_candidates(
        grasp_pose_world: np.ndarray, *, distance_m: float = 0.05,
    ) -> tuple[np.ndarray, ...]:
        """按 0/±5/±10°局部 X 排列，各自沿 TCP 局部 -Z 后退。"""
        grasp = np.asarray(grasp_pose_world, dtype=float)
        distance = float(distance_m)
        if grasp.shape != (4, 4) or not np.all(np.isfinite(grasp)):
            raise ValueError("grasp_pose_world 必须是有限的 4×4 矩阵")
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("pregrasp 距离必须为正的有限数")
        poses = []
        for degrees in _PREGRASP_ROTATIONS_DEG:
            angle = math.radians(degrees)
            c, s = math.cos(angle), math.sin(angle)
            rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
            pose = grasp.copy()
            pose[:3, :3] = grasp[:3, :3] @ rotation_x
            pose[:3, 3] = grasp[:3, 3] - pose[:3, 2] * distance
            poses.append(pose)
        return tuple(poses)

    def targets(
        self, grasp_pose_world: np.ndarray, axis: AxisSpec, *,
        start_angle_rad: float, settings: KnobSkillSettings,
        selected_pregrasp_pose_world: np.ndarray | None = None,
    ) -> tuple[KnobStageTarget, ...]:
        if axis.frame != "world":
            raise ValueError("Skill 阶段目标要求已解析到 world 的旋钮轴")
        path = generate_rotation_path(
            "knob", grasp_pose_world, axis,
            start_angle_rad=start_angle_rad,
            target_angle_rad=settings.target_angle_rad,
            step_rad=settings.rotation_step_rad,
        )
        grasp = path.waypoints[0].tcp_pose_world
        candidates = self.pregrasp_candidates(grasp, distance_m=settings.approach_distance_m)
        pregrasp = (
            candidates[0] if selected_pregrasp_pose_world is None
            else np.asarray(selected_pregrasp_pose_world, dtype=float)
        )
        if not any(np.allclose(pregrasp, candidate, atol=1e-9, rtol=0.0)
                   for candidate in candidates):
            raise ValueError("选中的 pregrasp 必须来自旧版五个局部 X 候选")
        last_joint = settings.use_last_joint_rotation
        end = path.waypoints[-1].tcp_pose_world
        retreat = pregrasp.copy()
        if last_joint:
            # 末关节转动后保持新的 TCP 姿态；另一分支保留原 pregrasp 姿态。
            retreat[:3, :3] = end[:3, :3]
        retreat[:3, 3] = grasp[:3, 3] - retreat[:3, 2] * settings.retreat_distance_m
        stages = [
            KnobStageTarget("open", None, None, start_angle_rad, start_angle_rad,
                            settings.open_width_m, action="gripper_open"),
            KnobStageTarget("pregrasp", MotionOperation.PLAN, pregrasp,
                            start_angle_rad, start_angle_rad, settings.open_width_m,
                            action="move_to_pose", collision_scope="full_scene",
                            pregrasp_candidates=candidates),
            KnobStageTarget("approach", MotionOperation.LINEAR_TCP, grasp,
                            start_angle_rad, start_angle_rad, settings.open_width_m,
                            action="linear_tcp", collision_scope="self_only"),
            KnobStageTarget("close", None, None, start_angle_rad, start_angle_rad,
                            0.0, action="gripper_grasp", gripper_is_grasping=True),
            KnobStageTarget("grasp_settle", None, None, start_angle_rad,
                            start_angle_rad, 0.0, action="wait",
                            gripper_is_grasping=True,
                            settle_time_s=settings.grasp_settle_time_s),
        ]
        if last_joint:
            stages.append(KnobStageTarget(
                f"rotate_{round(math.degrees(settings.target_angle_rad - start_angle_rad)):02d}",
                None, end, start_angle_rad, settings.target_angle_rad, 0.0,
                action="last_joint_rotation", gripper_is_grasping=True,
                joint_delta_rad=settings.target_angle_rad - start_angle_rad,
                visual_angle_commit=True,
            ))
        else:
            for previous, current in zip(path.waypoints, path.waypoints[1:]):
                degrees = round(math.degrees(current.angle_rad - start_angle_rad))
                stages.append(KnobStageTarget(
                    f"rotate_{degrees:02d}", MotionOperation.LINEAR_TCP,
                    current.tcp_pose_world, previous.angle_rad, current.angle_rad,
                    0.0, action="linear_tcp", collision_scope="self_only",
                    gripper_is_grasping=True,
                    visual_angle_commit=current is path.waypoints[-1],
                ))
        stages.extend((
            KnobStageTarget("release", None, None, settings.target_angle_rad,
                            settings.target_angle_rad, settings.release_width_m,
                            action="gripper_release"),
            KnobStageTarget("release_settle", None, None, settings.target_angle_rad,
                            settings.target_angle_rad, settings.release_width_m,
                            action="wait", settle_time_s=settings.release_settle_time_s),
            KnobStageTarget("retreat", MotionOperation.LINEAR_TCP, retreat,
                            settings.target_angle_rad, settings.target_angle_rad,
                            settings.release_width_m, action="linear_tcp",
                            collision_scope="full_scene",
                            retry_non_terminal_scales=settings.retreat_retry_non_terminal_scales,
                            non_linear_fallback=True),
        ))
        return tuple(stages)


__all__ = ["KnobSkillSettings", "KnobStageTarget", "RotateKnobSkill"]
