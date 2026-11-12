"""取出物体后的桌面 Place 纯规划阶段。

桌面 TCP 是旧 profile 给出的世界/SDK-base pose。调用方提供本次抓取后实际
T_tcp_object；本模块据此生成目标物体 pose，不把炉内初始 pose 当作事实。
完整场景、忽略普通场景及 detach 后场景由调用方分别准备和标识。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gene_agent.motion.model import (
    CollisionPolicy, MotionOperation, MotionRequest, PlanningSceneRef, PoseTarget,
)
from gene_agent.motion.planner import MotionPlanner

from gene_agent.skills.runtime.planning import SkillContext, SkillPlan, SkillStage


PLACE_STAGE_ORDER = ("preplace", "place", "release", "detach", "retreat")


@dataclass(frozen=True)
class PlacementTargets:
    preplace: PoseTarget
    place: PoseTarget
    retreat: PoseTarget
    target_object_pose: np.ndarray

    @classmethod
    def from_attachment(
        cls,
        table_tcp: PoseTarget,
        tcp_from_object: np.ndarray,
        *,
        preplace_distance_m: float = 0.1,
        retreat_distance_m: float = 0.05,
    ) -> "PlacementTargets":
        """沿世界 +Z 生成桌面上方点，保持旧版固定 TCP 朝向。"""
        if table_tcp.frame != "world_tcp":
            raise ValueError("桌面 TCP 必须已位于 world_tcp 坐标系")
        if preplace_distance_m <= 0 or retreat_distance_m <= 0:
            raise ValueError("桌面接近和撤离距离必须为正")
        attachment = PoseTarget("world_tcp", tcp_from_object).matrix
        preplace = table_tcp.matrix.copy()
        preplace[2, 3] += preplace_distance_m
        retreat = table_tcp.matrix.copy()
        retreat[2, 3] += retreat_distance_m
        return cls(PoseTarget("world_tcp", preplace), table_tcp,
                   PoseTarget("world_tcp", retreat), table_tcp.matrix @ attachment)


def _record(request: MotionRequest, result, name: str, attempts: list[dict],
            *, fallback_from: str | None = None) -> SkillStage:
    if result.request_id != request.request_id or result.operation is not request.operation:
        raise RuntimeError(f"planner 返回的 {name} 结果与请求不匹配")
    if (result.success and request.operation is not MotionOperation.IK
            and (result.goal_state is None or result.trajectory is None)):
        raise RuntimeError(f"成功的 {name} 必须包含实际终点和轨迹")
    attempts.append({"stage": name, "request": request.to_dict(), "result": result.to_dict()})
    details = {} if fallback_from is None else {"fallback_from": fallback_from}
    return SkillStage(name, "planned" if result.success else "failed",
                      request.target, request, result, details)


class PlaceObjectSkill:
    """先规划携物下降；松爪/detach 后再规划撤离。"""

    def plan_attached(
        self,
        planner: MotionPlanner,
        attached: SkillContext,
        ignore_world_scene: PlanningSceneRef,
        targets: PlacementTargets,
    ) -> SkillPlan:
        """复现旧线性 pre-place 筛选与最终直线下降。"""
        attempts: list[dict] = []
        ik_request = MotionRequest(
            "place:preplace_ik", MotionOperation.IK, attached.group,
            attached.start, targets.preplace, ignore_world_scene,
            CollisionPolicy(), seed=attached.seed,
        )
        ik_result = planner.evaluate_candidates((ik_request,))
        if len(ik_result) != 1:
            raise RuntimeError("pre-place batch IK 返回的候选数量不一致")
        ik_stage = _record(ik_request, ik_result[0], "preplace_ik", attempts)
        # IK 只提供可达性，不提供可执行轨迹。轨迹必须再由 linear TCP 规划。
        if ik_stage.status == "failed":
            return SkillPlan("place", None, (
                SkillStage("preplace", "failed", targets.preplace,
                           details={"operation": "linear_tcp", "reason": "pre-place IK 不可达"}),
                SkillStage("place", "not_executed", details={"operation": "linear_tcp"}),
                *self._pending_tail(),
            ), tuple(attempts))

        pre_request = MotionRequest(
            "place:preplace_linear", MotionOperation.LINEAR_TCP, attached.group,
            attached.start, targets.preplace, ignore_world_scene,
            CollisionPolicy(), seed=attached.seed,
        )
        pre_stage = _record(pre_request, planner.plan(pre_request), "preplace", attempts)
        if pre_stage.status == "failed":
            return SkillPlan("place", None, (
                pre_stage,
                SkillStage("place", "not_executed", details={"operation": "linear_tcp"}),
                *self._pending_tail(),
            ), tuple(attempts))

        place_request = MotionRequest(
            "place:place_linear", MotionOperation.LINEAR_TCP, attached.group,
            pre_stage.result.goal_state, targets.place, ignore_world_scene,
            CollisionPolicy(), seed=attached.seed,
        )
        place_stage = _record(place_request, planner.plan(place_request), "place", attempts)
        if place_stage.status == "failed":
            # 旧 PlaceSkill 在最终直线失败时允许同一目标的普通 TCP pose 兜底；
            # 必须把最终阶段标成实际的普通轨迹，不能继续宣称 linear 成功。
            fallback_request = MotionRequest(
                "place:place_pose_fallback", MotionOperation.PLAN, attached.group,
                pre_stage.result.goal_state, targets.place, ignore_world_scene,
                CollisionPolicy(), seed=attached.seed,
            )
            place_stage = _record(
                fallback_request, planner.plan(fallback_request), "place", attempts,
                fallback_from="linear_tcp",
            )
        return SkillPlan("place", "fixed_table_tcp_0", (
            pre_stage, place_stage, *self._pending_tail(),
        ), tuple(attempts))

    @staticmethod
    def _pending_tail() -> tuple[SkillStage, ...]:
        return (
            SkillStage("release", "not_executed", details={"operation": "gripper_command"}),
            SkillStage("detach", "not_executed", details={"operation": "attachment"}),
            SkillStage("retreat", "not_executed", details={"operation": "linear_tcp"}),
        )

    def plan_visual_release_and_retreat(
        self,
        detached_planner: MotionPlanner,
        initial: SkillPlan,
        detached: SkillContext,
        targets: PlacementTargets,
        *,
        release_width_m: float,
        release_settle_time_s: float = 1.5,
    ) -> SkillPlan:
        place = initial.stage("place")
        if (place.status != "planned" or place.result is None
                or detached.start != place.result.goal_state):
            raise ValueError("只有从实际 place 终点松爪并 detach 后才可规划撤离")
        if release_width_m <= 0 or release_settle_time_s < 0:
            raise ValueError("松爪宽度必须为正，稳定时间不能为负")
        stages = list(initial.stages[:2])
        stages.extend((
            SkillStage("release", "planned", details={
                "operation": "gripper_command", "width_m": float(release_width_m),
                "duration_s": float(release_settle_time_s),
            }),
            SkillStage("detach", "planned", details={
                "operation": "attachment", "object_id": detached.object_id,
                "object_pose_world": targets.target_object_pose.tolist(),
            }),
        ))
        attempts = list(initial.attempts)
        request = MotionRequest(
            "place:retreat", MotionOperation.LINEAR_TCP, detached.group,
            detached.start, targets.retreat, detached.scene,
            CollisionPolicy(), seed=detached.seed,
        )
        stages.append(_record(request, detached_planner.plan(request), "retreat", attempts))
        return SkillPlan("place", initial.selected_candidate_id,
                         tuple(stages), tuple(attempts))


__all__ = ["PlaceObjectSkill", "PlacementTargets", "PLACE_STAGE_ORDER"]
