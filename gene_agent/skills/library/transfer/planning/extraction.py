"""炉内固定五点取出的同步 Skill 规划阶段。

调用方负责把旧 profile 的 microwave-local TCP keypose 投影到当前世界坐标，
并提供阶段对应的场景快照。这里保留旧 Viser 的动作顺序；规划结果不表示
gripper 已真实夹住物体，也不负责修改 cuRobo world 或可视化状态。
"""

from __future__ import annotations

from dataclasses import dataclass

from gene_agent.motion.model import (
    CollisionPolicy, MotionOperation, MotionRequest, PlanningSceneRef, PoseTarget,
)
from gene_agent.motion.planner import MotionPlanner

from gene_agent.skills.runtime.planning import SkillContext, SkillPlan, SkillStage
from gene_agent.skills.runtime.poses import legacy_local_xy_pose_candidates


EXTRACT_STAGE_ORDER = (
    "open", "pre_extract_1", "pre_extract_2", "grasp", "close", "attach",
    "post_extract_1", "post_extract_2",
)


@dataclass(frozen=True)
class ExtractionTargets:
    """旧配置中的五个 TCP 点按原顺序转换后的世界目标。"""

    pre_extract_1: PoseTarget
    pre_extract_2: PoseTarget
    grasp: PoseTarget
    post_extract_1: PoseTarget
    post_extract_2: PoseTarget

    def __post_init__(self) -> None:
        values = (
            self.pre_extract_1, self.pre_extract_2, self.grasp,
            self.post_extract_1, self.post_extract_2,
        )
        if any(not isinstance(value, PoseTarget) or value.frame != "world_tcp"
               for value in values):
            raise ValueError("取出五个关键位姿必须是 world_tcp PoseTarget")


def _record_result(request: MotionRequest, result, name: str, attempts: list[dict]) -> SkillStage:
    if result.request_id != request.request_id or result.operation is not request.operation:
        raise RuntimeError(f"planner 返回的 {name} 结果与请求不匹配")
    if result.success and (result.goal_state is None or result.trajectory is None):
        raise RuntimeError(f"成功的 {name} 必须包含实际终点和轨迹")
    attempts.append({"stage": name, "request": request.to_dict(), "result": result.to_dict()})
    return SkillStage(name, "planned" if result.success else "failed",
                      request.target, request, result)


def _plan_with_legacy_local_fallback(
    planner: MotionPlanner, nominal: SkillStage, attempts: list[dict],
    *, limit_deg: float, step_deg: float,
) -> SkillStage:
    """名义直线失败后按旧版顺序 batch IK，再尝试保持直线的局部姿态。"""
    request = nominal.request
    candidates = legacy_local_xy_pose_candidates(
        request.target, limit_deg=limit_deg, step_deg=step_deg,
    )[1:]
    ik_requests = tuple(MotionRequest(
        f"{request.request_id}:local_ik_{index}", MotionOperation.IK,
        request.group, request.start, target, request.scene,
        request.collision_policy, seed=request.seed,
        planner_mode=request.planner_mode,
    ) for index, (_x, _y, target) in enumerate(candidates, start=1))
    try:
        reachable = planner.evaluate_candidates(ik_requests) if ik_requests else ()
    except NotImplementedError:
        reachable = ()
    if len(reachable) not in (0, len(ik_requests)):
        raise RuntimeError("局部姿态 batch IK 的返回数量与候选不一致")
    summary = {
        "candidate_count": len(candidates),
        "reachable_count": sum(bool(result.success) for result in reachable),
        "planned_count": 0,
        "requested_tcp_pose_world": request.target.matrix.tolist(),
    }
    for index, ((x_deg, y_deg, target), ik) in enumerate(
        zip(candidates, reachable), start=1,
    ):
        if not ik.success:
            continue
        alternative = MotionRequest(
            f"{request.request_id}:local_{index}", request.operation,
            request.group, request.start, target, request.scene,
            request.collision_policy, seed=request.seed,
            planner_mode=request.planner_mode,
        )
        stage = _record_result(alternative, planner.plan(alternative), nominal.name,
                               attempts)
        summary["planned_count"] += 1
        if stage.status == "planned":
            summary.update({
                "local_rotation_x_deg": x_deg,
                "local_rotation_y_deg": y_deg,
                "selected_tcp_pose_world": target.matrix.tolist(),
            })
            return SkillStage(stage.name, stage.status, stage.target, stage.request,
                              stage.result, {"local_pose_fallback": summary})
    # 保留初次失败的 request/result，不能把全部候选失败改报成最后一个错误。
    return SkillStage(nominal.name, nominal.status, nominal.target,
                      nominal.request, nominal.result,
                      {"local_pose_fallback": summary})


class ExtractObjectSkill:
    """先规划闭爪前动作，再从真实 attach 状态规划携物撤离。"""

    def plan_grasp(
        self,
        world_planner: MotionPlanner,
        ignore_world_planner: MotionPlanner,
        free: SkillContext,
        ignore_world_scene: PlanningSceneRef,
        targets: ExtractionTargets,
        *,
        pregrasp_width_m: float,
        grasp_width_m: float,
    ) -> SkillPlan:
        if pregrasp_width_m <= 0 or grasp_width_m < 0:
            raise ValueError("夹爪宽度必须有效")
        attempts: list[dict] = []
        stages = [SkillStage("open", "planned", details={
            "operation": "gripper_command", "width_m": float(pregrasp_width_m),
        })]
        state = free.start
        for name, target, operation, planner, scene in (
            ("pre_extract_1", targets.pre_extract_1, MotionOperation.PLAN,
             world_planner, free.scene),
            ("pre_extract_2", targets.pre_extract_2, MotionOperation.LINEAR_TCP,
             ignore_world_planner, ignore_world_scene),
            ("grasp", targets.grasp, MotionOperation.LINEAR_TCP,
             ignore_world_planner, ignore_world_scene),
        ):
            request = MotionRequest(
                f"extract:{name}", operation, free.group, state, target,
                scene, CollisionPolicy(), seed=free.seed,
            )
            stage = _record_result(request, planner.plan(request), name, attempts)
            stages.append(stage)
            if stage.status == "failed":
                break
            state = stage.result.goal_state
        for name, operation in (
            ("pre_extract_1", "plan"), ("pre_extract_2", "linear_tcp"),
            ("grasp", "linear_tcp"), ("close", "gripper_command"),
            ("attach", "attachment"), ("post_extract_1", "linear_tcp"),
            ("post_extract_2", "linear_tcp"),
        ):
            if name not in {stage.name for stage in stages}:
                stages.append(SkillStage(name, "not_executed", details={"operation": operation}))
        return SkillPlan("extract", None, tuple(stages), tuple(attempts))

    def plan_attached_retreat(
        self,
        attached_planner: MotionPlanner,
        initial: SkillPlan,
        attached: SkillContext,
        targets: ExtractionTargets,
        *,
        grasp_width_m: float,
        grasp_settle_time_s: float = 1.5,
        local_pose_fallback: bool = False,
        local_pose_fallback_limit_deg: float = 10.,
        local_pose_fallback_step_deg: float = 2.5,
    ) -> SkillPlan:
        grasp = initial.stage("grasp")
        if (grasp.status != "planned" or grasp.result is None
                or attached.start != grasp.result.goal_state):
            raise ValueError("只有从实际 grasp 终点 attach 后才可规划携物撤离")
        if grasp_width_m < 0 or grasp_settle_time_s < 0:
            raise ValueError("抓取宽度和闭爪稳定时间不能为负")
        stages = list(initial.stages[:4])
        stages.extend((
            SkillStage("close", "planned", details={
                "operation": "gripper_command", "width_m": float(grasp_width_m),
                "duration_s": float(grasp_settle_time_s),
            }),
            SkillStage("attach", "planned", details={
                "operation": "attachment", "object_id": attached.object_id,
            }),
        ))
        attempts = list(initial.attempts)
        state = attached.start
        for name, target in (
            ("post_extract_1", targets.post_extract_1),
            ("post_extract_2", targets.post_extract_2),
        ):
            request = MotionRequest(
                f"extract:{name}", MotionOperation.LINEAR_TCP,
                attached.group, state, target, attached.scene,
                CollisionPolicy(), seed=attached.seed,
            )
            stage = _record_result(request, attached_planner.plan(request), name, attempts)
            if stage.status == "failed" and local_pose_fallback:
                stage = _plan_with_legacy_local_fallback(
                    attached_planner, stage, attempts,
                    limit_deg=local_pose_fallback_limit_deg,
                    step_deg=local_pose_fallback_step_deg,
                )
            stages.append(stage)
            if stage.status == "failed":
                break
            state = stage.result.goal_state
        if len(stages) < len(EXTRACT_STAGE_ORDER):
            stages.append(SkillStage("post_extract_2", "not_executed",
                                     details={"operation": "linear_tcp"}))
        return SkillPlan("extract", "fixed_keypose_0", tuple(stages), tuple(attempts))


__all__ = ["ExtractObjectSkill", "ExtractionTargets", "EXTRACT_STAGE_ORDER"]
