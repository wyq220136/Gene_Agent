"""旧版固定 keypose 插入动作的同步纯视觉 Skill 阶段。

目标由任务 Knowledge 给出，本模块不依赖微波炉资产。阶段按旧版顺序：
普通规划到 pre-place → TCP 直线到 place → 释放并 detach → 两段 TCP
直线撤离。两个 planner 分别对应旧版“完整场景”与“仅自身碰撞”范围；
这种差别必须体现在各自的 PlanningSceneRef，不能在报告里写成同一场景。
"""

from __future__ import annotations

from dataclasses import dataclass

from gene_agent.motion.model import (
    CollisionPolicy, MotionOperation, MotionRequest, PlanningSceneRef, PoseTarget,
)
from gene_agent.motion.planner import MotionPlanner

from gene_agent.skills.runtime.planning import SkillContext, SkillPlan, SkillStage


INSERT_STAGE_ORDER = (
    "preplace", "insert", "release", "detach", "retreat_1", "retreat_2",
)


@dataclass(frozen=True)
class InsertionTargets:
    """完整的四个 TCP keypose；输入 frame 均为本次规划的 world_tcp。"""

    preplace: PoseTarget
    place: PoseTarget
    retreat_1: PoseTarget
    retreat_2: PoseTarget

    def __post_init__(self) -> None:
        if any(not isinstance(item, PoseTarget) or item.frame != "world_tcp"
               for item in (self.preplace, self.place, self.retreat_1, self.retreat_2)):
            raise ValueError("插入四个 keypose 必须是 world_tcp PoseTarget")


class InsertObjectSkill:
    def plan_attached(
        self,
        world_planner: MotionPlanner,
        ignore_world_planner: MotionPlanner,
        attached: SkillContext,
        ignore_world_scene: PlanningSceneRef,
        targets: InsertionTargets,
    ) -> SkillPlan:
        """复现旧 Viser 的单 pose 选择，随后规划真正的直线插入。

        先从 post-pick 实际终点规划 pre-place；使用这条路径的真实关节
        终点筛最终 place 普通 pose 轨迹。筛选通过后，执行段仍单独要求
        TCP 直线。即使筛选轨迹成功，直线失败也保留失败，不冒充完成。
        """
        attempts: list[dict] = []
        pre_request = MotionRequest(
            "insert:preplace", MotionOperation.PLAN, attached.group,
            attached.start, targets.preplace, attached.scene,
            CollisionPolicy(), seed=attached.seed,
        )
        pre_result = world_planner.plan(pre_request)
        if pre_result.request_id != pre_request.request_id or pre_result.operation is not MotionOperation.PLAN:
            raise RuntimeError("planner 返回的 pre-place 结果与请求不匹配")
        attempts.append({"stage": "preplace", "request": pre_request.to_dict(),
                         "result": pre_result.to_dict()})
        empty = (
            SkillStage("release", "not_executed", details={"operation": "gripper_command"}),
            SkillStage("detach", "not_executed", details={"operation": "attachment"}),
            SkillStage("retreat_1", "not_executed", details={"operation": "linear_tcp"}),
            SkillStage("retreat_2", "not_executed", details={"operation": "linear_tcp"}),
        )
        if not pre_result.success or pre_result.goal_state is None or pre_result.trajectory is None:
            return SkillPlan("insert", None, (
                SkillStage("preplace", "failed", targets.preplace, pre_request, pre_result),
                SkillStage("insert", "not_executed", details={"operation": "linear_tcp"}),
                *empty,
            ), tuple(attempts))

        screen_request = MotionRequest(
            "insert:place_screen", MotionOperation.PLAN, attached.group,
            pre_result.goal_state, targets.place, ignore_world_scene,
            CollisionPolicy(), seed=attached.seed,
        )
        screen_result = ignore_world_planner.plan(screen_request)
        if screen_result.request_id != screen_request.request_id or screen_result.operation is not MotionOperation.PLAN:
            raise RuntimeError("planner 返回的 place 筛选结果与请求不匹配")
        attempts.append({"stage": "place_screen", "request": screen_request.to_dict(),
                         "result": screen_result.to_dict()})
        pre_stage = SkillStage("preplace", "planned", targets.preplace, pre_request, pre_result)
        if not screen_result.success or screen_result.trajectory is None:
            return SkillPlan("insert", None, (
                pre_stage,
                SkillStage("insert", "failed", targets.place,
                           details={"operation": "linear_tcp", "reason": "place 普通轨迹筛选失败"}),
                *empty,
            ), tuple(attempts))

        insert_request = MotionRequest(
            "insert:place_linear", MotionOperation.LINEAR_TCP, attached.group,
            pre_result.goal_state, targets.place, ignore_world_scene,
            CollisionPolicy(), seed=attached.seed,
        )
        insert_result = ignore_world_planner.plan(insert_request)
        if (insert_result.request_id != insert_request.request_id
                or insert_result.operation is not MotionOperation.LINEAR_TCP):
            raise RuntimeError("planner 返回的插入直线结果与请求不匹配")
        attempts.append({"stage": "insert", "request": insert_request.to_dict(),
                         "result": insert_result.to_dict()})
        if insert_result.success and (insert_result.goal_state is None or insert_result.trajectory is None):
            raise RuntimeError("成功的插入直线必须包含终点状态和轨迹")
        result = SkillPlan("insert", "fixed_keypose_0", (
            pre_stage,
            SkillStage("insert", "planned" if insert_result.success else "failed",
                       targets.place, insert_request, insert_result),
            *empty,
        ), tuple(attempts))
        assert tuple(stage.name for stage in result.stages) == INSERT_STAGE_ORDER
        return result

    def plan_visual_release_and_retreat(
        self,
        detached_planner: MotionPlanner,
        initial: SkillPlan,
        detached: SkillContext,
        targets: InsertionTargets,
        *,
        release_width_m: float,
    ) -> SkillPlan:
        """视觉松爪/detach 后，按旧顺序分别规划两段直线撤离。"""
        insertion = initial.stage("insert")
        if (insertion.status != "planned" or insertion.result is None
                or insertion.result.goal_state is None
                or detached.start != insertion.result.goal_state):
            raise ValueError("只有插入成功并从其实际终点出发才可释放和撤离")
        if release_width_m <= 0:
            raise ValueError("松爪宽度必须为正")
        attempts = list(initial.attempts)
        stages = list(initial.stages[:2])
        stages.extend((
            SkillStage("release", "planned", details={
                "operation": "gripper_command", "width_m": float(release_width_m),
            }),
            SkillStage("detach", "planned", details={
                "operation": "attachment", "object_id": detached.object_id,
            }),
        ))
        state = detached.start
        for name, target in (("retreat_1", targets.retreat_1),
                             ("retreat_2", targets.retreat_2)):
            request = MotionRequest(
                f"insert:{name}", MotionOperation.LINEAR_TCP, detached.group,
                state, target, detached.scene, CollisionPolicy(), seed=detached.seed,
            )
            result = detached_planner.plan(request)
            if result.request_id != request.request_id or result.operation is not MotionOperation.LINEAR_TCP:
                raise RuntimeError(f"planner 返回的 {name} 结果与请求不匹配")
            attempts.append({"stage": name, "request": request.to_dict(),
                             "result": result.to_dict()})
            if result.success and (result.goal_state is None or result.trajectory is None):
                raise RuntimeError(f"成功的 {name} 必须包含终点状态和轨迹")
            stages.append(SkillStage(name, "planned" if result.success else "failed",
                                     target, request, result))
            if not result.success:
                break
            state = result.goal_state
        if len(stages) < len(INSERT_STAGE_ORDER):
            stages.append(SkillStage("retreat_2", "not_executed",
                                     details={"operation": "linear_tcp"}))
        assert tuple(stage.name for stage in stages) == INSERT_STAGE_ORDER
        return SkillPlan("insert", initial.selected_candidate_id, tuple(stages), tuple(attempts))


__all__ = ["InsertObjectSkill", "InsertionTargets", "INSERT_STAGE_ORDER"]
