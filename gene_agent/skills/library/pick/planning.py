"""最终抓取 IK 优先的最小同步规划；不依赖具体 motion 后端。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.grasp.model import GraspProposal
from gene_agent.motion.model import (
    CollisionAllowance, CollisionPolicy, MotionFailureKind, MotionOperation,
    MotionRequest, PlanningSceneRef, PoseTarget,
)
from gene_agent.motion.planner import MotionPlanner

from gene_agent.skills.library.pick.model import PICK_STAGE_ORDER, GRASP_STAGE_ORDER, GraspSettings
from gene_agent.skills.runtime.planning import SkillContext, SkillPlan, SkillStage
from gene_agent.skills.runtime.poses import offset_along_tcp_z


_LEGACY_ROTATIONS_DEG = (5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0)


def _legacy_pregrasp_targets(final: PoseTarget, distance_m: float) -> tuple[PoseTarget, ...]:
    """照搬旧 GraspSkill 的 0°、局部 X 和局部 Y 姿态候选及顺序。"""
    targets: list[PoseTarget] = []
    for axis, angle in ((None, 0.0), *(("x", v) for v in _LEGACY_ROTATIONS_DEG),
                        *(("y", v) for v in _LEGACY_ROTATIONS_DEG)):
        pose = np.array(final.matrix, copy=True)
        if axis is not None:
            pose[:3, :3] = final.matrix[:3, :3] @ Rotation.from_euler(
                axis, angle, degrees=True,
            ).as_matrix()
        pose[:3, 3] = final.matrix[:3, 3] - pose[:3, 2] * distance_m
        targets.append(PoseTarget(final.frame, pose))
    return tuple(targets)


def _joint_travel_check(
    start, joint_names, positions_rad, limit_rad: float | None, *,
    reason_code: str = "joint_travel_exceeded",
) -> dict:
    """检查实际候选关节路径是否需要从当前姿态大幅反扭。"""
    if limit_rad is None:
        return {"success": True, "enabled": False}
    names = tuple(joint_names)
    if names != start.joint_names:
        raise ValueError("pregrasp 姿态检查的关节顺序与当前状态不一致")
    positions = np.asarray(positions_rad, dtype=float)
    if positions.ndim == 1:
        positions = positions[None, :]
    if (positions.ndim != 2 or positions.shape[1] != len(names)
            or not np.isfinite(positions).all()):
        raise ValueError("pregrasp 姿态检查需要有限的 waypoint × joint 数组")
    delta = np.abs(positions - np.asarray(start.positions_rad, dtype=float)[None, :])
    per_joint = np.max(delta, axis=0)
    index = int(np.argmax(per_joint))
    maximum = float(per_joint[index])
    return {
        "success": bool(maximum <= limit_rad + 1e-12),
        "enabled": True,
        "reason_code": None if maximum <= limit_rad + 1e-12 else reason_code,
        "maximum_joint_travel_rad": maximum,
        "maximum_joint_travel_deg": float(np.rad2deg(maximum)),
        "maximum_joint_name": names[index],
        "per_joint_travel_rad": {
            name: float(value) for name, value in zip(names, per_joint)
        },
        "limit_rad": float(limit_rad),
        "limit_deg": float(np.rad2deg(limit_rad)),
        "source": "planned_joint_path_relative_to_current_executed_state",
    }


class GraspSkill:
    def plan(
        self,
        planner: MotionPlanner,
        proposals: Sequence[GraspProposal],
        context: SkillContext,
        settings: GraspSettings,
        *,
        contact_planner: MotionPlanner | None = None,
        contact_scene: PlanningSceneRef | None = None,
    ) -> SkillPlan:
        """按配置筛抓取候选；旧视觉模式在轨迹规划前筛全部局部姿态。"""
        if (contact_planner is None) != (contact_scene is None):
            raise ValueError("目标接触 planner 与场景快照必须同时提供")
        # 旧抓取语义：最终抓取和直线接近允许碰目标物体；pregrasp 的
        # IK 与路径依然使用完整世界障碍。不同阶段的实际快照写入各自请求，
        # 不能把一个放宽后的场景冒充为全流程的严格避碰结果。
        grasp_planner = contact_planner or planner
        grasp_scene = contact_scene or context.scene
        selected = tuple(proposals[:settings.max_grasp_candidates])
        if len({item.candidate_id for item in selected}) != len(selected):
            raise ValueError("抓取候选 ID 必须唯一")
        requests = tuple(
            MotionRequest(
                f"{item.candidate_id}:final_ik", MotionOperation.IK, context.group,
                context.start, PoseTarget("world_tcp", item.tcp_pose_world), grasp_scene,
                CollisionPolicy(), seed=context.seed,
            ) for item in selected
        )
        results = grasp_planner.evaluate_candidates(requests)
        if len(results) != len(requests):
            raise RuntimeError("planner 返回的 IK 结果数量与候选不一致")
        attempts: list[dict] = []
        for item, request, result in zip(selected, requests, results):
            if result.request_id != request.request_id or result.operation is not MotionOperation.IK:
                raise RuntimeError("planner 的批 IK 结果顺序或 operation 不匹配")
            attempts.append({"candidate_id": item.candidate_id, "stage": "final_grasp_ik",
                             "request": request.to_dict(), "result": result.to_dict()})

        eligible = tuple(
            (item, ik_request.target)
            for item, ik_request, ik_result in zip(selected, requests, results)
            if ik_result.success
        )
        if settings.candidate_selection == "legacy_ik":
            # 旧 Viser 先按顺序为所有最终 IK 可达的 grasp 生成 17 个
            # pregrasp，再以 128 为一批做 IK；第一个成功批次中的首个可达
            # 候选就是最终选项。随后执行阶段若路径失败，不偷偷换 grasp。
            records = tuple(
                (item, final, pregrasp, MotionRequest(
                    f"{item.candidate_id}:pregrasp_ik_{index:02d}",
                    MotionOperation.IK, context.group, context.start,
                    pregrasp, context.scene, CollisionPolicy(), seed=context.seed,
                ))
                for item, final in eligible
                for index, pregrasp in enumerate(
                    _legacy_pregrasp_targets(final, settings.approach_distance_m)
                )
            )
            viable: list[tuple[GraspProposal, PoseTarget, PoseTarget, float]] = []
            for offset in range(0, len(records), 128):
                batch = records[offset:offset + 128]
                prechecks = planner.evaluate_candidates(tuple(row[3] for row in batch))
                if len(prechecks) != len(batch):
                    raise RuntimeError("planner 返回的 pregrasp IK 数量与候选不一致")
                for (item, final, pregrasp, request), result in zip(batch, prechecks):
                    if result.request_id != request.request_id or result.operation is not MotionOperation.IK:
                        raise RuntimeError("planner 的 pregrasp IK 结果顺序或 operation 不匹配")
                    attempts.append({
                        "candidate_id": item.candidate_id, "stage": "pregrasp_ik",
                        "request": request.to_dict(), "result": result.to_dict(),
                    })
                    if not result.success or result.goal_state is None:
                        continue
                    posture = _joint_travel_check(
                        context.start, result.goal_state.joint_names,
                        result.goal_state.positions_rad,
                        settings.max_pregrasp_joint_travel_rad,
                        reason_code="pregrasp_joint_travel_exceeded",
                    )
                    if not posture["success"]:
                        attempts.append({
                            "candidate_id": item.candidate_id,
                            "stage": "pregrasp_posture_check",
                            "request_id": request.request_id,
                            "result": posture,
                        })
                        continue
                    viable.append((
                        item, final, pregrasp,
                        float(posture.get("maximum_joint_travel_rad", 0.)),
                    ))
                # 未启用姿态连续性约束时严格保留旧版“第一个 IK 成功即停”。
                if viable and settings.max_pregrasp_joint_travel_rad is None:
                    break
            if settings.max_pregrasp_joint_travel_rad is not None:
                # 保留抓取候选的外沿/基座侧优先级，只在同一候选内部优先
                # 尝试关节行程更小的 pregrasp 分支，减少腕部反扭。
                candidate_rank = {
                    item.candidate_id: index for index, item in enumerate(selected)
                }
                viable.sort(key=lambda row: (
                    candidate_rank[row[0].candidate_id], row[3],
                ))
                candidates_to_plan = tuple(row[:3] for row in viable)
            else:
                candidates_to_plan = tuple(row[:3] for row in viable[:1])
        else:
            candidates_to_plan = tuple(
                (item, final, offset_along_tcp_z(final, -settings.approach_distance_m))
                for item, final in eligible
            )

        for item, final, pregrasp in candidates_to_plan:
            pre_request = MotionRequest(
                f"{item.candidate_id}:pregrasp", MotionOperation.PLAN, context.group,
                context.start, pregrasp, context.scene, CollisionPolicy(), seed=context.seed,
                approximate_ik_budget=settings.pregrasp_approximate_ik_budget,
            )
            pre_result = planner.plan(pre_request)
            if (pre_result.request_id != pre_request.request_id
                    or pre_result.operation is not MotionOperation.PLAN):
                raise RuntimeError("planner 返回的 pregrasp 结果与请求不匹配")
            attempts.append({"candidate_id": item.candidate_id, "stage": "pregrasp",
                             "request": pre_request.to_dict(), "result": pre_result.to_dict()})
            if not pre_result.success or pre_result.goal_state is None or pre_result.trajectory is None:
                if (settings.candidate_selection == "legacy_ik"
                        and settings.max_pregrasp_joint_travel_rad is None):
                    return SkillPlan("grasp", item.candidate_id, (
                        SkillStage("open", "planned", details={
                            "width_m": float(item.parameters.get(
                                "pregrasp_opening_width_m", settings.open_width_m,
                            )),
                        }),
                        SkillStage("pregrasp", "failed", pregrasp, pre_request, pre_result),
                        SkillStage("approach", "not_executed"),
                        SkillStage("close", "not_executed"),
                    ), tuple(attempts))
                continue
            posture = _joint_travel_check(
                context.start, pre_result.trajectory.joint_names,
                pre_result.trajectory.positions_rad,
                settings.max_pregrasp_joint_travel_rad,
                reason_code="pregrasp_joint_travel_exceeded",
            )
            if posture["enabled"]:
                attempts.append({
                    "candidate_id": item.candidate_id,
                    "stage": "pregrasp_posture_check",
                    "request_id": pre_request.request_id,
                    "result": posture,
                })
            if not posture["success"]:
                continue

            # 指尖接触碗边仅属于直线接近阶段；pregrasp 仍使用严格碰撞检查。
            approach_policy = CollisionPolicy(
                "finger_object_contact",
                (CollisionAllowance("gripper_fingers", context.object_id, "抓取时的预期指尖接触"),),
            )
            approach_request = MotionRequest(
                f"{item.candidate_id}:approach", MotionOperation.LINEAR_TCP,
                context.group, pre_result.goal_state, final, grasp_scene,
                approach_policy, seed=context.seed,
            )
            approach_result = grasp_planner.plan(approach_request)
            if (approach_result.request_id != approach_request.request_id
                    or approach_result.operation is not MotionOperation.LINEAR_TCP):
                raise RuntimeError("planner 返回的 approach 结果与请求不匹配")
            attempts.append({"candidate_id": item.candidate_id, "stage": "approach",
                             "request": approach_request.to_dict(), "result": approach_result.to_dict()})
            if approach_result.success and approach_result.trajectory is None:
                raise RuntimeError("直线接近成功结果必须包含轨迹")
            if approach_result.success:
                approach_posture = _joint_travel_check(
                    pre_result.goal_state,
                    approach_result.trajectory.joint_names,
                    approach_result.trajectory.positions_rad,
                    settings.max_approach_joint_travel_rad,
                    reason_code="approach_joint_travel_exceeded",
                )
                if approach_posture["enabled"]:
                    attempts.append({
                        "candidate_id": item.candidate_id,
                        "stage": "approach_posture_check",
                        "request_id": approach_request.request_id,
                        "result": approach_posture,
                    })
                if not approach_posture["success"]:
                    continue
            approach_status = "planned" if approach_result.success else "failed"
            if (approach_result.failure is not None
                    and approach_result.failure.kind is MotionFailureKind.INVALID_REQUEST
                    and ("operation" in approach_result.failure.summary
                         or "不支持" in approach_result.failure.summary)):
                approach_status = "unsupported"
            # 启用姿态连续性约束时，候选筛选的语义是找到一条完整且不反扭的
            # pregrasp + approach 前缀。第一条安全 pregrasp 的直线接近失败
            # 不能阻止后续候选继续竞争；旧兼容模式仍保留首次候选行为。
            if (approach_status != "planned"
                    and (settings.max_pregrasp_joint_travel_rad is not None
                         or settings.max_approach_joint_travel_rad is not None)):
                continue
            later = "planned" if approach_result.success else "not_executed"
            # 旧版 GraspSkill 优先执行候选自带的预抓取开口，接触后再按
            # 该候选的 closing width 闭合。这里不能用任务默认开口覆盖
            # Bowl Ring 的 41 mm / 7 mm，否则视觉和碰撞快照都会不同。
            pregrasp_width = float(item.parameters.get(
                "pregrasp_opening_width_m", settings.open_width_m,
            ))
            if not np.isfinite(pregrasp_width) or pregrasp_width < item.closing_width_m:
                raise ValueError("候选预抓取开口必须有限且不小于闭合宽度")
            stages = (
                SkillStage("open", "planned", details={"width_m": pregrasp_width}),
                SkillStage("pregrasp", "planned", pregrasp, pre_request, pre_result),
                SkillStage("approach", approach_status, final, approach_request, approach_result),
                SkillStage("close", later, details={
                    "width_m": item.closing_width_m,
                    "candidate_nominal_closing_width_m": item.closing_width_m,
                }),
            )
            assert tuple(stage.name for stage in stages) == GRASP_STAGE_ORDER
            return SkillPlan("grasp", item.candidate_id, stages, tuple(attempts))

        # 这里可以区分最终 grasp 完全无 IK 解与 pregrasp 全部规划失败。
        any_ik = any(result.success for result in results)
        return SkillPlan("grasp", None, (
            SkillStage("open", "not_executed", details={
                "width_m": settings.open_width_m,
                "final_ik_checked": len(requests),
                "final_ik_success_count": sum(r.success for r in results),
            }),
            SkillStage("pregrasp", "failed" if any_ik else "not_executed"),
            SkillStage("approach", "not_executed"),
            SkillStage("close", "not_executed"),
        ), tuple(attempts))


def make_postpick_candidates(
    grasp_pose: np.ndarray, pregrasp_pose: np.ndarray, distance_m: float,
    *, translation_direction_world: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, ...]:
    """按旧 PickSkill 顺序生成撤离 pose，供视觉与在线物理执行共用。

    平移方向始终由已选 pregrasp 减 grasp 决定；局部旋转只改变
    TCP 姿态，不改变撤离终点。物理执行在闭爪后另行检查当前状态、
    附着物和每个候选的完整路径。
    """
    grasp = np.asarray(grasp_pose, dtype=float)
    pregrasp = np.asarray(pregrasp_pose, dtype=float)
    if grasp.shape != (4, 4) or pregrasp.shape != (4, 4):
        raise ValueError("grasp 和 pregrasp 必须是 4x4 pose")
    direction = (pregrasp[:3, 3] - grasp[:3, 3]
                 if translation_direction_world is None
                 else np.asarray(translation_direction_world, dtype=float))
    norm = float(np.linalg.norm(direction))
    distance = float(distance_m)
    if not np.isfinite(grasp).all() or not np.isfinite(pregrasp).all() \
            or not np.isfinite(distance) or distance <= 0 or norm <= 1e-9:
        raise ValueError("post-pick 撤离方向和距离必须有限且非零")
    translation = direction / norm * distance
    rotations = (5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0)
    candidates: list[np.ndarray] = []
    for axis, angle in ((None, 0.0), *(("x", v) for v in rotations),
                        *(("y", v) for v in rotations)):
        pose = np.array(grasp, copy=True)
        if axis is not None:
            pose[:3, :3] = grasp[:3, :3] @ Rotation.from_euler(
                axis, angle, degrees=True,
            ).as_matrix()
        pose[:3, 3] = grasp[:3, 3] + translation
        candidates.append(pose)
    return tuple(candidates)


class PickSkill:
    def plan(
        self,
        planner: MotionPlanner,
        proposals: Sequence[GraspProposal],
        context: SkillContext,
        settings: GraspSettings,
        *,
        contact_planner: MotionPlanner | None = None,
        contact_scene: PlanningSceneRef | None = None,
    ) -> SkillPlan:
        grasp = GraspSkill().plan(
            planner, proposals, context, settings,
            contact_planner=contact_planner, contact_scene=contact_scene,
        )
        # 抬升目标必须从闭爪后的实际 TCP 状态生成；静态规划在此只记录
        # 阶段契约，物体附着代理由执行时已验证的双指接触建立。
        lift = SkillStage("lift", "not_executed", details={
            "distance_m": settings.lift_distance_m,
            "path_kind": "linear_tcp",
            "reason": "需要闭爪后的实际关节状态和附着物碰撞快照",
        })
        hold = SkillStage("hold", "not_executed", details={
            "duration_s": settings.hold_duration_s,
            "reason": "需要抬升后的实际物体和夹爪状态",
        })
        stages = grasp.stages + (lift, hold)
        assert tuple(stage.name for stage in stages) == PICK_STAGE_ORDER
        return SkillPlan("pick", grasp.selected_candidate_id, stages, grasp.attempts)

    def plan_visual_retreat(
        self,
        planner: MotionPlanner,
        initial: SkillPlan,
        attached: SkillContext,
        settings: GraspSettings,
    ) -> SkillPlan:
        """按旧 PickSkill 的候选顺序筛选 post-pick，再做 TCP 直线规划。

        调用方必须在 approach 成功且闭爪后建立新的 attach 碰撞快照。
        撤离方向取已选 pregrasp 减去 grasp；局部姿态依次为 0°、
        X ±5/10/15/20°、Y ±5/10/15/20°。候选路径筛选与最终直线
        规划是两个独立事实：最终直线失败时不得悄悄换成下一候选。
        """
        approach = initial.stage("approach")
        pregrasp = initial.stage("pregrasp")
        close = initial.stage("close")
        if (approach.status != "planned" or approach.target is None
                or approach.result is None or approach.result.goal_state is None
                or pregrasp.target is None or close.status != "planned"):
            raise ValueError("只有完成 approach 和 close 后才能规划视觉撤离")
        if attached.start != approach.result.goal_state:
            raise ValueError("attach 后的规划起点必须是 approach 的实际终点")

        candidates = tuple(PoseTarget(approach.target.frame, pose) for pose in
                           make_postpick_candidates(
                               approach.target.matrix, pregrasp.target.matrix,
                               settings.lift_distance_m,
                           ))

        attempts = list(initial.attempts)
        selected: tuple[int, PoseTarget] | None = None
        # 旧版优先使用小 batch 完整轨迹筛选；当前抽象尚无 batch PLAN
        # 接口，故按完全相同的顺序逐个筛，不能把 IK 成功当成路径成功。
        for index, target in enumerate(candidates):
            request = MotionRequest(
                f"{initial.selected_candidate_id}:postpick_screen_{index:02d}",
                MotionOperation.PLAN, attached.group, attached.start, target,
                attached.scene, CollisionPolicy(), seed=attached.seed,
            )
            result = planner.plan(request)
            if result.request_id != request.request_id or result.operation is not MotionOperation.PLAN:
                raise RuntimeError("planner 返回的 post-pick 筛选结果与请求不匹配")
            attempts.append({
                "candidate_id": initial.selected_candidate_id,
                "stage": "postpick_screen", "index": index,
                "request": request.to_dict(), "result": result.to_dict(),
            })
            if result.success and result.trajectory is not None:
                selected = (index, target)
                break

        hold = SkillStage("hold", "not_executed", details={
            "duration_s": settings.hold_duration_s,
        })
        if selected is None:
            lift = SkillStage("lift", "failed", details={
                "reason": "全部 post-pick 候选都没有完整轨迹",
                "candidates_checked": len(candidates),
            })
        else:
            index, target = selected
            request = MotionRequest(
                f"{initial.selected_candidate_id}:postpick_linear",
                MotionOperation.LINEAR_TCP, attached.group, attached.start,
                target, attached.scene, CollisionPolicy(), seed=attached.seed,
            )
            result = planner.plan(request)
            if result.request_id != request.request_id or result.operation is not MotionOperation.LINEAR_TCP:
                raise RuntimeError("planner 返回的 post-pick 直线结果与请求不匹配")
            attempts.append({
                "candidate_id": initial.selected_candidate_id,
                "stage": "postpick_linear", "index": index,
                "request": request.to_dict(), "result": result.to_dict(),
            })
            if result.success and result.trajectory is None:
                raise RuntimeError("post-pick 直线成功结果必须包含轨迹")
            lift = SkillStage("lift", "planned" if result.success else "failed",
                              target, request, result,
                              {"screened_candidate_index": index, "path_kind": "linear_tcp"})
            if result.success:
                hold = SkillStage("hold", "planned", details={
                    "duration_s": settings.hold_duration_s,
                })
        stages = initial.stages[:4] + (lift, hold)
        assert tuple(stage.name for stage in stages) == PICK_STAGE_ORDER
        return SkillPlan("pick", initial.selected_candidate_id, stages, tuple(attempts))


def _pose(value) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("抓取 pose 必须是有限 4x4 矩阵")
    if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(
        transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
    ) or np.linalg.det(transform[:3, :3]) < 0:
        raise ValueError("抓取 pose 必须是合法刚体变换")
    return transform.copy()


def make_pregrasp_pose(final_pose, *, retreat_m: float = .08) -> np.ndarray:
    """沿 TCP -Z 从接触目标退回；候选 +Z 是朝物体的接近方向。"""
    retreat = float(retreat_m)
    if not np.isfinite(retreat) or retreat <= 0:
        raise ValueError("retreat_m 必须为正有限值")
    final = PoseTarget("world_tcp", _pose(final_pose))
    return np.array(offset_along_tcp_z(final, -retreat).matrix, copy=True)
