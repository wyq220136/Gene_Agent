"""Skill 阶段目标候选：保留 Knowledge 名义目标，并显式记录局部旋转偏差。"""

from __future__ import annotations

import inspect
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion.curobo.approximate import (
    ApproximateIKBudget,
    plan_with_approximate_ik,
)


def _local_rotation_candidates(target, x_degrees, y_degrees):
    target = np.asarray(target, dtype=float)
    rotations = {
        (float(x), float(y))
        for x in x_degrees for y in y_degrees
        if float(x) != 0. or float(y) != 0.
    }
    ordered = sorted(rotations, key=lambda item: (
        abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1]), item[0], item[1],
    ))
    rows = []
    for x_deg, y_deg in ordered:
        candidate = target.copy()
        candidate[:3, :3] = target[:3, :3] @ Rotation.from_euler(
            "xy", [x_deg, y_deg], degrees=True,
        ).as_matrix()
        rows.append((x_deg, y_deg, candidate))
    return tuple(rows)


def plan_with_local_rotation_search(
    planner,
    joint_names,
    start_positions,
    requested_pose,
    *,
    budget: ApproximateIKBudget | None,
    x_degrees=(0.,),
    y_degrees=(0.,),
    target_frame="world_tcp",
    linear_tcp=False,
    max_attempts=None,
):
    """用一次 batch IK 筛选名义目标和局部旋转，再规划首个可达目标。

    这与旧版 AffordToolAct 的门任务快路径一致：IK 负责低成本筛选，轨迹
    planner 只处理按原优先级选中的少量目标。只有全部 batch IK 候选失败
    时，才对名义目标运行带碰撞归因和近似 IK 的详细规划，避免正常执行
    为了生成不会被用户查看的失败报告而重复求解。

    局部旋转是 Skill 候选范围，不会改写 Knowledge。返回值始终保留名义
    目标、采用目标和实际角度偏差，使界面可以区分严格、近似和候选规划。
    """
    requested = np.asarray(requested_pose, dtype=float)
    candidates = _local_rotation_candidates(requested, x_degrees, y_degrees)
    if not candidates:
        return plan_with_approximate_ik(
            planner, joint_names, start_positions, requested,
            budget=budget, target_frame=target_frame, linear_tcp=linear_tcp,
            max_attempts=max_attempts,
        )

    # 名义目标始终位于第一个，后面的局部旋转继续使用原来的偏差优先级。
    # 一次 GPU batch 代替“完整规划名义目标失败后，再初始化 batch IK”。
    batch_rows = ((0.0, 0.0, requested), *candidates)
    batch_method = planner.solve_ik_batch
    batch_options = {"target_frame": target_frame}
    parameters = inspect.signature(batch_method).parameters.values()
    if (
        "diagnose_failures" in inspect.signature(batch_method).parameters
        or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    ):
        # 正常候选筛选只需要可达/不可达；失败对象归因留到整批无解时
        # 对名义目标做。其他 motion planner 不需要实现这个可选优化参数。
        batch_options["diagnose_failures"] = False
    ik_results = batch_method(
        joint_names, start_positions, [row[2] for row in batch_rows],
        **batch_options,
    )
    nominal_ik = ik_results[0]
    attempts = {
        "candidate_count": len(candidates),
        "batch_target_count": len(batch_rows),
        "nominal_ik_success": bool(nominal_ik.get("success")),
        "ik_success_count": sum(
            bool(result.get("success")) for result in ik_results[1:]
        ),
        "planned_candidates": [],
    }
    first_planning_failure = None
    for index, ((x_deg, y_deg, candidate), ik_result) in enumerate(
        zip(batch_rows, ik_results, strict=True)
    ):
        if not ik_result.get("success"):
            continue
        planned = plan_with_approximate_ik(
            planner, joint_names, start_positions, candidate,
            # 名义目标保留 Skill 显式声明的近似预算；局部旋转本身已经是
            # 选中的替代目标，不能在它上面继续静默扩大误差。
            budget=budget if index == 0 else None,
            target_frame=target_frame, linear_tcp=linear_tcp,
            max_attempts=max_attempts,
        )
        attempts["planned_candidates"].append({
            "local_rotation_x_deg": x_deg,
            "local_rotation_y_deg": y_deg,
            "success": bool(planned.get("success")),
            "failure_category": planned.get("failure_category"),
        })
        if not planned.get("success"):
            if first_planning_failure is None:
                first_planning_failure = planned
            continue
        if index == 0:
            return planned
        result = dict(planned)
        result["target_selection"] = {
            "kind": "local_rotation_candidate",
            "requested_tcp_pose_world": requested.tolist(),
            "selected_tcp_pose_world": candidate.tolist(),
            "local_rotation_x_deg": x_deg,
            "local_rotation_y_deg": y_deg,
            "position_error_m": 0.,
            "rotation_error_rad": float(np.deg2rad(np.hypot(x_deg, y_deg))),
            "strict_failure_category": nominal_ik.get("failure_category"),
        }
        result["target_selection_attempts"] = attempts
        return result

    # 全部 IK 候选失败时才运行详细名义目标诊断。若 batch IK 找到了目标，
    # 但所有已尝试轨迹都失败，保留第一条真实轨迹失败，避免重复规划。
    if first_planning_failure is None:
        failure = plan_with_approximate_ik(
            planner, joint_names, start_positions, requested,
            budget=budget, target_frame=target_frame, linear_tcp=linear_tcp,
            max_attempts=max_attempts,
        )
    else:
        failure = dict(first_planning_failure)
    failure["target_selection_attempts"] = attempts
    return failure


__all__ = ["plan_with_local_rotation_search"]
