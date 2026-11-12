"""对严格 IK 失败的阶段目标做有界替代，仍使用原 cuRobo 规划器。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ApproximateIKBudget:
    """Skill 阶段允许的原目标与实际末端目标误差；单位 m、rad。"""

    max_position_error_m: float
    max_rotation_error_rad: float

    def __post_init__(self) -> None:
        if (not math.isfinite(self.max_position_error_m)
                or not math.isfinite(self.max_rotation_error_rad)
                or self.max_position_error_m <= 0
                or self.max_rotation_error_rad <= 0):
            raise ValueError("近似 IK 误差上限必须是正的有限数")

    def to_dict(self) -> dict:
        return {"max_position_error_m": self.max_position_error_m,
                "max_rotation_error_rad": self.max_rotation_error_rad}


def _pose_error(requested: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(requested[:3, 3] - actual[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        requested[:3, :3] @ actual[:3, :3].T).as_rotvec()))
    return position, rotation


def plan_with_approximate_ik(
    planner, joint_names, start_positions, requested_pose,
    *, budget: ApproximateIKBudget | None,
    target_frame: str = "world_tcp", linear_tcp: bool = False,
    max_attempts: int | None = None,
) -> dict:
    """严格规划优先；仅目标 IK 无解时，重规划到安全候选的 FK pose。

    近似目标来源于同一次失败记录的 cuRobo IK 候选。返回的 success 只
    表示替代目标的轨迹通过正常规划，不能代表原目标已精确到达。
    """
    started = time.perf_counter()
    def finish(result):
        # 包含严格失败、候选检查和重规划，不能只记录最后一次替代规划耗时。
        result["duration_s"] = time.perf_counter() - started
        return result
    target = np.asarray(requested_pose, dtype=float)
    method = planner.plan_linear_tcp if linear_tcp else planner.plan
    options = {"target_frame": target_frame}
    if max_attempts is not None:
        options["max_attempts"] = int(max_attempts)
    strict = method(joint_names, start_positions, target, **options)
    if strict.get("success") or budget is None or strict.get("failure_category") != "goal_ik_no_solution":
        return finish(strict)
    diagnostic = strict.get("ik_diagnostic") or {}
    seeds = diagnostic.get("seeds") or []
    selected_index = diagnostic.get("selected_seed_index")
    collision_keys = ("joint_limit_violation", "self_collision", "scene_collision")
    candidates = []
    for index, seed in enumerate(seeds):
        try:
            position = float(seed["position_error_m"])
            rotation = float(seed["rotation_error_rad"])
        except (KeyError, TypeError, ValueError):
            continue
        # cuRobo 的原始 feasible=False 不说明单独是哪项约束失败。
        # 只允许有独立逐项约束检查能力的 planner 重新核验该种子。
        if ((not seed.get("constraint_feasible") and
             not hasattr(planner, "_collision_diagnostic"))
                or not np.isfinite([position, rotation]).all()
                or position > budget.max_position_error_m
                or rotation > budget.max_rotation_error_rad):
            continue
        score = position / budget.max_position_error_m + rotation / budget.max_rotation_error_rad
        candidates.append((score, index, seed))
    attempts = []
    screen = []
    for _, index, seed in sorted(candidates):
        try:
            q = np.asarray(seed.get("joint_positions_rad") if seed.get("joint_positions_rad") is not None
                           else diagnostic["selected_joint_positions_rad"] if index == selected_index else None,
                           dtype=float)
            if q.shape != (len(joint_names),) or not np.isfinite(q).all():
                screen.append({"seed_index": index, "reason": "invalid_joint_vector"})
                continue
            if not seed.get("constraint_feasible"):
                # 不得复用失败报告里只检查所选 seed 的缓存结论；逐种子
                # 对当前完整场景和关节限位重新查询。
                check = planner._collision_diagnostic(q)
            elif index == selected_index and diagnostic.get("collision_checked"):
                check = diagnostic.get("collision_check") or {}
            elif hasattr(planner, "_collision_diagnostic"):
                check = planner._collision_diagnostic(q)
            else:
                continue
            if any(key not in check or bool(check[key]) for key in collision_keys):
                screen.append({"seed_index": index, "reason": "constraint_check_rejected",
                               "constraint_check": check})
                continue
            substitute = np.asarray(planner.fk(joint_names, q), dtype=float)
            if substitute.shape != (4, 4) or not np.isfinite(substitute).all():
                screen.append({"seed_index": index, "reason": "invalid_fk_pose"})
                continue
            before_error = _pose_error(target, substitute)
            if (before_error[0] > budget.max_position_error_m
                    or before_error[1] > budget.max_rotation_error_rad):
                screen.append({"seed_index": index, "reason": "fk_error_above_budget",
                               "position_error_m": before_error[0],
                               "rotation_error_rad": before_error[1]})
                continue
            alternative = method(joint_names, start_positions, substitute, **options)
            attempts.append({"seed_index": index, "replan_success": bool(alternative.get("success")),
                             "replan_failure_category": alternative.get("failure_category")})
            screen.append({"seed_index": index, "reason": (
                "replan_success" if alternative.get("success") else "replan_failed"),
                "replan_failure_category": alternative.get("failure_category")})
            if not alternative.get("success"):
                continue
            positions = np.asarray(alternative.get("positions_rad"), dtype=float)
            if positions.ndim != 2 or positions.shape[1] != len(joint_names) or len(positions) < 2:
                continue
            endpoint = np.asarray(planner.fk(joint_names, positions[-1]), dtype=float)
            if endpoint.shape != (4, 4) or not np.isfinite(endpoint).all():
                continue
            endpoint_error = _pose_error(target, endpoint)
            if (endpoint_error[0] > budget.max_position_error_m
                    or endpoint_error[1] > budget.max_rotation_error_rad):
                continue
        except (TypeError, ValueError, IndexError, KeyError) as error:
            screen.append({"seed_index": index, "reason": "candidate_error",
                           "error": f"{type(error).__name__}: {error}"})
            continue
        result = dict(alternative)
        result["approximate_ik"] = {
            "approximate": True,
            "strict_ik_success": False,
            "constraint_feasible": True,
            "source_constraint_feasible": bool(seed.get("constraint_feasible")),
            "independent_constraint_check": True,
            "constraint_check": check,
            "requested_tcp_pose_world": target.tolist(),
            "planned_tcp_pose_world": substitute.tolist(),
            "planned_joint_positions_rad": q.tolist(),
            "planned_position_error_m": before_error[0],
            "planned_rotation_error_rad": before_error[1],
            "actual_endpoint_tcp_pose_world": endpoint.tolist(),
            "position_error_m": endpoint_error[0],
            "rotation_error_rad": endpoint_error[1],
            "budget": budget.to_dict(),
            "source_failure_category": strict.get("failure_category"),
            "selected_seed_index": index,
            "attempts": attempts,
            "candidate_screen": screen,
        }
        return finish(result)
    if attempts:
        strict["approximate_ik_attempts"] = attempts
    strict["approximate_ik_candidate_screen"] = screen
    return finish(strict)


__all__ = ["ApproximateIKBudget", "plan_with_approximate_ik"]
