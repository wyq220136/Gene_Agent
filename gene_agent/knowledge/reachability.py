"""与具体 IK 后端无关的位姿误差筛选；不推断碰撞或路径可行性。"""
from __future__ import annotations

import math


def assess_pose_ik(result, *, max_position_error_m: float,
                   max_rotation_error_deg: float) -> dict:
    """严格失败的候选可保留为近似 Knowledge 候选，不改原 IK 结果。"""
    position_limit = float(max_position_error_m)
    rotation_limit = float(max_rotation_error_deg)
    if (not math.isfinite(position_limit) or not math.isfinite(rotation_limit)
            or position_limit <= 0 or rotation_limit <= 0):
        raise ValueError("Knowledge 近似 IK 误差预算必须为正的有限数")
    position = float(result.position_error_m)
    rotation = float(result.rotation_error_deg)
    strict = bool(result.reachable)
    feasible = getattr(result, "constraint_feasible", None)
    backend = getattr(result, "backend", "mujoco")
    acceptable = (result.joint_positions is not None and
                  (feasible is True if backend == "curobo" else True) and
                  math.isfinite(position) and math.isfinite(rotation) and
                  position <= position_limit and rotation <= rotation_limit)
    return {
        "acceptable": bool(acceptable),
        "approximate": bool(acceptable and not strict),
        "strict_ik_success": strict,
        "constraint_feasible": feasible,
        "position_error_m": position,
        "rotation_error_deg": rotation,
        "max_position_error_m": position_limit,
        "max_rotation_error_deg": rotation_limit,
        "source": "curobo_ik_self_collision_no_scene" if backend == "curobo"
                  else "pose_ik_only_no_collision",
    }


__all__ = ["assess_pose_ik"]
