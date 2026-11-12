"""真机阶段的后置条件来源检查；不改变仿真任务的判定。"""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping

import numpy as np

from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.tasks.checks import CheckResult, resolve_unknown

# 来源属于执行边界协议。新增后端必须显式声明其现场证据名称，不能靠
# 「未出现在仿真黑名单」推断是真实测量；离线假设备测试也用这些语义名。
_REALWORLD_SUCCESS_SOURCES = frozenset({
    "manual_terminal", "vision", "camera", "observed_door_angle",
    "observed_robot_and_door", "realworld_reobserved_scene",
})


def has_realworld_success_evidence(result: Mapping | None) -> bool:
    """完整现场任务读取子任务报告时，只认可有真实来源的成功证据。"""
    if not isinstance(result, Mapping) or result.get("status") != "pass":
        return False
    source = result.get("source")
    evidence = result.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        return False
    predicted = result.get("predicted", False) or evidence.get("predicted", False)
    return isinstance(source, str) and source in _REALWORLD_SUCCESS_SOURCES and not predicted


def bind_check_to_scene(scene: dict, check: CheckResult) -> CheckResult:
    """将现场检查关联到当前观测帧，拒绝检查器返回的过期帧证据。"""
    evidence = dict(check.evidence)
    expected = {
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "source_frame_number": scene["provenance"]["source_frame_number"],
    }
    # 检查器可以省略来源字段；如果明确给出来源，就必须与本次观测相同。
    if any(key in evidence and evidence[key] != value
           for key, value in expected.items()):
        raise ValueError("现场检查证据与当前观测帧不一致")
    evidence.update(expected)
    return CheckResult(check.status, check.source, evidence, check.reason)


def resolve_realworld_check(
    result: CheckResult, *, ask: Callable[[str], str] | None = None,
    question: str = "本阶段是否达到现场目标？[y/n/?] ",
) -> CheckResult:
    """证据不明时允许人工确认；仿真来源的通过不能充当现场成功。"""
    checked = resolve_unknown(result, ask=ask, question=question)
    if checked is not result and checked.source == "manual_terminal":
        # 问题和回答由共享检查器保存；时间只属于现场人工确认，
        # 不改变 Viser/MuJoCo 的 CheckResult 输出格式。
        checked = CheckResult(
            checked.status, checked.source,
            {**checked.evidence, "answered_at_unix_ms": int(time.time() * 1000)},
            checked.reason,
        )
    if checked.passed and not has_realworld_success_evidence(checked.to_dict()):
        raise ValueError("仿真、预测或未声明的证据来源不能作为真机后置条件")
    return checked


__all__ = [
    "bind_check_to_scene", "has_realworld_success_evidence",
    "resolve_realworld_check", "execution_handoff_error",
    "check_observed_condition",
]


def execution_handoff_error(
    previous: dict, current: dict, execution: dict, *,
    max_joint_error_rad: float, max_width_error_m: float,
) -> str | None:
    """只接受执行后的新观测；预测 goal_state 不能代替实测终点。"""
    old = previous["provenance"]
    new = current["provenance"]
    if (int(new["source_frame_number"]) <= int(old["source_frame_number"])
            or float(new["source_timestamp_ms"]) <= float(old["source_timestamp_ms"])):
        return "执行后没有取得更新的现场观测帧"
    joints = _observed_start(current)
    if execution.get("source_format") == "realworld_motion_stage":
        measured = execution.get("measured_end")
        if measured is None:
            return "运动阶段没有实测终点"
        if tuple(measured["joint_names"]) != joints.joint_names:
            return "执行终点与新现场快照的关节顺序不同"
        error = float(np.max(np.abs(
            np.asarray(measured["positions_rad"], dtype=float)
            - np.asarray(joints.positions_rad, dtype=float)
        )))
        if not math.isfinite(error) or error > max_joint_error_rad:
            return f"新现场关节与执行终点不一致：{error:.5f} rad"
        group = current["snapshot"]["active_group"]
        old_width = float(old["gripper_widths_m"][group])
        new_width = float(new["gripper_widths_m"][group])
        if abs(old_width - new_width) > max_width_error_m:
            return "运动阶段夹爪开口意外变化"
        return None
    if execution.get("source_format") != "realworld_gripper_stage":
        return "阶段执行记录类型未知，不能交接状态"
    measured_width = execution.get("measured_end_width_m")
    if measured_width is None:
        return "夹爪阶段没有实测终点开口"
    group = current["snapshot"]["active_group"]
    current_width = float(new["gripper_widths_m"][group])
    width_error = abs(float(measured_width) - current_width)
    if not math.isfinite(width_error) or width_error > max_width_error_m:
        return f"新现场夹爪开口与执行终点不一致：{width_error:.5f} m"
    measured_start = execution.get("measured_joint_start")
    if measured_start is None or tuple(measured_start["joint_names"]) != joints.joint_names:
        return "夹爪阶段缺少可核对的实测关节起点"
    joint_error = float(np.max(np.abs(
        np.asarray(measured_start["positions_rad"], dtype=float)
        - np.asarray(joints.positions_rad, dtype=float)
    )))
    if not math.isfinite(joint_error) or joint_error > max_joint_error_rad:
        return f"夹爪阶段机械臂关节意外变化：{joint_error:.5f} rad"
    return None


def check_observed_condition(
    callback: Callable | None, *args, ask_manual: Callable[[str], str] | None,
    question: str,
) -> CheckResult:
    outcome = (CheckResult("unknown", "not_configured", {}, "未配置现场检查")
               if callback is None else callback(*args))
    if not isinstance(outcome, CheckResult):
        raise TypeError("现场子任务检查器必须返回 CheckResult")
    return resolve_realworld_check(outcome, ask=ask_manual, question=question)
