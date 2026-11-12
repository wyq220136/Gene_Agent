"""把已确认的微波炉 Knowledge 路径编排为可检查的 Skill 阶段。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np

from gene_agent.skills.contracts import CheckResult


@dataclass(frozen=True)
class DoorPhase:
    phase_id: str
    skill_id: str
    name: str
    tcp_pose_world: np.ndarray
    door_angle_rad: float
    gripper_width_m: float
    motion_kind: str
    contact_policy: str


def door_skill_sequence(part: str) -> tuple[str, ...]:
    """独立门子任务的顺序；默认 TCP 撤离由 Task 另行编排。"""
    if part not in {"open", "close"}:
        raise ValueError("门子任务只支持 open 或 close")
    return ("handle_pull", "push_open") if part == "open" else ("push_close",)


def door_skill_angle_range(
    skill_id: str, config: Mapping, observed_angle_rad: float | None,
) -> tuple[float, float]:
    """以更新后的实际门角生成推门起点，拉门目标与该观测互不覆盖。"""
    if observed_angle_rad is None or not math.isfinite(float(observed_angle_rad)):
        raise ValueError("生成门阶段需要有限的实际门角")
    actual = float(observed_angle_rad)
    if skill_id == "handle_pull":
        if abs(actual) > math.radians(1.):
            raise ValueError("抓把手拉门要求当前门接近关闭；请核对前置场景")
        return 0., float(config["pull_angle_deg"])
    if skill_id == "push_open":
        return float(np.rad2deg(actual)), float(config["push_angle_deg"])
    if skill_id == "push_close":
        return float(np.rad2deg(actual)), 0.
    raise ValueError(f"未知门 Skill：{skill_id}")


def door_angle_check(
    actual_rad: float | None, *, target_rad: float, tolerance_rad: float,
    source: str, predicted: bool = False,
) -> CheckResult:
    """只判断观测角；接触、稳定时间与碰撞仍由各后端提供独立证据。"""
    if (not math.isfinite(target_rad) or not math.isfinite(tolerance_rad)
            or tolerance_rad <= 0):
        raise ValueError("门角目标必须有限，容差必须为正有限数")
    evidence = {"target_rad": target_rad, "allowed_rad": tolerance_rad,
                "predicted": bool(predicted)}
    if predicted or actual_rad is None or not math.isfinite(float(actual_rad)):
        return CheckResult("unknown", source, evidence, "缺少实际门角观测")
    actual = float(actual_rad)
    error = abs(actual - target_rad)
    evidence.update(actual_rad=actual, error_rad=error)
    return CheckResult(
        "pass" if error <= tolerance_rad else "fail", source, evidence,
        "门角达到本 Skill 终点" if error <= tolerance_rad else "门角未达到本 Skill 终点",
    )


def door_observation_freshness(
    *, previous_frame, previous_time, current_frame, current_time, source: str,
) -> CheckResult:
    """执行后的门角必须来自新帧；hash 改变本身不能证明重新观测。"""
    values = (previous_frame, previous_time, current_frame, current_time)
    # 非有限采样仍需保存为 unknown；JSON 中用 null 表示该读数不可用。
    available = tuple(value is not None and math.isfinite(float(value)) for value in values)
    recorded = tuple(value if valid else None for value, valid in zip(values, available))
    fresh = all(available) and current_frame > previous_frame and current_time > previous_time
    return CheckResult(
        "pass" if fresh else "unknown", source,
        dict(zip(("previous_frame", "previous_time", "current_frame", "current_time"), recorded)),
        "执行后门观测已更新" if fresh else "执行后没有取得更新的门观测帧",
    )



def door_phases(handle=None, push_open=None, push_close=None) -> tuple[DoorPhase, ...]:
    """一个或多个门 Skill 共用阶段语义；不重算 Knowledge 的 TCP。"""
    result: list[DoorPhase] = []
    paths = []
    if handle is not None:
        paths.append(("handle_pull", handle.handle_pull_path,
                      handle.handle_pull_labels, handle.handle_pull_widths_m))
    if push_open is not None:
        paths.append(("push_open", push_open.door_path,
                      push_open.door_push_labels,
                      (.06,) * len(push_open.door_path.waypoints)))
    if push_close is not None:
        paths.append(("push_close", push_close.door_path,
                      push_close.door_push_labels,
                      (.06,) * len(push_close.door_path.waypoints)))
    if not paths:
        raise ValueError("至少提供一个门 Skill 的 Knowledge")
    for skill, path, labels, widths in paths:
        if len(path.waypoints) != len(labels) or len(labels) != len(widths):
            raise ValueError(f"{skill} Knowledge 阶段数量不一致")
        if skill == "handle_pull":
            # 旧 Pick/拉门 Skill 会在规划 pre-grasp 前显式设置夹爪开口。
            # 这是 Skill 的执行事件，不属于 Knowledge 的几何 waypoint。
            first = path.waypoints[0]
            result.append(DoorPhase(
                "handle_pull/00_open", skill, "open",
                np.asarray(first.tcp_pose_world, dtype=float).copy(),
                float(first.angle_rad), float(widths[0]), "gripper", "avoid_door",
            ))
        if skill == "push_close":
            # 旧 Task1 阶段三进入 pre-contact 前先闭合夹爪，让指尖外侧
            # 推门板。这个执行命令属于 Skill，不能混入门几何 Knowledge。
            first = path.waypoints[0]
            result.append(DoorPhase(
                "push_close/00_close", skill, "close",
                np.asarray(first.tcp_pose_world, dtype=float).copy(),
                float(first.angle_rad), 0., "gripper", "avoid_door",
            ))
        for index, (point, name, width) in enumerate(zip(path.waypoints, labels, widths, strict=True)):
            # pre-contact 和把手预抓取允许自由避障；操作接触路径维持 TCP 直线。
            motion = ("gripper" if name in {"close", "release"} else
                      "plan" if name in {"pregrasp", "pre_contact"} else "linear_tcp")
            policy = ("door_contact" if skill.startswith("push_") and
                      (name == "contact" or name.startswith(("push_", "retract_")))
                      else "handle_contact" if skill == "handle_pull" and
                      (name.startswith(("pull_", "swing_")) or name == "retreat")
                      else "avoid_door")
            stage_index = index + 1 if skill in {"handle_pull", "push_close"} else index
            if skill == "push_close":
                width = 0.
            result.append(DoorPhase(f"{skill}/{stage_index:02d}_{name}", skill, name,
                                    np.asarray(point.tcp_pose_world, dtype=float).copy(),
                                    float(point.angle_rad), float(width), motion, policy))
        if skill == "push_close" and not any(label.startswith("retract_") for label in labels):
            # 旧 Task1 关门后的第一个撤离候选：沿关闭门板外法向退出
            # 30 mm，保持接触末姿态。旧版还会按距离和局部 Y 倾角继续
            # 搜索候选；此处先恢复其首个候选，轨迹是否可行交给 planner。
            final = path.waypoints[-1]
            outward = np.asarray(
                push_close.door_candidate_sets[-1].candidates[0].surface_normal_world,
                dtype=float,
            )
            retreat = np.asarray(final.tcp_pose_world, dtype=float).copy()
            retreat[:3, 3] += .03 * outward
            result.append(DoorPhase(
                f"push_close/{len(labels) + 1:02d}_retreat", skill, "retreat",
                retreat, float(final.angle_rad), 0., "linear_tcp", "door_contact",
            ))
    return tuple(result)
