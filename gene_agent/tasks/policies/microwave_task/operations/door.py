"""微波炉门 Knowledge 的任务参数装配，供 Viewer 和现场规划共用。

这里只把 task profile 的最终生效值交给已有 Knowledge 生成器。
抓把手、推门候选、阶段 pose 的数值算法仍在原模块，不在这里复制。
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from gene_agent.visualization.grasp_tuner.microwave_scene import build_microwave_knowledge
from gene_agent.skills.library.door.sequence import DoorPhase
from gene_agent.knowledge.operation.door_push import make_door_push_precontact


def load_push_open_profile(root: str | Path, config: Mapping, group: str) -> dict | None:
    """沿用门 Viewer 的相对路径、版本与机器人组校验。"""
    profile_ref = config.get("push_open_profile")
    if not profile_ref:
        return None
    path = Path(profile_ref)
    path = path if path.is_absolute() else Path(root).resolve() / path
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("robot_group") != group:
        raise ValueError("逐角推门 profile 版本或机器人组与当前场景不符")
    return {**profile, "source": str(path)}


def build_door_knowledge(
    root: str | Path, case, config: Mapping, knowledge_state: Mapping,
    *, kind: str, start_deg: float, target_deg: float,
):
    """在给定场景上生成与原门 Viewer 完全相同的 Knowledge 目标。"""
    if kind not in {"handle_pull", "push_open", "push_close"}:
        raise ValueError(f"未知门 Skill：{kind}")
    tune = dict(knowledge_state.get(kind, {}))
    handle = tune.get("handle", {})
    pull = tune.get("handle_pull", {})
    transition = tune.get("door_push_transition", {})
    profile = (load_push_open_profile(root, config, case.snapshot["active_group"])
               if kind == "push_open" else None)
    short_retractions = bool(config.get("legacy_short_retractions", False))
    return build_microwave_knowledge(
        root, case, handle_parameters=handle,
        handle_pull_target_angle_rad=np.deg2rad(float(config["pull_angle_deg"])),
        handle_pull_open_width_m=float(config.get("handle_open_width_m", .08)),
        handle_pull_close_width_m=float(pull.get(
            "close_width_m", config.get("handle_close_width_m", .015))),
        handle_pull_pregrasp_distance_m=float(pull.get("pregrasp_distance_m", .03)),
        handle_pull_retreat_distance_m=float(pull.get("retreat_distance_m", .015)),
        handle_pull_retreat_up_m=float(pull.get("retreat_up_m", 0.)),
        door_push_parameters=tune.get("door_push", {}),
        door_push_waypoint_profile=profile,
        door_push_start_candidate_rank=int(
            tune.get("door_push_selection", {}).get("start_candidate_rank", 0)
        ),
        door_push_mode="right_close" if kind == "push_close" else "right_open",
        door_start_angle_rad=np.deg2rad(start_deg),
        door_target_angle_rad=np.deg2rad(target_deg),
        # Task1 已验证的默认 pre-contact 沿选中门面法向退出 50 mm。
        # 仅显式逐角 profile 或用户调参状态可以覆盖这个距离。
        door_push_pre_contact_distance_m=float(transition.get(
            "pre_contact_distance_m",
            profile["precontact_offset_m"] if profile is not None else .05,
        )),
        door_push_retract_up_m=(
            float(transition.get("retract_up_m", .02)) if short_retractions else None
        ),
        door_push_retract_local_z_m=(
            float(transition.get("retract_local_z_m", .03)) if short_retractions else None
        ),
        ik_scope="none",
    )


# 阶段误差预算由 Viewer、MuJoCo 和现场规划共用；不改变名义目标。

def approximate_ik_budget_for(config: Mapping, phase: DoorPhase) -> dict | None:
    """按阶段显式覆盖优先，其次使用阶段名称对应的默认预算。"""
    explicit = config.get("approximate_ik_by_stage", {})
    if phase.phase_id in explicit:
        return explicit[phase.phase_id]
    defaults = config.get("approximate_ik_defaults", {})
    # 连续推门点 push_01、push_02……共用 push 规则。首次 contact
    # 不继承此宽松预算，以免未碰到门就被视作已接触。
    key = "push" if phase.name.startswith("push_") else phase.name
    return defaults.get(key)


def local_rotation_search_for(config: Mapping, phase: DoorPhase) -> dict | None:
    """返回 Skill 允许的局部 x/y 旋转范围；不改变 Knowledge 名义 pose。"""
    explicit = config.get("local_rotation_search_by_stage", {})
    if phase.phase_id in explicit:
        return explicit[phase.phase_id]
    defaults = config.get("local_rotation_search_defaults", {})
    if phase.name.startswith("push_"):
        return defaults.get("push")
    return defaults.get(phase.name)


# 候选顺序与目标替换共用；碰撞场景和 planner 生命周期仍由调用方管理。

def select_legacy_push_candidate(planner, joint_names, positions, candidates):
    """复现旧同步阶段二每角度的两批 IK 筛选，按候选原顺序取首解。

    旧版先送入 16 个候选，只有整批失败才扩展到该角度其余候选。
    Task1 未开启 prefer_top_down，旧 preference_score 全为零；稳定排序
    因而保留候选原顺序，不能把 TCP 的几何朝下程度当成实际偏好分数。
    这里只筛选当前角度；随后的 TCP 直线规划仍按原 Skill 规则执行。
    """
    ordered = tuple(candidates)
    if not ordered:
        raise ValueError("推门角度没有 Knowledge 候选")
    checked = 0
    for group in (ordered[:16], ordered[16:]):
        if not group:
            continue
        results = planner.solve_ik_batch(
            joint_names, positions, [candidate.tcp_pose_world for candidate in group],
            diagnose_failures=False,
        )
        if len(results) != len(group):
            raise RuntimeError("推门候选 batch IK 返回数量不符")
        checked += len(group)
        reachable = [candidate for candidate, result in zip(group, results, strict=True)
                     if result.get("success")]
        if reachable:
            # preference_rank 是同一角度候选生成时的索引，与旧稳定排序一致。
            selected = min(reachable, key=lambda candidate: candidate.preference_rank)
            return selected, {
                "candidate_id": selected.candidate_id,
                "preference_rank": selected.preference_rank,
                "checked_count": checked,
                "total_count": len(ordered),
                "source": "AffordToolAct progressive batch IK: first 16, then remaining",
            }
    return None, {
        "candidate_id": None, "preference_rank": None,
        "checked_count": checked, "total_count": len(ordered),
        "source": "AffordToolAct progressive batch IK: first 16, then remaining",
    }


def apply_selected_push_candidate(phases, knowledge, phase_index: int, chosen) -> tuple:
    """把选中接触 pose 写入当前 Skill 阶段；不修改 Knowledge 原始候选。

    pre-contact 与其紧随的 contact 必须一起更新，退让距离保持现有
    Knowledge 的数值。逐角 push 只替换当前目标，不改变阶段顺序。
    """
    result = list(phases)
    phase = result[phase_index]
    if phase.name == "pre_contact":
        waypoint_pre, waypoint_contact = knowledge.door_path.waypoints[:2]
        distance = float(np.linalg.norm(
            waypoint_pre.tcp_pose_world[:3, 3]
            - waypoint_contact.tcp_pose_world[:3, 3]
        ))
        result[phase_index] = replace(
            phase, tcp_pose_world=make_door_push_precontact(chosen, distance_m=distance),
        )
        if phase_index + 1 >= len(result):
            raise RuntimeError("旧推门 Knowledge 缺少接触阶段")
        contact = result[phase_index + 1]
        if contact.skill_id != phase.skill_id or contact.name != "contact":
            raise RuntimeError("旧推门 Knowledge 缺少接触阶段")
        result[phase_index + 1] = replace(
            contact, tcp_pose_world=chosen.tcp_pose_world.copy(),
        )
    else:
        result[phase_index] = replace(
            phase, tcp_pose_world=chosen.tcp_pose_world.copy(),
        )
    return tuple(result)
