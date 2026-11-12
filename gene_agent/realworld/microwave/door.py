"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.motion.model import JointState, PlannerMode
from gene_agent.realworld.microwave.door_planning import (
    build_observed_door_skill_targets,
    load_planned_door_skill_stage,
    plan_observed_door_stage,
    rebase_observed_door_skill_targets,
    select_observed_door_phase_candidate,
)
from gene_agent.realworld.checks import has_realworld_success_evidence
from gene_agent.skills.library.door.sequence import (
    door_angle_check, door_observation_freshness, door_skill_sequence,
)


def _observed_start(scene: dict) -> JointState:
    group = scene["snapshot"]["active_group"]
    names = tuple(scene["robot_metadata"]["joint_groups"][group])
    values = scene["provenance"]["joint_positions_by_group_rad"][group]
    return JointState(names, tuple(values))


def _scene_state_summary(scene: dict) -> dict:
    """只记录恢复需要的实测量；详细碰撞仍由阶段快照保存。"""
    group = scene["snapshot"]["active_group"]
    source = scene["provenance"]
    return {
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "joint_state": _observed_start(scene).to_dict(),
        "gripper_width_m": float(source["gripper_widths_m"][group]),
        "door_angle_rad": float(source["door_angle_rad"]),
        "source_frame_number": source.get("source_frame_number"),
        "source_timestamp_ms": source.get("source_timestamp_ms"),
    }


def _measured_end_matches_scene(
    execution: dict, scene: dict, *, max_joint_error_rad: float,
    max_width_error_m: float,
) -> str | None:
    """下一阶段只接受新现场快照中实际读到的上阶段终态。"""
    if execution["source_format"] == "realworld_motion_stage":
        actual = execution.get("measured_end")
        if actual is None:
            return "运动阶段没有实测终点"
        current = _observed_start(scene)
        if tuple(actual["joint_names"]) != current.joint_names:
            return "执行终点与新现场快照的关节顺序不同"
        error = float(np.max(np.abs(
            np.asarray(actual["positions_rad"], dtype=float)
            - np.asarray(current.positions_rad, dtype=float)
        )))
        if error > max_joint_error_rad:
            return f"新现场关节与执行终点不一致：{error:.5f} rad"
    else:
        actual_width = execution.get("measured_end_width_m")
        if actual_width is None:
            return "夹爪阶段没有实测终点开口"
        group = scene["snapshot"]["active_group"]
        current_width = float(scene["provenance"]["gripper_widths_m"][group])
        error = abs(float(actual_width) - current_width)
        if error > max_width_error_m:
            return f"新现场夹爪开口与执行终点不一致：{error:.5f} m"
    return None


def run_observed_door_skill(
    *, skill_id: str,
    observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object],
    planner_pool: CuroboPlannerPool,
    output_dir: str | Path,
    config_path: str | Path,
    execute: bool = False,
    max_microwave_translation_m: float,
    max_microwave_rotation_rad: float,
    max_start_error_rad: float,
    max_gripper_width_error_m: float,
    max_door_angle_error_rad: float,
    seed: int = 123,
    planner_mode: PlannerMode | str | None = None,
    handle_grasp_force: float = 1.0,
) -> dict:
    """从现场状态运行一个门 Skill；未执行时只验证首个状态改变阶段。

    `observe_collision_scene` 必须在每次运动后提供新实测关节、夹爪和
    门角构成的同帧场景。`make_runtime` 只接受本阶段碰撞快照 hash；
    真实 SDK 连接、速度及安全阈值由调用方显式提供。把手拉门与推门
    是两个独立 Skill：调用方须在把手结束后重新观测，再生成推门目标。
    """
    limits = (max_start_error_rad, max_gripper_width_error_m,
              max_door_angle_error_rad)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError("现场起点、夹爪和门角容差必须为正有限数")
    force = float(handle_grasp_force)
    if not math.isfinite(force) or not 0. <= force <= 1.:
        raise ValueError("门把手抓取力必须在 0～1 范围内")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {"skill_id": skill_id, "execute": bool(execute),
              "success": False, "status": "not_started", "stages": rows,
              "first_failure": None,
              "claim": "现场逐阶段记录；规划或关节到达不等于物体操作成功"}

    def save() -> dict:
        (destination / "skill_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    try:
        scene = observe_collision_scene()
        initial_angle = scene["provenance"].get("door_angle_rad")
        if initial_angle is None or not math.isfinite(float(initial_angle)):
            report.update(status="unknown", first_failure={
                "code": "door_angle_unknown", "message": "缺少实际门角观测，不能生成门目标",
            })
            return save()
        targets = build_observed_door_skill_targets(
            scene, skill_id=skill_id, config_path=config_path,
        )
        for index in range(len(targets["phases"])):
            phase = targets["phases"][index]
            row = {"phase_id": phase.phase_id, "motion_kind": phase.motion_kind,
                   "status": "started", "scene_snapshot_sha256":
                   scene["snapshot"]["snapshot_sha256"]}
            rows.append(row)
            current_targets = rebase_observed_door_skill_targets(
                targets, scene,
                max_microwave_translation_m=max_microwave_translation_m,
                max_microwave_rotation_rad=max_microwave_rotation_rad,
            )
            current_targets, selection = select_observed_door_phase_candidate(
                scene, current_targets, phase_id=phase.phase_id,
                planner_pool=planner_pool,
                output_dir=destination / f"phase_{index:02d}" / "candidate_selection",
                scene_revision=index, seed=seed, planner_mode=planner_mode,
            )
            targets = current_targets
            phase = targets["phases"][index]
            if selection is not None:
                row["knowledge_candidate_selection"] = selection
                if not selection["success"]:
                    row["status"] = "failed"
                    report.update(status="failed", first_failure={
                        "phase_id": phase.phase_id, "code": "goal_ik_no_solution",
                        "message": "当前门角的旧 Knowledge 候选 IK 均不可达",
                        "candidate_selection": selection,
                    })
                    return save()
            if phase.motion_kind == "gripper":
                group = scene["snapshot"]["active_group"]
                runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
                row["execution"] = runtime.run_gripper_stage(
                    phase.phase_id, group=group,
                    target_width_m=phase.gripper_width_m,
                    planned_start=_observed_start(scene),
                    planned_start_width_m=float(
                        scene["provenance"]["gripper_widths_m"][group]
                    ),
                    max_width_error_m=max_gripper_width_error_m,
                    planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
                    execute=execute,
                    **({"command_mode": "grasp", "grasp_force": force}
                       if skill_id == "handle_pull" and phase.name == "close" else
                       {"command_mode": "release"}
                       if skill_id == "handle_pull" and phase.name == "release" else {}),
                )
            else:
                planned = plan_observed_door_stage(
                    scene, current_targets, phase_id=phase.phase_id,
                    planner_pool=planner_pool,
                    output_dir=destination / f"phase_{index:02d}",
                    scene_revision=index, seed=seed, planner_mode=planner_mode,
                )
                row["plan"] = planned
                if not planned["success"]:
                    row["status"] = "failed"
                    report.update(status="failed", first_failure={
                        "phase_id": phase.phase_id, "code": planned.get("failure_category"),
                        "message": planned.get("message") or "门阶段规划失败",
                    })
                    return save()
                runtime = make_runtime(planned["stage_collision_snapshot_sha256"])
                stage = load_planned_door_skill_stage(planned["skill_stage_ref"])
                row["execution"] = runtime.run_motion_stage(stage, execute=execute)
            execution = row["execution"]
            if not execution.get("success"):
                row["status"] = "failed"
                if execute:
                    # 真实命令可能已经执行一部分。停下后重新读现场，供人工
                    # 决定恢复起点；观测失败也保留原 SDK 错误，不自动回 home。
                    try:
                        row["observed_after_failure"] = _scene_state_summary(
                            observe_collision_scene()
                        )
                    except Exception as observation_error:
                        row["recovery_observation_error"] = (
                            f"{type(observation_error).__name__}: {observation_error}"
                        )
                report.update(status="failed", first_failure={
                    "phase_id": phase.phase_id,
                    **execution.get("failure", {"code": "execution_failed",
                                                "message": "阶段预检或执行失败"}),
                })
                return save()
            row["status"] = execution["status"]
            if not execute:
                # 不允许把未执行的机器人/夹爪终点当作下一阶段实际起点。
                report.update(status="validated_prefix", first_failure={
                    "phase_id": phase.phase_id, "code": "not_executed",
                    "message": "首段仅完成预检；继续规划需要先执行并重新观测",
                })
                return save()
            previous_source = scene["provenance"]
            scene = observe_collision_scene()
            current_source = scene["provenance"]
            freshness = door_observation_freshness(
                previous_frame=previous_source.get("source_frame_number"),
                previous_time=previous_source.get("source_timestamp_ms"),
                current_frame=current_source.get("source_frame_number"),
                current_time=current_source.get("source_timestamp_ms"),
                source="observed_door_frame",
            )
            row["observation_freshness"] = freshness.to_dict()
            if not freshness.passed:
                row["status"] = "unknown"
                report.update(status="unknown", first_failure={
                    "phase_id": phase.phase_id, "code": "door_observation_stale",
                    "message": freshness.reason,
                })
                return save()
            angle_check = door_angle_check(
                current_source.get("door_angle_rad"), target_rad=phase.door_angle_rad,
                tolerance_rad=max_door_angle_error_rad, source="observed_door_angle",
            )
            if angle_check.status == "unknown":
                row["status"] = "unknown"
                report.update(status="unknown", postcondition=angle_check.to_dict(),
                              first_failure={"phase_id": phase.phase_id,
                                             "code": "door_angle_unknown",
                                             "message": angle_check.reason})
                return save()
            row["observed_after"] = _scene_state_summary(scene)
            discontinuity = _measured_end_matches_scene(
                execution, scene, max_joint_error_rad=max_start_error_rad,
                max_width_error_m=max_gripper_width_error_m,
            )
            if discontinuity is not None:
                row["status"] = "failed"
                report.update(status="failed", first_failure={
                    "phase_id": phase.phase_id, "code": "observed_state_discontinuity",
                    "message": discontinuity,
                })
                return save()
            if phase.name.startswith(("pull_", "swing_", "push_")):
                observed = float(scene["provenance"]["door_angle_rad"])
                error = abs(observed - phase.door_angle_rad)
                row["door_angle_error_rad"] = error
                if error > max_door_angle_error_rad:
                    row["status"] = "failed"
                    report.update(status="failed", first_failure={
                        "phase_id": phase.phase_id, "code": "door_angle_not_reached",
                        "message": (f"实测门角误差 {error:.4f} rad 超过"
                                    f" {max_door_angle_error_rad:.4f} rad"),
                    })
                    return save()
            save()
        # 每段开始前都会检查柜体锚点，但最后一段执行后的新现场没有
        # 下一段来触发该检查。终点也必须复核，防止门角达标时柜体本身
        # 已被推走，却把这个门 Skill 误报为成功。
        try:
            final_targets = rebase_observed_door_skill_targets(
                targets, scene,
                max_microwave_translation_m=max_microwave_translation_m,
                max_microwave_rotation_rad=max_microwave_rotation_rad,
            )
        except ValueError as error:
            report.update(status="failed", first_failure={
                "phase_id": targets["phases"][-1].phase_id,
                "code": "target_anchor_changed", "message": str(error),
            })
            return save()
        report["target_rebase_final"] = final_targets["target_rebase"]
        final_angle = float(scene["provenance"]["door_angle_rad"])
        target_angle = targets["phases"][-1].door_angle_rad
        check = door_angle_check(
            final_angle, target_rad=target_angle,
            tolerance_rad=max_door_angle_error_rad, source="observed_door_angle",
        )
        report["postcondition"] = check.to_dict()
        report["success"] = check.passed
        report["status"] = "executed" if check.passed else "failed"
        if not check.passed:
            report["first_failure"] = {
                "code": "door_postcondition_failed", "message": check.reason,
            }
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception", "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


def run_observed_door_subtask(
    *, part: str,
    observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object],
    planner_pool: CuroboPlannerPool,
    output_dir: str | Path,
    config_path: str | Path,
    execute: bool = False,
    max_microwave_translation_m: float,
    max_microwave_rotation_rad: float,
    max_start_error_rad: float,
    max_gripper_width_error_m: float,
    max_door_angle_error_rad: float,
    seed: int = 123,
    planner_mode: PlannerMode | str | None = None,
    handle_grasp_force: float = 1.0,
) -> dict:
    """复用同一单 Skill 入口：抓把手拉门后，按新实测门角生成内侧推门。

    关门只从操作者准备或前一子任务留下的现场状态推门，不自动补跑
    开门。`execute=False` 只验证当前首段并停止；不会串联预测状态。
    """
    skill_order = door_skill_sequence(part)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    result = {
        "part": part, "execute": bool(execute), "success": False,
        "status": "not_started", "skills": rows, "first_failure": None,
        "claim": "现场门子任务；只在逐段执行及实测门角通过后报告成功",
    }

    def save() -> dict:
        (output / "door_subtask_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result

    for skill_id in skill_order:
        skill = run_observed_door_skill(
            skill_id=skill_id,
            observe_collision_scene=observe_collision_scene,
            make_runtime=make_runtime,
            planner_pool=planner_pool,
            output_dir=output / skill_id,
            config_path=config_path,
            execute=execute,
            max_microwave_translation_m=max_microwave_translation_m,
            max_microwave_rotation_rad=max_microwave_rotation_rad,
            max_start_error_rad=max_start_error_rad,
            max_gripper_width_error_m=max_gripper_width_error_m,
            max_door_angle_error_rad=max_door_angle_error_rad,
            seed=seed, planner_mode=planner_mode,
            handle_grasp_force=handle_grasp_force,
        )
        rows.append({"skill_id": skill_id, "success": skill["success"],
                     "status": skill["status"],
                     "result_ref": str(output / skill_id / "skill_result.json"),
                     "postcondition": skill.get("postcondition")})
        if not skill["success"]:
            result.update(status=skill["status"], first_failure={
                "skill_id": skill_id,
                **(skill.get("first_failure") or {
                    "code": "skill_incomplete", "message": "门 Skill 尚未完成",
                }),
            })
            return save()
        if (skill.get("status") != "executed"
                or not has_realworld_success_evidence(skill.get("postcondition"))):
            # 把手后置证据未知时就停止，不能先推门，再由完整 Task 拦截。
            rows[-1].update(success=False, status="unknown")
            result.update(status="unknown", first_failure={
                "skill_id": skill_id, "code": "door_postcondition_unverified",
                "message": "门 Skill 执行或实际门角后置条件未确认",
            })
            return save()
    result.update(success=True, status="executed")
    return save()


__all__ = []
