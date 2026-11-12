"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from gene_agent.realworld.checks import bind_check_to_scene as _bind_check_to_scene
from gene_agent.realworld.checks import check_observed_condition as _check
from gene_agent.realworld.checks import execution_handoff_error as _handoff_error
from gene_agent.realworld.microwave.place_planning import (
    plan_observed_table_descent,
    plan_observed_table_detached_retreat,
    plan_observed_table_place_prefix,
)
from gene_agent.realworld.microwave.scene import bowl_state_summary as _summary
from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.tasks.checks import CheckResult
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.library.transfer import place as run_place
from gene_agent.realworld.microwave.pick import _operation_state, _command_end_state


def run_observed_table_place(
    *, grasp_report: dict,
    observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    table_height_alignment: dict, output_dir: str | Path,
    execute: bool = False,
    max_table_height_delta_m: float,
    max_table_translation_m: float, max_table_rotation_rad: float,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    max_hold_translation_m: float, max_hold_rotation_rad: float,
    max_start_error_rad: float, max_gripper_width_error_m: float,
    check_hold: Callable[[dict], CheckResult] | None = None,
    check_place_path: Callable[[dict, dict], CheckResult] | None = None,
    check_release: Callable[[dict], CheckResult] | None = None,
    check_table: Callable[[dict], CheckResult] | None = None,
    ask_manual: Callable[[str], str] | None = None,
    seed: int = 123, settle_duration_s: float = 1.5,
) -> dict:
    """独立桌面 Place 或完整 Task 中的同一子任务入口。

    人工准备的持碗状态可用现有 manual-held 报告进入本函数；完整
    Task 则传刚执行的取碗抓取事实。真机没有 MuJoCo weld/detach，
    release 后的物体支撑与最终留桌必须另用现场证据确认。
    """
    limits = (max_table_height_delta_m,
              max_table_translation_m, max_table_rotation_rad,
              max_bowl_translation_m, max_bowl_rotation_rad,
              max_hold_translation_m, max_hold_rotation_rad,
              max_start_error_rad, max_gripper_width_error_m)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("桌面放置位姿、关节与夹爪容差必须为正有限数")
    if not math.isfinite(settle_duration_s) or settle_duration_s < 0:
        raise ValueError("桌面松爪等待时间必须是非负有限秒数")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {
        "schema_version": 1, "subtask": "place_bowl_on_table",
        "execute": bool(execute), "status": "not_started",
        "table_place_success": False, "first_failure": None,
        "stages": rows,
        "claim": "桌面 Place 需现场释放、桌面支撑和撤离后置条件，不以规划成功替代",
    }

    def save() -> dict:
        (output / "table_place_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    def stop(name: str, code: str, message: str, *, status: str = "failed") -> dict:
        report.update(status=status, first_failure={
            "stage": name, "code": code, "message": message,
        })
        return save()

    def run_motion(name: str, planned: dict, scene: dict, *, stage_hash: str):
        nonlocal observed
        stage = planned["stage"]
        row = {"stage": name, "status": "started",
               "start": _summary(scene),
               "plan": {**planned, "stage": stage.to_dict()}}
        rows.append(row)
        if not planned["success"]:
            row["status"] = "failed"
            return scene, stop(name, "motion_planning_failed",
                               "当前现场状态没有可执行的桌面 Place 轨迹")
        runtime = make_runtime(stage_hash)
        if execute:
            observed = _command_end_state({}, name)
        execution = runtime.run_motion_stage(stage, execute=execute)
        if execute:
            observed = _command_end_state(execution, name)
        row["execution"] = execution
        if not execution.get("success"):
            row["status"] = "failed"
            failure = execution.get("failure") or {}
            return scene, stop(name, failure.get("code", "motion_failed"),
                               failure.get("message", "桌面 Place 预检或执行失败"))
        row["status"] = execution["status"]
        if not execute:
            return scene, stop(name, "not_executed",
                               "本阶段只预检，执行并重观测后才能继续",
                               status="validated_prefix")
        after = observe_collision_scene()
        observed = _operation_state(after)
        row["observed_after"] = _summary(after)
        error = _handoff_error(
            scene, after, execution,
            max_joint_error_rad=max_start_error_rad,
            max_width_error_m=max_gripper_width_error_m,
        )
        if error is not None:
            row["status"] = "failed"
            return after, stop(name, "observed_state_discontinuity", error)
        save()
        return after, None

    try:
        scene = observe_collision_scene()
        held = _bind_check_to_scene(scene, _check(
            check_hold, scene, ask_manual=ask_manual,
            question="桌面 Place 前 Bowl 是否仍被稳定夹持？[y/n/?] ",
        ))
        report["initial_hold_check"] = held.to_dict()
        if not held.passed:
            return stop("preplace", "hold_not_confirmed", held.reason,
                        status="unknown" if held.status == "unknown" else "failed")
        observed = _operation_state(scene)
        group = targets = descent = released_scene = settled_scene = release_report = None
        def perform(name):
            nonlocal scene, observed, targets, held, descent, released_scene, settled_scene, release_report, group
            if name == 'preplace':
                prefix = plan_observed_table_place_prefix(
                    scene, grasp_report, current_hold_check=held,
                    table_height_alignment=table_height_alignment,
                    planner_pool=planner_pool, output_dir=output / "prefix_plan",
                    seed=seed,
                )
                report["prefix_plan_ref"] = str(output / "prefix_plan/table_plan.json")
                if not prefix["success"]:
                    return stop("preplace", "place_prefix_planning_failed",
                                "原 Place Skill 没有完整可执行前缀")
                targets = prefix["targets"]
                planned = {"success": True, "stage": prefix["preplace_stage"],
                           "stage_collision_snapshot_sha256": prefix[
                               "stage_collision_snapshot_sha256"]}
                if execute:
                    safety = _check(
                        check_place_path, scene, planned, ask_manual=ask_manual,
                        question="preplace 携物路径已检查周围物体，确认执行？[y/n/?] ",
                    )
                    report["preplace_path_check"] = safety.to_dict()
                    if not safety.passed:
                        return stop("preplace", "place_path_not_confirmed", safety.reason,
                                    status="unknown" if safety.status == "unknown" else "failed")
                scene, stopped = run_motion(
                    "preplace", planned, scene,
                    stage_hash=planned["stage_collision_snapshot_sha256"],
                )
                if stopped is not None:
                    return {"success": False}
                return {"success": True}
            if name == 'hold_check':
                held = _bind_check_to_scene(scene, _check(
                    check_hold, scene, ask_manual=ask_manual,
                    question="preplace 后 Bowl 是否仍被稳定夹持？[y/n/?] ",
                ))
                report["post_preplace_hold_check"] = held.to_dict()
                if not held.passed:
                    stop("place", "hold_not_confirmed", held.reason,
                                status="unknown" if held.status == "unknown" else "failed")
                return {"success": True, "check": held}
            if name == 'transfer_path_check':
                descent = plan_observed_table_descent(
                    scene, targets, grasp_report, current_hold_check=held,
                    table_height_alignment=table_height_alignment,
                    planner_pool=planner_pool, output_dir=output / "descent_plan",
                    max_table_height_delta_m=max_table_height_delta_m,
                    max_hold_translation_m=max_hold_translation_m,
                    max_hold_rotation_rad=max_hold_rotation_rad,
                    seed=seed,
                )
                if not descent["success"]:
                    report["descent_plan"] = {**descent, "stage": descent["stage"].to_dict()}
                    return stop("place", "descent_planning_failed", "桌面下降无可执行轨迹")
                if execute:
                    safety = _check(
                        check_place_path, scene, descent, ask_manual=ask_manual,
                        question="place 携物下降路径已检查，确认执行？[y/n/?] ",
                    )
                    report["place_path_check"] = safety.to_dict()
                    if not safety.passed:
                        stop("place", "place_path_not_confirmed", safety.reason,
                                    status="unknown" if safety.status == "unknown" else "failed")

                if not execute:
                    safety = CheckResult("unknown", "not_executed", {}, "路径尚未执行")
                return {"success": True, "check": safety}
            if name == 'place':
                scene, stopped = run_motion(
                    "place", descent, scene,
                    stage_hash=descent["stage_collision_snapshot_sha256"],
                )
                if stopped is not None:
                    return {"success": False}
                return {"success": True}
            if name == 'release':
                group = scene["snapshot"]["active_group"]
                release_width = float(scene["robot_metadata"]["grippers"][group][
                    "width_range_m"][1])
                runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                release = runtime.run_gripper_stage(
                    "release", group=group, target_width_m=release_width,
                    planned_start=_observed_start(scene),
                    planned_start_width_m=float(scene["provenance"]["gripper_widths_m"][group]),
                    max_width_error_m=max_gripper_width_error_m,
                    planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
                    execute=execute, command_mode="release",
                )
                if execute:
                    observed = _command_end_state(release, name)
                release_row = {"stage": "release", "start": _summary(scene),
                               "execution": release, "status": release["status"]}
                rows.append(release_row)
                if not release.get("success"):
                    failure = release.get("failure") or {}
                    return stop("release", failure.get("code", "release_failed"),
                                failure.get("message", "真机松爪失败"))
                released_scene = observe_collision_scene()
                observed = _operation_state(released_scene)
                release_row["observed_after"] = _summary(released_scene)
                error = _handoff_error(
                    scene, released_scene, release,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
                if error is not None:
                    return stop("release", "observed_state_discontinuity", error)
                return {"success": True}
            if name == 'detach':
                rows.append({"stage": "detach", "status": "planning_only",
                             "claim": "真机没有 MuJoCo weld；后续规划不再携带 Bowl"})
                return {"success": True}
            if name == 'settle':
                observed = _command_end_state({}, name)
                if settle_duration_s:
                    time.sleep(settle_duration_s)
                settled_scene = observe_collision_scene()
                observed = _operation_state(settled_scene)
                old_source = released_scene["provenance"]
                new_source = settled_scene["provenance"]
                if (int(new_source["source_frame_number"])
                        <= int(old_source["source_frame_number"])
                        or float(new_source["source_timestamp_ms"])
                        <= float(old_source["source_timestamp_ms"])):
                    return stop("release", "observed_state_discontinuity",
                                "松爪等待后没有取得更新的现场观测帧")
                if (np.max(np.abs(
                        np.asarray(_observed_start(released_scene).positions_rad)
                        - np.asarray(_observed_start(settled_scene).positions_rad)
                    )) > max_start_error_rad
                        or abs(float(old_source["gripper_widths_m"][group])
                               - float(new_source["gripper_widths_m"][group]))
                        > max_gripper_width_error_m):
                    return stop("release", "observed_state_discontinuity",
                                "松爪等待期间机械臂或夹爪状态变化")
                report["settle_observation"] = _summary(settled_scene)
                return {"success": True}
            if name == 'release_check':
                released = _bind_check_to_scene(settled_scene, _check(
                    check_release, settled_scene, ask_manual=ask_manual,
                    question="Bowl 已离开夹爪并稳定落在桌面？[y/n/?] ",
                ))
                report["release_postcondition"] = released.to_dict()
                if not released.passed:
                    stop("release", "release_not_confirmed", released.reason,
                                status="unknown" if released.status == "unknown" else "failed")
                    return {'success': True, 'check': released}
                release_report = {
                    "status": "executed", "release_success": True,
                    "source_frame_number": released_scene["provenance"]["source_frame_number"],
                    "bowl_pose_world_after_release": settled_scene["provenance"][
                        "aligned_object_poses_world"]["Bowl/Bowl_0"],
                    "postcondition": released.to_dict(),
                }
                report["release_report"] = release_report

                scene = settled_scene
                return {"success": True, "check": released}
            if name == 'retreat':
                retreat = plan_observed_table_detached_retreat(
                    settled_scene, targets, release_report,
                    planner_pool=planner_pool, output_dir=output / "retreat_plan",
                    max_table_translation_m=max_table_translation_m,
                    max_table_rotation_rad=max_table_rotation_rad,
                    max_bowl_translation_m=max_bowl_translation_m,
                    max_bowl_rotation_rad=max_bowl_rotation_rad,
                    seed=seed,
                )
                scene, stopped = run_motion(
                    "retreat", retreat, settled_scene,
                    stage_hash=retreat["stage_collision_snapshot_sha256"],
                )
                if stopped is not None:
                    return {"success": False}
                return {"success": True}
            if name == "final_check":
                final = _bind_check_to_scene(
                    scene, _check(
                        check_table, scene, ask_manual=ask_manual,
                        question="撤离后 Bowl 碗口朝上且稳定留在桌面？[y/n/?] ",
                    ),
                )
                report["postcondition"] = final.to_dict()
                report["table_place_success"] = final.passed
                report["status"] = ("executed" if final.passed else
                                    "unknown" if final.status == "unknown" else "failed")
                if not final.passed:
                    report["first_failure"] = {
                        "stage": "retreat", "code": "table_place_not_confirmed",
                        "message": final.reason,
                    }
                else:
                    report["post_place_scene_snapshot_sha256"] = scene[
                        "snapshot"]["snapshot_sha256"]
                return {"success": True, "check": final}
            raise ValueError(f'不支持的搬运阶段：{name}')

        def guarded_perform(name):
            try:
                result = perform(name)
                return {'success': False} if result is report else result
            except Exception as error:
                report.update(status='failed', first_failure={
                    'stage': name, 'code': 'stage_exception',
                    'message': f'{type(error).__name__}: {error}',
                    'traceback': traceback.format_exc()})
                return {'success': False}
        flow = run_place(CallbackOperationBackend(
            observe=lambda: observed, perform=guarded_perform,
            evidence_mode='realworld_observation', execution_enabled=execute, failure_details=lambda stage: report.get('first_failure')))
        report['shared_flow'] = flow.to_dict()
        if not flow.success:
            report['table_place_success'] = False
            if report['first_failure'] is None:
                report.update(status='unknown', first_failure={
                    'stage': flow.failure_stage, 'code': flow.failure_code,
                    'message': '共享搬运流程拒绝继续'})
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception",
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


__all__ = ['run_observed_table_place']
