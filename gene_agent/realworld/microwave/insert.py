"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Callable

from gene_agent.realworld.checks import bind_check_to_scene, resolve_realworld_check
from gene_agent.realworld.checks import check_observed_condition as _check
from gene_agent.realworld.checks import execution_handoff_error as _handoff_error
from gene_agent.realworld.microwave.attachment import (
    bind_current_hold_check,
    same_observation_as_hold_report,
)
from gene_agent.realworld.microwave.insert_planning import (
    plan_observed_bowl_detached_retreat_stage,
    plan_observed_bowl_insert_linear,
    plan_observed_bowl_insert_prefix,
)
from gene_agent.realworld.microwave.scene import bowl_state_summary as _summary
from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.tasks.checks import CheckResult
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.library.transfer import insert as run_insert
from gene_agent.realworld.microwave.pick import _operation_state, _command_end_state


def run_observed_bowl_insert(
    *, pick_report: dict,
    observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    microwave_frame_alignment: dict, output_dir: str | Path,
    execute: bool = False,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_hold_translation_m: float, max_hold_rotation_rad: float,
    max_start_error_rad: float, max_gripper_width_error_m: float,
    check_hold: Callable[[dict], CheckResult] | None = None,
    check_insert_path: Callable[[dict, dict], CheckResult] | None = None,
    check_release: Callable[[dict], CheckResult] | None = None,
    check_place: Callable[[dict], CheckResult] | None = None,
    ask_manual: Callable[[str], str] | None = None,
    seed: int = 123, insert_depth_offset_m: float = 0.,
    retreat_rotation_local_deg=(0., 0., 0.),
    settle_duration_s: float = .5,
) -> dict:
    """运行 preplace→insert→release→detach→两段直线 retreat。

    独立入口和完整 Task 应调用同一函数。真实 SDK、现场观测及检查器
    由调用方注入；默认仅预检 preplace。旧 insert/retreat 的规划 world
    被停用，真实 insert 下发前须由检查器明确审查路径安全。
    """
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_hold_translation_m, max_hold_rotation_rad,
              max_start_error_rad, max_gripper_width_error_m)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError("现场放碗位姿、关节与夹爪容差必须为正有限数")
    if not math.isfinite(float(settle_duration_s)) or settle_duration_s < 0:
        raise ValueError("放碗等待时间必须是非负有限秒数")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {
        "schema_version": 1, "subtask": "insert_bowl",
        "execute": bool(execute), "status": "not_started",
        "insert_success": False, "first_failure": None,
        "stages": rows,
        "claim": "逐段实际状态与现场后置条件；规划直线不等于 Bowl 已进入炉内",
    }

    def save() -> dict:
        (output / "insert_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    def stop(code: str, message: str, *, status: str = "failed") -> dict:
        report.update(status=status, first_failure={"code": code, "message": message})
        return save()

    def execute_motion(name: str, planned: dict, scene: dict, *, scene_hash_key: str):
        nonlocal observed
        stage = planned["stage"]
        row = {"stage": name, "plan": {**planned, "stage": stage.to_dict()},
               "start": _summary(scene)}
        rows.append(row)
        if not planned["success"]:
            row["status"] = "failed"
            return row, scene, stop("motion_planning_failed", f"{name} 未获得可执行轨迹")
        runtime = make_runtime(planned[scene_hash_key])
        if execute:
            observed = _command_end_state({}, name)
        execution = runtime.run_motion_stage(stage, execute=execute)
        if execute:
            observed = _command_end_state(execution, name)
        row["execution"] = execution
        if not execution.get("success"):
            row["status"] = "failed"
            if execute:
                try:
                    recovery = observe_collision_scene()
                    observed = _operation_state(recovery)
                    row["observed_after_failure"] = _summary(recovery)
                except Exception as error:
                    row["recovery_observation_error"] = f"{type(error).__name__}: {error}"
            failure = execution.get("failure") or {}
            return row, scene, stop(failure.get("code", "motion_execution_failed"),
                                    failure.get("message", f"{name} 预检或执行失败"))
        row["status"] = execution["status"]
        if not execute:
            return row, scene, stop("not_executed", "首段只预检；不能沿预测终点继续",
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
            return row, after, stop("observed_state_discontinuity", error)
        save()
        return row, after, None

    try:
        scene = observe_collision_scene()
        initial_hold = None
        manual = pick_report.get("source_format") == "manual_prepared_held_bowl"
        origin_snapshot = (pick_report.get("source_collision_snapshot_sha256")
                           if manual else pick_report.get("post_lift_scene_snapshot_sha256"))
        origin_frame = (pick_report.get("source_frame_number") if manual
                        else pick_report.get("post_lift_source_frame_number"))
        origin_time = (pick_report.get("source_timestamp_ms") if manual
                       else pick_report.get("post_lift_source_timestamp_ms"))
        if origin_snapshot is None or origin_frame is None or origin_time is None:
            raise ValueError("放碗前缺少持碗来源帧，不能沿用历史夹持证据")
        same_observation = same_observation_as_hold_report(
            scene, snapshot_sha256=origin_snapshot,
            frame_number=origin_frame, timestamp_ms=origin_time,
        )
        if not same_observation:
            initial_hold = bind_current_hold_check(
                scene, _check(
                    check_hold, scene, ask_manual=ask_manual,
                    question="当前 Bowl 是否仍被夹爪稳定夹持？[y/n/?] ",
                ),
            )
            report["initial_hold_check"] = initial_hold.to_dict()
            if not initial_hold.passed:
                return stop("hold_not_confirmed", initial_hold.reason,
                            status="unknown" if initial_hold.status == "unknown" else "failed")
        observed = _operation_state(scene)
        held = targets = linear = released_scene = settled_scene = release_report = None
        def perform(name):
            nonlocal scene, observed, targets, held, linear, released_scene, settled_scene, release_report
            if name == 'preplace':
                prefix = plan_observed_bowl_insert_prefix(
                    scene, pick_report,
                    microwave_frame_alignment=microwave_frame_alignment,
                    planner_pool=planner_pool, output_dir=output / "preplace_plan",
                    current_hold_check=None if initial_hold is None else initial_hold.to_dict(),
                    seed=seed, insert_depth_offset_m=insert_depth_offset_m,
                    retreat_rotation_local_deg=retreat_rotation_local_deg,
                )
                targets = prefix["targets"]
                row, scene, stopped = execute_motion(
                    "preplace", {"success": prefix["success"],
                                 "stage": prefix["preplace_stage"],
                                 "attached_collision_snapshot_sha256": prefix[
                                     "attached_collision_snapshot_sha256"],
                                 "plan_ref": str(output / "preplace_plan/insert_plan.json")},
                    scene, scene_hash_key="attached_collision_snapshot_sha256",
                )
                if stopped is not None:
                    return {"success": False}
                return {"success": True}
            if name == 'hold_check':
                held = bind_current_hold_check(
                    scene, _check(
                        check_hold, scene, ask_manual=ask_manual,
                        question="preplace 后 Bowl 是否仍被夹爪稳定夹持？[y/n/?] ",
                    ),
                )
                report["post_preplace_hold_check"] = held.to_dict()
                if not held.passed:
                    stop("hold_not_confirmed", held.reason,
                                status="unknown" if held.status == "unknown" else "failed")
                return {"success": True, "check": held}
            if name == 'transfer_path_check':
                linear = plan_observed_bowl_insert_linear(
                    scene, targets, pick_report,
                    microwave_frame_alignment=microwave_frame_alignment,
                    hold_check=held.to_dict(), planner_pool=planner_pool,
                    output_dir=output / "insert_plan",
                    max_microwave_translation_m=max_microwave_translation_m,
                    max_microwave_rotation_rad=max_microwave_rotation_rad,
                    max_hold_translation_m=max_hold_translation_m,
                    max_hold_rotation_rad=max_hold_rotation_rad, seed=seed,
                )
                if not linear["success"]:
                    report["insert_plan"] = {**linear, "stage": linear["stage"].to_dict()}
                    return stop("insert_planning_failed", "当前实测状态无直线放入轨迹")
                if execute:
                    safety = _check(
                        check_insert_path, scene, linear, ask_manual=ask_manual,
                        question="insert 路径已检查周围物体和允许接触，确认执行？[y/n/?] ",
                    )
                    report["insert_path_check"] = safety.to_dict()
                    if not safety.passed:
                        stop("insert_path_not_approved", safety.reason,
                                    status="unknown" if safety.status == "unknown" else "failed")

                if not execute:
                    safety = CheckResult("unknown", "not_executed", {}, "路径尚未执行")
                return {"success": True, "check": safety}
            if name == 'insert':
                row, scene, stopped = execute_motion(
                    "insert", linear, scene,
                    scene_hash_key="stage_collision_snapshot_sha256",
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
                release_row = {"stage": "release", "execution": release,
                               "start": _summary(scene), "status": release["status"]}
                rows.append(release_row)
                if not release.get("success"):
                    return stop((release.get("failure") or {}).get("code", "release_failed"),
                                (release.get("failure") or {}).get("message", "松爪失败"))
                released_scene = observe_collision_scene()
                observed = _operation_state(released_scene)
                release_row["observed_after"] = _summary(released_scene)
                error = _handoff_error(
                    scene, released_scene, release,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
                if error is not None:
                    return stop("observed_state_discontinuity", error)
                return {"success": True}
            if name == 'detach':
                # 真机没有 MuJoCo weld；detach 只更新后续规划碰撞状态。
                rows.append({"stage": "detach", "status": "executed",
                             "mode": "planning_attachment_removed",
                             "source_frame_number": released_scene["provenance"]["source_frame_number"]})
                return {"success": True}
            if name == 'settle':
                observed = _command_end_state({}, name)
                time.sleep(settle_duration_s)
                settled_scene = observe_collision_scene()
                observed = _operation_state(settled_scene)
                if (int(settled_scene["provenance"]["source_frame_number"])
                        <= int(released_scene["provenance"]["source_frame_number"])
                        or float(settled_scene["provenance"]["source_timestamp_ms"])
                        <= float(released_scene["provenance"]["source_timestamp_ms"])):
                    return stop("observed_state_discontinuity",
                                "松爪等待后没有取得更新的现场观测帧")
                report["settle_duration_s"] = settle_duration_s
                return {"success": True}
            if name == 'release_check':
                released = bind_check_to_scene(settled_scene, _check(
                    check_release, settled_scene, ask_manual=ask_manual,
                    question="Bowl 已离开夹爪并稳定留在炉内？[y/n/?] ",
                ))
                report["release_postcondition"] = released.to_dict()
                if not released.passed:
                    stop("release_not_confirmed", released.reason,
                                status="unknown" if released.status == "unknown" else "failed")
                    return {'success': True, 'check': released}
                release_report = {
                    "status": "executed", "release_success": True,
                    "source_frame_number": released_scene["provenance"]["source_frame_number"],
                    "source_timestamp_ms": released_scene["provenance"]["source_timestamp_ms"],
                    "bowl_pose_world_after_release": settled_scene["provenance"][
                        "aligned_object_poses_world"]["Bowl/Bowl_0"],
                    "postcondition": released.to_dict(),
                }
                report["release_report"] = release_report
                scene = settled_scene
                return {"success": True, "check": released}
            if name in {"retreat_1", "retreat_2"}:
                index = 1 if name == "retreat_1" else 2
                planned = plan_observed_bowl_detached_retreat_stage(
                    scene, targets, release_report,
                    stage_name=name,
                    microwave_frame_alignment=microwave_frame_alignment,
                    planner_pool=planner_pool,
                    output_dir=output / f"retreat_{index}_plan",
                    max_microwave_translation_m=max_microwave_translation_m,
                    max_microwave_rotation_rad=max_microwave_rotation_rad,
                    max_bowl_translation_m=max_hold_translation_m,
                    max_bowl_rotation_rad=max_hold_rotation_rad,
                    seed=seed, scene_revision=13 + index,
                )
                row, scene, stopped = execute_motion(
                    name, planned, scene,
                    scene_hash_key="stage_collision_snapshot_sha256",
                )
                if stopped is not None:
                    return {"success": False}
                return {"success": True}
            if name == "final_check":
                placed = bind_check_to_scene(
                    scene, _check(
                        check_place, scene, ask_manual=ask_manual,
                        question="Bowl 碗口朝上、稳定处于微波炉内？[y/n/?] ",
                    ),
                )
                report["postcondition"] = placed.to_dict()
                report["insert_success"] = placed.passed
                report["status"] = "executed" if placed.passed else (
                    "unknown" if placed.status == "unknown" else "failed"
                )
                if not placed.passed:
                    report["first_failure"] = {
                        "code": "place_not_confirmed", "message": placed.reason,
                    }
                return {"success": True, "check": placed}
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
        flow = run_insert(CallbackOperationBackend(
            observe=lambda: observed, perform=guarded_perform,
            evidence_mode='realworld_observation', execution_enabled=execute, failure_details=lambda stage: report.get('first_failure')))
        report['shared_flow'] = flow.to_dict()
        if not flow.success:
            report['insert_success'] = False
            if report['first_failure'] is None:
                report.update(status='unknown', first_failure={
                    'stage': flow.failure_stage, 'code': flow.failure_code,
                    'message': '共享搬运流程拒绝继续'})
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception", "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


__all__ = ['run_observed_bowl_insert']
