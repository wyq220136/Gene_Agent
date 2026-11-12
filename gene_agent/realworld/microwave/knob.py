"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from gene_agent.knowledge.geometry.transforms import pose_delta as _pose_delta
from gene_agent.realworld.checks import bind_check_to_scene, resolve_realworld_check
from gene_agent.realworld.checks import execution_handoff_error as _handoff_error
from gene_agent.realworld.microwave.knob_planning import (
    build_observed_knob_targets,
    plan_observed_knob_motion_stage,
    select_observed_knob_grasp,
)
from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.tasks.checks import CheckResult
from gene_agent.skills.library.knob import run as run_knob
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.realworld.microwave.pick import _operation_state, _command_end_state


def _summary(scene: dict) -> dict:
    source = scene["provenance"]
    group = scene["snapshot"]["active_group"]
    return {
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "joint_state": _observed_start(scene).to_dict(),
        "gripper_width_m": float(source["gripper_widths_m"][group]),
        "knob_angle_rad": float(source["knob_angle_rad"]),
        "microwave_pose_world": source["aligned_object_poses_world"][
            "Microwave/Microwave_0"],
    }


def _check(
    callback: Callable | None, *args,
    ask_manual: Callable[[str], str] | None, question: str,
) -> CheckResult:
    outcome = (CheckResult("unknown", "not_configured", {}, "未配置现场检查")
               if callback is None else callback(*args))
    if not isinstance(outcome, CheckResult):
        raise TypeError("旋钮现场检查器必须返回 CheckResult")
    checked = resolve_realworld_check(outcome, ask=ask_manual, question=question)
    return checked


def _wait_handoff_error(
    previous: dict, current: dict, *, max_joint_error_rad: float,
    max_width_error_m: float,
) -> str | None:
    old, new = previous["provenance"], current["provenance"]
    if (int(new["source_frame_number"]) <= int(old["source_frame_number"])
            or float(new["source_timestamp_ms"]) <= float(old["source_timestamp_ms"])):
        return "等待后没有取得更新的现场观测帧"
    before, after = _observed_start(previous), _observed_start(current)
    if before.joint_names != after.joint_names:
        return "等待前后机器人关节顺序不一致"
    if max(abs(a - b) for a, b in zip(before.positions_rad, after.positions_rad)) > max_joint_error_rad:
        return "等待时机械臂关节意外变化"
    group = previous["snapshot"]["active_group"]
    if abs(float(old["gripper_widths_m"][group])
           - float(new["gripper_widths_m"][group])) > max_width_error_m:
        return "等待时夹爪开口意外变化"
    return None


def run_observed_knob_skill(
    *, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    output_dir: str | Path, task_path: str | Path | None = None,
    execute: bool = False,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_knob_angle_error_rad: float, max_start_error_rad: float,
    max_gripper_width_error_m: float,
    check_grasp: Callable[[dict], CheckResult] | None = None,
    check_rotation: Callable[[dict], CheckResult] | None = None,
    check_last_joint_path: Callable[[dict, object], CheckResult] | None = None,
    ask_manual: Callable[[str], str] | None = None,
    seed: int = 123, last_joint_speed_scale: float = 0.1,
    grasp_force: float = 0.5,
) -> dict:
    """执行 open→候选选择→接近→夹持→逐段转动→释放→撤离。

    显式使用旧六候选 profile 时直接旋转末关节；此段不做 cuRobo 场景碰撞规划，
    必须先明确检查接触路径。非零 tip X profile 继续走 TCP 直线段。
    每个转动段都检查实测旋钮角；最终还需独立现场检查来源。
    """
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_knob_angle_error_rad, max_start_error_rad,
              max_gripper_width_error_m, last_joint_speed_scale)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("旋钮现场容差必须为正有限值")
    force = float(grasp_force)
    if not math.isfinite(force) or not 0. <= force <= 1.:
        raise ValueError("旋钮抓取力必须在 0～1 范围内")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {
        "schema_version": 1, "skill_id": "rotate_knob_0",
        "execute": bool(execute), "status": "not_started",
        "rotation_success": False, "stages": rows, "first_failure": None,
        "claim": "逐段实测机器人与旋钮角；不使用 MuJoCo 运动学旋钮提交",
    }

    def save() -> dict:
        (output / "skill_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    def stop(code: str, message: str, *, status: str = "failed") -> dict:
        report.update(status=status, first_failure={"code": code, "message": message})
        save()
        return {'success': False}

    try:
        scene = observe_collision_scene()
        options = {} if task_path is None else {"task_path": task_path}
        initial = build_observed_knob_targets(scene, **options)
        report["task_config_sha256"] = initial["task_config_sha256"]
        report["grasp_profile_sha256"] = initial["grasp_profile_sha256"]
        group = scene["snapshot"]["active_group"]
        open_width = initial["settings"].open_width_m
        selection = None
        observed = _operation_state(scene)

        def perform(name):
            nonlocal scene, selection, observed
            if name == 'open':
                runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                execution = runtime.run_gripper_stage(
                    "open", group=group, target_width_m=open_width,
                    planned_start=_observed_start(scene),
                    planned_start_width_m=float(scene["provenance"]["gripper_widths_m"][group]),
                    max_width_error_m=max_gripper_width_error_m,
                    planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
                    execute=execute,
                )
                if execute:
                    observed = _command_end_state(execution, name)
                rows.append({"stage": "open", "start": _summary(scene),
                             "execution": execution, "status": execution["status"]})
                if not execution.get("success"):
                    failure = execution.get("failure") or {}
                    return stop(failure.get("code", "open_failed"),
                                failure.get("message", "旋钮开爪失败"))
                if not execute:
                    return stop("not_executed", "只验证开爪；不能沿预测状态规划后续阶段",
                                status="validated_prefix")
                next_scene = observe_collision_scene()
                observed = _operation_state(next_scene)
                rows[-1]["observed_after"] = _summary(next_scene)
                handoff = _handoff_error(
                    scene, next_scene, execution,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
                if handoff:
                    return stop("observed_state_discontinuity", handoff)
                scene = next_scene
                selection = select_observed_knob_grasp(
                    scene, planner_pool=planner_pool,
                    output_dir=output / "selection", task_path=task_path, seed=seed,
                )
                report["selection_ref"] = str(output / "selection/selection.json")
                report["selected_candidate_id"] = selection.get("selected_candidate_id")
                if not selection["success"]:
                    return stop("grasp_selection_failed", selection.get("failure_reason")
                                or "旋钮 grasp/pregrasp 候选不可达")
                report["stage_contract"] = [stage.name for stage in selection["targets"]]
                return {'success': True}
            anchor = selection['source']['microwave_pose_world']
            if name == 'grasp_check':
                check = bind_check_to_scene(
                    scene, _check(
                        check_grasp, scene, ask_manual=ask_manual,
                        question="夹爪是否可靠接触并夹住旋钮？[y/n/?] ",
                    ),
                )
                report["grasp_postcondition"] = check.to_dict()
                if not check.passed:
                    stop("knob_grasp_not_confirmed", check.reason,
                                status="unknown" if check.status == "unknown" else "failed")

                return {'success': True, 'check': check}
            if name == 'rotation_check':
                # 每段开始前会复核微波炉本体锚点，但最后一次撤离后没有下一段。
                # 终点仍须使用刚采到的现场 pose 检查，不能只凭旋钮角达标。
                final_microwave_delta = _pose_delta(
                    anchor,
                    scene["provenance"]["aligned_object_poses_world"]["Microwave/Microwave_0"],
                    field="旋钮操作终点微波炉 pose",
                )
                report["final_microwave_anchor_error_m_rad"] = final_microwave_delta
                if (final_microwave_delta[0] > max_microwave_translation_m
                        or final_microwave_delta[1] > max_microwave_rotation_rad):
                    return stop("microwave_anchor_changed", "旋钮撤离后微波炉本体 pose 已改变")
                final_angle = float(scene["provenance"]["knob_angle_rad"])
                target_angle = float(selection["source"]["target_angle_rad"])
                if abs(final_angle - target_angle) > max_knob_angle_error_rad:
                    return stop("knob_angle_not_reached", "完成撤离后旋钮未保持目标角")
                check = bind_check_to_scene(
                    scene, _check(
                        check_rotation, scene, ask_manual=ask_manual,
                        question="旋钮是否由实际接触带动并保持目标角？[y/n/?] ",
                    ),
                )
                report["rotation_postcondition"] = check.to_dict()
                report["rotation_success"] = check.passed
                report["status"] = "executed" if check.passed else (
                    "unknown" if check.status == "unknown" else "failed"
                )
                if not check.passed:
                    report["first_failure"] = {
                        "code": "rotation_not_confirmed", "message": check.reason,
                    }

                return {'success': True, 'check': check}
            index, stage_target = next((i, t) for i, t in enumerate(selection['targets'])
                                       if t.name == name)
            row = {"stage": name, "start": _summary(scene), "status": "started"}
            rows.append(row)
            microwave_delta = _pose_delta(
                anchor,
                scene["provenance"]["aligned_object_poses_world"]["Microwave/Microwave_0"],
                field="微波炉现场 pose",
            )
            if (microwave_delta[0] > max_microwave_translation_m
                    or microwave_delta[1] > max_microwave_rotation_rad):
                row["status"] = "failed"
                return stop("microwave_anchor_changed", "旋钮操作期间微波炉 pose 已改变")
            if (abs(float(scene["provenance"]["knob_angle_rad"])
                    - stage_target.start_angle_rad) > max_knob_angle_error_rad):
                row["status"] = "failed"
                return stop("knob_start_angle_changed", "当前旋钮角与阶段起点不一致")
            if stage_target.action == "wait":
                observed = _command_end_state({}, name)
                time.sleep(stage_target.settle_time_s)
                next_scene = observe_collision_scene()
                observed = _operation_state(next_scene)
                row["execution"] = {
                    "source_format": "realworld_wait_stage", "status": "executed",
                    "duration_s": stage_target.settle_time_s,
                    "claim": "只等待并重新读取现场状态；不声明接触成功",
                }
                handoff = _wait_handoff_error(
                    scene, next_scene, max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
            elif stage_target.action.startswith("gripper_"):
                runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                execution = runtime.run_gripper_stage(
                    name, group=group,
                    target_width_m=stage_target.gripper_width_m,
                    planned_start=_observed_start(scene),
                    planned_start_width_m=float(
                        scene["provenance"]["gripper_widths_m"][group]),
                    max_width_error_m=max_gripper_width_error_m,
                    planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
                    execute=True,
                    **({"command_mode": "grasp", "grasp_force": force}
                       if stage_target.action == "gripper_grasp" else
                       {"command_mode": "release"}
                       if stage_target.action == "gripper_release" else {}),
                )
                observed = _command_end_state(execution, name)
                row["execution"] = execution
                if not execution.get("success"):
                    row["status"] = "failed"
                    failure = execution.get("failure") or {}
                    return stop(failure.get("code", "gripper_execution_failed"),
                                failure.get("message", f"{name} 夹爪命令失败"))
                next_scene = observe_collision_scene()
                observed = _operation_state(next_scene)
                handoff = _handoff_error(
                    scene, next_scene, execution,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
            elif stage_target.action == "last_joint_rotation":
                # 旧版直接关节命令不经过场景碰撞规划，因此必须由现场
                # 路径检查器明确审查旋钮及周边；证据未知时不下发命令。
                safety = _check(
                    check_last_joint_path, scene, stage_target,
                    ask_manual=ask_manual,
                    question="末关节旋转路径与接触已检查，确认执行？[y/n/?] ",
                )
                report["last_joint_path_check"] = safety.to_dict()
                if not safety.passed:
                    row["status"] = "failed"
                    return stop("last_joint_path_not_approved", safety.reason,
                                status="unknown" if safety.status == "unknown" else "failed")
                if stage_target.joint_delta_rad is None:
                    row["status"] = "failed"
                    return stop("last_joint_delta_missing", "旋钮末关节阶段缺少角度增量")
                runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                execution = runtime.run_last_joint_stage(
                    name, group=group,
                    joint_delta_rad=stage_target.joint_delta_rad,
                    planned_start=_observed_start(scene),
                    planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
                    speed_scale=last_joint_speed_scale, execute=True,
                )
                observed = _command_end_state(execution, name)
                row["execution"] = execution
                if not execution.get("success"):
                    row["status"] = "failed"
                    failure = execution.get("failure") or {}
                    return stop(failure.get("code", "last_joint_execution_failed"),
                                failure.get("message", "末关节旋钮动作失败"))
                next_scene = observe_collision_scene()
                observed = _operation_state(next_scene)
                handoff = _handoff_error(
                    scene, next_scene, execution,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
            else:
                planned = plan_observed_knob_motion_stage(
                    scene, selection, stage_name=name,
                    planner_pool=planner_pool,
                    output_dir=output / f"stage_{index:02d}_{name}",
                    max_microwave_translation_m=max_microwave_translation_m,
                    max_microwave_rotation_rad=max_microwave_rotation_rad,
                    max_knob_angle_error_rad=max_knob_angle_error_rad,
                    seed=seed, scene_revision=index,
                )
                row["plan"] = {**planned, "stage": planned["stage"].to_dict()}
                if not planned["success"]:
                    row["status"] = "failed"
                    return stop("motion_planning_failed", f"{name} 没有可执行轨迹")
                runtime = make_runtime(planned["stage_collision_snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                execution = runtime.run_motion_stage(planned["stage"], execute=True)
                observed = _command_end_state(execution, name)
                row["execution"] = execution
                if not execution.get("success"):
                    row["status"] = "failed"
                    failure = execution.get("failure") or {}
                    return stop(failure.get("code", "motion_execution_failed"),
                                failure.get("message", f"{name} 真机轨迹失败"))
                next_scene = observe_collision_scene()
                observed = _operation_state(next_scene)
                handoff = _handoff_error(
                    scene, next_scene, execution,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
            row["observed_after"] = _summary(next_scene)
            if handoff:
                row["status"] = "failed"
                return stop("observed_state_discontinuity", handoff)
            actual_angle = float(next_scene["provenance"]["knob_angle_rad"])
            error = abs(actual_angle - stage_target.end_angle_rad)
            row["knob_angle_error_rad"] = error
            if error > max_knob_angle_error_rad:
                row["status"] = "failed"
                return stop("knob_angle_not_reached",
                            f"{name} 后实测旋钮角误差 {error:.4f} rad 超限")
            row["status"] = "executed"
            scene = next_scene
            save()
            return {'success': True}

        def guarded_perform(name):
            nonlocal observed
            try:
                return perform(name)
            except Exception as error:
                failure_traceback = traceback.format_exc()
                stop('stage_exception', f'{type(error).__name__}: {error}')
                report['first_failure'].update(stage=name, traceback=failure_traceback)
                # 发令或等待后无法返回终态时，重新采集实际状态；仍失败则保留 unavailable。
                if execute and observed.source == 'unavailable':
                    try:
                        observed = _operation_state(observe_collision_scene())
                    except Exception as observation_error:
                        report['first_failure']['recovery_observation_error'] = (
                            f'{type(observation_error).__name__}: {observation_error}')
                return {'success': False}

        flow = run_knob(CallbackOperationBackend(
            observe=lambda: observed, perform=guarded_perform,
            evidence_mode='realworld_observation', execution_enabled=execute, failure_details=lambda stage: report.get('first_failure')),
            target_names=lambda: [target.name for target in selection['targets']])
        report['shared_flow'] = flow.to_dict()
        if not flow.success:
            report['rotation_success'] = False
            if report['first_failure'] is None:
                report.update(status='unknown', first_failure={
                    'code': flow.failure_code, 'stage': flow.failure_stage,
                    'message': '共享旋钮流程拒绝继续'})
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception", "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


__all__ = ['run_observed_knob_skill']
