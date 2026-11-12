"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    load_object_operation_profile,
)
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.checks import bind_check_to_scene as _bind_check_to_scene
from gene_agent.realworld.checks import check_observed_condition as _check
from gene_agent.realworld.checks import execution_handoff_error as _handoff_error
from gene_agent.realworld.microwave.extract_planning import (
    plan_observed_bowl_extract_motion_stage,
    plan_observed_bowl_extract_prefix,
    plan_observed_bowl_extract_retreat_stage,
)
from gene_agent.realworld.microwave.scene import bowl_state_summary as _summary
from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.realworld.scene import observed_tcp_pose as _observed_tcp_pose
from gene_agent.tasks.checks import CheckResult
from gene_agent.skills.library.transfer import extract as run_extract
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.realworld.microwave.pick import _operation_state, _command_end_state

_ROOT = Path(__file__).resolve().parents[3]


def run_observed_bowl_extract(
    *, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    microwave_frame_alignment: dict, output_dir: str | Path,
    execute: bool = False,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    max_hold_translation_m: float, max_hold_rotation_rad: float,
    max_start_error_rad: float, max_gripper_width_error_m: float,
    check_grasp: Callable[[dict], CheckResult] | None = None,
    check_hold: Callable[[dict], CheckResult] | None = None,
    check_contact_path: Callable[[dict, dict], CheckResult] | None = None,
    check_extracted: Callable[[dict], CheckResult] | None = None,
    ask_manual: Callable[[str], str] | None = None,
    seed: int = 123, grasp_settle_time_s: float = 1.5,
    grasp_force: float = 0.5,
) -> dict:
    """单独执行取碗，或者由完整任务直接调用；前置场景由人准备。

    保留旧 open→三段进炉→close→attach→两段携物直线撤离。
    attach 只是规划碰撞转移，真机夹持和最终取出必须有现场证据。
    `execute=False` 只预检第一段并停下，不沿预测状态规划后续动作。
    """
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_bowl_translation_m, max_bowl_rotation_rad,
              max_hold_translation_m, max_hold_rotation_rad,
              max_start_error_rad, max_gripper_width_error_m)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("取碗位姿、关节与夹爪容差必须为正有限数")
    if not math.isfinite(grasp_settle_time_s) or grasp_settle_time_s < 0:
        raise ValueError("闭爪等待时间必须是非负有限秒数")
    force = float(grasp_force)
    if not math.isfinite(force) or not 0. <= force <= 1.:
        raise ValueError("取碗抓取力必须在 0～1 范围内")
    profile = load_object_operation_profile(_ROOT / "data/task_1", "Bowl_0")
    open_width = float(profile["grasp_pregrasp_opening_width_m"])
    close_width = float(profile["grasp_opening_width_m"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {
        "schema_version": 1, "subtask": "extract_bowl",
        "execute": bool(execute), "status": "not_started",
        "extract_success": False, "first_failure": None,
        "stages": rows,
        "claim": "必须由现场夹持、携物撤离和取出后置检查共同证明取碗成功",
    }

    def save() -> dict:
        (output / "extract_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    def stop(name: str, code: str, message: str, *, status: str = "failed") -> dict:
        report.update(status=status, first_failure={
            "stage": name, "code": code, "message": message,
        })
        save()
        return {'success': False}

    def run_gripper(name: str, width: float, scene: dict):
        nonlocal observed
        row = {"stage": name, "status": "started", "start": _summary(scene)}
        rows.append(row)
        runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
        if execute:
            observed = _command_end_state({}, name)
        execution = runtime.run_gripper_stage(
            name, group="right", target_width_m=width,
            planned_start=_observed_start(scene),
            planned_start_width_m=float(scene["provenance"]["gripper_widths_m"]["right"]),
            max_width_error_m=max_gripper_width_error_m,
            planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
            execute=execute,
            **({"command_mode": "grasp", "grasp_force": force}
               if name == "close" else {}),
        )
        if execute:
            observed = _command_end_state(execution, name)
        row["execution"] = execution
        if not execution.get("success"):
            row["status"] = "failed"
            failure = execution.get("failure") or {}
            return scene, stop(name, failure.get("code", "gripper_failed"),
                               failure.get("message", "夹爪阶段失败"))
        row["status"] = execution["status"]
        if not execute:
            return scene, stop(name, "not_executed",
                               "首段只预检；执行并重新观测后才能规划下一段",
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
                               "当前现场状态没有可执行的取碗轨迹")
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
                               failure.get("message", "取碗运动预检或执行失败"))
        row["status"] = execution["status"]
        if not execute:
            return scene, stop(name, "not_executed", "本阶段只预检，不能沿预测终点继续",
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
        observed = _operation_state(scene)
        prefix = targets = grasp_report = tcp_bowl = grasp_check = None
        hold_checks = {}

        def perform(name):
            nonlocal scene, prefix, targets, grasp_report, tcp_bowl, grasp_check, observed
            if name == 'open':
                scene, stopped = run_gripper("open", open_width, scene)
                if stopped is not None:
                    return stopped
                return {'success': True}
            if name in {'pre_extract_1', 'pre_extract_2', 'grasp'}:
                index = ('pre_extract_1', 'pre_extract_2', 'grasp').index(name)
                if name == 'pre_extract_1':
                    prefix = plan_observed_bowl_extract_prefix(
                        scene, microwave_frame_alignment=microwave_frame_alignment,
                        planner_pool=planner_pool, output_dir=output / "prefix_plan",
                        seed=seed, max_gripper_width_error_m=max_gripper_width_error_m,
                    )
                    report["prefix_plan_ref"] = str(output / "prefix_plan/extract_plan.json")
                    if not prefix["success"]:
                        return stop("pre_extract_1", "extract_prefix_planning_failed",
                                    "原取碗 Skill 前缀没有完整可执行轨迹")
                    targets = prefix["targets"]
                planned = ({"success": True, "stage": prefix["pre_extract_1_stage"],
                            "stage_collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"]}
                           if index == 0 else plan_observed_bowl_extract_motion_stage(
                               scene, targets, stage_name=name,
                               microwave_frame_alignment=microwave_frame_alignment,
                               planner_pool=planner_pool,
                               output_dir=output / f"stage_{index:02d}_{name}",
                               max_microwave_translation_m=max_microwave_translation_m,
                               max_microwave_rotation_rad=max_microwave_rotation_rad,
                               max_bowl_translation_m=max_bowl_translation_m,
                               max_bowl_rotation_rad=max_bowl_rotation_rad,
                               seed=seed,
                           ))
                if index > 0 and execute:
                    path = _check(
                        check_contact_path, scene, planned, ask_manual=ask_manual,
                        question=f"{name} 忽略普通 world 障碍，确认该路径可执行？[y/n/?] ",
                    )
                    report[f"{name}_path_check"] = path.to_dict()
                    if not path.passed:
                        return stop(name, "contact_path_not_confirmed", path.reason,
                                    status="unknown" if path.status == "unknown" else "failed")
                scene, stopped = run_motion(
                    name, planned, scene,
                    stage_hash=planned["stage_collision_snapshot_sha256"],
                )
                if stopped is not None:
                    return stopped
                return {'success': True}
            if name == 'close':
                scene, stopped = run_gripper("close", close_width, scene)
                if stopped is not None:
                    return stopped
                if grasp_settle_time_s:
                    observed = _command_end_state({}, name)
                    time.sleep(grasp_settle_time_s)
                    settled = observe_collision_scene()
                    observed = _operation_state(settled)
                    old_source = scene["provenance"]
                    new_source = settled["provenance"]
                    if (int(new_source["source_frame_number"])
                            <= int(old_source["source_frame_number"])
                            or float(new_source["source_timestamp_ms"])
                            <= float(old_source["source_timestamp_ms"])):
                        return stop("close", "observed_state_discontinuity",
                                    "闭爪等待后没有取得更新的现场观测帧")
                    old_joints = np.asarray(_observed_start(scene).positions_rad)
                    new_joints = np.asarray(_observed_start(settled).positions_rad)
                    old_width = float(old_source["gripper_widths_m"]["right"])
                    new_width = float(new_source["gripper_widths_m"]["right"])
                    if (np.max(np.abs(old_joints - new_joints)) > max_start_error_rad
                            or abs(old_width - new_width) > max_gripper_width_error_m):
                        return stop("close", "observed_state_discontinuity",
                                    "闭爪等待期间机械臂或夹爪状态变化")
                    scene = settled
                    report["grasp_settle_observation"] = _summary(scene)
                return {'success': True}
            if name == 'grasp_check':
                grasp_check = _bind_check_to_scene(scene, _check(
                    check_grasp, scene, ask_manual=ask_manual,
                    question="Bowl 是否已被夹爪稳定夹住？[y/n/?] ",
                ))
                report["grasp_check"] = grasp_check.to_dict()
                if not grasp_check.passed:
                    stop("close", "grasp_not_confirmed", grasp_check.reason,
                                status="unknown" if grasp_check.status == "unknown" else "failed")
                return {'success': True, 'check': grasp_check}
            if name == 'attach':
                bowl = validate_pose(
                    scene["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"],
                    field_name="闭爪后 T_world_bowl",
                )
                tcp_bowl = validate_pose(
                    np.linalg.inv(_observed_tcp_pose(scene)) @ bowl,
                    field_name="闭爪后 T_tcp_bowl",
                )
                grasp_report = {
                    "execute": True, "status": "executed", "grasp_success": True,
                    "selected_candidate_id": "fixed_keypose_0",
                    "physical_grasp_check": grasp_check.to_dict(),
                    "source_collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
                    "source_frame_number": scene["provenance"]["source_frame_number"],
                    "T_tcp_bowl_after_close": tcp_bowl.tolist(),
                }
                report["grasp"] = grasp_report
                rows.append({"stage": "attach", "status": "planning_only",
                             "source_frame_number": scene["provenance"]["source_frame_number"],
                             "claim": "真机没有仿真 weld；附着物只进入后续规划碰撞"})
                return {'success': True}
            if name.endswith('_hold_check'):
                name = name.removesuffix('_hold_check')
                held = _bind_check_to_scene(scene, _check(
                    check_hold, scene, ask_manual=ask_manual,
                    question=f"{name} 前 Bowl 是否仍被稳定夹持？[y/n/?] ",
                ))
                report[f"{name}_hold_check"] = held.to_dict()
                if not held.passed:
                    stop(name, "hold_not_confirmed", held.reason,
                                status="unknown" if held.status == "unknown" else "failed")
                hold_checks[name] = held
                return {'success': True, 'check': held}
            if name in {'post_extract_1', 'post_extract_2'}:
                index = ('post_extract_1', 'post_extract_2').index(name) + 1
                held = hold_checks[name]
                planned = plan_observed_bowl_extract_retreat_stage(
                    scene, targets, grasp_report,
                    stage_name=name, current_hold_check=held,
                    T_tcp_bowl_after_close=tcp_bowl,
                    microwave_frame_alignment=microwave_frame_alignment,
                    planner_pool=planner_pool,
                    output_dir=output / f"retreat_{index}_plan",
                    max_microwave_translation_m=max_microwave_translation_m,
                    max_microwave_rotation_rad=max_microwave_rotation_rad,
                    max_hold_translation_m=max_hold_translation_m,
                    max_hold_rotation_rad=max_hold_rotation_rad,
                    seed=seed,
                )
                if execute:
                    path = _check(
                        check_contact_path, scene, planned, ask_manual=ask_manual,
                        question=f"{name} 携物路径已检查，确认执行？[y/n/?] ",
                    )
                    report[f"{name}_path_check"] = path.to_dict()
                    if not path.passed:
                        return stop(name, "contact_path_not_confirmed", path.reason,
                                    status="unknown" if path.status == "unknown" else "failed")
                scene, stopped = run_motion(
                    name, planned, scene,
                    stage_hash=planned["stage_collision_snapshot_sha256"],
                )
                if stopped is not None:
                    return stopped
                return {'success': True}
            if name == 'extract_check':
                extracted = _bind_check_to_scene(
                    scene, _check(
                        check_extracted, scene, ask_manual=ask_manual,
                        question="Bowl 已完整退出炉腔且仍被夹爪夹持？[y/n/?] ",
                    ),
                )
                report["postcondition"] = extracted.to_dict()
                report["extract_success"] = extracted.passed
                report["status"] = ("executed" if extracted.passed else
                                    "unknown" if extracted.status == "unknown" else "failed")
                if not extracted.passed:
                    report["first_failure"] = {
                        "stage": "post_extract_2", "code": "extract_not_confirmed",
                        "message": extracted.reason,
                    }
                else:
                    report["post_extract_scene_snapshot_sha256"] = scene[
                        "snapshot"]["snapshot_sha256"]
                    report["post_extract_source_frame_number"] = scene[
                        "provenance"]["source_frame_number"]
                return {'success': True, 'check': extracted}
            raise ValueError(f'未知取出阶段 {name}')

        def guarded_perform(name):
            nonlocal observed
            try:
                return perform(name)
            except Exception as error:
                failure_traceback = traceback.format_exc()
                stop(name, 'stage_exception', f'{type(error).__name__}: {error}')
                report['first_failure'].update(stage=name, traceback=failure_traceback)
                # 发令或等待后无法返回终态时，重新采集实际状态；仍失败则保留 unavailable。
                if execute and observed.source == 'unavailable':
                    try:
                        observed = _operation_state(observe_collision_scene())
                    except Exception as observation_error:
                        report['first_failure']['recovery_observation_error'] = (
                            f'{type(observation_error).__name__}: {observation_error}')
                return {'success': False}

        flow = run_extract(CallbackOperationBackend(
            observe=lambda: observed, perform=guarded_perform,
            evidence_mode='realworld_observation', execution_enabled=execute, failure_details=lambda stage: report.get('first_failure')))
        report['shared_flow'] = flow.to_dict()
        if not flow.success:
            report['extract_success'] = False
            if report['first_failure'] is None:
                report.update(status='unknown', first_failure={
                    'stage': flow.failure_stage, 'code': flow.failure_code,
                    'message': '共享取出流程拒绝继续'})
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception",
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


__all__ = ['run_observed_bowl_extract']
