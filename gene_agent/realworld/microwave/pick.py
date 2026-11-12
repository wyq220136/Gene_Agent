"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.model import JointState
from gene_agent.realworld.checks import bind_check_to_scene, resolve_realworld_check
from gene_agent.realworld.checks import execution_handoff_error as _handoff_error
from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment
from gene_agent.realworld.microwave.pick_planning import (
    plan_observed_bowl_grasp_stages,
    plan_observed_bowl_lift,
    plan_observed_bowl_motion_stage,
)
from gene_agent.realworld.microwave.scene import bowl_state_summary as _summary
from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.skills import load_grasp_settings
from gene_agent.skills.library.pick.skill import run_pick_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.skills.library.pick.model import GRASP_STAGE_ORDER
from gene_agent.tasks.checks import CheckResult

def _operation_state(scene):
    source = scene.get('provenance', {})
    return OperationState(f"{scene['snapshot']['snapshot_sha256']}:{source.get('source_frame_number')}",
                          'realworld_observation', source)


def _command_end_state(execution, name):
    """优先保存SDK实测终点；未知终态不能退回到发令前状态。"""
    measured = execution.get('measured_end')
    if measured is not None:
        return OperationState(f'sdk:{name}', 'realworld_sdk', measured)
    width = execution.get('measured_end_width_m')
    if width is not None and math.isfinite(float(width)):
        return OperationState(f'sdk_partial:{name}', 'realworld_sdk_partial',
                              {'gripper_width_m': float(width), 'joint_state_unknown': True})
    return OperationState(f'unavailable:{name}', 'unavailable',
                          {'state_unavailable': True, 'reason': '等待执行后重新观测'})


def _run_observed_pick_flow(report, *, observe, perform, execute, **options):
    def guarded_perform(name):
        try:
            return perform(name)
        except Exception as error:
            report.update(status='failed', first_failure={'code': 'stage_exception',
                'stage': name, 'message': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc()})
            return {'success': False}
    flow = run_pick_flow(CallbackOperationBackend(observe=observe, perform=guarded_perform,
        evidence_mode='realworld_observation', execution_enabled=execute, failure_details=lambda stage: report.get('first_failure')), **options)
    report['shared_flow'] = flow.to_dict()
    return flow



def run_observed_bowl_grasp(
    *, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    output_dir: str | Path, execute: bool = False,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    max_start_error_rad: float, max_gripper_width_error_m: float,
    check_grasp: Callable[[dict], CheckResult] | None = None,
    ask_manual: Callable[[str], str] | None = None,
    seed: int = 123, grasp_force: float = 0.5,
) -> dict:
    """按原 PickSkill 抓取前缀运行；夹持成功必须另有现场证据。

    独立调试和完整 Task 均调用此函数，前置场景由调用方准备。默认
    `execute=False` 只预检 open 并停止；执行模式每段后读新观测。
    这里只完成 grasp，不宣称已抬升碗或完成整个 pick。
    """
    limits = (max_start_error_rad, max_gripper_width_error_m,
              max_bowl_translation_m, max_bowl_rotation_rad)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("现场关节、夹爪与 Bowl pose 容差必须为正有限数")
    force = float(grasp_force)
    if not math.isfinite(force) or not 0. <= force <= 1.:
        raise ValueError("现场抓碗夹取力必须在 0～1 范围内")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {
        "schema_version": 1, "skill_id": "grasp_bowl",
        "execute": bool(execute), "status": "not_started",
        "grasp_success": False, "stages": rows,
        "first_failure": None,
        "claim": "只验证现场抓取与夹持；未规划或执行 lift/hold",
    }

    def save() -> dict:
        (output / "grasp_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    try:
        scene = observe_collision_scene()
        selection = plan_observed_bowl_grasp_stages(
            scene, output_dir=output / "selection", seed=seed,
        )
        report["selection_ref"] = str(output / "selection/selection.json")
        report["selected_candidate_id"] = selection.get("selected_candidate_id")
        if not selection.get("grasp_prefix_planned"):
            report.update(status="failed", first_failure={
                "code": "grasp_selection_failed",
                "message": "原 PickSkill 没有生成完整的抓取前缀",
            })
            return save()
        plan_rows = {row["name"]: row for row in selection["plan"]["stages"]}
        observed = _operation_state(scene)
        def perform(name):
            nonlocal scene, observed
            if name == 'grasp_check':
                check = (CheckResult(
                    "unknown", "not_configured", {}, "未配置真机 Bowl 夹持判定"
                ) if check_grasp is None else check_grasp(scene))
                if not isinstance(check, CheckResult):
                    raise TypeError("夹持检查器必须返回 CheckResult")
                check = bind_check_to_scene(
                    scene, resolve_realworld_check(
                        check, ask=ask_manual, question="碗是否已被夹爪稳定夹住？[y/n/?] ",
                    ),
                )
                report["physical_grasp_check"] = check.to_dict()
                report["grasp_success"] = check.passed
                report["status"] = "executed" if check.passed else (
                    "unknown" if check.status == "unknown" else "failed"
                )
                if not check.passed:
                    report["first_failure"] = {
                        "code": "grasp_not_confirmed", "message": check.reason,
                    }
                return {'success': True, 'check': check}
            index = GRASP_STAGE_ORDER.index(name)
            row = {"stage": name, "status": "started",
                   "scene_snapshot_sha256": scene["snapshot"]["snapshot_sha256"]}
            rows.append(row)
            if name in {"open", "close"}:
                group = scene["snapshot"]["active_group"]
                target_width = float(plan_rows[name]["details"]["width_m"])
                runtime = make_runtime(scene["snapshot"]["snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                execution = runtime.run_gripper_stage(
                    name, group=group, target_width_m=target_width,
                    planned_start=_observed_start(scene),
                    planned_start_width_m=float(
                        scene["provenance"]["gripper_widths_m"][group]
                    ),
                    max_width_error_m=max_gripper_width_error_m,
                    planned_scene_snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
                    execute=execute,
                    **({"command_mode": "grasp", "grasp_force": force}
                       if name == "close" else {}),
                )
            else:
                planned = plan_observed_bowl_motion_stage(
                    scene, selection, stage_name=name,
                    planner_pool=planner_pool,
                    output_dir=output / f"stage_{index:02d}_{name}",
                    max_bowl_translation_m=max_bowl_translation_m,
                    max_bowl_rotation_rad=max_bowl_rotation_rad,
                    max_gripper_width_error_m=max_gripper_width_error_m,
                    scene_revision=index,
                )
                row["plan"] = {**planned, "stage": planned["stage"].to_dict()}
                if not planned["success"]:
                    row["status"] = "failed"
                    report.update(status="failed", first_failure={
                        "code": planned.get("failure_code") or "motion_planning_failed",
                        "stage": name,
                        "message": ("抓碗目标接触复核发现非法碰撞"
                                    if planned.get("failure_code") == "collision_postcheck_rejected"
                                    else "抓碗运动阶段没有可执行轨迹"),
                        "contact_postcheck": planned.get("contact_postcheck"),
                    })
                    return {'success': False}
                runtime = make_runtime(planned["stage_collision_snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, name)
                execution = runtime.run_motion_stage(
                    planned["stage"], execute=execute,
                )
            row["execution"] = execution
            if execute:
                observed = _command_end_state(execution, name)
            if not execution.get("success"):
                row["status"] = "failed"
                if execute:
                    try:
                        recovery = observe_collision_scene()
                        observed = _operation_state(recovery)
                        row["observed_after_failure"] = _summary(recovery)
                    except Exception as observation_error:
                        row["recovery_observation_error"] = (
                            f"{type(observation_error).__name__}: {observation_error}"
                        )
                report.update(status="failed", first_failure={
                    "stage": name,
                    **execution.get("failure", {
                        "code": "execution_failed", "message": "抓碗阶段预检或执行失败",
                    }),
                })
                return {'success': False}
            row["status"] = execution["status"]
            if not execute:
                report.update(status="validated_prefix", first_failure={
                    "stage": name, "code": "not_executed",
                    "message": "当前阶段仅完成预检；下一段需要先执行并重新观测",
                })
                return {'success': False}
            next_scene = observe_collision_scene()
            observed = _operation_state(next_scene)
            row["observed_after"] = _summary(next_scene)
            error = _handoff_error(
                scene, next_scene, execution,
                max_joint_error_rad=max_start_error_rad,
                max_width_error_m=max_gripper_width_error_m,
            )
            if error is not None:
                row["status"] = "failed"
                report.update(status="failed", first_failure={
                    "stage": name, "code": "observed_state_discontinuity",
                    "message": error,
                })
                return {'success': False}
            scene = next_scene
            save()
            return {'success': True}
        flow = _run_observed_pick_flow(report, observe=lambda: observed, perform=perform,
                                      execute=execute, stop_after='grasp')
        if not flow.success and report['first_failure'] is None:
            report.update(status='unknown', grasp_success=False, first_failure={
                'code': flow.failure_code, 'message': '共享抓取流程拒绝继续'})
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception",
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_PATH = _ROOT / "data/task_1/skills/skill_profiles.json"


def _post_grasp_continuity_error(
    last: dict, scene: dict, *, max_joint_error_rad: float,
    max_width_error_m: float, max_bowl_translation_m: float,
    max_bowl_rotation_rad: float,
) -> str | None:
    """夹持检查后的再次观测不能偷偷改变机器人、夹爪或 Bowl。"""
    group = scene["snapshot"]["active_group"]
    current_names = tuple(scene["robot_metadata"]["joint_groups"][group])
    previous_joints = last["joint_state"]
    if tuple(previous_joints["joint_names"]) != current_names:
        return "抓取后活动臂关节顺序已变化"
    previous_q = np.asarray(previous_joints["positions_rad"], dtype=float)
    current_q = np.asarray(scene["provenance"]["joint_positions_by_group_rad"][group],
                           dtype=float)
    if previous_q.shape != current_q.shape or not np.isfinite(current_q).all():
        return "抓取后活动臂关节读数无效"
    joint_error = float(np.max(np.abs(previous_q - current_q)))
    if joint_error > max_joint_error_rad:
        return f"抓取后活动臂关节变化 {joint_error:.5f} rad"
    width = float(scene["provenance"]["gripper_widths_m"][group])
    width_error = abs(width - float(last["gripper_width_m"]))
    if not math.isfinite(width_error) or width_error > max_width_error_m:
        return f"抓取后夹爪开口变化 {width_error:.5f} m"
    old_pose = validate_pose(last["bowl_pose_world"], field_name="抓取后 Bowl pose")
    new_pose = validate_pose(
        scene["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"],
        field_name="附着前 Bowl pose",
    )
    translation = float(np.linalg.norm(new_pose[:3, 3] - old_pose[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        old_pose[:3, :3].T @ new_pose[:3, :3],
    ).as_rotvec()))
    if translation > max_bowl_translation_m or rotation > max_bowl_rotation_rad:
        return f"抓取后 Bowl pose 变化 {translation:.4f} m / {rotation:.4f} rad"
    return None


def run_observed_bowl_pick(
    *, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    output_dir: str | Path, execute: bool = False,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    max_start_error_rad: float, max_gripper_width_error_m: float,
    grasp_check: Callable[[dict], CheckResult] | None = None,
    check_pick: Callable[[dict], CheckResult] | None = None,
    ask_manual: Callable[[str], str] | None = None,
    seed: int = 123, grasp_force: float = 0.5,
) -> dict:
    """单独或由完整 Task 调用同一 Pick；每段消费新的实测状态。

    默认只预检抓取首段。`execute=True` 才经注入的 runtime 下发命令；
    夹持检查通过后才建立规划附着，lift 后等待原 profile 的 hold 时间
    并检查碗是否随夹爪抬起。没有后置证据时结果保持 unknown。
    """
    limits = (max_bowl_translation_m, max_bowl_rotation_rad,
              max_start_error_rad, max_gripper_width_error_m)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("现场 Bowl pose、关节和夹爪容差必须为正有限数")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "subtask": "pick_bowl",
        "execute": bool(execute), "status": "not_started",
        "pick_success": False, "first_failure": None,
        "claim": "抓取和直线抬升逐段实测；规划附着本身不是物理成功",
    }

    def save() -> dict:
        (output / "pick_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    try:
        grasp = run_observed_bowl_grasp(
            observe_collision_scene=observe_collision_scene,
            make_runtime=make_runtime, planner_pool=planner_pool,
            output_dir=output / "grasp", execute=execute,
            max_bowl_translation_m=max_bowl_translation_m,
            max_bowl_rotation_rad=max_bowl_rotation_rad,
            max_start_error_rad=max_start_error_rad,
            max_gripper_width_error_m=max_gripper_width_error_m,
            check_grasp=grasp_check, ask_manual=ask_manual,
            seed=seed, grasp_force=grasp_force,
        )
        report["grasp"] = grasp
        report["grasp_result_ref"] = str(output / "grasp/grasp_result.json")
        if not grasp.get("grasp_success"):
            report.update(status=grasp.get("status", "failed"),
                          first_failure=grasp.get("first_failure") or {
                              "code": "grasp_incomplete", "message": "抓取尚未完成",
                          })
            return save()
        last_scene = grasp['stages'][-1]['observed_after']
        observed = OperationState(str(last_scene['source_frame_number']), 'realworld_observation', last_scene)
        scene = selection = attachment = execution = None
        def perform(name):
            nonlocal scene, selection, attachment, execution, observed
            if name == 'grasp_check':
                evidence = grasp.get('physical_grasp_check', {})
                check = CheckResult(evidence.get('status', 'unknown'),
                    evidence.get('source', 'not_configured'), evidence.get('evidence', {}),
                    evidence.get('reason', '缺少此前抓取的夹持证据'))
                check = resolve_realworld_check(check)
                if not check.passed:
                    report.update(status='unknown' if check.status == 'unknown' else 'failed',
                        first_failure={'code': 'grasp_not_confirmed', 'message': check.reason})
                return {'success': True, 'check': check}
            if name == 'establish_hold':
                scene = observe_collision_scene()
                observed = _operation_state(scene)
                source = scene["provenance"]
                if (int(source["source_frame_number"]) <= int(last_scene["source_frame_number"])
                        or float(source["source_timestamp_ms"]) <= float(last_scene["source_timestamp_ms"])):
                    raise ValueError("Bowl grasp 后没有取得新的现场观测帧")
                continuity_error = _post_grasp_continuity_error(
                    last_scene, scene, max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                    max_bowl_translation_m=max_bowl_translation_m,
                    max_bowl_rotation_rad=max_bowl_rotation_rad,
                )
                if continuity_error is not None:
                    report.update(status="failed", first_failure={
                        "code": "post_grasp_state_changed", "message": continuity_error,
                    })
                    return {'success': False}
                report["post_grasp_scene_snapshot_sha256"] = scene["snapshot"]["snapshot_sha256"]
                selection = json.loads(Path(grasp["selection_ref"]).read_text(encoding="utf-8"))
                attachment = prepare_observed_bowl_attachment(
                    scene, grasp, output_dir=output / "attachment",
                )
                report["attachment_ref"] = str(output / "attachment/attachment.json")
                report["attachment_snapshot_sha256"] = attachment["snapshot"]["snapshot_sha256"]
                observed = _operation_state(scene)
                return {'success': True}
            if name == 'lift':
                lift = plan_observed_bowl_lift(
                    scene, selection, grasp, attachment,
                    planner_pool=planner_pool, output_dir=output / "lift",
                )
                report["lift_plan_ref"] = str(output / "lift/lift_plan.json")
                report["lift_plan"] = {**lift, "lift_stage": lift["lift_stage"].to_dict()}
                if not lift["success"]:
                    report.update(status="failed", first_failure={
                        "code": "lift_planning_failed",
                        "message": "旧 PickSkill 未找到完整可执行的 TCP 直线 lift",
                    })
                    return {'success': False}
                runtime = make_runtime(lift["attached_collision_snapshot_sha256"])
                if execute:
                    observed = _command_end_state({}, 'lift')
                execution = runtime.run_motion_stage(lift["lift_stage"], execute=execute)
                report["lift_execution"] = execution
                if execute:
                    observed = _command_end_state(execution, 'lift')
                if not execution.get("success"):
                    if execute:
                        try:
                            recovery = observe_collision_scene()
                            observed = _operation_state(recovery)
                            report["observed_after_failure"] = {
                                "collision_snapshot_sha256": recovery["snapshot"]["snapshot_sha256"],
                                "source_frame_number": recovery["provenance"]["source_frame_number"],
                                "joint_positions_by_group_rad": recovery["provenance"][
                                    "joint_positions_by_group_rad"],
                            }
                        except Exception as observation_error:
                            report["recovery_observation_error"] = (
                                f"{type(observation_error).__name__}: {observation_error}"
                            )
                    report.update(status="failed", first_failure={
                        **execution.get("failure", {"code": "lift_execution_failed",
                                                     "message": "lift 预检或执行失败"}),
                    })
                    return {'success': False}
                if not execute:
                    report.update(status="validated_prefix", first_failure={
                        "code": "not_executed", "message": "lift 仅预检，不能检查未发生的抬升",
                    })
                    return {'success': False}
                measured = execution.get('measured_end')
                if measured is None:
                    raise ValueError('lift执行成功但缺少SDK实测终态')
                observed = OperationState('lift:' + scene['snapshot']['snapshot_sha256'], 'realworld_sdk', measured)
                return {'success': True}
            if name == 'hold_check':
                settings = load_grasp_settings(_PROFILE_PATH, selection["skill_profile"])
                time.sleep(settings.hold_duration_s)
                report["hold_duration_s"] = settings.hold_duration_s
                after_lift = observe_collision_scene()
                observed = _operation_state(after_lift)
                error = _handoff_error(
                    scene, after_lift, execution,
                    max_joint_error_rad=max_start_error_rad,
                    max_width_error_m=max_gripper_width_error_m,
                )
                if error is not None:
                    report.update(status="failed", first_failure={
                        "code": "observed_state_discontinuity", "message": error,
                    })
                    return {'success': False}
                report["post_lift_scene_snapshot_sha256"] = after_lift["snapshot"]["snapshot_sha256"]
                report["post_lift_source_frame_number"] = after_lift["provenance"]["source_frame_number"]
                report["post_lift_source_timestamp_ms"] = after_lift["provenance"]["source_timestamp_ms"]
                check = (CheckResult(
                    "unknown", "not_configured", {}, "未配置 Bowl 抬升后置条件检查"
                ) if check_pick is None else check_pick(after_lift))
                if not isinstance(check, CheckResult):
                    raise TypeError("Pick 后置检查器必须返回 CheckResult")
                check = bind_check_to_scene(
                    after_lift, resolve_realworld_check(
                        check, ask=ask_manual, question="碗是否随夹爪抬起并稳定保持？[y/n/?] ",
                    ),
                )
                report["postcondition"] = check.to_dict()
                report["pick_success"] = check.passed
                report["status"] = "executed" if check.passed else (
                    "unknown" if check.status == "unknown" else "failed"
                )
                if not check.passed:
                    report["first_failure"] = {
                        "code": "pick_not_confirmed", "message": check.reason,
                    }
                observed = _operation_state(after_lift)
                return {'success': True, 'check': check}
            raise ValueError(f'不支持的Pick阶段：{name}')
        flow = _run_observed_pick_flow(report, observe=lambda: observed, perform=perform,
                                      execute=execute, resume_after_grasp=True)
        if not flow.success:
            report['pick_success'] = False
            if report['first_failure'] is None:
                report.update(status='unknown', first_failure={
                    'code': flow.failure_code, 'message': '共享Pick流程拒绝继续'})
        return save()
    except Exception as error:
        report["status"] = "failed"
        report["first_failure"] = {
            "code": "stage_exception", "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        return save()


__all__ = ['run_observed_bowl_grasp', 'run_observed_bowl_pick']
