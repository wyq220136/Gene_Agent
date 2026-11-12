"""真机完整任务与独立子任务入口；共用同一组操作函数。"""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from time import perf_counter
from pathlib import Path
from typing import Callable, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.checks import has_realworld_success_evidence
from gene_agent.realworld.microwave.door import (
    _measured_end_matches_scene,
    _scene_state_summary,
    run_observed_door_subtask,
)
from gene_agent.realworld.microwave.door_planning import (
    load_planned_door_skill_stage,
    plan_observed_door_stage,
)
from gene_agent.realworld.microwave.extract import run_observed_bowl_extract
from gene_agent.realworld.microwave.insert import run_observed_bowl_insert
from gene_agent.realworld.microwave.knob import run_observed_knob_skill
from gene_agent.realworld.microwave.pick import run_observed_bowl_pick
from gene_agent.realworld.microwave.place import run_observed_table_place
from gene_agent.realworld.scene_drift import SceneDriftLimits, bind_live_scene_guard
from gene_agent.tasks.policies.microwave_task.operations.bowl import check_bowl_access
from gene_agent.tasks.policies.microwave_task.policy import (
    default_tcp_transition, MicrowaveTaskPolicy, microwave_task_steps,
    run_microwave_task1_flow,
)
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult

_PHASE_ID = "task/open_to_close/return_default_tcp"
_BEFORE_REOPEN_PHASE_ID = "task/before_reopen/return_default_tcp"


def _home_tcp_pose(scene: dict) -> np.ndarray:
    """在私有 MuJoCo 数据里用当前机器人 profile 的 home 关节求 TCP。

    这里只做 FK，不推进动力学，也不改动现场快照。旧 Task1 的默认
    TCP 属于机器人 home frame，微波炉平移时不应跟着物体平移。
    """
    model = mujoco.MjModel.from_xml_path(str(scene["scene_xml_path"]))
    data = mujoco.MjData(model)
    qpos = np.asarray(scene["snapshot"]["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("现场 qpos 与默认 TCP FK 模型不一致")
    data.qpos[:] = qpos
    robot = scene["robot_metadata"]
    group = scene["snapshot"]["active_group"]
    for name, value in robot["home_joint_positions"].items():
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = float(value)
    mujoco.mj_forward(model, data)
    site = data.site(robot["tcp_sites"][group])
    target = np.eye(4)
    target[:3, :3] = np.asarray(site.xmat, dtype=float).reshape(3, 3)
    target[:3, 3] = np.asarray(site.xpos, dtype=float)
    return validate_pose(target, field_name="机器人 home TCP")


def run_observed_door_return_default_tcp(
    *, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    output_dir: str | Path, config_path: str | Path,
    execute: bool = False, max_start_error_rad: float,
    max_gripper_width_error_m: float, max_door_angle_error_rad: float,
    max_microwave_translation_m: float = .01,
    max_microwave_rotation_rad: float = .03,
    seed: int = 123, planner_mode=None,
    phase_id: str = _PHASE_ID,
    motion_kind: str = "linear_tcp",
    contact_policy: str = "door_contact",
) -> dict:
    """复用旧门阶段规划；第二轮开门前按旧 Task1 使用完整场景碰撞。

    默认参数保持已验收的开门后直线撤离及其 fallback。第二轮开门前
    则使用普通路径规划，不移除门碰撞；两种过渡都属于完整 Task，
    不属于独立开门 Skill。执行后读取新观测，确认机械臂到达且门角不变。
    """
    if (phase_id, motion_kind, contact_policy) not in {
        (_PHASE_ID, "linear_tcp", "door_contact"),
        (_BEFORE_REOPEN_PHASE_ID, "plan", "avoid_door"),
    }:
        raise ValueError("Task 默认 TCP 过渡阶段、路径类型与碰撞策略不匹配")
    limits = (max_start_error_rad, max_gripper_width_error_m,
              max_door_angle_error_rad, max_microwave_translation_m,
              max_microwave_rotation_rad)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("任务过渡容差必须为正有限值")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "stage_id": phase_id,
        "execute": bool(execute), "status": "not_started", "success": False,
        "first_failure": None,
        "claim": "从实测状态回默认 TCP；规划与执行均不代表门状态自动成功",
    }

    def save() -> dict:
        (output / "transition_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    def stop(code: str, message: str, *, status: str = "failed") -> dict:
        report.update(status=status, first_failure={"code": code, "message": message})
        return save()

    try:
        scene = observe_collision_scene()
        report["observed_start"] = _scene_state_summary(scene)
        config_source = Path(config_path).expanduser().resolve()
        config_bytes = config_source.read_bytes()
        config = json.loads(config_bytes)
        if config.get("schema_version") != 1:
            raise ValueError("门 Task 过渡配置版本必须为 1")
        default_tcp = _home_tcp_pose(scene)
        override = (config.get("task_transition_overrides_world") or {}).get(phase_id)
        target = (default_tcp if override is None else validate_pose(
            override, field_name=phase_id))
        source = scene["provenance"]
        group = scene["snapshot"]["active_group"]
        phase = default_tcp_transition(
            target, phase_id=phase_id,
            door_angle_rad=float(source["door_angle_rad"]),
            gripper_width_m=float(source["gripper_widths_m"][group]),
        )
        report["target_tcp_pose_world"] = target.tolist()
        report["target_source"] = ("robot_home_fk" if override is None
                                   else "task_transition_overrides_world")
        report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        targets = {
            "skill_id": "task_transition", "phases": (phase,),
            "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
            "config_path": str(config_source),
            "config_sha256": report["config_sha256"],
        }
        planned = plan_observed_door_stage(
            scene, targets, phase_id=phase_id, planner_pool=planner_pool,
            output_dir=output / "planning", scene_revision=0,
            seed=seed, planner_mode=planner_mode,
        )
        report["plan"] = planned
        if not planned["success"]:
            return stop(planned.get("failure_category") or "motion_planning_failed",
                        planned.get("message") or "开门后无法安全撤回默认 TCP")
        runtime = make_runtime(planned["stage_collision_snapshot_sha256"])
        stage = load_planned_door_skill_stage(planned["skill_stage_ref"])
        execution = runtime.run_motion_stage(stage, execute=execute)
        report["execution"] = execution
        if not execution.get("success"):
            failure = execution.get("failure") or {}
            if execute:
                try:
                    report["observed_after_failure"] = _scene_state_summary(
                        observe_collision_scene())
                except Exception as error:
                    report["recovery_observation_error"] = (
                        f"{type(error).__name__}: {error}")
            return stop(failure.get("code", "motion_execution_failed"),
                        failure.get("message", "默认 TCP 过渡执行失败"))
        if not execute:
            return stop("not_executed", "只预检本段；没有实际新状态可供下一子任务使用",
                        status="validated_prefix")
        after = observe_collision_scene()
        report["observed_after"] = _scene_state_summary(after)
        old_source, new_source = scene["provenance"], after["provenance"]
        if (int(new_source["source_frame_number"]) <= int(old_source["source_frame_number"])
                or float(new_source["source_timestamp_ms"])
                <= float(old_source["source_timestamp_ms"])):
            return stop("observed_state_discontinuity", "撤离后没有更新的现场观测帧")
        mismatch = _measured_end_matches_scene(
            execution, after, max_joint_error_rad=max_start_error_rad,
            max_width_error_m=max_gripper_width_error_m,
        )
        if mismatch:
            return stop("observed_state_discontinuity", mismatch)
        width_error = abs(float(new_source["gripper_widths_m"][group])
                          - float(old_source["gripper_widths_m"][group]))
        if width_error > max_gripper_width_error_m:
            return stop("observed_state_discontinuity", "撤离期间夹爪开口意外变化")
        angle_error = abs(float(new_source["door_angle_rad"])
                          - float(old_source["door_angle_rad"]))
        if angle_error > max_door_angle_error_rad:
            return stop("door_changed_during_transition", "撤离期间门角变化超过容差")
        before_pose = validate_pose(old_source["aligned_object_poses_world"][
            "Microwave/Microwave_0"], field_name="撤离前微波炉 pose")
        after_pose = validate_pose(new_source["aligned_object_poses_world"][
            "Microwave/Microwave_0"], field_name="撤离后微波炉 pose")
        translation = float(np.linalg.norm(after_pose[:3, 3] - before_pose[:3, 3]))
        rotation = float(np.linalg.norm(Rotation.from_matrix(
            before_pose[:3, :3].T @ after_pose[:3, :3]).as_rotvec()))
        if (translation > max_microwave_translation_m
                or rotation > max_microwave_rotation_rad):
            return stop("microwave_changed_during_transition",
                        "撤离期间微波炉本体移动超过容差")
        report["postcondition"] = {
            "status": "pass", "source": "observed_robot_and_door",
            "evidence": {
                "source_frame_number": new_source["source_frame_number"],
                "collision_snapshot_sha256": after["snapshot"]["snapshot_sha256"],
                "door_angle_error_rad": angle_error,
                "gripper_width_error_m": width_error,
                "microwave_translation_m": translation,
                "microwave_rotation_rad": rotation,
            },
        }
        if phase_id == _BEFORE_REOPEN_PHASE_ID:
            # 旧 Task1 回到第二轮开门默认 TCP 后固定等待 5 秒。
            # 只在真实执行且后置条件通过时等待，预检和失败不阻塞。
            time.sleep(5.0)
            report["settle_wait_s"] = 5.0
        report.update(status="executed", success=True)
        return save()
    except Exception as error:
        report.update(status="failed", first_failure={
            "code": "stage_exception", "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        return save()


SUBTASK_NAMES = (
    "open_door", "close_door", "pick_bowl", "insert_bowl",
    "rotate_knob", "extract_bowl", "place_bowl_on_table",
    "load_bowl", "unload_bowl",
)


def run_observed_microwave_subtask(
    *, name: str, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    output_dir: str | Path, options: Mapping,
    execute: bool = False, prepared_hold_report: Mapping | None = None,
    scene_drift_limits: SceneDriftLimits | None = None,
) -> dict:
    """由人准备前置场景，只运行所选子任务，原样返回该子任务报告。

    这只是调用入口；独立子任务与完整 Task 使用同一函数。放入和桌面
    Place 需要明确传入人工准备的持碗报告，后续仍由原子任务检查同帧
    状态。这里不自动补抓碗、开门、回默认 TCP，也不伪造已执行报告。
    """
    if name not in SUBTASK_NAMES:
        raise ValueError(f"可选真机子任务：{', '.join(SUBTASK_NAMES)}")
    if name in {'load_bowl', 'unload_bowl'}:
        if prepared_hold_report is not None:
            raise ValueError('组合操作从实际 Pick/Extract 开始，不能同时注入旧持碗报告')
        return run_observed_microwave_task1(
            observe_collision_scene=observe_collision_scene, make_runtime=make_runtime,
            planner_pool=planner_pool, output_dir=output_dir, subtask_options=options,
            execute=execute, scene_drift_limits=scene_drift_limits, selection=name)
    if scene_drift_limits is not None:
        # 同一个子任务函数仍负责目标和阶段；这里只在输入边界绑定
        # 发令前的现场重观测，不改变已有 Skill 或规划逻辑。
        observe_collision_scene, make_runtime = bind_live_scene_guard(
            observe_collision_scene, make_runtime, limits=scene_drift_limits,
        )
    reserved = {"observe_collision_scene", "make_runtime", "planner_pool",
                "output_dir", "execute", "part", "pick_report", "grasp_report",
                "scene_drift_limits"}
    overlapping = reserved.intersection(options)
    if overlapping:
        raise ValueError(f"子任务 options 不能覆盖调用边界：{sorted(overlapping)}")
    if name in {"insert_bowl", "place_bowl_on_table"} and prepared_hold_report is None:
        raise ValueError("放入或桌面 Place 前必须提供人工准备的持碗报告")
    runners = {
        "open_door": run_observed_door_subtask,
        "close_door": run_observed_door_subtask,
        "pick_bowl": run_observed_bowl_pick,
        "insert_bowl": run_observed_bowl_insert,
        "rotate_knob": run_observed_knob_skill,
        "extract_bowl": run_observed_bowl_extract,
        "place_bowl_on_table": run_observed_table_place,
    }
    kwargs = {**dict(options),
              "observe_collision_scene": observe_collision_scene,
              "make_runtime": make_runtime, "planner_pool": planner_pool,
              "output_dir": output_dir, "execute": execute}
    if name in {"open_door", "close_door"}:
        kwargs["part"] = "open" if name == "open_door" else "close"
    elif name == "insert_bowl":
        kwargs["pick_report"] = dict(prepared_hold_report)
    elif name == "place_bowl_on_table":
        kwargs["grasp_report"] = dict(prepared_hold_report)
    return runners[name](**kwargs)




def _postcondition_evidence(kind: str, outcome: dict) -> tuple[object, bool]:
    """读取子任务自己的后置判定，不凭一个布尔成功标志推进完整任务。"""
    if kind == "door":
        # 开门包含抓把手和推门两个 Skill；关门只有推门。任一 Skill 的
        # 后置检查缺失或未知，都不能把完整 Task 推进到下一动作。
        skills = outcome.get("skills")
        rows = skills if isinstance(skills, list) else []
        checked = bool(rows) and all(
            isinstance(row, dict)
            and has_realworld_success_evidence(row.get("postcondition"))
            for row in rows
        )
        return {"status": "pass" if checked else "unknown",
                "source": "door_skill_postconditions", "skills": rows}, checked
    evidence = (outcome.get("rotation_postcondition") if kind == "knob"
                else outcome.get("postcondition"))
    checked = has_realworld_success_evidence(evidence)
    return evidence, checked


def run_observed_microwave_task1(
    *, observe_collision_scene: Callable[[], dict],
    make_runtime: Callable[[str], object], planner_pool,
    output_dir: str | Path, subtask_options: Mapping[str, Mapping],
    execute: bool = False,
    scene_drift_limits: SceneDriftLimits | None = None,
    include_knob: bool = True, selection: str = "all",
) -> dict:
    """按旧 Task1 顺序调用同一批现场子任务，每段消费新的现场观测。

    `subtask_options` 分别提供 door/pick/insert/knob/extract/table 的现场
    容差、标定及后置检查器；任务级撤离复用 door 配置。完整 Task 不生成
    第二套操作目标或轨迹，也不隐式重置关节；失败或检查证据未知即停止。
    `include_knob=False` 沿用旧真机无旋钮流程；第二次开门前的 Task
    过渡仍保留。`execute=False` 只预检第一段，
    绝不把预测终点交给后续子任务。
    """
    from gene_agent.tasks.policies.microwave_task.subtasks import resolve_subtask
    if selection not in {'all', 'load_bowl', 'unload_bowl'}:
        raise ValueError('现场组合入口只接受 all、load_bowl、unload_bowl')
    resolve_subtask(selection, policy=MicrowaveTaskPolicy(include_knob=include_knob))
    if scene_drift_limits is not None:
        # 完整任务和单独子任务使用同一个现场复核绑定器；每段仍由原
        # 观察回调提供新场景，不能把上一段的成功当作下一段的现场证据。
        observe_collision_scene, make_runtime = bind_live_scene_guard(
            observe_collision_scene, make_runtime, limits=scene_drift_limits,
        )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    report = {
        "schema_version": 1, "task": "microwave_task1", "execute": bool(execute),
        "status": "not_started", "success": False, "steps": rows,
        "first_failure": None,
        "requested_scope": selection, "task_success": None, "operation_success": False,
        "preparation": "current" if selection != "all" else "full_task",
        "include_knob": bool(include_knob),
        "skipped_steps": [] if include_knob else ["rotate_knob"],
        "claim": "仅逐段现场检查通过才报告真机任务完成；预检不代表执行",
    }

    def save() -> dict:
        (output / "task_result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report

    runners = {
        "door": run_observed_door_subtask,
        "transition": run_observed_door_return_default_tcp,
        "pick": run_observed_bowl_pick,
        "insert": run_observed_bowl_insert,
        "knob": run_observed_knob_skill,
        "extract": run_observed_bowl_extract,
        "table": run_observed_table_place,
    }
    pick_result = None
    extract_result = None
    policy = MicrowaveTaskPolicy(include_knob=include_knob)
    specs = {step.name: step for step in microwave_task_steps(policy)}
    current = OperationState("task_start", "unavailable", {
        "reason": "尚未调用现场子任务；Task 不自行推断机器人状态"})

    def finish(step_id, outcome, success):
        nonlocal current
        # Task 交接的是下层执行报告的引用，不伪造另一份实测机器人状态。
        # 后续子任务仍重新观测并核对 SDK 实测终态与场景来源。
        outcome = outcome if isinstance(outcome, dict) else {}
        terminal = ((outcome.get("shared_flow") or {}).get("end_state")
                    or outcome.get("observed_after")
                    or outcome.get("observed_after_failure"))
        terminal_values = terminal.get("values", terminal) if isinstance(terminal, dict) else {}
        unknown = (not terminal or terminal.get("source") in {
            "unavailable", "realworld_sdk_partial", "prediction", "preview"}
            or terminal_values.get("state_unavailable")
            or terminal_values.get("joint_state_unknown")
            or terminal_values.get("predicted"))
        current = OperationState(f"report:{output / step_id}", "subtask_report", {
            "result_ref": rows[-1]["result_ref"], "terminal_state": terminal,
            "physical_state_known": not bool(unknown),
        })
        row = rows[-1]
        row['wall_time_s'] = perf_counter() - stage_started[0]
        if not success:
            row['failure'] = report.get('first_failure')
        save()
        return {"success": bool(success), "check": CheckResult(
            "pass" if success else "unknown", "realworld_subtask_checks",
            {"postcondition": rows[-1].get("postcondition"),
             "result_ref": rows[-1]["result_ref"]})}

    stage_started = [0.]

    def perform(step_id):
        stage_started[0] = perf_counter()
        nonlocal pick_result, extract_result
        spec = specs[step_id]
        kind, door_part = spec.kind, spec.door_part
        success_key, result_name = spec.success_key, spec.result_name
        folder = output / step_id
        row = {"step_id": step_id, "subtask": kind, "status": "started",
               "result_ref": str(folder / result_name)}
        rows.append(row)
        save()
        try:
            if selection != 'all' and kind in {'pick', 'insert', 'extract'}:
                # 使用实际现场观测；不要求真机提供 MuJoCo 内部证据。
                actual = observe_collision_scene()
                angle = actual.get('provenance', {}).get('door_angle_rad')
                if not check_bowl_access(angle, target_angle_rad=np.pi/2).passed:
                    report.update(status='failed', first_failure={
                        'step_id': step_id, 'code': 'precondition_failed',
                        'message': '门角未知或未达到已打开范围（90±5°）；请先独立开门并完成必要过渡'})
                    row.update(status='failed', success=False)
                    return finish(step_id, {}, False)
            options = dict(subtask_options.get(
                "door" if kind == "transition" else kind, {}))
            if kind == "transition":
                # 抓把手的力只属于门 Skill；回默认 TCP 的 Task 过渡段
                # 复用门场景容差，但绝不能收到夹爪接触参数。
                options.pop("handle_grasp_force", None)
                if step_id == "before_reopen_transition":
                    # 旧完整 Task 在旋钮后先用完整场景碰撞回默认 TCP，
                    # 再开始第二轮开门；这里不能沿用开门后的接触白名单。
                    options.update(
                        phase_id="task/before_reopen/return_default_tcp",
                        motion_kind="plan", contact_policy="avoid_door",
                    )
            args = {**options,
                    "observe_collision_scene": observe_collision_scene,
                    "make_runtime": make_runtime, "planner_pool": planner_pool,
                    "output_dir": folder, "execute": execute}
            if door_part is not None:
                args["part"] = door_part
            if kind == "insert":
                # 只传刚完成的 Pick 事实，不能凭假定的 Bowl 附着继续。
                args["pick_report"] = pick_result
            elif kind == "table":
                # 桌面放置复用取碗时闭爪后实测的 TCP→Bowl 关系。
                args["grasp_report"] = extract_result["grasp"]
            outcome = runners[kind](**args)
            if not isinstance(outcome, dict):
                raise TypeError(f"{step_id} 子任务必须返回 dict 报告")
        except Exception as error:
            report.update(status="failed", first_failure={
                "step_id": step_id, "code": "subtask_exception",
                "message": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            row["status"] = "failed"
            return finish(step_id, locals().get("outcome", {}), False)
        row["status"] = outcome.get("status", "unknown")
        row["success"] = outcome.get(success_key) is True
        evidence, verified = _postcondition_evidence(kind, outcome)
        row["postcondition"] = evidence
        if not execute:
            # 即使替身子任务误报 success，预检也没有真实状态供下一段消费。
            report.update(status="validated_prefix", first_failure={
                "step_id": step_id, "code": "not_executed",
                "message": "仅验证第一段；执行并重新观测后才能继续完整任务",
            })
            return finish(step_id, locals().get("outcome", {}), False)
        if not row["success"]:
            failure = outcome.get("first_failure") or {
                "code": "subtask_incomplete", "message": f"{step_id} 未确认完成",
            }
            report.update(status=outcome.get("status", "failed"),
                          first_failure={"step_id": step_id, **failure})
            return finish(step_id, locals().get("outcome", {}), False)
        if outcome.get("status") != "executed" or not verified:
            # 下层若误报 success，也不能把未知现场状态传给后续子任务。
            row.update(status="unknown", success=False)
            report.update(status="unknown", first_failure={
                "step_id": step_id, "code": "postcondition_unverified",
                "message": f"{step_id} 返回成功，但执行状态或现场后置条件未确认",
            })
            return finish(step_id, locals().get("outcome", {}), False)
        if kind == "pick":
            pick_result = outcome
        elif kind == "extract":
            extract_result = outcome
        return finish(step_id, outcome, True)

    flow = run_microwave_task1_flow(CallbackOperationBackend(
        observe=lambda: current, perform=perform, evidence_mode="realworld",
        execution_enabled=execute), policy=policy, selection=selection,
        preparation="prefix" if selection == "all" else "current")
    report["shared_flow"] = flow.to_dict()
    if flow.success:
        report.update(status="executed", success=True)
    report['operation_success'] = bool(flow.success)
    report['task_success'] = bool(flow.success) if selection == 'all' and execute else None
    report['actual_final'] = dict(current.values)
    report['timing'] = {'target_wall_s': sum(r.get('wall_time_s', 0.) for r in rows),
                        'source': 'perf_counter including observation/planning/execution/manual checks'}
    return save()


__all__ = ['run_observed_door_return_default_tcp', 'SUBTASK_NAMES', 'run_observed_microwave_subtask', 'run_observed_microwave_task1']
