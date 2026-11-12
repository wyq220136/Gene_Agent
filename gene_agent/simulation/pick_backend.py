"""MuJoCo 抓取适配：物理执行、接触诊断和证据；阶段顺序由共享流程控制。"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.skills.library.pick.model import PICK_STAGE_OPERATIONS, PICK_STAGE_ORDER, GraspSettings
from gene_agent.skills.library.pick.planning import make_postpick_candidates, _pose, make_pregrasp_pose
from gene_agent.skills.library import pick
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _build_pick_backend(
    runtime,
    output_dir,
    *,
    candidate: dict,
    motion_stage,
    object_id: str,
    group: str,
    diagnostics=None,
    evidence_writer=None,
    failure_handler=None,
    open_width_m: float = .08,
    close_width_m: float = 0.0,
    close_duration_s: float = 1.0,
    retreat_m: float = .08,
    lift_m: float = 0.0,
    settings: GraspSettings | None = None,
    execute_lift: bool = True,
    visual_recorder=None,
    after_contact_verified=None,
    close_gripper_command=None,
    postpick_candidate_motion=None,
    postpick_translation_direction_world=None,
    selected_pregrasp_pose=None,
    contact_classifier=None,
    lift_target_builder=None,
    close_position_kp_scale: float | None = None,
    hold_position_kp_scale: float | None = None,
    _request=None,
) -> CallbackOperationBackend:
    """最薄的操作编排：每段都查询并消费上一段的实际 state ID。"""
    for adapter in (motion_stage, postpick_candidate_motion):
        owner = getattr(adapter, '__self__', adapter)
        bound_group = getattr(owner, 'group', None)
        if bound_group is not None and bound_group != group:
            raise ValueError(f'Pick 机器人组 {group} 与运动适配绑定的 {bound_group} 不一致')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if settings is not None:
        open_width_m = settings.open_width_m
        close_width_m = settings.close_width_m
        retreat_m = settings.approach_distance_m
        lift_m = settings.lift_distance_m if execute_lift else 0.0
    hold_duration_s = settings.hold_duration_s if settings is not None else .5
    if not np.isfinite(close_duration_s) or close_duration_s <= 0:
        raise ValueError("close_duration_s 必须为正有限秒数")
    final_pose = _pose(candidate["T_world_tcp"])
    # 自动候选筛选已选出的 pregrasp 可能有局部 X/Y 姿态修正；执行时
    # 必须原样使用该 pose，不能重新沿 grasp 的局部 Z 推导一个新目标。
    pregrasp_pose = (_pose(selected_pregrasp_pose) if selected_pregrasp_pose is not None
                     else make_pregrasp_pose(final_pose, retreat_m=retreat_m))
    if diagnostics is None or evidence_writer is None:
        raise ValueError("PickExecutor需要显式诊断和阶段证据写入接口")
    report = {
        "schema_version": 1,
        "mode": "synchronous_pick_skill",
        "candidate_id": candidate["candidate_id"],
        "instance_id": object_id,
        "robot_group": group,
        "status": "running",
        "success": False,
        "failure_stage": None,
        "stage_state_ids": {},
        "stages": {},
        "skill_settings": asdict(settings) if settings is not None else {
            "approach_distance_m": retreat_m, "lift_distance_m": lift_m,
            "open_width_m": open_width_m, "close_width_m": close_width_m,
            "hold_duration_s": hold_duration_s,
        },
        "execute_lift": bool(lift_m > 0),
        "close_duration_s": close_duration_s,
        "close_position_kp_scale": close_position_kp_scale,
        "stage_contract": [
            {"name": name, "operation": PICK_STAGE_OPERATIONS[name]}
            for name in PICK_STAGE_ORDER
        ],
    }

    def report_failure(stage: str, kind: str, details: dict) -> None:
        if failure_handler is None:
            return
        try:
            failure_handler(stage, kind, details)
        except Exception as error:
            report["failure_capture_error"] = f"{type(error).__name__}: {error}"
            _write_json(output / "result.json", report)

    def observe(stage, target_pose=None):
        observer = getattr(runtime, "replay_sample", runtime.query)
        observation = observer(object_names=(object_id,), robot_groups=(group,))
        report["stage_state_ids"][stage] = observation["state_id"]
        evidence_writer(stage, observation, target_pose)
        return observation

    initial = observe('initial', pregrasp_pose)
    after_open = closed_observation = lifted = initial
    closed = contact = descriptor = classifier = target = None
    active_stage = ['initial']
    def open_step():
        nonlocal after_open
        open_args = dict(group=group, width_m=open_width_m, duration_s=.3,
                         planned_state_id=initial["state_id"], timeout_s=.6,
                         replay_object_names=(object_id,))
        if visual_recorder is not None:
            visual_recorder.begin_stage("open", runtime.model, runtime.data)
            open_args["step_observer"] = visual_recorder.observer("open")
        opened = runtime.execute_gripper_command(**open_args)
        if visual_recorder is not None:
            visual_recorder.end_stage("open", runtime.model, runtime.data)
        report["stages"]["open"] = opened
        if not opened.get("success"):
            report["status"] = "failed"
            report["failure_stage"] = "open"
            _write_json(output / "result.json", report)
            report_failure("open", "execution_failed", opened)
            return opened
        after_open = observe("opened", pregrasp_pose)
        return opened

    def move_step():
        nonlocal after_open
        stage = active_stage[0]
        target = pregrasp_pose if stage == 'pregrasp' else final_pose
        result = motion_stage(stage, target, after_open['state_id'])
        report['stages'][stage] = result
        if not result.get('success'):
            report.update(status='failed', failure_stage=stage)
            return result
        after_open = observe(stage, final_pose)
        return result

    def close_step():
        nonlocal closed, closed_observation
        close_args = dict(group=group, width_m=close_width_m, duration_s=close_duration_s,
                          planned_state_id=after_open["state_id"], timeout_s=close_duration_s + .3,
                          replay_object_names=(object_id,))
        if close_position_kp_scale is not None:
            close_args["position_kp_scale"] = float(close_position_kp_scale)
        if visual_recorder is not None:
            visual_recorder.begin_stage("close", runtime.model, runtime.data)
            close_args["step_observer"] = visual_recorder.observer("close")
        closed = (close_gripper_command or runtime.execute_gripper_command)(**close_args)
        if visual_recorder is not None:
            visual_recorder.end_stage("close", runtime.model, runtime.data)
        report["stages"]["close"] = closed
        closed_observation = observe("closed", final_pose)
        return closed

    def grasp_check():
        nonlocal contact, descriptor, classifier
        contact_diagnostics = diagnostics()
        descriptor = getattr(runtime, "robot_metadata", {}).get("grippers", {}).get(group)
        classifier = contact_classifier or classify_grasp_contacts
        if descriptor:
            contact = classifier(
                contact_diagnostics.get("contacts", []), descriptor, object_body=object_id
            )
        else:
            contact = {
                "success": False, "bilateral_pad_contact": False,
                "pad_sides": {"positive": False, "negative": False},
                "contact_count": 0, "undesired_gripper_contacts": [], "contacts": [],
            }
        report["contact"] = contact
        report["contact_success"] = bool(closed.get("success") and contact["success"])
        report["success"] = bool(report["contact_success"] and lift_m <= 0)
        report["failure_stage"] = None if report["contact_success"] else "close_contact"
        if not report["contact_success"]:
            report["status"] = "failed"
        report["closed_state_id"] = closed_observation["state_id"]
        _write_json(output / "diagnostics_closed.json", contact_diagnostics)
        _write_json(output / "result.json", report)
        if not report["contact_success"]:
            report_failure(
                "close_contact",
                "execution_failed" if not closed.get("success") else "postcondition_failed",
                contact,
            )
            return {'success': False, 'check': CheckResult('fail', 'mujoco_contact', contact),
                    'failure': {'code':'grasp_contact_rejected', 'source':'mujoco_contact',
                        'message':'未满足双侧指垫接触，或存在不允许的夹爪接触',
                        'report_ref':'diagnostics_closed.json'}}
        return {'success': True, 'check': CheckResult('pass', 'mujoco_contact', contact)}

    def establish_hold():
        nonlocal closed_observation
        if after_contact_verified is not None:
            try:
                report["simulation_attachment"] = after_contact_verified(contact)
                # 固定会改变 MuJoCo 积分状态，后续规划必须使用新的实际 state ID。
                closed_observation = observe("attached", final_pose)
            except Exception as error:
                report["status"] = "failed"
                report["failure_stage"] = "simulation_attachment"
                report["simulation_attachment_error"] = f"{type(error).__name__}: {error}"
                _write_json(output / "result.json", report)
                report_failure("simulation_attachment", "execution_failed", report)
                return {'success': False}
        return {'success': True}

    def lift_step():
        nonlocal target, lifted
        # 附着代理由外部 motion_stage 显式提交；执行器只消费当前实际
        # 观测并在末端检查物体位移与夹持，不把规划附着当物理成功。
        if postpick_candidate_motion is not None:
            # 默认与视觉 Pick 共用旧版 17 个撤离目标；具体任务可以显式
            # 提供世界方向，把接近轴撤离与任务语义上的抬升解耦。
            candidates = make_postpick_candidates(
                final_pose, pregrasp_pose, lift_m,
                translation_direction_world=postpick_translation_direction_world,
            )
            lift = postpick_candidate_motion(
                "lift", candidates, closed_observation["state_id"],
            )
            chosen = (lift.get("plan") or {}).get("selected_candidate_index")
            if lift.get("success") and (not isinstance(chosen, int)
                                        or not 0 <= chosen < len(candidates)):
                raise ValueError("post-pick 成功结果缺少有效候选索引")
            target = candidates[chosen] if chosen is not None else candidates[0]
        else:
            target = np.asarray(closed_observation["robots"][group]["T_world_tcp"], dtype=float)
            target = target.copy()
            if lift_target_builder is not None:
                target = _pose(lift_target_builder(target))
            else:
                target[2, 3] += lift_m
            lift = motion_stage("lift", target, closed_observation["state_id"])
        report["stages"]["lift"] = lift
        if not lift.get("success"):
            report["status"] = "failed"
            report["success"] = False
            report["failure_stage"] = "lift"
            report["physical_pick_success"] = False
            _write_json(output / "result.json", report)
            return lift

        lifted = observe('lifted', target)
        return lift

    def hold_check():
        start_z = float(np.asarray(closed_observation["objects"][object_id]["T_world_object"])[2, 3])
        lifted_z = float(np.asarray(lifted["objects"][object_id]["T_world_object"])[2, 3])
        hold_args = dict(duration_s=hold_duration_s, planned_state_id=lifted["state_id"],
                         replay_object_names=(object_id,), replay_group=group)
        if hold_position_kp_scale is not None:
            hold_args["position_kp_scale"] = float(hold_position_kp_scale)
        if visual_recorder is not None:
            visual_recorder.begin_stage("hold", runtime.model, runtime.data)
            hold_args["step_observer"] = visual_recorder.observer("hold")
        held_result = runtime.execute_hold(**hold_args)
        if visual_recorder is not None:
            visual_recorder.end_stage("hold", runtime.model, runtime.data)
        report["stages"]["hold"] = held_result
        _write_json(output / "executions/hold.json", held_result)
        held = observe("held", target)
        held_z = float(np.asarray(held["objects"][object_id]["T_world_object"])[2, 3])
        held_contact = classifier(
            diagnostics().get("contacts", []), descriptor, object_body=object_id,
        )
        report["lift_height_m"] = lifted_z - start_z
        report["hold_height_m"] = held_z - start_z
        report["hold_drop_m"] = lifted_z - held_z
        report["hold_contact"] = held_contact
        closed_tcp = np.asarray(closed_observation["robots"][group]["T_world_tcp"])
        held_tcp = np.asarray(held["robots"][group]["T_world_tcp"])
        closed_object = np.asarray(closed_observation["objects"][object_id]["T_world_object"])
        held_object = np.asarray(held["objects"][object_id]["T_world_object"])
        closed_relative = np.linalg.inv(closed_tcp) @ closed_object
        held_relative = np.linalg.inv(held_tcp) @ held_object
        report["hold_relative_translation_drift_m"] = float(np.linalg.norm(
            held_relative[:3, 3] - closed_relative[:3, 3],
        ))
        report["physical_pick_success"] = bool(
            held_result["success"] and held_contact["success"]
            and report["lift_height_m"] >= min(.05, .7 * lift_m)
            and report["hold_height_m"] >= min(.05, .7 * lift_m)
            and report["hold_drop_m"] <= .01
            and report["hold_relative_translation_drift_m"] <= .025
        )
        if report.get("simulation_attachment"):
            report["simulation_assisted_pick_success"] = bool(
                held_result["success"]
                and report["lift_height_m"] >= min(.05, .7 * lift_m)
                and report["hold_height_m"] >= min(.05, .7 * lift_m)
                and report["hold_drop_m"] <= .01
                and report["hold_relative_translation_drift_m"] <= .025
            )
            # weld 接管了物体动力学，此运行不构成纯物理抓取成功证据。
            report["physical_pick_success"] = False
            report["success"] = report["simulation_assisted_pick_success"]
        else:
            report["success"] = report["physical_pick_success"]
        report["status"] = "completed" if report["success"] else "failed"
        report["failure_stage"] = None if report["success"] else "hold_postcondition"
        _write_json(output / "result.json", report)
        if not report["success"]:
            report_failure("hold_postcondition", "postcondition_failed", {
                "lift_height_m": report["lift_height_m"],
                "hold_height_m": report["hold_height_m"],
                "hold_drop_m": report["hold_drop_m"],
                "hold_relative_translation_drift_m": report["hold_relative_translation_drift_m"],
                "hold_contact_success": held_contact["success"],
            })

        return {'success': True, 'check': CheckResult('pass' if report['success'] else 'fail', 'mujoco_hold', {'simulation_assisted': bool(report.get('simulation_attachment'))})}

    callbacks = {'open': open_step, 'pregrasp': move_step, 'approach': move_step,
                 'close': close_step, 'grasp_check': grasp_check, 'establish_hold': establish_hold,
                 'lift': lift_step, 'hold_check': hold_check}
    def perform(stage):
        active_stage[0] = stage
        try:
            row = callbacks[stage]()
        except Exception as error:
            import traceback
            (output / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
            row = {'success': False, 'failure': {'code': 'stage_exception', 'source': 'mujoco_pick',
                'message': f'{type(error).__name__}: {error}', 'report_ref': 'exception.txt'}}
        if stage in {'pregrasp', 'approach', 'lift'} and row.get('success'):
            if (row.get('evidence_mode') == 'planning_only'
                    or ('execution' in row and row['execution'] is None)):
                row = {**row, 'success': False, 'failure': {
                    'code': 'motion_not_executed', 'source': 'mujoco_pick',
                    'message': '运动仅完成规划，未提供实际执行结果', 'report_ref': 'result.json'}}
            elif isinstance(row.get('execution'), dict) and not row['execution'].get('success'):
                row = {**row, 'success': False}
        if not row.get('success') and not row.get('failure'):
            # 实际执行失败时，成功的规划状态不能覆盖 SDK/执行器的直接错误。
            execution = row.get('execution') or {}
            original = (execution if isinstance(execution, dict) and not execution.get('success', True)
                        else row.get('first_failure') or row.get('planning') or row.get('plan') or {})
            if not isinstance(original, dict):
                original = {}
            row['failure'] = {'code': original.get('code') or original.get('failure_category')
                or original.get('failure_type') or row.get('failure_type') or 'stage_failed',
                'source': 'mujoco_execution' if original is execution else 'mujoco_pick',
                'message': str(original.get('message') or original.get('error') or original.get('status')
                               or row.get('error') or stage),
                'report_ref': 'result.json'}
        return row
    def operation_state():
        observation = getattr(runtime, 'replay_sample', runtime.query)(object_names=(object_id,), robot_groups=(group,))
        return OperationState(observation['state_id'], 'mujoco', observation)
    request = _request or pick.PickRequest(object_id, group, settings or GraspSettings(
        approach_distance_m=retreat_m, lift_distance_m=lift_m if lift_m > 0 else .08,
        open_width_m=open_width_m, close_width_m=close_width_m,
        hold_duration_s=hold_duration_s), lift=lift_m > 0)
    def finish(flow):
        report['shared_flow'] = flow.to_dict()
        report['requested_scope'] = flow.requested_scope
        report['operation_success'] = flow.success and flow.executed
        report['actual_final'] = flow.to_dict()['end_state']
        report.setdefault('physical_pick_success', False)
        if not flow.success:
            report['success'] = False
            report['status'] = 'failed'
            report['failure_stage'] = report.get('failure_stage') or flow.failure_stage
            report['failure'] = flow.failure or {'code': flow.failure_code,
                'stage': flow.failure_stage, 'source': 'pick'}
        elif flow.requested_scope != 'pick' or lift_m <= 0:
            report.update(status='completed', success=True)
        _write_json(output / 'result.json', report)
    backend = CallbackOperationBackend(observe=operation_state, perform=perform, evidence_mode='mujoco')
    backend.request, backend.report, backend.finish = request, report, finish
    return backend


def make_pick_backend(runtime, output_dir, *, request: pick.PickRequest, candidate,
                      motion_stage, **adapters):
    """绑定候选、证据与设备；只装配，动作由 pick.run 执行。

    机器人组与参数唯一来自 request。场景特例集中在调用方适配，不进入 Skill。
    """
    return _build_pick_backend(runtime, output_dir, candidate=candidate,
        motion_stage=motion_stage, object_id=request.object_id, group=request.group,
        settings=request.settings, execute_lift=request.lift, _request=request, **adapters)


def run_simulation_pick(*args, stop_after='pick', **kwargs):
    """兼容旧场景装配；复用公开 Pick 流程与同一 backend。"""
    backend = _build_pick_backend(*args, **kwargs)
    pick.run(backend, backend.request, stop_after=stop_after)
    return backend.report
