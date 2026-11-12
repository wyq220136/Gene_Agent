"""把现有旋钮 Skill 的规划轨迹接到 MuJoCo；旋钮跟随方式显式记录。"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.feedback.state import state_id
from gene_agent.skills.library.knob.skill import run_knob_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult


def run_planned_knob_execution(runtime, planned: dict, output_dir: Path, *, stop_after=None) -> dict:
    """执行旧 Viser 轨迹，并区分接触驱动与旧版运动学旋钮更新。

    机器人始终通过 actuator 运动。旧 Viser 只在最后一个旋转段结束时
    更新旋钮关节；若物理接触尚未带到目标角，本函数保留该行为并在
    报告中标明 kinematic_commit，不能把它计为全物理旋钮成功。
    """
    if not planned.get("success") or planned.get("source_format") != "kinematic_plan":
        raise ValueError("旋钮必须先得到完整的旧 Viser Skill 规划")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, data = runtime.model, runtime.data
    if not np.allclose(data.qpos, planned["input_qpos"], atol=1e-5, rtol=0):
        raise ValueError("旋钮规划起点与当前 MuJoCo 状态不一致，必须重新规划")
    joint = model.joint(planned["joint_name"])
    door = model.joint("microwave_door_joint")
    target = float(planned["joint_target_angle_rad"])
    rows: list[dict] = []
    physical_angle_before_commit = None
    committed = False
    stage_map = {stage['name']: stage for stage in planned['stages']}

    def observe():
        return OperationState(state_id(model, data), 'mujoco_execution', {
            'qpos': np.asarray(data.qpos).tolist(),
            'qvel': np.asarray(getattr(data, 'qvel', [])).tolist(),
            'knob_angle_rad': float(data.qpos[joint.qposadr[0]]),
            'kinematic_commit_applied': committed,
        })

    def perform(name):
        nonlocal committed, physical_angle_before_commit
        if name == 'grasp_check':
            return {'success': True, 'check': CheckResult(
                'unknown', 'mujoco_contact_unverified', {},
                '该旧版旋钮执行器没有物理夹持验收')}
        if name == 'kinematic_eligibility_check':
            # 现有混合仿真按配置允许旋钮运动学跟随；该许可不是夹持接触证据。
            return {'success': True, 'check': CheckResult(
                'pass', 'simulation_declared_kinematic_knob',
                {'physical_grasp_verified': False,
                 'mode': 'explicit_legacy_knob_follow'},
                '仅允许声明的混合仿真旋钮动作，不证明物理夹持')}
        if name == 'rotation_check':
            reached = abs(float(data.qpos[joint.qposadr[0]]) - target) <= np.deg2rad(3.)
            return {'success': True, 'check': CheckResult(
                'pass' if reached else 'fail', 'mujoco_execution',
                {'kinematic_commit_applied': committed,
                 'actual_knob_angle_rad': float(data.qpos[joint.qposadr[0]]),
                 'target_angle_rad': target})}
        stage = stage_map[name]
        start = state_id(model, data)
        if stage["action"].startswith("gripper_"):
            execution = runtime.execute_gripper_command(
                group=planned["robot_group"], width_m=float(stage["gripper_width_m"]),
                duration_s=.35, planned_state_id=start,
                replay_object_names=("bowl", "knob_0"),
            )
        elif stage["action"] == "wait":
            execution = runtime.execute_hold(
                duration_s=max(.05, float(stage.get("settle_time_s", .1))),
                planned_state_id=start, replay_object_names=("bowl", "knob_0"),
                replay_group=planned["robot_group"],
            )
        else:
            trajectory = (stage.get("result") or {}).get("trajectory")
            if not trajectory:
                raise ValueError(f"旋钮阶段 {name} 缺少真实规划轨迹")
            execution = runtime.execute_joint_trajectory(
                group=planned["robot_group"], joint_names=trajectory["joint_names"],
                times_s=trajectory["times_s"], positions_rad=trajectory["positions_rad"],
                planned_state_id=start, endpoint_tolerance_rad=.2,
                settle_timeout_s=.3, replay_object_names=("bowl", "knob_0"),
            )
        row = {
            "stage_id": f"rotate_knob/{name}",
            "planning_operation": stage.get("operation"),
            "start_state_id": start,
            "execution": execution,
            "actual_knob_angle_rad": float(data.qpos[joint.qposadr[0]]),
            "actual_door_angle_rad": float(data.qpos[door.qposadr[0]]),
        }
        if name.startswith("rotate_") and stage.get("visual_angle_commit") \
                and execution["success"]:
            physical_angle_before_commit = row["actual_knob_angle_rad"]
            if abs(physical_angle_before_commit - target) > np.deg2rad(3.):
                # 原版视觉 Skill 的旋钮只作运动学跟随。只在这里提交一次，
                # 后续阶段仍从同一物理状态继续，不重新放置机器人或碗。
                data.qpos[joint.qposadr[0]] = target
                mujoco.mj_forward(model, data)
                committed = True
                row["kinematic_commit"] = {
                    "from_rad": physical_angle_before_commit,
                    "to_rad": target,
                    "source": "legacy_viser_knob_follow",
                }
        row["end_state_id"] = state_id(model, data)
        rows.append(row)
        (output / "stages.json").write_text(
            json.dumps(rows, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )
        return {'success': bool(execution['success']), 'execution': execution}

    observed = observe()
    first_failure = None

    def guarded_perform(name):
        nonlocal observed, first_failure
        start = observed.state_id
        try:
            outcome = perform(name)
            observed = observe()
            return outcome
        except Exception as error:
            # actuator 可能已完成部分运动；保留异常，再读取实际终态。
            first_failure = {'stage': name, 'code': 'stage_exception',
                             'exception_type': type(error).__name__,
                             'message': str(error), 'traceback': traceback.format_exc()}
            try:
                observed = observe()
            except Exception as observation_error:
                observed = OperationState(f'unavailable:{name}', 'unavailable', {
                    'state_unavailable': True, 'reason': str(observation_error)})
                first_failure['recovery_observation_error'] = str(observation_error)
            stage = stage_map.get(name, {})
            failure_row = {
                'stage_id': f'rotate_knob/{name}',
                'planning_operation': stage.get('operation'),
                'start_state_id': start, 'end_state_id': observed.state_id,
                'execution': {'success': False, 'failure_type': 'stage_exception'},
                **first_failure,
            }
            if rows and rows[-1]['stage_id'] == failure_row['stage_id']:
                failure_row['execution_before_exception'] = rows[-1]['execution']
                rows[-1].update(failure_row)
            else:
                rows.append(failure_row)
            (output / 'stages.json').write_text(
                json.dumps(rows, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
            return {'success': False}

    flow = run_knob_flow(CallbackOperationBackend(
        observe=lambda: observed, perform=guarded_perform, evidence_mode='mujoco_mixed_knob',
        failure_details=lambda name: first_failure),
        target_names=lambda: [stage['name'] for stage in planned['stages']],
        grasp_mode='declared_kinematic', stop_after=stop_after)
    complete = len(rows) == len(planned['stages']) and all(
        row['execution']['success'] for row in rows)
    final_angle = float(data.qpos[joint.qposadr[0]])
    result = {
        "schema_version": 1,
        "first_failure": first_failure,
        "shared_flow": flow.to_dict(),
        "skill": "rotate_knob_0",
        "source_format": "mujoco_physical_robot_with_explicit_knob_follow",
        "planned_report_ref": str(output.parent / "knob_plan" / "skill_rotate_knob_0_result.json"),
        "execution_success": bool(flow.success),
        "operation_success": bool(flow.success),
        "requested_scope": flow.requested_scope,
        "actual_final": flow.to_dict()["end_state"],
        "goal_success": (None if stop_after and flow.success else
                         bool(complete and flow.success and abs(final_angle - target) <= np.deg2rad(3.))),
        "physical_rotation_success": bool(
            complete and physical_angle_before_commit is not None
            and abs(physical_angle_before_commit - target) <= np.deg2rad(3.)
        ),
        "kinematic_commit_applied": committed,
        "evidence_mode": "mixed_kinematic_knob" if committed else "physical",
        "physical_angle_before_commit_rad": physical_angle_before_commit,
        "final_knob_angle_rad": final_angle,
        "target_knob_angle_rad": target,
        "failure_stage": (f"rotate_knob/{flow.failure_stage}" if not flow.success else None),
        "final_state_id": observed.state_id,
        "stage_count": len(rows),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["run_planned_knob_execution"]
