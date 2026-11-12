"""用实际刷子状态逐段执行清扫；每段由调用方提供规划和 MuJoCo 执行器。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from gene_agent.knowledge.sweep import SweepLayout


from .model import SweepContacts


def run_sweep_skill(
    layout: SweepLayout, *, contacts: SweepContacts, initial_brush_z: float,
    move_stage: Callable, target_for_brush: Callable,
    current_brush_xyz: Callable, stage_prefix: str = "",
    sweep_targets: tuple[str, ...] = (),
    contact_target_for_brush: Callable | None = None,
    sweep_end_for_current_pose: Callable | None = None,
    prepared_targets: dict | None = None,
    retreat_lift_m: float = .09,
    retreat_delta_world_m: np.ndarray | None = None,
    contact_target_name: str | None = None,
    stop_after: str | None = None,
    evidence_mode: str = "execution",
) -> dict:
    """执行预备、直线下降、直线扫动与撤离；失败时保持实际现场。"""
    if evidence_mode not in {'execution', 'prediction'}:
        raise ValueError('清扫模式必须为 execution 或 prediction')
    if stop_after not in {None, 'complete', 'move_behind_cube', 'lower_to_contact', 'sweep', 'retreat'}:
        raise ValueError('未知清扫停止范围')
    if not isinstance(contacts, SweepContacts):
        raise ValueError('清扫必须显式提供接触角色')

    def was_executed(row):
        # 字段缺失不是运动证据：规划失败可能尚未向设备发令。
        if row.get('evidence_mode') == 'planning_only':
            return False
        if 'executed' in row:
            return bool(row['executed'])
        if 'execution' in row:
            return row['execution'] is not None
        # 兼容已完成动作、仅返回成功标志的原有执行回调。
        return bool(row.get('success'))

    def finish(success, failed=None):
        row = results.get(failed,{})
        executed = evidence_mode == 'execution' and any(was_executed(r) for r in results.values())
        return {'success':bool(success), 'executed':executed, 'evidence_mode':evidence_mode,
                'operation_success':bool(success and executed),
                'requested_scope':stop_after or 'complete', 'failure_stage':failed,
                'failure':row.get('failure') or ({'code':row.get('failure_type','stage_failed'),
                    'source':'sweep_motion','message':str(row.get('error') or failed)} if failed else None),
                'stages':results}
    def move(*args, **kwargs):
        row = move_stage(*args, **kwargs)
        if evidence_mode == 'execution' and row.get('success') and (
                row.get('evidence_mode') == 'planning_only'
                or row.get('executed') is False
                or ('execution' in row and row['execution'] is None)):
            return {**row, 'success':False, 'failure':{'code':'motion_not_executed',
                    'source':'sweep_motion','message':'仅规划结果不能算实际清扫执行'}}
        return row
    contact_start = layout.brush_start_world_xyz.copy()
    contact_start[2] = initial_brush_z
    lifted_start = contact_start.copy()
    lifted_start[2] += .055
    sweep_contacts = (*contacts.table_pairs,
                      *((contacts.tool, name) for name in sweep_targets),
                      (contacts.tool, contacts.receiver))
    lower_contacts = (*contacts.table_pairs,
                      *((contacts.tool, name) for name in
                        ((contact_target_name,) if contact_target_name else sweep_targets)))
    results = {}
    stages = (
        ("move_behind_cube", lifted_start, "joint", 3.0, contacts.grip_pairs),
        ("lower_to_contact", contact_start, "linear_tcp", 2.0, lower_contacts),
    )
    for name, desired, path_kind, duration_s, allowed_contacts in stages:
        stage_name = stage_prefix + name
        target = (np.asarray(prepared_targets[name]) if prepared_targets is not None else
                  contact_target_for_brush(desired)
                  if name == "lower_to_contact" and contact_target_for_brush is not None
                  else target_for_brush(desired))
        result = move(stage_name, target, path_kind, duration_s,
                            allowed_contacts=allowed_contacts)
        results[stage_name] = result
        if not result["success"]:
            return finish(False, stage_name)
        if name == stop_after:
            return finish(True)

    # 已经执行过下降，扫动高度必须取实际刷子位置；倾斜接触姿态会改变
    # 夹爪与刷子中心的相对高度，继续用抓取前 z 会把刷面压进桌面。
    sweep_z = (float(np.asarray(current_brush_xyz(), dtype=float)[2])
               if contact_target_for_brush is not None or prepared_targets is not None else initial_brush_z)
    end_xyz = (sweep_end_for_current_pose() if sweep_end_for_current_pose is not None
               else layout.brush_end_world_xyz)
    sweep_end = np.r_[np.asarray(end_xyz)[:2], sweep_z]
    sweep_name = stage_prefix + "sweep"
    result = move(sweep_name, target_for_brush(sweep_end), "linear_tcp", 5.0,
                        allowed_contacts=sweep_contacts)
    results[sweep_name] = result
    if not result["success"]:
        return finish(False, sweep_name)
    if stop_after == 'sweep':
        return finish(True)

    # 撤离以实际执行后的刷子 pose 为基准，避免把规划终点当作真实状态。
    retreat_xyz = np.asarray(current_brush_xyz(), dtype=float).copy()
    retreat_xyz += (np.asarray(retreat_delta_world_m, dtype=float)
                    if retreat_delta_world_m is not None else [0., 0., retreat_lift_m])
    retreat_name = stage_prefix + "retreat"
    result = move(retreat_name, target_for_brush(retreat_xyz), "linear_tcp", 2.5,
                        allowed_contacts=sweep_contacts)
    results[retreat_name] = result
    return finish(result["success"], None if result["success"] else retreat_name)


run = run_sweep_skill
