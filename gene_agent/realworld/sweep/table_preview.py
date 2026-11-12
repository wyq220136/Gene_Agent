"""桌面首帧的抓刷与清扫前瞻。复用抓取几何和 Sweep Skill，没有真机执行器。"""
from __future__ import annotations
from gene_agent.motion.configuration import configured_motion, current_motion_config

import json
from pathlib import Path
import time
import mujoco
import numpy as np

from gene_agent.skills.library.pick.brush import brush_pick_targets, brush_lift_target
from gene_agent.robots.pika_geometry import pika_grasp_center_tcp, pika_predict_contact_width
from gene_agent.skills.library.pick.skill import run_pick_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.skills.library.pick.brush import load_brush_pick_profile
from gene_agent.tasks.checks import CheckResult
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.robots.pika_frames import model_frames, pose_from_mujoco
from gene_agent.skills.library.pick.planning import make_postpick_candidates
from .planning import _RecordingPlanner, check_world_contacts, plan_sweep, write_json
from .preview_motion import plan_stage

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[3] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs


@configured_motion("site")
def preview_table_task(model, data, robot, site, output: Path, *, stop_after='task'):
    """用实际 FK 接续各规划段；不伪造闭爪接触、方块位移或任务成功。"""
    started = time.monotonic()
    if stop_after not in {'pregrasp','approach','pick','task'}:
        raise ValueError('未知预览阶段范围')
    initial = data.qpos.copy()
    measured_brush = pose_from_mujoco(data.body('brush'))
    names = robot['joint_groups']['left']
    corrected = any(row['shift_world_z_m'] > 0 for row in site.get('support_adjustment',{}).get('objects',{}).values())
    frames = [{'stage': 'initial_support_adjusted' if corrected else 'initial_camera_observation',
               'qpos': initial.tolist(), 'predicted': corrected}]
    pick = {'success': False, 'stages': [], 'attachment_evidence': 'hypothetical_after_planned_grasp',
            'contact_verified': False, 'physics_stepped': False}
    report = {'schema_version': 1, 'task': 'sweep_realworld_plan_only',
              'source': site['snapshot_source'], 'brush_variant': 'old', 'skill_id': 'sweep_cube',
              'planner_backend': 'curobo', 'motion': current_motion_config().to_dict(), 'execution_enabled': False, 'task_success': None,
              'evidence_mode': 'kinematic_plan_preview', 'planning_complete': False,
              'scene_xml_path': str(output/'scene/scene.xml'), 'selected_index': None,
              'candidates': [], 'status': 'pick_planning_failed', 'pick': pick,
              'planner_invoked': False,
              'preview_scope':stop_after,'requested_scope_complete':False,
              'support_adjustment': site.get('support_adjustment'),
              'robot_state_source': site['robot_state_source']}
    pool = CuroboPlannerPool(planner_factory=_RecordingPlanner, max_entries=2)
    from gene_agent.skills.library.pick.brush import brush_pick_source_hashes
    report['code_sources_sha256'] = brush_pick_source_hashes()
    grip = None
    def joints():
        return [float(data.qpos[model.joint(n).qposadr[0]]) for n in names]
    def width():
        return float(np.linalg.norm(data.site('pika_left_inner').xpos-data.site('pika_right_inner').xpos))
    def gripper_preview(stage, target_width):
        start = width()
        maximum = robot['grippers']['left']['width_range_m'][1]
        if maximum < start <= maximum+1e-9:
            start = maximum  # 两site距离的浮点舍入，不截断真实超限输入。
        for value in np.linspace(start, target_width, 10):
            model_frames(model, data, robot, joints(), float(value))
            check = check_world_contacts(model, data, robot, [joints()], None, TABLE_CONTACTS,
                                           base_table_mount=site.get('base_table_mounting_contact', False))
            if not check['success']:
                pick['stages'].append({'name': stage, 'success': False, 'postcheck': check})
                pick['failure_stage'] = stage
                report['failure_stage'] = stage
                report['status'] = 'initial_world_collision' if stage == 'open_assumed' else 'gripper_preview_collision'
                return False
            frames.append({'stage': stage, 'qpos': data.qpos.tolist(), 'predicted': True})
        pick['stages'].append({'name': stage, 'success': True, 'width_m': target_width,
                               'source': 'kinematic_assumption_no_gripper_command'})
        return True
    try:
        concept = output/'scene/concepts/brush.json'
        profile = load_brush_pick_profile('preview', overrides=site.get('pick_overrides'))
        pick['effective_profile'] = profile.to_dict()
        targets = brush_pick_targets(measured_brush, concept, profile,
            grasp_center_tcp_m=pika_grasp_center_tcp(robot,'left'))
        grasp, pregrasp = targets['approach'], targets['pregrasp']
        pick['targets'] = {n: v.tolist() for n, v in targets.items()}
        serial = 0
        def observe_pick():
            nonlocal serial
            serial += 1
            return OperationState(str(serial), 'prediction', {'joint_positions_rad': joints(),
                'gripper_width_m': width(), 'T_world_tcp': pose_from_mujoco(data.site('pika_tcp')).tolist()})
        def perform_pick(stage):
            nonlocal grip
            if stage in {'open', 'close'}:
                target_width = profile.values['open_width_m']
                if stage == 'close':
                    from gene_agent.scene.rigid_geometry import visual_mesh_object
                    geometry = pika_predict_contact_width(robot,'left',
                        np.linalg.inv(pose_from_mujoco(data.site('pika_tcp'))) @ measured_brush,
                        visual_mesh_object(model,data,'brush'),open_width_m=target_width)
                    pick['grasp_geometry'] = geometry
                    target_width = geometry['predicted_width_m']
                return {'success': gripper_preview(stage+'_assumed', target_width)}
            if stage in {'grasp_check', 'hold_check'}:
                return {'success': True, 'check': CheckResult('unknown', 'prediction', {},
                    '仅规划中的刚性夹持假设，没有接触或物理证据')}
            if stage == 'establish_hold':
                grip = np.linalg.inv(pose_from_mujoco(data.site('pika_tcp'))) @ measured_brush
                pick['T_tcp_brush'] = grip.tolist()
                return {'success': True}
            if stage == 'lift':
                target = brush_lift_target(pose_from_mujoco(data.site('pika_tcp')), pregrasp, profile)
                pick['targets']['lift'] = target.tolist()
            else:
                target = targets[stage]
            contacts = (('brush', 'table'),) if stage == 'pregrasp' else TABLE_CONTACTS
            report['planner_invoked'] = True
            row, samples = plan_stage(model, data, robot, pool, output/'pick'/stage,
                stage, target, 'joint' if stage == 'pregrasp' else 'linear_tcp', contacts, grip,
                base_table_mount=site.get('base_table_mounting_contact', False))
            pick['stages'].append(row); frames.extend(samples)
            return row
        flow = run_pick_flow(CallbackOperationBackend(observe=observe_pick, perform=perform_pick,
            evidence_mode='prediction'), stop_after=stop_after if stop_after != 'task' else 'pick',
            verify_hold=profile.values['verify_hold_for_pick'])
        pick['shared_flow'] = flow.to_dict()
        if not flow.success:
            pick['failure_stage'] = flow.failure_stage
            return report
        if stop_after in {'pregrasp', 'approach', 'pick'}:
            pick['success'] = stop_after == 'pick'
            report.update(status='prefix_planning_complete', requested_scope_complete=True, planned_through=stop_after)
            return report
        pick['success'] = True
        # 退出 pick 的 GPU planner 以后再初始化 sweep，避免两个 pool 同驻显存。
        pool.clear()
        site = {**site, 'planning_start_source': 'predicted_after_pick'}
        sweep = plan_sweep(model, data, robot, grip, site, output)
        sweep_frames = json.loads((output/'preview_frames.json').read_text(encoding='utf-8'))
        for f in sweep_frames:
            f['predicted'] = True
            if f['stage'].startswith('initial'):
                f['stage'] = 'predicted_after_pick'
        frames.extend(sweep_frames)
        report = {**sweep, 'pick': pick, 'robot_state_source': site['robot_state_source'],
                  'preview_scope':'task','requested_scope_complete':bool(sweep['planning_complete']),
                  'support_adjustment':site.get('support_adjustment'),
                  'gripper_hold': 'hypothetical_after_planned_grasp_no_contact_evidence'}
        return report
    finally:
        pool.clear()
        report['planning_duration_s'] = time.monotonic()-started
        report['initial_object_poses_world'] = {
            n: pose_from_mujoco(data.body(n)).tolist() for n in ('brush', 'cube', 'dustpan')}
        # 初始事实必须来自首帧，不能被规划后的刚体预测覆盖。
        report['initial_object_poses_world']['brush'] = measured_brush.tolist()
        write_json(output/'preview_frames.json', frames)
        report['preview_frames_path'] = str(output/'preview_frames.json')
        write_json(output/'result.json', report)
        data.qpos[:] = initial;mujoco.mj_forward(model, data)
