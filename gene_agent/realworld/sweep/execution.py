"""现场清扫装配：复用微波炉 Runtime、刷子 Knowledge 和同步 Sweep Skill。"""
from __future__ import annotations
from gene_agent.motion.configuration import configured_motion

import json
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np

from gene_agent.knowledge.sweep import make_brush_grasp_pose, make_sweep_layout, make_sweep_pose_sequences, correct_sequence_table_clearance
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.robots.pika_frames import model_frames, pose_from_mujoco, pose_error
from gene_agent.scene.rigid_geometry import collision_vertices_object
from gene_agent.skills.library.pick.brush import brush_pick_targets, brush_lift_target
from gene_agent.robots.pika_geometry import pika_grasp_center_tcp
from gene_agent.skills.library.pick.skill import run_pick_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.skills.library.pick.brush import load_brush_pick_profile
from gene_agent.skills.library.pick.planning import make_postpick_candidates
from gene_agent.skills.library.sweep.skill import run_sweep_skill
from gene_agent.tasks.checks import CheckResult, resolve_unknown
from .planning import _RecordingPlanner, _set_prediction, check_world_contacts, write_json
from .preview_motion import plan_stage

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[3] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs


def preview_ready_for_stop(report, stop_after):
    """执行前只要求请求的前缀；后续清扫失败不能阻断pregrasp验收。"""
    if stop_after not in {'pregrasp','approach','pick','task'}:
        raise ValueError('未知阶段范围')
    if report.get('planning_complete'):
        return True
    if stop_after=='task':
        return False
    required=['open_assumed','pregrasp']
    if stop_after in {'approach','pick'}:required.append('approach')
    if stop_after=='pick':required.extend(['close_assumed','lift'])
    stages={s['name']:s for s in report.get('pick',{}).get('stages',[])}
    return all(stages.get(name,{}).get('success') is True for name in required)


def check_execution_input(snapshot, site, config):
    if snapshot.get('source')!='first_frame_preview' or snapshot.get('robot_state_source')!='sdk_read_only':
        raise ValueError('执行需要实景首帧与实测机器人初态，不接受仿真或假定待机')
    if snapshot.get('input_mode')!='new_camera_frame':
        raise ValueError('执行输入必须来自新相机帧')
    age=time.time()-float(snapshot['observed_at_unix_ms'])/1000
    if not np.isfinite(age) or not -1 <= age <= float(config['max_observation_age_s']):
        raise ValueError(f'首帧已过期（{age:.1f}s），请重新拍照规划')


def read_execution_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    # 兼容旧现场文件，但不把手填布尔值当作几何证据或执行许可。
    config.pop('tcp_verified',None)
    config.pop('tcp_verification_source',None)
    if config.get('schema_version')!=1:raise ValueError('执行配置版本无效')
    limits=('max_observation_age_s','max_start_error_rad','max_width_error_m',
            'max_endpoint_translation_m','max_endpoint_rotation_deg','motion_speed_scale',
            'max_tcp_speed_mps','canfd_frequency_hz','min_tcp_above_table_m')
    for name in limits:
        if not np.isfinite(config[name]) or float(config[name])<=0:
            raise ValueError(f'{name}需要正有限数')
    if config['motion_speed_scale']>1:raise ValueError('速度比例不能超过1')
    if not str(config.get('gripper_port','')).strip():raise ValueError('缺少Pika串口')
    return config


@configured_motion("config")
def execute_observed_sweep(model,data,robot,site,snapshot,config,backend,output:Path,*,
                           stop_after='pregrasp',ask=input,stage_planner=plan_stage,pool_factory=CuroboPlannerPool):
    """首帧静态场景假设；每段从 SDK 实测状态重新规划，不重放预览轨迹。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    check_execution_input(snapshot,site,config)
    names=robot['joint_groups']['left']
    report={'schema_version':1,'task':'sweep_cube_into_dustpan','skill_id':'sweep_cube',
        'source_format':'realworld_sweep_execution','source':getattr(backend,'evidence_mode','realworld_sdk'), 'brush_variant':'old',
        'robot_state_source':'sdk_read_only',
        'execution_enabled':True,'planner_backend':'curobo','status':'running','motion_complete':False,
        'task_success':None,'stop_after':stop_after,'stages':[],
        'scene_update_policy':'first_frame_static_obstacles_and_manual_confirmed_rigid_grasp',
        'scene_xml_path':str(Path(site['execution_scene_xml']).resolve()),
        'joint_name_mapping':{n:f'left_joint_{i}' for i,n in enumerate(names,1)},
        'physical_robot_ip':site['execution_robot_ip'],
        'device_evidence_mode':getattr(backend,'evidence_mode','real_sdk'),
        'tcp_validation':{'mode':'automatic_model_sdk_fk_consistency',
            'max_translation_m':.003,'max_rotation_deg':1.,
            'profile':site.get('frame_profile'),
            'physical_contact_point_status':'not_assessed_by_fk_consistency'},
        'support_adjustment':site.get('support_adjustment'),'contact_evidence':'manual_not_force_sensor'}
    pool=pool_factory(planner_factory=_RecordingPlanner,max_entries=2)
    from gene_agent.skills.library.pick.brush import brush_pick_source_hashes
    report['code_sources_sha256'] = brush_pick_source_hashes()
    grip=None
    hold_width=None
    frames=[]
    initial_qpos=data.qpos.copy()
    def save():
        write_json(output/'execution_result.json',report)
        write_json(output/'executed_frames.json',frames)
        return report
    def sync():
        state=backend.call('state')
        if hold_width is not None and abs(float(state['gripper_width_m'])-hold_width)>config['max_width_error_m']:
            raise ValueError('夹爪开口偏离已确认夹持状态，停止刚性attachment接续')
        q=np.asarray(state['joint_positions_rad'],float)
        model_frames(model,data,robot,q,float(state['gripper_width_m']))
        if grip is not None:_set_prediction(model,data,robot,q,grip)
        state['T_world_tcp_model']=pose_from_mujoco(data.site('pika_tcp')).tolist()
        if 'T_sdk_tcp' in state:
            world_from_sdk=np.linalg.inv(site['T_sdk_urdf_base'])
            fk_error=pose_error(np.asarray(state['T_world_tcp_model']),world_from_sdk@state['T_sdk_tcp'])
            state['sdk_model_fk_error']=fk_error
            if fk_error['translation_m']>.003 or fk_error['rotation_deg']>1.:
                raise ValueError(f'现场 SDK 与规划模型 TCP 不一致：{fk_error}')
        return state
    def check_initial(state):
        initial=np.asarray(snapshot['joint_positions_rad'],float)
        if np.max(np.abs(initial-state['joint_positions_rad']))>config['max_start_error_rad']:
            raise ValueError('实机关节偏离拍照初态；请重新采集')
    def check_actual(state,contacts):
        return check_world_contacts(model,data,robot,[state['joint_positions_rad']],grip,contacts,
            base_table_mount=site.get('base_table_mounting_contact',False))
    def motion(stage,target,path_kind,duration_s,*,allowed_contacts):
        before=sync()
        before_qpos=data.qpos.copy()
        try:
            row,_=stage_planner(model,data,robot,pool,output/'stages'/stage,stage,target,path_kind,
                allowed_contacts,grip,base_table_mount=site.get('base_table_mounting_contact',False))
        finally:
            # plan_stage内部只更新预测状态；发令与实测接续不从预测终点开始。
            data.qpos[:]=before_qpos;mujoco.mj_forward(model,data)
        row['observed_before']=before
        row['attachment']='manual_confirmed_rigid_grasp' if grip is not None else None
        report['stages'].append(row);save()
        if not row['success']:
            return {'success':False, 'executed':False,
                    'failure_type':row.get('failure_type', 'planning_failed')}
        row['planning_success']=True
        row['success']=False
        row['execution_status']='not_sent'
        save()
        execution=backend.call('motion',{'stage':row})
        row['execution']=execution
        row['execution_status']=execution.get('status','unknown')
        after=sync()
        row['observed_after']=after
        frames.append({'stage':stage,'qpos':data.qpos.tolist(),'source':'SDK_measured_with_rigid_object_hypothesis'})
        if execution.get('success') and execution.get('status')=='executed':
            error=pose_error(target,np.asarray(after['T_world_tcp_model']))
            post=check_actual(after,allowed_contacts)
            row['actual_tcp_target_error']=error;row['actual_world_postcheck']=post
            row['success']=bool(post['success'] and error['translation_m']<=config['max_endpoint_translation_m']
                                and error['rotation_deg']<=config['max_endpoint_rotation_deg'])
        else:row['success']=False
        save()
        return {'success':row['success'], 'execution':execution,
                'executed':execution.get('status') == 'executed'}
    def gripper(stage,target):
        before=sync()
        # 只为当前夹爪阶段导出真实宽度/起点下的完整碰撞快照。
        from gene_agent.motion.curobo.scene import export_collision_scene
        collision=export_collision_scene(model,data,robot,'left',output/'stages'/stage/'collision')
        row={'name':stage,'target_width_m':target,'observed_before':before,'success':False}
        report['stages'].append(row);save()
        initial_gripper_qpos=data.qpos.copy()
        try:
            for width in np.linspace(before['gripper_width_m'],target,20):
                model_frames(model,data,robot,before['joint_positions_rad'],float(width))
                check=check_actual(before,TABLE_CONTACTS if stage=='close' else (('brush','table'),))
                if not check['success']:
                    row['gripper_geometry_postcheck']=check
                    save();return False
        finally:
            data.qpos[:]=initial_gripper_qpos;mujoco.mj_forward(model,data)
        execution=backend.call('gripper',{'name':stage,'target_width_m':target,
            'joint_positions_rad':before['joint_positions_rad'],'gripper_width_m':before['gripper_width_m'],
            'snapshot_sha256':collision['snapshot_sha256']})
        row['execution']=execution;row['observed_after']=sync()
        row['success']=bool(execution.get('success') and execution.get('status')=='executed')
        frames.append({'stage':stage,'qpos':data.qpos.tolist(),'source':'SDK_measured'})
        save();return row['success']
    def fail(stage):
        report.update(status='failed',failure_stage=stage);return save()
    try:
        state=sync();check_initial(state)
        frames.append({'stage':'initial_sdk_measured','qpos':data.qpos.tolist(),'source':'SDK_measured'})
        report['initial_observed']=state
        initial_check=check_actual(state,TABLE_CONTACTS)
        report['initial_world_postcheck']=initial_check
        if not initial_check['success']:return fail('initial_collision')
        brush=pose_from_mujoco(data.body('brush'))
        profile = load_brush_pick_profile('realworld', overrides=site.get('pick_overrides'))
        report['pick_profile'] = profile.to_dict()
        targets = brush_pick_targets(brush, Path(site['execution_concept_dir'])/'brush.json', profile,
            grasp_center_tcp_m=pika_grasp_center_tcp(robot,'left'))
        grasp, pregrasp = targets['approach'], targets['pregrasp']
        serial = 0
        def observe_pick():
            nonlocal serial
            serial += 1
            return OperationState(str(serial), report['source'], sync())
        def perform_pick(stage):
            nonlocal grip, hold_width
            if stage in {'open', 'close'}:
                return {'success': gripper(stage, profile.values['open_width_m'] if stage == 'open' else 0.)}
            if stage == 'grasp_check':
                check = resolve_unknown(CheckResult('unknown','no_contact_sensor',{},'夹爪停稳不能证明夹住刷柄'),
                    ask=ask, question='刷柄是否已被Pika稳定夹住？[y/n/?] ')
                report['grasp_check'] = check.to_dict(); save()
                return {'success': True, 'check': check}
            if stage == 'establish_hold':
                grasp_state = sync()
                last_close = next(row for row in reversed(report['stages']) if row['name']=='close')['observed_after']
                if np.max(np.abs(np.asarray(grasp_state['joint_positions_rad'])-last_close['joint_positions_rad'])) > config['max_start_error_rad']:
                    report['failure_stage'] = 'state_changed_during_grasp_check'
                    return {'success': False}
                if abs(grasp_state['gripper_width_m']-last_close['gripper_width_m']) > config['max_width_error_m']:
                    report['failure_stage'] = 'gripper_changed_during_grasp_check'
                    return {'success': False}
                grip = np.linalg.inv(pose_from_mujoco(data.site('pika_tcp'))) @ brush
                hold_width = float(grasp_state['gripper_width_m'])
                report['T_tcp_brush'] = grip.tolist()
                return {'success': True}
            if stage == 'hold_check':
                check = resolve_unknown(CheckResult('unknown','no_slip_sensor',{},'需确认抬升后未滑动'),
                    ask=ask, question='抬升后刷子是否稳定且可继续清扫？[y/n/?] ')
                report['lift_hold_check'] = check.to_dict(); save()
                return {'success': True, 'check': check}
            target = (brush_lift_target(pose_from_mujoco(data.site('pika_tcp')), pregrasp, profile)
                      if stage == 'lift' else targets[stage])
            return motion(stage, target, 'joint' if stage == 'pregrasp' else 'linear_tcp',
                3. if stage != 'lift' else 2.5,
                allowed_contacts=(('brush','table'),) if stage=='pregrasp' else TABLE_CONTACTS)
        flow = run_pick_flow(CallbackOperationBackend(observe=observe_pick, perform=perform_pick,
            evidence_mode=report['source']), stop_after=stop_after if stop_after != 'task' else 'pick',
            verify_hold=stop_after == 'task' or profile.values['verify_hold_for_pick'])
        report['pick_flow'] = flow.to_dict()
        if not flow.success:
            aliases = {'grasp_check':'grasp_not_confirmed', 'hold_check':'lift_hold_not_confirmed'}
            return fail(report.get('failure_stage') or aliases.get(flow.failure_stage, flow.failure_stage))
        if stop_after != 'task':
            report.update(status='stopped_as_requested', motion_complete=True); return save()
        sync()
        poses={n:pose_from_mujoco(data.body(n)) for n in ('brush','cube','dustpan')}
        concepts=Path(site['execution_concept_dir'])
        table_z=float(data.geom('scene_table').xpos[2]+model.geom('scene_table').size[2])
        candidates=make_sweep_pose_sequences(pose_from_mujoco(data.site('pika_tcp')),poses,concepts,
            table_top_z_m=table_z,initial_brush_z=poses['brush'][2,3],orientation_policy='grip_relative')
        # 从本次完整预览选中的几何候选开始；每段轨迹仍从实测关节重新规划。
        index=int(site.get('execution_candidate_index',0))
        if index>=len(candidates):raise ValueError('清扫候选序号已失效')
        candidate=correct_sequence_table_clearance(candidates[index],grip,
            collision_vertices_object(model,data,'brush'),table_z)
        targets=candidate['targets'];report['sweep_candidate']=candidate
        def target_for_brush(xyz):
            target=np.asarray(targets['sweep']).copy();target[:3,3]=np.asarray(xyz)-target[:3,:3]@grip[:3,3]
            return target
        result=run_sweep_skill(make_sweep_layout(poses,concepts), contacts=SWEEP_CONTACTS_PROFILE, sweep_targets=('cube',),initial_brush_z=poses['brush'][2,3],
            move_stage=motion,target_for_brush=target_for_brush,current_brush_xyz=lambda:data.body('brush').xpos.copy(),
            prepared_targets=targets,sweep_end_for_current_pose=lambda:(np.asarray(targets['sweep'])@grip)[:3,3],
            retreat_delta_world_m=np.asarray(candidate['retreat_delta_world_m']))
        report['sweep']=result
        if not result['success']:return fail(result['failure_stage'])
        report['motion_complete']=True
        task_check=resolve_unknown(CheckResult('unknown','no_final_visual_check',{},'运动完成不能证明方块入簸箕'),
            ask=ask,question='方块是否已完整扫入簸箕？[y/n/?] ')
        report['task_check']=task_check.to_dict()
        report['task_success']=True if task_check.passed else False if task_check.status=='fail' else None
        report['status']='executed' if task_check.passed else 'task_failed' if task_check.status=='fail' else 'task_unknown'
        return save()
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        save()
        raise
    finally:
        pool.clear()
