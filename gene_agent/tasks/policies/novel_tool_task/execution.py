"""原生能力检查和共享Skill的装配；任务目标不从隐藏评分状态生成。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from gene_agent.motion import JointState,PoseTarget,CollisionPolicy
from gene_agent.motion.curobo.process import ProcessPlanningSession


def select_group(candidate,robot,state):
    """可选按当前目标与base距离选臂；不读取初态设置或专家选臂。"""
    if candidate.get('group_selection')=='nearest_target':
        target=np.asarray(state.values['objects']['target']['T_world_object'])[:2,3]
        return min(robot['joint_groups'],key=lambda g:np.linalg.norm(np.asarray(robot['transforms'][g]['world_from_urdf_base'])[:2,3]-target))
    if candidate.get('group_selection') is not None:raise ValueError('未知选臂方式')
    group=candidate.get('group',next(iter(robot['joint_groups'])))
    if group not in robot['joint_groups']:raise ValueError('未知机器人组')
    return group


def run_operation(operation, *, runtime,robot,start,recorder,options,output):
    """每episode保持一个规划worker/pool，阶段间传递真实执行状态。"""
    with ProcessPlanningSession(robot,python=options.get('gene_planner_python') or sys.executable,
            pythonpath=options.get('gene_planner_pythonpath'),log_path=output/'planner-worker.log',
            request_output_dir=output/'motion_requests' if options.get('gene_save_motion_requests') else None,
            diagnostic_options=dict(mode=options.get('gene_diagnostic_mode','off'),budget_s=options.get('gene_diagnostic_budget_s',5.))) as session:
        dependency=session.describe()
        recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id='planner_setup',kind='dependency',**dependency))
        if not dependency['api_available']:
            raise RuntimeError(f"规划worker缺少所需cuRobo API；实际来源 {dependency['curobo_path']}，解释器 {dependency['python']}")
        # scene provider由组合根注入原环境；runtime不暴露native_env。
        scene_provider=options['_scene_provider']
        if operation in {'pick','hit'}:
            from gene_agent.skills.library import pick
            from gene_agent.skills.library.pick.model import PickRequest,GraspSettings
            from .pick_backend import make_pick_backend
            path=Path(options['gene_candidate']);candidate=json.loads(path.read_text())
            # 将有效几何输入随run保存，后续调参不能改写本次候选的来源事实。
            source_inputs={}
            for key in ('concept_ref','grasp_profile_ref'):
                if candidate.get(key):
                    source=Path(candidate[key]);payload=source.read_bytes()
                    source_inputs[key]=dict(path=str(source.resolve()),sha256=hashlib.sha256(payload).hexdigest(),content=json.loads(payload))
            (output/'geometry_inputs.json').write_text(json.dumps(source_inputs,ensure_ascii=False,indent=2))
            preferred=select_group(candidate,robot,start)
            groups=[preferred]+([g for g in robot['joint_groups'] if g!=preferred] if candidate.get('search_other_group') else [])
            candidate={**candidate,'_screen_hit':operation=='hit'}
            for group in groups:
                (output/'candidate.resolved.json').write_text(json.dumps({**candidate,'selected_group':group},ensure_ascii=False,indent=2))
                request=PickRequest('tool',group,settings=GraspSettings(**candidate.get('skill_parameters',{})))
                recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id='candidate',kind='input',
                    candidate_ref=str(path.resolve()),candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    annotation_source=candidate.get('source','explicit_user_geometry'),group=group))
                backend=make_pick_backend(runtime=runtime,scene_provider=scene_provider,planning_session=session,
                    request=request,candidate=candidate,recorder=recorder,robot_metadata=robot,failure_image_capture=options.get("_failure_image_capture") if options.get("gene_diagnostic_mode")=="on_failure" else None)
                result=pick.run(backend,request,stop_after='pick' if operation=='hit' else options.get('gene_stop_after','pregrasp'))
                if result.success:break
                # 仅筛选未通过时换臂；已经尝试接近/夹持/执行失败则正常终止。
                if result.failure_code!='no_feasible_grasp_candidate':return result
            if operation=='pick' or not result.success:return result
            from gene_agent.skills.library import hit
            from .hit_backend import NativeHitBackend
            parameters=candidate['hit_parameters']
            return hit.run(NativeHitBackend(backend,parameters,target_body=options['_target_body']),
                withdraw=float(parameters.get('withdraw_distance_m',0.))>0)
        group=next(iter(robot['joint_groups']))
        snapshot=scene_provider.snapshot(state=start,group=group,collision_policy=CollisionPolicy())
        row=start.values['robots'][group]
        target=np.asarray(row['T_world_tcp']).copy();target[2,3]+=.01
        with session.stage(snapshot=snapshot,start=JointState(tuple(row['joint_names']),tuple(row['joint_positions_rad'])),group=group,scene_revision=start.values['scene_revision']) as motion:
            result=motion.plan_linear(PoseTarget('world_tcp',target),request_id='native_motion/linear')
        recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id='linear_motion',kind='planning',state_id=start.state_id,result=result.to_dict()))
        (output/'motion_result.json').write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2))
        if not result.success:return result
        outcome=runtime.execute_trajectory(result.trajectory,group=group,stage_id='linear_motion',expected_state_id=start.state_id)
        recorder.append_state('linear_motion_end',outcome.end)
        recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id='linear_motion',kind='execution',state_id=outcome.end.state_id,
                                   success=outcome.success,details=outcome.details,failure=outcome.failure))
        return outcome
