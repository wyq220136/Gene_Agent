"""抓取前的候选筛选；预测目标只作筛选，不能替代实际持物规划/执行检查。"""
import json
import time
import numpy as np
from gene_agent.motion import CollisionAllowance,CollisionPolicy,JointState,PoseTarget
from gene_agent.knowledge.geometry.transforms import validate_pose
from gene_agent.knowledge.operation.hit import align_hit_tcp_candidates


def _bounded_alignment_candidate(result,budget):
    """仅保留待重查的有界IK候选；不改写失败标志、不把关节解当轨迹。"""
    diagnostic=getattr(getattr(result,'failure',None),'details',{}).get('ik_diagnostic') or {}
    for index,seed in enumerate(diagnostic.get('seeds') or []):
        try:
            errors=np.asarray([seed['position_error_m'],seed['rotation_error_rad']],float)
            q=np.asarray(seed['joint_positions_rad'],float)
        except (KeyError,TypeError,ValueError):continue
        if (errors.shape!=(2,) or not np.isfinite(errors).all() or np.any(errors<0)
                or errors[0]>budget['max_position_error_m'] or errors[1]>budget['max_rotation_error_rad']
                or q.ndim!=1 or not q.size or not np.isfinite(q).all()):continue
        check=seed.get('collision_check') or {}
        if index==diagnostic.get('selected_seed_index') and diagnostic.get('collision_checked'):
            check=diagnostic.get('collision_check') or check
        if any(check.get(k) for k in ('joint_limit_violation','self_collision','scene_collision')):continue
        return True
    return False


def first_feasible_candidate(results,stride,*,alignment_budget=None):
    for i in range(len(results)//stride):
        row=results[i*stride:(i+1)*stride]
        if row[0].success and (stride==1 or any(r.success for r in row[1:])
                or (alignment_budget is not None and any(_bounded_alignment_candidate(r,alignment_budget) for r in row[1:]))):return i
    return None


def screen_approach_path(backend,snapshot,start,revision,grasp,*,index,approach_snapshot=None,approach_policy=None):
    """执行前预检两段连接；直线起点来自预抓取规划终态，不改变真实场景。"""
    group=backend.request.group
    axis=np.asarray(backend.robot['grippers'][group]['approach_axis_tcp'],float)
    if axis.shape!=(3,) or not np.isclose(np.linalg.norm(axis),1.):
        raise ValueError('approach_axis_tcp必须单位向量')
    pre=grasp.copy();pre[:3,3]-=pre[:3,:3]@axis*backend.request.settings.approach_distance_m
    kwargs=dict(snapshot=snapshot,group=group,scene_revision=revision)
    with backend.session.stage(start=start,**kwargs) as motion:
        budget=getattr(backend.request.settings,'pregrasp_approximate_ik_budget',None)
        plan=motion.plan_to(PoseTarget('world_tcp',pre),request_id=f'candidate/{group}/{index}/pregrasp',
            **({'approximate_ik_budget':budget} if budget is not None else {}))
    evidence={'pregrasp':plan.to_dict()}
    if not plan.success:return False,evidence
    if approach_snapshot is not None:kwargs['snapshot']=approach_snapshot
    if approach_policy is not None:kwargs['collision_policy']=approach_policy
    with backend.session.stage(start=plan.goal_state,**kwargs) as motion:
        line=motion.plan_linear(PoseTarget('world_tcp',grasp),request_id=f'candidate/{group}/{index}/approach')
    evidence['approach']=line.to_dict()
    return line.success,evidence


def screen_held_lift(backend,observed,grasp,local,start,*,index):
    """仅预测理想夹持的抬升可行性；保留当前张爪模型，实际闭爪后必须重新检查。"""
    group=backend.request.group
    link_from_tcp=np.asarray(backend.robot['transforms'][group]['link_from_tcp'])
    held=dict(object_id=backend.request.object_id,T_link_object=(link_from_tcp@np.linalg.inv(local)).tolist())
    policy=CollisionPolicy('lift_support',(CollisionAllowance('gene_held_tool','table','从实际支撑面抬起工具'),))
    snapshot=backend.scene.snapshot(state=observed,group=group,collision_policy=policy,held_object=held)
    target=grasp.copy();target[2,3]+=backend.request.settings.lift_distance_m
    with backend.session.stage(snapshot=snapshot,start=start,group=group,
            scene_revision=observed.values['scene_revision'],collision_policy=policy) as motion:
        result=motion.plan_linear(PoseTarget('world_tcp',target),request_id=f'candidate/{group}/{index}/predicted_lift')
    return result.success,result.to_dict()


def evaluate_candidate_ik(session,targets,stride,stage_kwargs,*,contact_snapshot=None,contact_policy=None):
    """抓取终态沿用接近接触规则；预测对齐仍独立检查严格碰撞约束。"""
    results=[None]*len(targets)
    batches=[(list(range(len(targets))),stage_kwargs)]
    if contact_snapshot is not None:
        batches=[([i for i in range(len(targets)) if i%stride==0],
                  dict(stage_kwargs,snapshot=contact_snapshot,collision_policy=contact_policy)),
                 ([i for i in range(len(targets)) if i%stride!=0],stage_kwargs)]
    for indices,kwargs in batches:
        if not indices:continue
        with session.stage(**kwargs) as motion:
            batch=motion.evaluate_ik([targets[i] for i in indices],
                request_ids=[f"candidate/{stage_kwargs['group']}/{i}" for i in indices])
        if len(batch)!=len(indices):raise ValueError('候选IK返回数量不匹配')
        for i,result in zip(indices,batch):results[i]=result
    return results


def select_grasp(backend, start):
    """筛选父级wall包括子规划/场景，不与子项重复相加；异常也保留时间。"""
    started = time.perf_counter()
    try:
        return _select_grasp(backend, start)
    finally:
        backend.recorder.append_event(dict(episode_id=start.values['episode_id'],
            stage_id='candidate', kind='candidate_timing', state_id=start.state_id,
            group=backend.request.group, wall_s=time.perf_counter()-started))


def _select_grasp(backend,start):
    """不移动机器人；同一真实场景批查抓取和预测对齐TCP的IK/碰撞。"""
    pool=backend.candidate.get('grasp_candidates')
    if not pool:return True
    group=backend.request.group
    use_hit=bool(backend.candidate.get('_screen_hit'))
    alignment_budget=(backend.candidate.get('hit_parameters',{}).get('approximate_ik_by_stage',{}).get('align') if use_hit else None)
    observed=backend.runtime.observe(object_names=(backend.request.object_id,'target') if use_hit else (backend.request.object_id,),robot_groups=(group,))
    if observed.state_id!=start.state_id:raise ValueError('候选检查时实际状态改变')
    tool=np.asarray(observed.values['objects'][backend.request.object_id]['T_world_object'])
    targets=[];locals_=[];stride=None
    for candidate in pool:
        local=validate_pose(candidate['T_object_tcp']);locals_.append(local)
        grasp=tool@local;goals=[grasp]
        if use_hit:
            p=backend.candidate['hit_parameters'];target=np.asarray(observed.values['objects']['target']['T_world_object'])
            goals.extend(align_hit_tcp_candidates(tool,grasp,target,strike_point=p['strike_point_object_m'],strike_normal=p['strike_normal_object'],target_point=p['target_point_object_m'],clearance_m=p['clearance_m'],yaw_offsets_rad=p.get('screen_yaw_offsets_rad',[0.,-.2617993878,-.5235987756])))
        stride=len(goals);targets.extend(PoseTarget('world_tcp',g) for g in goals)
    snapshot=backend.scene.snapshot(state=observed,group=group,collision_policy=CollisionPolicy())
    robot=observed.values['robots'][group]
    approach_options={}
    if backend.candidate.get('allow_finger_tool_contact'):
        contact=backend.approach_collision_policy()
        approach_options=dict(approach_policy=contact,approach_snapshot=backend.scene.snapshot(state=observed,group=group,collision_policy=contact))
    results=evaluate_candidate_ik(backend.session,targets,stride,
        dict(snapshot=snapshot,start=JointState(tuple(robot['joint_names']),tuple(robot['joint_positions_rad'])),group=group,scene_revision=observed.values['scene_revision']),
        contact_snapshot=approach_options.get('approach_snapshot'),contact_policy=approach_options.get('approach_policy'))
    selected=first_feasible_candidate(results,stride,alignment_budget=alignment_budget)
    path_screens=[]
    if backend.candidate.get('screen_approach_path'):
        selected=None
        for i,local in enumerate(locals_):
            row=results[i*stride:(i+1)*stride]
            if first_feasible_candidate(row,stride,alignment_budget=alignment_budget) is None:continue
            okay,record=screen_approach_path(backend,snapshot,
                JointState(tuple(robot['joint_names']),tuple(robot['joint_positions_rad'])),
                observed.values['scene_revision'],tool@local,index=i,**approach_options)
            if okay and backend.candidate.get('screen_held_lift'):
                end=record['approach']['goal_state']
                okay,record['predicted_lift']=screen_held_lift(backend,observed,tool@local,local,
                    JointState(tuple(end['joint_names']),tuple(end['positions_rad'])),index=i)
            path_screens.append(dict(candidate_index=i,**record))
            if okay:
                selected=i;break
    evidence=dict(state_id=start.state_id,group=group,selected_index=selected,stride=stride,
        scope='predicted geometry only; optional held lift uses ideal grasp/current opening; actual closed-gripper attachment and execution checked later',
        alignment_ik_budget=alignment_budget,
        alignment_validation=('deferred_to_actual_planning' if selected is not None and stride>1
            and not any(r.success for r in results[selected*stride+1:(selected+1)*stride]) else 'strict_ik' if stride>1 else 'not_requested'),
        path_screens=path_screens,candidates=pool,results=[r.to_dict() for r in results])
    (backend.recorder.root/f'candidate_screen_{group}.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    backend.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id='pregrasp',kind='candidate_selection',state_id=start.state_id,group=group,selected_index=selected,candidate_count=len(pool)))
    if selected is None:return False
    backend.local=locals_[selected]
    backend.candidate['selected_grasp']=pool[selected]
    (backend.recorder.root/f'candidate_selected_{group}.json').write_text(json.dumps(backend.candidate,ensure_ascii=False,indent=2))
    return True
