"""有预算的失败证据采集；不执行动作、不修改正式求解结果。"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import time
import numpy as np
from functools import lru_cache


@dataclass(frozen=True)
class DiagnosticOptions:
    mode: str = 'off'
    budget_s: float = 5.
    max_solver_calls: int = 1
    max_candidates: int = 3

    def __post_init__(self):
        if self.mode not in ('off','on_failure') or not math.isfinite(self.budget_s) or self.budget_s < 0:
            raise ValueError('诊断mode/budget无效')
        if not 0 <= self.max_solver_calls <= 1 or not 0 <= self.max_candidates <= 3:
            raise ValueError('诊断最多一次补充IK、三个候选')


class DiagnosticBudget:
    """软时限：单个CUDA调用不可强行打断，记录超支并停止后续检查。"""
    def __init__(self, options):
        self.options=options; self.started=time.monotonic(); self.solver_calls=0
        self.skipped=[]; self.checks=[]

    def allow(self, name, *, solver=False):
        if (time.monotonic()-self.started >= self.options.budget_s or
                solver and self.solver_calls >= self.options.max_solver_calls):
            self.skipped.append(name);return False
        self.checks.append(name)
        if solver:self.solver_calls+=1
        return True

    def to_dict(self):
        elapsed=time.monotonic()-self.started
        return dict(budget_s=self.options.budget_s,elapsed_s=elapsed,over_budget=elapsed>self.options.budget_s,
                    solver_calls=self.solver_calls,checks=self.checks,skipped=self.skipped,hard_timeout=False)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=lambda v:v.tolist()).encode()).hexdigest()


def build_sphere_identity(links, spheres_local, *, model_hash, snapshot_hash, group,
                          held_object=None, gripper_links=()):
    """编号只在本次有效模型/快照内有意义；来源缺失时不猜Concept部件。"""
    rows=[];held_index=0;held_object=held_object or {}
    for index,(link,sphere) in enumerate(zip(links,spheres_local)):
        role='held' if link=='gene_held_tool' else 'gripper' if link in gripper_links else 'robot'
        oid=held_object.get('object_id','unknown_held_object') if role=='held' else 'robot'
        source={}
        if role=='held':
            sources=held_object.get('sphere_sources',[])
            if held_index<len(sources):source=dict(sources[held_index])
            held_index+=1
        rows.append(dict(model_hash=model_hash,snapshot_hash=snapshot_hash,sphere_index=index,
            role=role,group=group,link=link,object_id=oid,**source,
            center_local_m=np.asarray(sphere[:3],float).tolist(),radius_m=float(sphere[3]),local_frame=link,
            label=f'{group}/{oid}/{link}/sphere#{index}'))
    return rows


def candidate_diagnostics(planner, joint_names, target, seeds, budget, *, tolerances=None):
    """仅比较本次有限搜索；独立FK与约束检查，近似候选从不直接执行。"""
    from .approximate import _pose_error
    tolerances=tolerances or {}
    pt=tolerances.get('position_tolerance_m');rt=tolerances.get('rotation_tolerance_rad')
    known=all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (pt,rt))
    pscale,rscale=(pt,rt) if known else (.005,.05)
    def score(s):
        return float(s.get('position_error_m',float('inf')))/pscale + float(s.get('rotation_error_rad',float('inf')))/rscale
    ordered=sorted(seeds,key=score)
    # 同时给已标为可行的候选检查机会，避免三个碰撞候选耗尽预算。
    selected=ordered[:1]+[s for s in ordered[1:] if s.get('constraint_feasible')]+[s for s in ordered[1:] if not s.get('constraint_feasible')]
    checked=[];errors=[]
    for seed in selected[:budget.options.max_candidates]:
        if not budget.allow('candidate_fk_and_constraints'):break
        try:
            q=np.asarray(seed['joint_positions_rad'],float)
            if q.shape!=(len(joint_names),) or not np.isfinite(q).all():raise ValueError('候选关节向量无效')
            fk=np.asarray(planner.fk(joint_names,q),float)
            if fk.shape!=(4,4) or not np.isfinite(fk).all():raise ValueError('FK无效')
            pos,rot=_pose_error(np.asarray(target),fk)
            check=planner._collision_diagnostic(q) if budget.allow('candidate_constraint_query') else {}
            keys=('joint_limit_violation','self_collision','scene_collision')
            feasible=not any(check[k] for k in keys) if all(check.get(k) is not None for k in keys) else None
            checked.append(dict(seed_index=seed.get('index'),joint_names=list(joint_names),joint_positions_rad=q.tolist(),
                requested_tcp_pose_world=np.asarray(target).tolist(),fk_tcp_pose_world=fk.tolist(),
                translation_delta_world_m=(fk[:3,3]-np.asarray(target)[:3,3]).tolist(),
                position_error_m=pos,rotation_error_rad=rot,solver_position_error_m=seed.get('position_error_m'),
                solver_rotation_error_rad=seed.get('rotation_error_rad'),constraint_feasible=feasible,
                strict_pose_converged=bool(pos<pt and rot<rt) if known else None,
                position_tolerance_m=pt,rotation_tolerance_rad=rt,constraint_check=check))
        except Exception as exc:errors.append(dict(seed_index=seed.get('index'),error=f'{type(exc).__name__}: {exc}'))
    metric=lambda c:c['position_error_m']/pscale+c['rotation_error_rad']/rscale
    feasible=[c for c in checked if c['constraint_feasible'] is True]
    return dict(checked=checked,nearest_checked=min(checked,key=metric) if checked else None,
        nearest_checked_feasible=min(feasible,key=metric) if feasible else None,
        searched_candidate_count=len(seeds),checked_count=len(checked),errors=errors,
        ranking=f'position_error/{pscale}m + rotation_error/{rscale}rad; checked subset only',
        ranking_scale_source='recorded_solver_tolerances' if known else 'display_ranking_only_not_acceptance',
        scope='pose candidates only; no global nearest or executable path guarantee')


def retain_failed_trajectories(values, joint_count):
    """保留后端真实轨迹的batch边界，最多两条，不插值或补造路径。"""
    values=np.asarray(values,float)
    if values.ndim<2 or values.shape[-1]!=joint_count:
        raise ValueError('失败轨迹维度不匹配')
    paths=values.reshape(-1,values.shape[-2],joint_count)
    return [dict(positions_rad=path.tolist(),trajectory_index=i,returned_trajectory_count=len(paths),
                 source='backend_failed_js_solution',original_shape=list(values.shape))
            for i,path in enumerate(paths[:2])]


def check_failed_trajectories(planner, paths, budget):
    """失败后的旁路采样：最多两条、每条64点，不削减正式轨迹验证。"""
    count=0;first=None
    for path in paths[:2]:
        if not budget.allow('failed_trajectory_samples'):break
        positions=path['positions_rad']
        ids=np.unique(np.linspace(0,len(positions)-1,min(64,len(positions))).astype(int))
        validity=planner._configuration_validity([positions[i] for i in ids])
        path['checked_waypoint_indices']=ids.tolist();path['checked_validity']=list(validity)
        path['check_scope']='sampled_only_not_continuous_path_proof';count+=len(validity)
        if False in validity:
            index=int(ids[validity.index(False)]);path['first_invalid_waypoint']=index
            if first is None:first=(path['trajectory_index'],index,positions[index])
    return dict(sampled_waypoints=count,first_invalid=first)


def select_failure_configuration(raw, paths, candidate, start):
    """球级证据绑定明确的候选或轨迹点；起点失败优先检查实际起点。"""
    if raw.get('failure_category')=='start_collision':return list(start),dict(kind='start')
    if candidate:return candidate['joint_positions_rad'],dict(kind='ik_candidate',seed_index=candidate.get('seed_index'))
    selected=(raw.get('ik_diagnostic') or {}).get('selected_joint_positions_rad')
    if selected is not None:return selected,dict(kind='ik_candidate',seed_index=(raw.get('ik_diagnostic') or {}).get('selected_seed_index'))
    for path in paths:
        if path.get('first_invalid_waypoint') is not None:
            i=path['first_invalid_waypoint']
            return path['positions_rad'][i],dict(kind='failed_trajectory',trajectory_index=path.get('trajectory_index',0),waypoint_index=i)
    return list(start),dict(kind='start',reason='no_checked_invalid_trajectory_point')


def effective_spheres(planner, q, snapshot):
    """从有效cuRobo模型导出真实球顺序和FK；全部球保留供离线查看。"""
    kin=planner.kinematics;params=kin.config.kinematics_config
    state=kin.compute_kinematics(planner._state(planner.joint_names,q))
    spheres=state.robot_spheres.detach().cpu().numpy().reshape(-1,4)
    ids=params.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
    names={int(v):k for k,v in params.link_name_to_idx_map.items()}
    local=params.link_spheres.detach().cpu().numpy().reshape(-1,4)
    gripper=snapshot.get('gripper_snapshot') or {}
    gripper_links=set(gripper.get('added_spheres_by_link',{}))
    urdf=planner.robot_config.get('kinematics',{}).get('urdf_path')
    if urdf and gripper.get('joint_names'):
        import xml.etree.ElementTree as ET
        for joint in ET.parse(urdf).getroot().findall('joint'):
            if joint.get('name') in gripper['joint_names'] and joint.find('child') is not None:
                gripper_links.add(joint.find('child').get('link'))
    identities=build_sphere_identity([names[int(i)] for i in ids],local,
        model_hash=digest(planner.robot_config),snapshot_hash=planner.scene_snapshot_sha256,
        group=snapshot.get('active_group','unknown'),held_object=planner.held_object,
        gripper_links=gripper_links)
    for row,sphere in zip(identities,spheres):
        row['center_base_m']=sphere[:3].tolist()
        row['center_world_m']=(planner.world_from_base[:3,:3]@sphere[:3]+planner.world_from_base[:3,3]).tolist()
        row['effective_radius_m']=float(sphere[3])
    return identities,spheres


def sphere_collisions(planner, q, snapshot, budget):
    """self使用原pair/padding；world逐mesh检查，所有enable状态finally恢复。"""
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo.types import DeviceCfg
    identities,spheres=effective_spheres(planner,q,snapshot)
    kin=planner.kinematics;cfg=kin.get_self_collision_config();torch=planner.torch
    pairs=cfg.collision_pairs.detach().cpu().numpy().reshape(-1,2)
    pad=cfg.sphere_padding.detach().cpu().numpy().reshape(-1)
    hits=[]
    for a,b in pairs:
        overlap=spheres[a,3]+spheres[b,3]+pad[a]+pad[b]-np.linalg.norm(spheres[a,:3]-spheres[b,:3])
        if spheres[a,3]>0 and spheres[b,3]>0 and overlap>1e-7:
            hits.append(dict(kind='self',first=identities[a],second=identities[b],value=float(overlap),
                value_type='sphere_overlap_with_self_padding',unit='m',allowed=False,
                position_world_m=((np.array(identities[a]['center_world_m'])+identities[b]['center_world_m'])/2).tolist(),
                position_kind='sphere_centers_midpoint_not_mesh_contact'))
    if budget.allow('world_sphere_pairs'):
        if planner._collision_checker is None:planner._configuration_validity([q])
        world=planner._collision_checker.scene_model
        records={r['obstacle_name']:r for r in snapshot.get('records',[]) if r.get('obstacle_name')}
        states=[]
        for key in ('cuboids','meshes','voxels'):
            collection=getattr(world.data,key,None)
            if collection is not None:
                names=tuple(collection.get_names(0));enabled=[bool(collection.enable[0,i].item()) for i in range(len(names))]
                states.append((collection,names,enabled))
        query=torch.as_tensor(spheres,device='cuda',dtype=torch.float32).reshape(1,1,-1,4)
        buffer=CollisionBuffer.from_shape(query.shape,DeviceCfg())
        try:
            for col,names,enabled in states:
                for name,on in zip(names,enabled):
                    if not on:continue
                    if not budget.allow('mesh:'+str(name)):break
                    for other,other_names,other_on in states:
                        for obj,active in zip(other_names,other_on):other.set_enabled(obj,active and other is col and obj==name,0)
                    costs=world.get_sphere_distance_raw(query,buffer,torch.tensor([1.],device='cuda'),torch.tensor([0.],device='cuda'))
                    values=costs.detach().cpu().numpy().reshape(-1)
                    for idx in np.flatnonzero(values>1e-7):
                        entity=identities[int(idx)];policy=planner.mounting_contact_policy
                        allowed=entity['link'] in policy.get('robot_links',[]) and str(name) in policy.get('obstacle_names',[])
                        hits.append(dict(kind='environment',first=entity,
                            second=dict(object_id=records.get(str(name),{}).get('body_name',str(name)),mesh_id=str(name),
                                        source=records.get(str(name),{})),value=float(values[idx]),
                            value_type='curobo_raw_sphere_collision_cost',unit='backend_cost',allowed=allowed,
                            position_world_m=entity['center_world_m'],position_kind='sphere_center_not_mesh_contact'))
        finally:
            for col,names,enabled in states:
                for name,on in zip(names,enabled):col.set_enabled(name,on,0)
    return dict(spheres=identities,pairs=hits[:20],pair_count=len(hits),truncated=len(hits)>20,
                complete=not bool(budget.skipped),scene_snapshot_sha256=planner.scene_snapshot_sha256)


@lru_cache(maxsize=4)
def _cached_dependency_signature(root):
    """进程内依赖源码不热替换；每个包只读取一次源码/配置指纹。"""
    from pathlib import Path
    root=Path(root)
    suffixes={'.py','.so','.cu','.cuh','.cpp','.h','.yaml','.yml'}
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in suffixes}


def dependency_signature(root, *, fresh=False):
    # 采集进程不热替换依赖；离线校验必须重新读取，不能沿用采集缓存。
    return (_cached_dependency_signature.__wrapped__(root) if fresh
            else _cached_dependency_signature(root))


def capture_request_context(planner, request, snapshot):
    """保存与正式planner相同的源配置、状态和资产身份，不执行额外IK。"""
    replay=dict(robot_config=planner._source_robot_config,joint_names=planner.joint_names,
        planner_joint_names=planner.planner_joint_names,world_from_base=planner.world_from_base.tolist(),
        link_from_tcp=planner.link_from_tcp.tolist(),tool_frame=planner.tool_frame,scene_model=planner.scene_model,
        scene_scope=planner.scene_scope,scene_snapshot_sha256=planner.scene_snapshot_sha256,
        mounting_contact_policy=planner.mounting_contact_policy,gripper_snapshot=planner._gripper_snapshot,
        held_object=planner.held_object,attachment_spheres_tool=planner.attachment_spheres_tool,
        planner_mode=planner.planner_mode.value,random_seed=planner.random_seed,
        collision_mode=planner.collision_mode.value)
    from pathlib import Path
    if planner.collision_mode.value == "robot_only":
        replay["robot_protection_scene"] = planner.scene_model
    asset_hashes={}
    def fingerprint(value):
        if isinstance(value,dict):
            for v in value.values():fingerprint(v)
        elif isinstance(value,(list,tuple)):
            for v in value:fingerprint(v)
        elif isinstance(value,str) and value.startswith('/'):
            path=Path(value)
            if path.is_file():asset_hashes[value]=hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint(replay)
    import sys, curobo
    environment=dict(python=sys.version,executable=sys.executable,curobo_module=curobo.__file__)
    fingerprint(curobo.__file__)
    code_hashes={}
    for source in Path(__file__).resolve().parents[1].rglob('*.py'):
        code_hashes[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
    return dict(schema_version=1, environment=environment, asset_hashes=asset_hashes,
        code_hashes=code_hashes, dependency_root=str(Path(curobo.__file__).resolve().parent),
        dependency_hashes=dependency_signature(str(Path(curobo.__file__).resolve().parent)),
        request=request.to_dict(), request_sha256=digest(request.to_dict()),
        snapshot=snapshot, snapshot_payload_sha256=digest(snapshot),
        effective_robot_config=planner.robot_config, replay_config=replay,
        replay_config_sha256=digest(replay))


def collect_failure_diagnostics(planner, request, raw, snapshot, options):
    """在原结果旁路附加证据；异常由调用者记下，原失败不会被覆盖。"""
    budget=getattr(planner,'_failure_diagnostic_budget',None) or DiagnosticBudget(options)
    target=np.asarray(request.target.matrix,float)
    if request.target.frame!='world_tcp':
        return dict(schema_version=1,status='unsupported_target_frame',budget=budget.to_dict())
    ik=raw.get('ik_diagnostic') or {};seeds=ik.get('seeds') or []
    if not seeds and raw.get('solver_diagnostic',{}).get('joint_positions_rad'):
        ik=raw['solver_diagnostic'];seeds=[dict(ik,index=0)]
    candidates=candidate_diagnostics(planner,request.start.joint_names,target,seeds,budget,tolerances=ik)
    trajectories=raw.get('diagnostic_trajectories',[])
    if not trajectories and not raw.get('success') and raw.get('positions_rad'):
        trajectories=retain_failed_trajectories(raw['positions_rad'],len(request.start.joint_names))
        for path in trajectories:path['source']='backend_returned_postcheck_rejected_positions'
        invalid=(raw.get('collision_verification') or {}).get('first_invalid_index')
        if invalid is not None:trajectories[0]['first_invalid_waypoint']=invalid
        else:check_failed_trajectories(planner,trajectories,budget)
    q,configuration=select_failure_configuration(raw,trajectories,candidates['nearest_checked'],request.start.positions_rad)
    collisions=None;errors=[]
    if budget.allow('sphere_identity_and_pairs'):
        try:collisions=sphere_collisions(planner,q,snapshot,budget)
        except Exception as exc:errors.append(f'spheres: {type(exc).__name__}: {exc}')
    captured = capture_request_context(planner, request, snapshot)
    return dict(captured, raw_failure=raw, source_batch_size=raw.get('batch_size',1),
        status='partial' if errors or budget.skipped else 'collected',candidates=candidates,
        invocation='evaluate_candidates' if 'batch_size' in raw else ('solve_ik' if request.operation.value=='ik' else 'plan'),
        configuration_ref=configuration["kind"],configuration_identity=configuration,configuration_joint_positions_rad=list(q),collisions=collisions,
        failed_trajectories=trajectories,trajectory_status='available' if trajectories else 'not_returned',
        errors=errors,budget=budget.to_dict(),media={'actual':'not_captured','diagnostic':'deferred'})


def request_from_dict(req):
    from ..model import MotionRequest, JointState, PoseTarget, PlanningSceneRef, CollisionPolicy, CollisionAllowance, PivotLimits
    cp = req['collision_policy']
    return MotionRequest(request_id=req['request_id'],operation=req['operation'],group=req['group'],
        start=JointState(tuple(req['start']['joint_names']),tuple(req['start']['positions_rad'])),
        target=PoseTarget(req['target']['frame'],req['target']['matrix']),scene=PlanningSceneRef(**req['scene']),
        collision_policy=CollisionPolicy(cp['policy_id'],tuple(CollisionAllowance(**a) for a in cp['allowances'])),
        seed=req['seed'],planner_mode=req['planner_mode'],approximate_ik_budget=req.get('approximate_ik_budget'),
        pivot_limits=PivotLimits(**req['pivot_limits']) if req.get('pivot_limits') else None)


def save_planning_request(planner, requests, results, snapshot, output, *, invocation=None):
    """显式启用的请求记录；batch完整保存，一次批次只写一份现场。"""
    from pathlib import Path
    from uuid import uuid4
    requests, results = tuple(requests), tuple(results)
    payload = capture_request_context(planner, requests[0], snapshot)
    payload.update(mode='frozen_motion_request',source_batch_size=len(requests),
                   invocation=invocation or ('solve_ik' if requests[0].operation.value=='ik' else 'plan'),
                   requests=[r.to_dict() for r in requests],results=[r.to_dict() for r in results])
    payload['requests_sha256'] = digest(payload['requests'])
    out = Path(output);out.mkdir(parents=True, exist_ok=True)
    path = out / (uuid4().hex + '.json')
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist()))
    return path


def replan_bundle(path, output, *, warm_repeats=1, allow_code_change=False):
    """离线真实cuRobo重规划；沿用原约束和adapter，绝不调用执行器/仿真。"""
    from pathlib import Path
    import traceback
    from gene_agent.feedback.motion_diagnostics import validate_diagnostic_bundle
    from .core import SynchronousPlanner
    from .adapter import CuroboPlannerAdapter
    if not isinstance(warm_repeats, int) or not 0 <= warm_repeats <= 5:
        raise ValueError('warm_repeats须在0到5之间')
    source = Path(path).resolve()
    raw = json.loads(source.read_text());bundle=raw.get('diagnostic', raw)
    changes = validate_diagnostic_bundle(bundle,verify_environment=True,allow_code_change=allow_code_change)
    requests = bundle.get('requests') or [bundle['request']]
    if bundle.get('requests') and digest(requests) != bundle.get('requests_sha256'):
        raise ValueError('requests hash不匹配')
    if bundle.get('source_batch_size',1) != len(requests):
        raise ValueError('原batch目标集合缺失，不能改成单目标重规划')
    requests = [request_from_dict(r) for r in requests]
    if requests[0].to_dict() != bundle['request']:
        raise ValueError('batch首个请求与现场request不一致')
    config=bundle['replay_config']
    for request in requests:
        if request.seed != config.get('random_seed') or request.planner_mode.value != config.get('planner_mode'):
            raise ValueError('请求seed/mode与保存的solver配置不一致')
    out = Path(output);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,mode='offline_motion_replan_no_execution',source_bundle=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_changes=changes,
        sampling='同seed重置采样；新solver首调用cold，后续warm；不保证重现原进程采样历史',
        dependency_coverage='package_sources' if bundle.get('dependency_hashes') else 'legacy_module_only',attempts=[])
    planner=None
    try:
        started=time.perf_counter();planner=SynchronousPlanner(**config)
        report['wrapper_setup_s']=time.perf_counter()-started
        adapter=CuroboPlannerAdapter(planner,group=requests[0].group,scene_revision=requests[0].scene.revision)
        for index in range(warm_repeats+1):
            planner.reset_seed()
            started=time.perf_counter()
            if bundle.get("invocation")=="evaluate_candidates" or len(requests)>1:
                results=adapter.evaluate_candidates(requests)
            else:
                request=requests[0]
                results=(adapter.solve_ik(request) if request.operation.value=='ik' else adapter.plan(request),)
            report['attempts'].append(dict(phase='cold' if index==0 else 'warm',
                call_wall_s=time.perf_counter()-started,results=[r.to_dict() for r in results]))
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc()
        raise
    finally:
        if planner is not None:planner.close()
        (out/'replan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda v:v.tolist(),allow_nan=False))
    return out/'replan.json'


def recheck_bundle(path, output, *, budget_s=5.):
    """只从保存的请求重查候选/碰撞，不加载仿真或调用机器人执行。"""
    from pathlib import Path
    from gene_agent.feedback.motion_diagnostics import validate_diagnostic_bundle
    from gene_agent.motion.model import MotionRequest, JointState, PoseTarget, PlanningSceneRef, CollisionPolicy, CollisionAllowance, PivotLimits
    from .core import SynchronousPlanner
    source=json.loads(Path(path).read_text());bundle=source.get('diagnostic',source)
    if not bundle:raise ValueError('历史记录缺完整诊断现场；只允许摘要读取')
    validate_diagnostic_bundle(bundle,verify_environment=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    request=request_from_dict(bundle['request'])
    planner=SynchronousPlanner(**bundle['replay_config'])
    try:
        result=collect_failure_diagnostics(planner,request,bundle.get('raw_failure',{}),bundle['snapshot'],DiagnosticOptions(budget_s=budget_s))
        result['source_bundle']=str(Path(path).resolve());result['mode']='offline_candidate_recheck_no_motion'
        payload=json.dumps(result,ensure_ascii=False,indent=2,default=lambda v:v.tolist(),allow_nan=False)
        (out/'diagnostic.json').write_text(payload)
        return out/'diagnostic.json'
    finally:
        planner.close()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--budget-s',type=float,default=5.)
    args=parser.parse_args();print(recheck_bundle(args.bundle,args.output,budget_s=args.budget_s))


if __name__=='__main__':main()
