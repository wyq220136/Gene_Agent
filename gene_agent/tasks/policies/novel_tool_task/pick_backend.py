"""共享Pick的原生绑定：目标来自Concept候选，执行/诊断与阶段顺序分离。"""
from __future__ import annotations
from dataclasses import replace
import json
import time
import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose,pose_delta
from gene_agent.motion import JointState,PoseTarget,CollisionPolicy,CollisionAllowance
from gene_agent.skills.contracts import StageOutcome,CheckResult


class NativePickBackend:
    """不复制Pick阶段循环；不同instance通过候选/设置表达差异。"""
    evidence_mode='native_physical'

    def __init__(self, *, runtime,scene_provider,planning_session,request,candidate,recorder,robot_metadata=None,failure_image_capture=None):
        self.failure_image_capture=failure_image_capture
        self.runtime=runtime;self.scene=scene_provider;self.session=planning_session
        self.request=request;self.candidate=dict(candidate);self.recorder=recorder
        self.robot=robot_metadata or getattr(planning_session,'robot',{})
        initial=self.observe()
        world_object=np.asarray(initial.values['objects'][request.object_id]['T_world_object'])
        if 'T_object_tcp' in candidate:
            self.local=validate_pose(candidate['T_object_tcp'])
        elif 'T_world_tcp' in candidate:
            self.local=validate_pose(np.linalg.inv(world_object)@validate_pose(candidate['T_world_tcp']))
        else:raise ValueError('抓取候选需要明确的T_object_tcp')
        self.anchor=None;self.lift_start=None

    def observe(self):
        """每个阶段只使用允许的当前工具/机器人观测。"""
        return self.runtime.observe(object_names=(self.request.object_id,),robot_groups=(self.request.group,))

    def _target(self,state):
        """当前物体pose乘固定局部候选，不从出生setting推导目标。"""
        return np.asarray(state.values['objects'][self.request.object_id]['T_world_object'])@self.local

    def _move(self,name,start):
        """pregrasp普通路径；approach严格直线；失败原样返回。"""
        if name=='pregrasp':
            from .candidate_selection import select_grasp
            if not select_grasp(self,start):
                return StageOutcome(name,False,start,start,details={'execution_status':'not_executed'},failure=dict(code='no_feasible_grasp_candidate',source='candidate_selection',message='候选抓取/预测对齐均未通过碰撞IK筛查'))
        group=self.request.group;target=self._target(start).copy()
        if name=='pregrasp':
            axis=np.asarray(self.robot.get('grippers',{}).get(group,{}).get('approach_axis_tcp',[0,0,1]),dtype=float)
            if axis.shape!=(3,) or not np.isclose(np.linalg.norm(axis),1.):raise ValueError('approach_axis_tcp必须单位向量')
            target[:3,3]-=target[:3,:3]@axis*self.request.settings.approach_distance_m
        elif name=='lift':
            self.lift_start=start
            target=np.asarray(start.values['robots'][group]['T_world_tcp']).copy()
            target[2,3]+=self.request.settings.lift_distance_m
        cp=self.approach_collision_policy() if name=='approach' else CollisionPolicy()
        if name=='lift':cp=CollisionPolicy('lift_support',(CollisionAllowance('gene_held_tool','table','从实际支撑面抬起工具'),))
        budget=self.request.settings.pregrasp_approximate_ik_budget if name=='pregrasp' else None
        return self.move_target(name,start,target,linear=name in {'approach','lift'},collision_policy=cp,
            **({'approximate_ik_budget':budget} if budget is not None else {}))

    def approach_collision_policy(self):
        """仅接近阶段显式允许夹爪指部接触待抓工具；掌部/板面等保持检查。"""
        if not self.candidate.get('allow_finger_tool_contact'):return CollisionPolicy()
        links=self.robot['grippers'][self.request.group].get('contact_links',())
        if not links:raise ValueError('指部接触需要源机器人模型声明contact_links')
        return CollisionPolicy('grasp_finger_contact',tuple(
            CollisionAllowance(link,self.request.object_id,'直线抓取接近中的预期指部/工具接触') for link in links))

    def move_target(self,name,start,target,*,linear,collision_policy,time_scale=1.,approximate_ik_budget=None):
        """原生持物运动桥：Pick和Hit共用实际状态、场景、规划和执行记录。"""
        if not np.isfinite(time_scale) or time_scale<1.:raise ValueError('time_scale只允许有限减速倍率>=1')
        planning_started=time.perf_counter()
        group=self.request.group;cp=collision_policy;held=None
        if self.anchor is not None:
            link_from_tcp=np.asarray(self.robot['transforms'][group]['link_from_tcp'])
            tcp=np.asarray(start.values['robots'][group]['T_world_tcp'])
            obj=np.asarray(start.values['objects'][self.request.object_id]['T_world_object'])
            held=dict(object_id=self.request.object_id,T_link_object=(link_from_tcp@np.linalg.inv(tcp)@obj).tolist())
        extra={} if held is None else {'held_object':held}
        snapshot_started=time.perf_counter()
        snapshot=self.scene.snapshot(state=start,group=group,collision_policy=cp,**extra)
        snapshot_duration=time.perf_counter()-snapshot_started
        row=start.values['robots'][group]
        with self.session.stage(snapshot=snapshot,start=JointState(tuple(row['joint_names']),tuple(row['joint_positions_rad'])),
                group=group,scene_revision=start.values['scene_revision'],collision_policy=cp) as motion:
            method=motion.plan_linear if linear else motion.plan_to
            plan=method(PoseTarget('world_tcp',target),request_id=f'native/{name}',
                **({'approximate_ik_budget':approximate_ik_budget} if approximate_ik_budget is not None else {}))
        self.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id=name,kind='planning',state_id=start.state_id,
            target_world_tcp=target.tolist(),planning=plan.to_dict(),
            ik_policy=dict(mode='strict' if approximate_ik_budget is None else 'bounded_approximate',budget=approximate_ik_budget),
            scene_snapshot_s=snapshot_duration,planning_wall_s=time.perf_counter()-planning_started))
        if not plan.success:
            try:
                recording_started=time.monotonic()
                from gene_agent.feedback.motion_diagnostics import save_motion_diagnostic, summarize_motion_result
                event=dict(episode_id=start.values['episode_id'],stage_id=name,state_id=start.state_id,target_world_tcp=target.tolist(),planning=plan.to_dict())
                path=save_motion_diagnostic(self.recorder.root,event,snapshot)
                bundle=plan.failure.details.get('diagnostic') or {}
                budget=bundle.get('budget') or {}
                if self.failure_image_capture and budget.get('budget_s',0)-budget.get('elapsed_s',0)-(time.monotonic()-recording_started)>.25:
                    payload=json.loads(path.read_text())
                    try:
                        payload['media']['actual']=self.failure_image_capture(path.with_suffix('.actual.png'),stage_id=name,state_id=start.state_id)
                    except Exception as media_error:
                        payload['media']['actual']=dict(status='unknown',reason=f'{type(media_error).__name__}: {media_error}')
                    b=payload['diagnostic']['budget']
                    b['collector_elapsed_s']=b['elapsed_s']
                    b['recording_and_image_elapsed_s']=time.monotonic()-recording_started
                    b['elapsed_s']+=b['recording_and_image_elapsed_s']
                    b['over_budget']=b['elapsed_s']>b['budget_s']
                    payload['planning']['failure']['details']['diagnostic']=payload['diagnostic']
                    payload['summary']=summarize_motion_result(payload['planning'],snapshot).to_dict()
                    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
                    path.with_suffix('.md').write_text(summarize_motion_result(payload['planning'],snapshot).to_markdown()+'\n证据：'+str(path)+'\n')
                self.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id=name,kind='diagnostic_summary',
                    state_id=start.state_id,diagnostic_ref=str(path),summary=summarize_motion_result(plan,snapshot).to_dict()))
            except Exception as exc:
                self.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id=name,kind='diagnostic_unavailable',error=f'{type(exc).__name__}: {exc}'))
            failure=dict(code=plan.failure.kind.value,source='motion',message=plan.failure.summary)
            return StageOutcome(name,False,start,start,details={'execution_status':'not_executed','planning':plan.to_dict()},failure=failure)
        if plan.trajectory is None:raise RuntimeError('规划成功却缺少轨迹')
        trajectory=replace(plan.trajectory,times_s=tuple(t*float(time_scale) for t in plan.trajectory.times_s))
        outcome=self.runtime.execute_trajectory(trajectory,group=group,stage_id=name,expected_state_id=start.state_id)
        outcome=replace(outcome,details={**outcome.details,'planning':plan.to_dict(),'target_world_tcp':target.tolist(),'execution_time_scale':float(time_scale)})
        approximation=plan.metrics.get('approximate_ik') or {}
        if outcome.success and approximation.get('approximate'):
            # 计划FK通过预算不代表实际执行也满足；不让后续阶段继承名义目标。
            actual=np.asarray(outcome.end.values['robots'][group]['T_world_tcp'])
            position,rotation=pose_delta(target,actual,field='approximate_execution')
            budget=approximation['budget']
            okay=position<=budget['max_position_error_m'] and rotation<=budget['max_rotation_error_rad']
            evidence=dict(approximate_ik=approximation,actual_tcp_pose_world=actual.tolist(),
                requested_tcp_pose_world=target.tolist(),position_error_m=position,rotation_error_rad=rotation,budget=budget)
            check=CheckResult('pass' if okay else 'fail','actual_tcp_pose',evidence,'实际TCP与阶段误差预算检查')
            outcome=replace(outcome,success=bool(okay),check=check,failure=None if okay else dict(
                code='approximate_ik_postcondition_failed',source='actual_pose',message='实际TCP超出阶段近似IK误差预算'))
        return outcome

    def _check(self,name,start):
        """只读实际aperture/pose；内部接触/评分状态不作为策略条件。"""
        group=self.request.group;robot=start.values['robots'][group]
        tcp=np.asarray(robot['T_world_tcp']);obj=np.asarray(start.values['objects'][self.request.object_id]['T_world_object'])
        relative=np.linalg.inv(tcp)@obj
        evidence={};status='unknown';reason='缺少可验证的夹持观测'
        if name=='grasp_check':
            width=robot.get('gripper',{}).get('width_m');expected=self.candidate.get('contact_width_m')
            interval=self.candidate.get('contact_width_range_m')
            if interval is not None:
                interval=np.asarray(interval,dtype=float)
                if interval.shape!=(2,) or not np.isfinite(interval).all() or not 0<interval[0]<=interval[1]:
                    raise ValueError('contact_width_range_m必须是正有限递增区间')
            elif expected is not None:interval=np.array([float(expected)-.004,float(expected)+.004])
            if width is not None and interval is not None:
                error=float(np.linalg.norm(tcp[:3,3]-self._target(start)[:3,3]))
                okay=interval[0]<=float(width)<=interval[1] and float(width)>self.request.settings.close_width_m+.001 and error<=.008
                evidence=dict(actual_width_m=width,expected_width_m=expected,contact_width_range_m=interval.tolist(),tcp_error_m=error,provisional=True)
                status='pass' if okay else 'fail';reason='实际开口受阻和工具位置检查；夹持仍须抬升/保持验证'
                if okay:self.anchor=relative.copy()
        elif self.anchor is not None and self.lift_start is not None:
            translation,rotation=pose_delta(self.anchor,relative,field='tool_in_tcp')
            old=np.asarray(self.lift_start.values['objects'][self.request.object_id]['T_world_object'])
            lift=float(obj[2,3]-old[2,3]);okay=translation<=.01 and rotation<=np.deg2rad(10) and lift>=.6*self.request.settings.lift_distance_m
            evidence=dict(relative_translation_m=translation,relative_rotation_rad=rotation,actual_lift_m=lift)
            status='pass' if okay else 'fail';reason='实际工具相对TCP保持和高度变化'
        check=CheckResult(status,'actual_pose_and_gripper',evidence,reason)
        return StageOutcome(name,check.passed,start,start,check=check,details={'execution_status':'not_executed','observation_only':True})

    def execute(self,stage,start):
        """按共享流程要求执行单阶段，不插入reset/补路或仿真weld。"""
        if self.observe().state_id!=start.state_id:raise ValueError('Pick阶段起点state已过期')
        name=stage.name;settings=self.request.settings
        if name in {'pregrasp','approach','lift'}:result=self._move(name,start)
        elif name in {'open','close'}:
            result=self.runtime.execute_gripper(group=self.request.group,
                width_m=settings.open_width_m if name=='open' else settings.close_width_m,
                duration_s=2. if name=='open' else 1.,stage_id=name,expected_state_id=start.state_id)
            if name=='open' and result.success:
                actual=result.end.values['robots'][self.request.group]['gripper']['width_m']
                if actual<settings.open_width_m-.004:
                    result=replace(result,success=False,failure=dict(code='gripper_tracking_error',source='robotwin',message='实际开口小于pregrasp所需净空'))
        elif name=='hold_check':
            held=self.runtime.hold(duration_s=settings.hold_duration_s,stage_id=name,expected_state_id=start.state_id)
            if held.success:
                checked=self._check(name,held.end)
                result=replace(checked,start=start,details={**held.details,'observation_only':False})
            else:result=held
        elif name=='grasp_check':result=self._check(name,start)
        elif name=='establish_hold':
            # 记录实际夹持关系；后续规划转移真实几何，不建立物理weld。
            result=StageOutcome(name,self.anchor is not None,start,start,details={'execution_status':'not_executed','simulation_weld':False})
        else:raise ValueError(f'未知Pick阶段: {name}')
        if not result.end.values.get('state_unavailable'):self.recorder.append_state(name+'_end',result.end)
        self.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id=name,kind='stage_outcome',
            state_id=result.end.state_id,success=result.success,check=result.check.to_dict() if result.check else None,
            failure=result.failure,execution_status=result.execution_status))
        return result

    def finish(self,result):
        """局部操作结果独立保存，不改原benchmark评分。"""
        (self.recorder.root/'pick_result.json').write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2))


def make_pick_backend(**kwargs):
    """显式装配一个Pick，不增加全局工厂或按物体复制策略。"""
    return NativePickBackend(**kwargs)
