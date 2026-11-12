"""共享Hit在原生benchmark的接线；只消费当前pose与本项目功能几何。"""
from dataclasses import replace
import json
import numpy as np
from gene_agent.knowledge.operation.hit import align_hit_tcp_candidates
from gene_agent.knowledge.geometry.transforms import pose_delta,validate_pose
from gene_agent.motion import CollisionPolicy,CollisionAllowance
from gene_agent.motion.curobo.approximate import ApproximateIKBudget
from gene_agent.skills.contracts import StageOutcome,CheckResult


class NativeHitBackend:
    evidence_mode='native_physical'

    def __init__(self,pick_backend,parameters,*,target_id='target',target_body):
        self.motion=pick_backend;self.parameters=dict(parameters);self.target_id=target_id;self.target_body=target_body
        if self.motion.anchor is None:raise ValueError('Hit必须从已验证的Pick终态开始')
        for key in ('strike_point_object_m','strike_normal_object','target_point_object_m'):
            value=np.asarray(self.parameters[key],float)
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError(f'{key}必须为有限三维')
        clearance=float(self.parameters['clearance_m']);stroke=float(self.parameters['stroke_m'])
        if not 0<clearance<stroke or not np.isfinite(stroke):raise ValueError('敲击距离必须满足0<clearance<stroke')
        withdraw=float(self.parameters.get('withdraw_distance_m',0.))
        if not np.isfinite(withdraw) or withdraw<0:raise ValueError('withdraw_distance_m必须为非负有限米数')
        policies=self.parameters.get('approximate_ik_by_stage',{})
        if not isinstance(policies,dict) or set(policies)-{'align','withdraw'}:
            raise ValueError('Hit近似IK预算仅允许align/withdraw阶段；strike保持严格')
        self.ik_budgets={name:ApproximateIKBudget(**budget).to_dict() for name,budget in policies.items()}
        if 'withdraw' in self.ik_budgets and self.ik_budgets['withdraw']['max_position_error_m']>=withdraw:
            raise ValueError('撤离位置误差预算必须小于withdraw_distance_m，不能覆盖整个撤离动作')

    def observe(self):
        return self.motion.runtime.observe(object_names=(self.motion.request.object_id,self.target_id),robot_groups=(self.motion.request.group,))

    def execute(self,stage,start):
        p=self.parameters;group=self.motion.request.group
        tool=np.asarray(start.values['objects'][self.motion.request.object_id]['T_world_object'])
        tcp=np.asarray(start.values['robots'][group]['T_world_tcp']);target=np.asarray(start.values['objects'][self.target_id]['T_world_object'])
        relative=np.linalg.inv(tcp)@tool
        translation,rotation=pose_delta(self.motion.anchor,relative,field='tool_in_tcp')
        if translation>.01 or rotation>np.deg2rad(10):
            return StageOutcome(stage.name,False,start,start,failure={'code':'lost_hold','source':'actual_pose','message':'工具相对夹爪已滑移'},details={'execution_status':'not_executed'})
        normal=target[:3,2];point=tool[:3,:3]@p['strike_point_object_m']+tool[:3,3]
        surface=target[:3,:3]@p['target_point_object_m']+target[:3,3]
        if stage.name=='align':
            goals=align_hit_tcp_candidates(tool,tcp,target,strike_point=p['strike_point_object_m'],strike_normal=p['strike_normal_object'],target_point=p['target_point_object_m'],clearance_m=p['clearance_m'],yaw_offsets_rad=p.get('alignment_yaw_offsets_rad',[0.]))
            cp=CollisionPolicy();linear=False
        elif stage.name=='strike':
            delta=point-surface;lateral=np.linalg.norm(delta-normal*(delta@normal))
            angle=float(np.arccos(np.clip((tool[:3,:3]@p['strike_normal_object'])@(-normal),-1,1)))
            if lateral>.008 or angle>.12:
                return StageOutcome(stage.name,False,start,start,failure={'code':'alignment_error','source':'actual_pose','message':f'敲击面横向误差{lateral:.6f}m/角度{angle:.6f}rad'},details={'execution_status':'not_executed'})
            goal=tcp.copy();goal[:3,3]-=normal*(float(delta@normal)+p['stroke_m']-p['clearance_m'])
            goals=(goal,)
            cp=CollisionPolicy('strike_contact',(CollisionAllowance('gene_held_tool',self.target_body,'敲击工具与目标的预期接触'),));linear=True
        elif stage.name=='withdraw':
            # 从执行后的TCP沿目标法向撤离，不复用敲击前状态或隐藏评分条件。
            distance=float(p.get('withdraw_distance_m',0.))
            if distance<=0:raise ValueError('撤离阶段需要显式正withdraw_distance_m')
            goal=tcp.copy();goal[:3,3]+=normal*distance
            goals=(goal,)
            cp=CollisionPolicy('withdraw_contact',(CollisionAllowance('gene_held_tool',self.target_body,'从敲击预期接触位置直线撤离'),));linear=True
        else:raise ValueError(f'未知Hit阶段: {stage.name}')
        for index,goal in enumerate(goals):
            self.motion.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id=stage.name,
                kind='alignment_candidate' if stage.name=='align' else f'{stage.name}_target',candidate_index=index,
                state_id=start.state_id,target_world_tcp=goal.tolist()))
            # 运送与接触速度分别调节；只延长时间，不改变路径或后置检查。
            scales={'align':p.get('transport_time_scale',1.),'strike':p.get('strike_time_scale',1.),'withdraw':1.}
            result=self.motion.move_target(stage.name,start,goal,linear=linear,collision_policy=cp,
                time_scale=float(scales[stage.name]),**({'approximate_ik_budget':self.ik_budgets[stage.name]} if stage.name in self.ik_budgets else {}))
            # 只在尚未执行且实际状态未改变的规划失败后换目标；执行失败不回到旧起点。
            if result.success or result.execution_status!='not_executed' or result.end.state_id!=start.state_id or (result.failure or {}).get('source')!='motion':
                break
        # 只有正常完成才检查动作后置条件；原评测提前终止的partial事实原样保留。
        if result.success:
            end=result.end.values;actual_tcp=np.asarray(end['robots'][group]['T_world_tcp'])
            actual_tool=np.asarray(end['objects'][self.motion.request.object_id]['T_world_object'])
            approximation=(result.details.get('planning',{}).get('metrics',{}).get('approximate_ik') or {})
            effective_goal=goal
            if approximation.get('approximate'):
                if stage.name not in self.ik_budgets:raise ValueError('未启用的阶段返回了近似IK结果')
                effective_goal=validate_pose(approximation['actual_endpoint_tcp_pose_world'])
            drift,turn=pose_delta(self.motion.anchor,np.linalg.inv(actual_tcp)@actual_tool,field='tool_in_tcp')
            actual_point=actual_tool[:3,:3]@p['strike_point_object_m']+actual_tool[:3,3]
            expected_tool=effective_goal@np.linalg.inv(tcp)@tool
            expected_point=expected_tool[:3,:3]@p['strike_point_object_m']+expected_tool[:3,3]
            error=float(np.linalg.norm(actual_point-expected_point))
            okay=bool(drift<=.01 and turn<=np.deg2rad(10) and error<=.008)
            stage_evidence={}
            if stage.name=='align':
                # 即便悬停目标允许近似，仍须满足随后精确敲击的侧向/法向要求。
                delta=actual_point-surface
                lateral=float(np.linalg.norm(delta-normal*(delta@normal)))
                normal_error=float(np.arccos(np.clip((actual_tool[:3,:3]@p['strike_normal_object'])@(-normal),-1,1)))
                okay=bool(okay and lateral<=.008 and normal_error<=.12 and delta@normal>0)
            elif stage.name=='withdraw':
                # FK替代只放宽端点；仍须真正沿法向离开，并保持作用点在目标表面之外。
                # 8mm沿用作用点执行误差上限，不从隐藏的benchmark评分读取净空条件。
                position_budget=self.ik_budgets.get('withdraw',{}).get('max_position_error_m',0.)
                minimum=max(0.,float(p['withdraw_distance_m'])-position_budget-.008)
                withdrawal=float((actual_point-point)@normal)
                surface_clearance=float((actual_point-surface)@normal)
                okay=bool(okay and withdrawal>0 and withdrawal>=minimum and surface_clearance>0)
                stage_evidence=dict(actual_withdrawal_m=withdrawal,minimum_withdrawal_m=minimum,
                    surface_clearance_m=surface_clearance)
            check=CheckResult('pass' if okay else 'fail','actual_tool_pose',
                dict(relative_translation_m=drift,relative_rotation_rad=turn,strike_point_error_m=error,
                    **stage_evidence,
                    **({'approximate_ik':approximation} if approximation.get('approximate') else {})),
                '实际工具保持与作用点终态检查')
            result=replace(result,success=okay,check=check,
                failure=None if okay else dict(code='tool_postcondition_failed',source='actual_pose',message='工具滑移或作用点未达到阶段目标'))
        self.motion.recorder.append_state(stage.name+'_end',result.end)
        self.motion.recorder.append_event(dict(episode_id=start.values['episode_id'],stage_id=stage.name,kind='stage_outcome',state_id=result.end.state_id,success=result.success,failure=result.failure,execution_status=result.execution_status))
        return result

    def finish(self,result):
        (self.motion.recorder.root/'hit_result.json').write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2))
