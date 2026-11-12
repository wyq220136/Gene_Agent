"""cuRobo 纯规划适配：共享阶段、预测状态；不调用物理或真实设备。"""
from pathlib import Path
import json
import copy
import time
import traceback
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.motion.curobo import SynchronousPlanner
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.motion.curobo.approximate import ApproximateIKBudget, plan_with_approximate_ik
from gene_agent.scene.visual_inputs import load_visual_inputs
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.skills.library.pick.skill import run_pick_flow
from gene_agent.skills.library.transfer.skill import run_transfer_flow
from gene_agent.skills.library.transfer.skill import run_extract_flow
from gene_agent.skills.library.knob.skill import run_knob_flow
from gene_agent.skills.library.pick.brush import brush_pick_targets, brush_lift_target, brush_grasp_cross_section
from gene_agent.robots.pika_geometry import pika_grasp_center_tcp, pika_predict_contact_width
from gene_agent.skills.library.pick.brush import load_brush_pick_profile
from gene_agent.skills.library.sweep.skill import run_sweep_skill
from gene_agent.skills.library.door.planning import DoorSkill, DoorSkillSettings
from gene_agent.skills.library.knob.planning import RotateKnobSkill, KnobSkillSettings
from gene_agent.knowledge.operation import AxisSpec
from gene_agent.knowledge.sweep import make_sweep_layout, make_sweep_contact_pose_candidates
from gene_agent.tasks.checks import CheckResult
from gene_agent.tasks.planning.knowledge import proposals, SKILLS

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[3] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs

ROOT=Path(__file__).resolve().parents[3]


def write(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,default=convert,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


class VisualPlanningRun:
    def __init__(self,task,source,output,*,skill=None,seed=123,randomize=False,tilt=-20.,collision_mode="full"):
        if collision_mode not in {'full','robot_only'}:raise ValueError('未知碰撞模式')
        self.collision_mode=collision_mode
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.inputs=load_visual_inputs(task,source,self.output/'inputs',seed=seed,randomize=randomize)
        self.task=task;self.skill=skill;self.tilt=tilt;self.seed=seed
        self.pool=CuroboPlannerPool(planner_factory=SynchronousPlanner,max_entries=4)
        self.q={g:r.start_positions_rad.copy() for g,r in self.inputs.robots.items()}
        self.width={g:r.home_width_m for g,r in self.inputs.robots.items()}
        self.attachments={};self.counter=0;self.prefix='';self.failed=None
        self.report=dict(schema_version=1,task=task,record_type='pure_visual_plan',
            success=None,task_success=None,planning_success=False,status='planning',
            planner_backend='curobo',execution_backend='none',physics_used=False,
            evidence_mode='pure_visual_plan',collision_mode=collision_mode,
            environment_collision_enabled=collision_mode=='full',planning_attachment_enabled=False,
            attachment_collision_limitation='本纯视觉适配器只预测附着，不生成持物碰撞球',
            self_collision_check=True,selection=skill or 'task',source=str(Path(source).resolve()),
            seed=seed,stages=[],initial_assumptions=[],checks=[],knowledge=[],failure_stage=None,
            started_at=time.time(),predicted_cube_motion=False)
        self.initial=self.state()
        self.publish()

    def publish(self):
        from gene_agent.feedback.live_record import publish_plan
        self.report.update(initial_state=self.initial,final_state=self.state())
        publish_plan(self.output,self.report)

    def state(self):
        return dict(joints={g:q.tolist() for g,q in self.q.items()},widths=dict(self.width),
            objects={n:np.asarray(o['T_world_object']).tolist() for n,o in self.inputs.objects.items()},
            object_joints=copy.deepcopy(self.inputs.object_joint_positions),
            attachments={n:dict(group=a['group'],T_tcp_object=a['T_tcp_object'].tolist()) for n,a in self.attachments.items()})

    def _sync(self):
        for g,r in self.inputs.robots.items():r.set_state(self.q[g],self.width[g])
        for name,a in self.attachments.items():
            self.inputs.objects[name]['T_world_object']=self.inputs.robots[a['group']].fk(self.q[a['group']])@a['T_tcp_object']

    def planner(self,group):
        self._sync();settings,snapshot=self.inputs.prepare_planner(group,self.width[group],collision_mode=self.collision_mode,held_objects=self.attachments)
        return self.pool.acquire(settings,snapshot,random_seed=self.seed,collision_mode=self.collision_mode)

    def move(self,name,target,kind='joint',*,group,allowed_contacts=(),duration_s=None,**_):
        self._sync();before=self.state();name=f'{self.prefix}{name}'
        print(f'cuRobo {self.collision_mode}：{name} ({kind})',flush=True)
        self.report['active_stage']=name
        self.report['active_target_tcp_world']=np.asarray(target).tolist()
        self.publish()
        lease=self.planner(group);planner=lease.planner
        # 接近/接触保持严格；只有原有允许预位/撤离语义的阶段启用近似能力。
        budget=None
        if name.endswith('pregrasp'):budget=ApproximateIKBudget(.025,.065)
        elif name.endswith('move_behind_cube'):budget=ApproximateIKBudget(.010,.065)
        elif name.endswith('retreat'):budget=ApproximateIKBudget(.003,.08)
        result=plan_with_approximate_ik(planner,self.inputs.robots[group].joint_names,self.q[group],target,
            budget=budget,linear_tcp=kind=='linear_tcp',max_attempts=2)
        result['planner_cache']={'cache_hit':lease.cache_hit,'setup_duration_s':lease.setup_duration_s,
                                 'rebuild_reason':lease.rebuild_reason}
        planpath=f'plans/{len(self.report["stages"]):03d}.json';write(self.output/planpath,result)
        row=dict(name=name,group=group,path_kind=kind,success=bool(result['success']),
            target_tcp_world=np.asarray(target).tolist(),plan_file=planpath,start_state=before,
            failure_type=result.get('failure_category'),duration_s=result.get('duration_s'),
            approximation=result.get('approximate_ik'),planner_cache=result['planner_cache'])
        if result['success']:
            positions=np.asarray(result['positions_rad'],float)
            if positions.ndim!=2 or len(positions)<2:raise ValueError('cuRobo 没有返回真实多点轨迹')
            self.q[group]=positions[-1].copy();self._sync()
            actual=self.inputs.robots[group].fk(self.q[group])
            row.update(positions_rad=positions.tolist(),times_s=(np.arange(len(positions))*result['interpolation_dt_s']).tolist(),
                actual_tcp_world=actual.tolist(),position_error_m=float(np.linalg.norm(actual[:3,3]-np.asarray(target)[:3,3])),
                rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(np.asarray(target)[:3,:3]@actual[:3,:3].T).as_rotvec())))
        else:self.failed=name
        row['end_state']=self.state();self.report['stages'].append(row);self.counter+=1
        self.publish()
        write(self.output/'progress.json',dict(stage=name,planning_success=result['success'],stage_count=len(self.report['stages'])))
        return dict(success=bool(result['success']),failure_type=row.get('failure_type'))

    def event(self,name,*,group=None,width=None,attach=None,detach=None,check=False):
        before=self.state()
        if width is not None:self.width[group]=width
        if attach:
            self.attachments[attach]=dict(group=group,T_tcp_object=np.linalg.inv(self.inputs.robots[group].fk(self.q[group]))@self.inputs.objects[attach]['T_world_object'])
        if detach:self.attachments.pop(detach,None)
        self._sync();self.counter+=1
        row=dict(name=self.prefix+name,path_kind='prediction',group=group,success=True,
                 start_state=before,end_state=self.state(),assumption=True)
        if check:
            row['check']=CheckResult('unknown','prediction',{'predicted':True},'纯视觉模式没有物理接触或任务终态证据').to_dict()
            self.report['checks'].append(row['check'])
        self.report['stages'].append(row)
        self.publish()
        return row

    def backend(self,perform):
        return CallbackOperationBackend(observe=lambda:OperationState(str(self.counter),'prediction',{'predicted':True}),
            perform=perform,evidence_mode='prediction',execution_enabled=False)

    def choose(self,kind,*,group=None):
        candidates=proposals(self.inputs,kind,tilt=self.tilt)
        self.report['knowledge'].extend([{**p,'pose':p['pose'].tolist()} for p in candidates])
        trials=[]
        trial_path=f'candidate_checks/{len(self.report["knowledge"]):04d}_{kind}.json'
        self.report.setdefault('candidate_checks',[]).append({'skill':self.prefix,'path':trial_path})
        for p in candidates:
            g=group or p['group'];lease=self.planner(g)
            raw=lease.planner.solve_ik(self.inputs.robots[g].joint_names,self.q[g],p['pose'])
            trials.append(dict(candidate=p['name'],success=bool(raw['success']),diagnostic=raw))
            if raw['success']:
                write(self.output/trial_path,trials)
                return p
        write(self.output/trial_path,trials)
        reasons='；'.join(dict.fromkeys(str(t['diagnostic'].get('message') or t['diagnostic'].get('failure_category') or t['diagnostic'].get('status') or '未提供诊断') for t in trials))
        raise ValueError(f'{kind} 没有通过 cuRobo IK 和机器人碰撞检查的候选：{reasons}；详见 {trial_path}')

    def pick(self,object_name):
        group='left' if object_name=='brush' else 'right';kind='brush_grasp' if object_name=='brush' else 'bowl_grasp'
        self.width[group]=.04 if object_name=='brush' else .041
        chosen=self.choose(kind,group=group);grasp=chosen['pose'];pre=grasp.copy()
        profile=None
        predicted_close_width=0.
        if object_name=='brush':
            profile=load_brush_pick_profile('preview',overrides={'grasp_tilt_deg':self.tilt})
            center=pika_grasp_center_tcp(self.inputs.robots[group].metadata,group)
            targets=brush_pick_targets(self.inputs.objects['brush']['T_world_object'],self.inputs.concepts_dir/'brush.json',profile,
                                       grasp_center_tcp_m=center)
            pre=targets['pregrasp'];grasp=targets['approach']
            geometry=brush_grasp_cross_section(self.inputs.objects['brush']['T_world_object'],
                self.inputs.objects['brush']['mesh_path'],grasp,center)
            contact=pika_predict_contact_width(self.inputs.robots[group].metadata,group,
                np.linalg.inv(grasp) @ self.inputs.objects['brush']['T_world_object'],
                self.inputs.objects['brush']['mesh_path'],open_width_m=self.width[group])
            predicted_close_width=contact['predicted_width_m']
            self.report['grasp_geometry']={**geometry,'section_width_m':geometry['predicted_width_m'],
                **contact,'grasp_center_tcp_m':center.tolist()}
        else:pre[:3,3]-=.08*grasp[:3,2]
        def perform(name):
            if name=='pregrasp':return self.move(name,pre,group=group)
            if name=='approach':return self.move(name,grasp,'linear_tcp',group=group)
            if name=='lift':
                actual=self.inputs.robots[group].fk(self.q[group])
                if profile:target=brush_lift_target(actual,pre,profile)
                else:target=actual.copy();target[2,3]+=.07
                return self.move(name,target,'linear_tcp',group=group)
            if name=='close' and object_name=='brush':
                # 使用 approach 的实际 FK 终态；不能拿名义目标代替执行终态。
                contact=pika_predict_contact_width(self.inputs.robots[group].metadata,group,
                    np.linalg.inv(self.inputs.robots[group].fk(self.q[group])) @ self.inputs.objects['brush']['T_world_object'],
                    self.inputs.objects['brush']['mesh_path'],open_width_m=self.width[group])
                self.report['grasp_geometry'].update(contact)
                close_width=contact['predicted_width_m']
            else:close_width=predicted_close_width
            self.event(name,group=group,width=self.width[group] if name=='open' else close_width if name=='close' else None,
                       attach=object_name if name=='establish_hold' else None,check=name.endswith('check'))
            if name=='close':
                self.report['stages'][-1].update(command_width_m=0.,predicted_width_m=close_width)
            return dict(success=True,check=CheckResult('unknown','prediction',{'predicted':True}) if name.endswith('check') else None)
        return run_pick_flow(self.backend(perform),verify_hold=True).success

    def assume_hold(self,object_name):
        group='left' if object_name=='brush' else 'right'
        self.report['initial_assumptions'].append(f'独立 Skill 初态假定已持有 {object_name}；初始关节来自成功 cuRobo IK，不代表执行过抓取。')
        chosen=self.choose('brush_grasp' if object_name=='brush' else 'bowl_grasp',group=group)
        raw=self.planner(group).planner.solve_ik(self.inputs.robots[group].joint_names,self.q[group],chosen['pose'])
        write(self.output/f'initialization/{object_name}_ik.json',raw)
        if not raw['success']:raise ValueError('独立持物初态 IK 失败')
        q=np.asarray(raw['positions_rad'],float).reshape(-1,7)[0]
        self.q[group]=q
        width=0.
        if object_name=='brush':
            geometry=pika_predict_contact_width(self.inputs.robots[group].metadata,group,
                np.linalg.inv(self.inputs.robots[group].fk(q)) @ self.inputs.objects['brush']['T_world_object'],
                self.inputs.objects['brush']['mesh_path'],open_width_m=.04)
            self.report['grasp_geometry']=geometry
            width=geometry['predicted_width_m']
        self.event('assumed_initial_hold',group=group,width=width,attach=object_name)

    def sweep(self):
        if 'brush' not in self.attachments:self.assume_hold('brush')
        for target_name in ('cube','cube_red','cube_blue'):
            if target_name not in self.inputs.objects:continue
            old=self.prefix;self.prefix+=target_name+'/'
            layout=make_sweep_layout({n:o['T_world_object'] for n,o in self.inputs.objects.items()},self.inputs.concepts_dir,target_name=target_name)
            grip=self.attachments['brush']['T_tcp_object']
            def target(xyz):
                brush=np.eye(4);brush[:3,:3]=self.inputs.objects['brush']['T_world_object'][:3,:3];brush[:3,3]=xyz
                return brush@np.linalg.inv(grip)
            def contact(xyz):
                rows=make_sweep_contact_pose_candidates(target(xyz),self.inputs.robots['left'].fk(self.q['left']),
                    self.inputs.objects['brush']['T_world_object'],self.inputs.concepts_dir/'brush.json',
                    sweep_direction_world_xy=layout.direction_world_xy,table_top_z_m=self.inputs.table_top_z_m)
                for row in rows:
                    raw=self.planner('left').planner.solve_ik(self.inputs.robots['left'].joint_names,self.q['left'],row['target_tcp_world'])
                    if raw['success']:return np.asarray(row['target_tcp_world'])
                raise ValueError('刷面接触 Knowledge 候选均未通过 cuRobo IK')
            result=run_sweep_skill(layout, contacts=SWEEP_CONTACTS_PROFILE, evidence_mode="prediction",initial_brush_z=float(self.initial['objects']['brush'][2][3]),
                move_stage=lambda n,p,k,d,**kw:self.move(n,p,k,group='left',**kw),target_for_brush=target,
                current_brush_xyz=lambda:self.inputs.objects['brush']['T_world_object'][:3,3],sweep_targets=(target_name,),
                contact_target_for_brush=contact)
            self.prefix=old
            if not result['success']:return False
            # 不伪造方块位移；只展示刷子的规划轨迹，任务是否扫入保持未知。
        return True

    def door(self,closing=False):
        current=float(self.inputs.object_joint_positions.get('microwave',{}).get('door_hinge',0.))
        if closing and current<.1:
            self.report['initial_assumptions'].append('独立关门初态：门运动学设为 70°，未执行开门。')
            self.inputs.set_object_joints('microwave',{'door_hinge':np.deg2rad(70.)});current=np.deg2rad(70.)
        grasp=self.choose('handle_grasp')['pose']
        urdf=self.inputs.object_urdfs['microwave'][0];joint=urdf.joint_map['door_hinge']
        base=self.inputs.objects['microwave']['T_world_object'];origin=base@joint.origin
        declared=json.loads((ROOT/'data/task_1/skills/door_task.json').read_text(encoding='utf-8'))['door_axis_in_microwave']
        point=base[:3,:3]@np.asarray(declared['point_m'])+base[:3,3]
        direction=base[:3,:3]@np.asarray(declared['direction'])
        urdf_direction=origin[:3,:3]@np.asarray(joint.axis)
        delta=point-origin[:3,3]
        if np.linalg.norm(np.cross(delta,direction))>.001 or not np.allclose(direction,urdf_direction,atol=1e-6):
            raise ValueError('显式门轴与当前 URDF 不一致')
        axis=AxisSpec(point,direction,'world','data/task_1/skills/door_task.json:door_axis_in_microwave')
        settings=DoorSkillSettings(.04,0. if closing else np.deg2rad(70.),np.deg2rad(10.),.08,0.,.08)
        rows=DoorSkill().targets(grasp,axis,start_angle_rad=current,settings=settings)
        for s in rows:
            if s.operation is None:self.event(s.name,group='left',width=s.gripper_width_m)
            elif not self.move(s.name,s.tcp_pose_world,'linear_tcp' if s.operation.value=='linear_tcp' else 'joint',group='left')['success']:return False
            if s.name.startswith('swing'):
                self.inputs.set_object_joints('microwave',{'door_hinge':s.end_angle_rad})
                self.report['stages'][-1]['end_state']=self.state()
        return True

    def knob(self):
        current=float(self.inputs.object_joint_positions.get('microwave',{}).get('power_knob_rotation',0.))
        if abs(current)>1e-9:raise ValueError('当前旋钮预览要求起始角为0，不能忽略已有角度')
        chosen=self.choose('knob_grasp');group=chosen['group'];link=self.inputs.object_link_pose('microwave','power_knob')
        joint=self.inputs.object_urdfs['microwave'][0].joint_map['power_knob_rotation']
        origin=self.inputs.object_link_pose('microwave',joint.parent)@joint.origin
        axis=AxisSpec(origin[:3,3],origin[:3,:3]@np.asarray(joint.axis),'world','microwave URDF power_knob_rotation')
        settings=KnobSkillSettings(.05,np.deg2rad(45.),np.deg2rad(15.),.08,0.,.08,tip_rotations_x_rad=(.01,),grasp_settle_time_s=0.,release_settle_time_s=0.)
        targets=RotateKnobSkill().targets(chosen['pose'],axis,start_angle_rad=0.,settings=settings)
        by_name={t.name:t for t in targets}
        def perform(name):
            if name.endswith('check'):
                self.event(name,check=True);return dict(success=True,check=CheckResult('unknown','prediction',{'predicted':True}))
            t=by_name[name]
            if t.operation is not None:
                result=self.move(name,t.tcp_pose_world,'linear_tcp' if t.operation.value=='linear_tcp' else 'joint',group=group)
                if result['success'] and name.startswith('rotate'):
                    self.inputs.set_object_joints('microwave',{'power_knob_rotation':t.end_angle_rad});self.report['stages'][-1]['end_state']=self.state()
                return result
            self.event(name,group=group,width=t.gripper_width_m);return dict(success=True)
        return run_knob_flow(self.backend(perform),target_names=lambda:list(by_name)).success

    def transfer(self,extract=False):
        from gene_agent.tasks.policies.microwave_task.operations.bowl import load_object_operation_profile,fixed_place_targets,fixed_extract_targets
        profile=load_object_operation_profile(ROOT/'data/task_1','Bowl_0')
        world=self.inputs.objects['microwave']['T_world_object']@self.inputs.legacy_microwave_transform
        if extract:
            local=np.asarray(profile['microwave_from_placed_object'])
            if self.skill=='extract_bowl':
                self.inputs.objects['bowl']['T_world_object']=world@local
                self.report['initial_assumptions'].append('独立取碗：碗由已有 profile 放到炉内；仅是初态假设。')
            a,b,g,c,d=fixed_extract_targets(world,profile)
            targets={'pre_extract_1':a,'pre_extract_2':b,'grasp':g,'post_extract_1':c,'post_extract_2':d}
        else:
            if 'bowl' not in self.attachments:self.assume_hold('bowl')
            t=fixed_place_targets(world,profile,self.attachments['bowl']['T_tcp_object'])
            targets={'preplace':t.pre_place_tcp,'insert':t.place_tcp,'retreat_1':t.post_place_1_tcp,'retreat_2':t.post_place_2_tcp}
        def perform(name):
            if name in targets:return self.move(name,targets[name],'joint' if name in {'preplace','pre_extract_1'} else 'linear_tcp',group='right')
            if name.endswith('check'):
                self.event(name,check=True);return dict(success=True,check=CheckResult('unknown','prediction',{'predicted':True}))
            self.event(name,group='right',width=.08 if name in {'open','release'} else 0. if name=='close' else None,
                       attach='bowl' if name=='attach' else None,detach='bowl' if name=='detach' else None)
            return dict(success=True)
        if extract:return run_extract_flow(self.backend(perform)).success
        return run_transfer_flow(self.backend(perform),operation='insert').success

    def place(self):
        if 'bowl' not in self.attachments:raise ValueError('归位需要上一阶段留下的持碗预测状态')
        grasp=self.attachments['bowl']['T_tcp_object']
        final=np.asarray(self.initial['objects']['bowl'])@np.linalg.inv(grasp)
        pre=final.copy();pre[2,3]+=.07
        retreat=final.copy();retreat[2,3]+=.10
        targets={'preplace':pre,'place':final,'retreat':retreat}
        def perform(name):
            if name in targets:return self.move(name,targets[name],'joint' if name=='preplace' else 'linear_tcp',group='right')
            if name.endswith('check'):
                self.event(name,check=True);return dict(success=True,check=CheckResult('unknown','prediction',{'predicted':True}))
            self.event(name,group='right',width=.08 if name=='release' else None,detach='bowl' if name=='detach' else None)
            return dict(success=True)
        return run_transfer_flow(self.backend(perform),operation='place').success

    def run(self):
        try:
            if self.skill and self.skill not in SKILLS[self.task]:raise ValueError('该任务不支持所选 Skill')
            sequence=([self.skill] if self.skill else ['pick_brush','sweep_cube'] if self.task=='sweep'
                      else ['open_door','pick_bowl','insert_bowl','close_door','rotate_knob','open_door','extract_bowl','place_bowl','close_door'])
            methods={'pick_brush':lambda:self.pick('brush'),'sweep_cube':self.sweep,'pick_bowl':lambda:self.pick('bowl'),
                'open_door':self.door,'close_door':lambda:self.door(True),'rotate_knob':self.knob,
                'insert_bowl':self.transfer,'extract_bowl':lambda:self.transfer(True),'place_bowl':self.place}
            complete=True
            for index,name in enumerate(sequence):
                self.prefix=f'{index:02d}/{name}/'
                if not methods[name]():complete=False;break
            self.report['planning_success']=complete
        except Exception as exc:
            self.report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            self.failed=self.failed or 'preparation_or_knowledge'
        finally:
            self.report.update(status='planned' if self.report['planning_success'] else 'planning_failed',
                failure_stage=self.failed,initial_state=self.initial,final_state=self.state(),
                finished_at=time.time(),mujoco_imported='mujoco' in sys.modules)
            self.pool.clear();write(self.output/'result.json',self.report);self.publish()
        if self.report['mujoco_imported']:raise RuntimeError('纯视觉路径意外导入 MuJoCo，拒绝验收')
        return self.report
