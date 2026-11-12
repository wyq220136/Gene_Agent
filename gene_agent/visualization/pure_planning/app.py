"""冻结输入、Knowledge 与 cuRobo 轨迹显示；支持持续接收同一运行记录。"""
from pathlib import Path
import json
import time
import copy
from threading import RLock
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from viser.extras import ViserUrdf
from gene_agent.scene.visual_inputs import load_visual_inputs
from gene_agent.tasks.planning.knowledge import proposals
from .graphics import TargetGripper,load_geometry,line_segments,pose_args


class PurePlanningApp:
    def __init__(self,server,*,task=None,source=None,run=None,knowledge=None,output=None,
                 tilt=-20.,skill=None,report=None,seed=123,randomize=False,initial_only=False):
        self.server=server;self.playing=False;self.user_paused=False;self.last_tick=time.monotonic();self._lock=RLock()
        self.report=report or (json.loads((Path(run)/'result.json').read_text(encoding='utf-8')) if run else None)
        self.inputs=load_visual_inputs(task,source,output,seed=seed,randomize=randomize) if not run else load_visual_inputs(
            self.report['task'],Path(run)/'inputs/inputs.json',Path(output)/'replay_inputs')
        self.skill=skill;self.handles={};self.robot_views={};self.overlay=[];self.path_handles=[]
        self._updating=False;self._drawn_stage=None
        server.scene.set_up_direction('+z')
        server.gui.configure_theme(dark_mode=False,control_width='large',show_logo=False,brand_color=(53,112,169))
        server.gui.add_markdown('## 本次运行 · 纯视觉初态\n**无 MuJoCo，尚未规划或执行。** 可拖动视角检查本次输入摆位。' if initial_only else '## Knowledge / Skill · 纯视觉检查\n金色夹爪是目标，蓝线是 cuRobo 轨迹。**无 MuJoCo、无物理执行。** 附着和关节变化仅为预测。')
        self.status=server.gui.add_markdown('')
        with server.gui.add_folder('视角与显示'):
            self.camera=server.gui.add_dropdown('相机',options=('全景','目标近景','夹爪侧面'),initial_value='全景')
            self.show_target=server.gui.add_checkbox('显示金色目标夹爪',initial_value=True)
            self.show_axes=server.gui.add_checkbox('显示目标与实际 TCP 坐标轴',initial_value=True)
            self.show_path=server.gui.add_checkbox('显示蓝色路径',initial_value=True)
            self.show_concept=server.gui.add_checkbox('显示 Concept 功能部件',initial_value=False)
        self.camera.on_update(lambda _:self.set_camera())
        self.show_concept.on_update(lambda _:self.draw_concepts())
        server.scene.add_grid('/grid',width=2.,height=2.,cell_size=.1,section_size=.5)
        for group,r in self.inputs.robots.items():
            root=f'/robots/{group}';server.scene.add_frame(root,show_axes=False,**pose_args(r.world_from_base))
            self.robot_views[group]=ViserUrdf(server,r.urdf,root_node_name=root,mesh_color_override=(.30,.44,.60,1.))
            self.robot_views[group].update_cfg(r.joint_configuration)
        palette={'brush':(.18,.48,.73),'bowl':(.21,.64,.65),'cube':(.20,.72,.30),
                 'cube_red':(.85,.19,.20),'cube_blue':(.18,.32,.86),'dustpan':(.77,.65,.43),'microwave':(.46,.52,.59)}
        for name,obj in self.inputs.objects.items():
            self.handles[name]=[]
            for index,row in enumerate(obj['visuals']):
                handle=self.add_geometry(f'/objects/{name}/{index}',row,obj['T_world_object']@row['T_object_visual'],palette.get(name))
                self.handles[name].append(handle)
        for index,row in enumerate(self.inputs.static_visuals):
            transform=(self.inputs.objects[row['parent']]['T_world_object']@row['T_object_visual']
                       if row.get('parent') else np.asarray(row['T_world_visual']))
            self.add_geometry(f'/static/{index}',row,transform)
        self.target=server.scene.add_frame('/requested_target',axes_length=.055,axes_radius=.002,visible=False)
        self.actual=server.scene.add_frame('/actual_tcp',axes_length=.045,axes_radius=.002,visible=False)
        self.gripper=TargetGripper(server);self.focus=None
        self.concept_handles=[]
        self.camera.on_update(lambda _:self.set_camera())
        @server.on_client_connect
        def connected(client):self.set_camera(client)
        if initial_only:
            for control in (self.show_target,self.show_axes,self.show_path):control.visible=False
            self.status.content='**本次运行场景初态；无 MuJoCo、尚未规划或执行。**'
            self.set_camera();return
        if knowledge:self.setup_knowledge(knowledge,tilt,output)
        else:self.setup_playback()
        self.set_camera()

    def set_camera(self,client=None):
        center=np.mean([o['T_world_object'][:3,3] for o in self.inputs.objects.values()],axis=0)
        if self.camera.value!='全景' and self.focus is not None:
            center=np.asarray(self.focus)
            offset=[.38,-.35,.20] if self.camera.value=='夹爪侧面' else [.35,-.40,.30]
        else:offset=[-.85,.65,.62] if self.inputs.task=='microwave' else [-.65,.65,.55]
        if self.camera.value!='全景' and self.inputs.task=='microwave' and self.focus is not None:
            outward=-self.inputs.objects['microwave']['T_world_object'][:3,1]
            lateral=np.cross([0.,0.,1.],outward)
            offset=.42*outward+.12*lateral+np.array([0.,0.,.10])
        for c in ([client] if client else self.server.get_clients().values()):
            c.camera.position=tuple(center+np.asarray(offset));c.camera.look_at=tuple(center);c.camera.up_direction=(0.,0.,1.)
        self.server.initial_camera.position=tuple(center+np.asarray(offset));self.server.initial_camera.look_at=tuple(center)

    def add_geometry(self,name,row,T,color=None):
        mesh=load_geometry(row)
        if color is not None:mesh.visual.face_colors=np.tile([*[int(255*v) for v in color],255],(len(mesh.faces),1))
        elif not row.get('mesh_path'):mesh.visual.face_colors=np.tile([*[int(255*v) for v in row.get('color',[.65,.7,.75])[:3]],255],(len(mesh.faces),1))
        return self.server.scene.add_mesh_trimesh(name,mesh,**pose_args(T))

    def show_goal(self,pose,group,width):
        self.focus=np.asarray(pose)[:3,3]
        self.target.visible=self.show_axes.value
        for k,v in pose_args(np.asarray(pose)).items():setattr(self.target,k,v)
        if self.show_target.value:self.gripper.select(self.inputs.robots[group],np.asarray(pose),width)
        else:self.gripper.hide()

    def setup_knowledge(self,selection,tilt,output):
        self.knowledge_selection=selection;self.output=Path(output);self.tilt=tilt
        self.knowledge_initial_joints=copy.deepcopy(self.inputs.object_joint_positions)
        self.knowledge_paths=[]
        self.status.content='**几何候选，未做 IK / 碰撞筛选；不是机械臂已到达状态。**'
        self.choice=self.server.gui.add_dropdown('候选 / 路径点',options=('等待生成',))
        self.detail=self.server.gui.add_markdown('')
        self.tune={}
        with self.server.gui.add_folder('几何参数（修改后重新生成）',expand_by_default=False):
            self.tune['tilt']=self.server.gui.add_number('抓刷倾角 (°)',initial_value=float(tilt),step=1.,visible=selection=='brush_grasp')
            if selection in {'door_push','handle_pull','knob_turn'}:
                self.tune['target_angle_rad']=self.server.gui.add_number('目标角 (rad)',initial_value=1.48353 if selection=='door_push' else .43633 if selection=='handle_pull' else .45,step=.05)
            if selection in {'door_push','handle_pull','knob_turn'}:
                self.tune['candidate_rank']=self.server.gui.add_number('候选顺序',initial_value=0,step=1)
            if selection=='door_push':
                self.tune['start_angle_rad']=self.server.gui.add_number('起始门角 (rad)',initial_value=.523599,step=.05)
                self.tune['mode']=self.server.gui.add_dropdown('推门方向',options=('right_open','left_open','right_close','left_close'))
            self.tune_error=self.server.gui.add_markdown('')
            self.server.gui.add_button('重新生成候选与路径').on_click(lambda _:self.regenerate())
        self.choice.on_update(lambda _:self.select_knowledge())
        self.show_target.on_update(lambda _:self.select_knowledge());self.show_axes.on_update(lambda _:self.select_knowledge())
        self.show_path.on_update(lambda _:[setattr(h,'visible',self.show_path.value) for h in self.knowledge_paths])
        self.rebuild_knowledge()

    def regenerate(self):
        try:
            self.rebuild_knowledge();self.tune_error.content='已更新；候选与参数保存在本次查看目录的 knowledge.json。'
        except (ValueError,IndexError) as error:
            self.tune_error.content='参数无效：'+str(error)

    def rebuild_knowledge(self):
        with self._lock:
            parameters={k:v.value for k,v in self.tune.items() if k!='tilt'}
            displayed=copy.deepcopy(self.inputs.object_joint_positions)
            try:
                for name,joints in self.knowledge_initial_joints.items():self.inputs.set_object_joints(name,joints)
                rows=proposals(self.inputs,self.knowledge_selection,tilt=self.tune['tilt'].value,**({'parameters':parameters} if parameters else {}))
            finally:
                for name,joints in displayed.items():self.inputs.set_object_joints(name,joints)
            if not rows:raise ValueError('Knowledge 没有产生候选')
            for h in self.overlay:h.remove()
            self.overlay=[];self.knowledge_paths=[];self.rows=rows
            self._updating=True
            self.choice.options=tuple(f'{i}: {r["name"]}' for i,r in enumerate(self.rows));self.choice.value=self.choice.options[0]
            self._updating=False
            points=np.array([r['pose'][:3,3] for r in self.rows])
            self.overlay.append(self.server.scene.add_point_cloud('/knowledge/candidates',points=points,
                colors=np.tile([65,139,206],(len(points),1)),point_size=.005))
            groups={}
            for row in self.rows:groups.setdefault(row['path_id'],[]).append(row['pose'][:3,3]) if row.get('path_id') else None
            for i,values in enumerate(groups.values()):
                h=line_segments(self.server,f'/knowledge/path_{i}',values)
                if h:
                    h.visible=self.show_path.value;self.overlay.append(h);self.knowledge_paths.append(h)
            from gene_agent.feedback.live_record import atomic_json
            atomic_json(self.output/'knowledge.json',dict(task=self.inputs.task,selection=self.knowledge_selection,candidates=self.rows,evidence_mode='knowledge_geometry_only'))
            self.select_knowledge()

    def select_knowledge(self):
        if self._updating or not hasattr(self,'rows'):return
        with self._lock:
            row=self.rows[int(self.choice.value.split(':')[0])]
            for name,values in row.get('object_joint_positions',{}).items():self.inputs.set_object_joints(name,values)
            self.sync_objects()
            self.show_goal(row['pose'],row['group'],row.get('width_m',.04))
            for h in getattr(self,'selected_overlays',[]):h.remove()
            self.selected_overlays=[]
            point=row.get('contact_point_world')
            if point is None and row.get('contact_points_world'):point=row['contact_points_world'][0]
            normal=row.get('surface_normal_world')
            if point is not None and normal is not None:
                h=line_segments(self.server,'/knowledge/contact_normal',[point,np.asarray(point)+np.asarray(normal)*.08],(225,95,65),4.)
                if h:self.selected_overlays.append(h)
            summary={k:row[k] for k in ('parameters','waypoint_label','angle_rad','width_m','axis_point_world','axis_direction_world') if k in row}
            self.detail.content='**'+row['name']+'**\n几何候选，未筛 IK / 碰撞；单位 m / rad。\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:v.tolist())+'\n```'
            if self.camera.value!='全景':self.set_camera()

    def setup_playback(self):
        gui=self.server.gui;self.stages=[]
        self.follow=gui.add_checkbox('跟随新增阶段（暂停浏览不停止任务）',initial_value=True)
        self.stage=gui.add_dropdown('阶段',options=('等待首个阶段',),disabled=True)
        self.position=gui.add_slider('阶段进度',min=0.,max=1.,step=.005,initial_value=0.,disabled=True)
        self.speed=gui.add_slider('播放速度',min=.1,max=3.,step=.1,initial_value=1.)
        gui.add_button('播放 / 暂停').on_click(lambda _:self.toggle_play())
        gui.add_button('回到起点').on_click(lambda _:self.rewind())
        self.info=gui.add_markdown('等待首个阶段；尚无可播放轨迹。')
        self.stage.on_update(lambda _:self.changed_stage())
        self.position.on_update(lambda _:self.update())
        for control in (self.show_target,self.show_axes,self.show_path):control.on_update(lambda _:self.update())
        self.update_report(self.report)

    def toggle_play(self):
        self.playing=not self.playing;self.user_paused=not self.playing

    def rewind(self):
        self.follow.value=False;self.playing=False;self.user_paused=True
        if self.stages:self.stage.value=self.stage.options[0];self.position.value=0.;self.update()

    def changed_stage(self):
        if self._updating:return
        self.position.value=0.;self._drawn_stage=None;self.update()

    def update_report(self,report):
        with self._lock:
            previous=len(self.stages);self.report=report
            self.stages=[r for r in report['stages'] if not self.skill or f'/{self.skill}/' in r['name']]
            self.status.content=f"**状态：{report['status']}**；已记录 {len(self.stages)} 阶段；任务成功：**未验证**\n场景碰撞：{'检查' if report.get('environment_collision_enabled') else '忽略'}；持物碰撞球未接入。\n"+'\n'.join(report.get('initial_assumptions',[]))+ ('\n**失败：** '+report['error'] if report.get('error') else '')+ ('\n正在计算：'+report['active_stage'] if report['status']=='planning' and report.get('active_stage') else '')
            if not self.stages and report.get('knowledge'):
                row=report['knowledge'][-1]
                self.show_goal(row['pose'],row['group'],row.get('width_m',.04))
            if self.stages:
                old=self.stage.value;self._updating=True
                self.stage.options=tuple(f'{i}: {s["name"]}' for i,s in enumerate(self.stages))
                self.stage.value=old if old in self.stage.options else self.stage.options[0]
                self.stage.disabled=self.position.disabled=False;self._updating=False
                if not self.user_paused and (not previous or (self.follow.value and len(self.stages)>previous)):self.playing=True
                self.update()

    def sync_objects(self):
        for name,obj in self.inputs.objects.items():
            for handle,visual in zip(self.handles[name],obj['visuals']):
                for k,v in pose_args(obj['T_world_object']@visual['T_object_visual']).items():setattr(handle,k,v)

    def draw_concepts(self):
        # Concept 格式较多，只复用项目已有生成器，不猜测 primitive 参数约定。
        for h in self.concept_handles:h.remove()
        self.concept_handles=[]
        if not self.show_concept.value:return
        from gene_agent.visualization.concept_editor.model import ConceptEditorModel,concept_mesh
        import trimesh
        for name,obj in self.inputs.objects.items():
            path=obj.get('concept_path')
            if not path:continue
            try:
                model=ConceptEditorModel.from_json(path)
                parts=[concept_mesh(model.concept_at(i)) for i in range(len(model.parts))]
                mesh=trimesh.util.concatenate([trimesh.Trimesh(vertices=p.vertices,faces=p.faces,process=False) for p in parts])
            except (ValueError,AttributeError,KeyError,FileNotFoundError):continue
            mesh.visual.face_colors=np.tile([245,190,55,90],(len(mesh.faces),1))
            self.concept_handles.append(self.server.scene.add_mesh_trimesh('/concept/'+name,mesh,**pose_args(obj['T_world_object'])))

    def update(self):
        if self._updating or not self.stages:return
        with self._lock:
            index=int(self.stage.value.split(':')[0]);row=self.stages[index];t=self.position.value
            before=row['start_state'];after=row['end_state'];qs={g:np.asarray(q) for g,q in before['joints'].items()}
            if row.get('positions_rad'):
                points=np.asarray(row['positions_rad']);x=t*(len(points)-1);lo=min(int(x),len(points)-2);fraction=x-lo
                qs[row['group']]=points[lo]*(1-fraction)+points[lo+1]*fraction
            elif t>=1.:qs={g:np.asarray(q) for g,q in after['joints'].items()}
            for group,r in self.inputs.robots.items():
                cfg=r.set_state(qs[group],(1-t)*before['widths'][group]+t*after['widths'][group]);self.robot_views[group].update_cfg(cfg)
            for name,values in before.get('object_joints',{}).items():
                end=after.get('object_joints',{}).get(name,values)
                self.inputs.set_object_joints(name,{k:(1-t)*v+t*end.get(k,v) for k,v in values.items()})
            active=before['attachments'] if t<1 else after['attachments']
            for name,obj in self.inputs.objects.items():
                start=np.asarray(before['objects'][name]);end=np.asarray(after['objects'][name]);T=start.copy()
                T[:3,3]=(1-t)*start[:3,3]+t*end[:3,3]
                T[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack([start[:3,:3],end[:3,:3]])))(t).as_matrix()
                if name in active:
                    a=active[name];T=self.inputs.robots[a['group']].fk(qs[a['group']])@np.asarray(a['T_tcp_object'])
                obj['T_world_object']=T
            self.sync_objects()
            if row.get('target_tcp_world'):
                self.show_goal(row['target_tcp_world'],row['group'],after['widths'][row['group']])
                actual=self.inputs.robots[row['group']].fk(qs[row['group']]);self.actual.visible=self.show_axes.value
                for k,v in pose_args(actual).items():setattr(self.actual,k,v)
            else:
                previous=next((r for r in reversed(self.stages[:index]) if r.get('target_tcp_world')),None)
                if previous:
                    group=previous['group'];self.show_goal(previous['target_tcp_world'],group,after['widths'][group])
                    actual=self.inputs.robots[group].fk(qs[group]);self.actual.visible=self.show_axes.value
                    for k,v in pose_args(actual).items():setattr(self.actual,k,v)
                else:self.target.visible=False;self.actual.visible=False;self.gripper.hide()
            signature=(index,self.show_path.value)
            if self._drawn_stage!=signature:
                for h in self.path_handles:h.remove()
                self.path_handles=[]
                if self.show_path.value:
                    for path_index,path_row in enumerate(self.stages):
                        if not path_row.get('positions_rad'):continue
                        robot=self.inputs.robots[path_row['group']];points=np.asarray(path_row['positions_rad'])
                        sampled=points[np.unique(np.linspace(0,len(points)-1,min(70,len(points))).astype(int))]
                        xyz=[robot.fk(q)[:3,3] for q in sampled]
                        h=line_segments(self.server,f'/trajectory/{path_index}',xyz,
                                        (65,139,206) if path_index==index else (131,174,204),4. if path_index==index else 2.)
                        if h:self.path_handles.append(h)
                self._drawn_stage=signature
            self.info.content=f"**{row['name']}**\n规划通过：{row['success']}；路径：{row['path_kind']}\n终点平移误差：{row.get('position_error_m','未提供')} m；转角：{row.get('rotation_error_rad','未提供')} rad\n近似策略：{row.get('approximation') or '严格'}\n原始返回：{row.get('plan_file','状态假设/检查')}"

    def tick(self):
        now=time.monotonic();dt=now-self.last_tick;self.last_tick=now
        if not self.playing or not hasattr(self,'stages') or not self.stages:return
        row=self.stages[int(self.stage.value.split(':')[0])]
        duration=max(.3,(row.get('times_s') or [0.,.5])[-1]);next_time=self.position.value+dt*self.speed.value/duration
        if next_time>=1.:
            self.position.value=1.;index=int(self.stage.value.split(':')[0])+1
            if index<len(self.stages):self.stage.value=self.stage.options[index]
            elif not self.follow.value or self.report['status']!='planning':self.playing=False
        else:self.position.value=next_time
