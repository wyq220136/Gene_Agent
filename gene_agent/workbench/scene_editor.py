"""完整场景中的摆位草稿；复用 MuJoCo visual，只在保存时创建新场景。"""
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.visualization.pick_bowl.scene import _geom_pose, _pose


LABELS={'brush':'刷子','cube':'绿方块','cube_red':'红方块','cube_blue':'蓝方块','dustpan':'簸箕','microwave':'微波炉','bowl':'碗','table':'桌面'}


def screen_to_xy_plane(camera,screen_xy,z):
    """OpenCV 相机射线与世界水平面相交；screen_xy 为归一化图像坐标。"""
    u,v=screen_xy
    direction=np.array([(u-.5)*2*np.tan(camera.fov/2)*camera.aspect,
                        (v-.5)*2*np.tan(camera.fov/2),1.])
    direction=Rotation.from_quat(np.asarray(camera.wxyz)[[1,2,3,0]]).apply(direction)
    origin=np.asarray(camera.position,dtype=float)
    if abs(direction[2])<1e-6:raise ValueError('视角接近平行桌面，请稍微俯视后拖动物体')
    distance=(z-origin[2])/direction[2]
    if distance<=0:raise ValueError('拖动平面位于相机后方，请调整视角')
    return origin+distance*direction


class SceneEditor:
    def __init__(self,server,case,renderer,*,workbench_url,output):
        self.server=server
        self.case=case
        self.renderer=renderer
        self.url=workbench_url.rstrip('/')
        self.output=Path(output)
        self.lock=threading.RLock()
        self.syncing=False
        self.saved=None;self.dirty=False
        self.mesh_drags={}
        self.bounds={}
        self.base={}
        self.poses={}
        self.geoms={}
        for name in case.scene['objects']:
            body=case.model.body(name)
            pose=np.eye(4)
            pose[:3,:3]=case.data.xmat[body.id].reshape(3,3)
            pose[:3,3]=case.data.xpos[body.id]
            self.base[name]=pose
            self.poses[name]=pose.copy()
            self.geoms[name]=[]
            roots=getattr(case,'visual_roots',{}).get(name,(name,))
            root_ids={case.model.body(n).id for n in roots}
            for gid,handle in renderer.visual_handles.items():
                owner=int(case.model.geom_bodyid[gid])
                while owner and owner not in root_ids:
                    owner=int(case.model.body_parentid[owner])
                if owner in root_ids:
                    self.geoms[name].append((handle,_geom_pose(case.data,gid)))
                    # async 回调按 Viser 消息顺序处理，避免 start/update/end 在线程池里乱序。
                    async def click(event,n=name):
                        self.select(n)
                    async def drag(event,n=name):
                        self._mesh_drag(n,event)
                    handle.on_click(click)
                    handle.on_drag(drag)
            from gene_agent.scene.rigid_geometry import collision_vertices_object
            points=[]
            for root_name in roots:
                vertices=collision_vertices_object(case.model,case.data,root_name)
                root_body=case.data.body(root_name)
                world=vertices@root_body.xmat.reshape(3,3).T+root_body.xpos
                points.extend((world-pose[:3,3])@pose[:3,:3])
            points=np.asarray(points)
            self.bounds[name]=(points.min(axis=0),points.max(axis=0))
        # 桌面是 world 下的 geom，不是可动 body；沿用同一选中/拖动草稿。
        table=case.model.geom('scene_table')
        pose=_geom_pose(case.data,table.id)
        self.base['table']=pose.copy();self.poses['table']=pose.copy()
        self.table_size=2*np.asarray(table.size).copy();self.table_base_size=self.table_size.copy()
        self.bounds['table']=(-self.table_size/2,self.table_size/2)
        handle=renderer.visual_handles[table.id]
        self.geoms['table']=[(handle,pose.copy())]
        async def table_click(event):self.select('table')
        async def table_drag(event):self._mesh_drag('table',event)
        handle.on_click(table_click);handle.on_drag(table_drag)
        server.gui.configure_theme(dark_mode=False,control_width='medium',show_logo=False)
        server.gui.add_markdown('## 场景编辑\n左键直接拖动物体沿桌面移动；单击显示选中边框。拖空白处旋转视角，滚轮缩放。箭头和圆环可精确调整位置与旋转。')
        names=tuple(self.poses)
        self.labels={LABELS.get(n,n):n for n in names}
        self.selection=server.gui.add_dropdown('选择物体',options=tuple(self.labels),initial_value=next(iter(self.labels)))
        self.info=server.gui.add_markdown('')
        self.status=server.gui.add_markdown(
            ('当前为任务构建初态，尚未推进物理。' if case.task=='microwave' else '当前是名义摆位的落稳初态。')
            +'编辑草稿尚未验证可达性或任务成功。')
        self.control=server.scene.add_transform_controls('/scene_editor/selected',scale=.18,depth_test=False,line_width=4.)
        self.control.on_update(self._control_update)
        self.selection_box=server.scene.add_box('/scene_editor/selection',
            dimensions=(.1,.1,.1),wireframe=True,color=(255,166,40),opacity=.9)
        self.selection.on_update(lambda _:self._sync())
        with server.gui.add_folder('精确调整当前物体',expand_by_default=False):
            self.fields=[server.gui.add_number(label,initial_value=0.,step=.005 if i<3 else 1.)
                         for i,label in enumerate(('X (m)','Y (m)','Z (m)','Roll (°)','Pitch (°)','Yaw (°)'))]
            for field in self.fields:field.on_update(self._numbers)
        with server.gui.add_folder('桌面尺寸（米）',expand_by_default=True):
            server.gui.add_markdown('选择“桌面”可拖动位置；Z 为桌面顶高。移动桌面不会搬动物体。保存时摆位边界随桌面更新。')
            self.table_fields=[server.gui.add_number(label,initial_value=float(value),step=.01 if i<2 else .005)
                               for i,(label,value) in enumerate(zip(('长度 X (m)','宽度 Y (m)','厚度 (m)'),self.table_size))]
            for field in self.table_fields:field.on_update(self._table_numbers)
        server.gui.add_button('重置当前物体').on_click(self._reset_one)
        server.gui.add_button('重置整个场景').on_click(self._reset_all)
        self.save_button=server.gui.add_button('保存为新场景并用于任务',color='blue')
        self.save_button.on_click(self._save)
        self._sync()

    @property
    def selected(self):
        return self.labels[self.selection.value]

    def select(self,name):
        with self.lock:
            self.selection.value=next(label for label,n in self.labels.items() if n==name)
            self._sync()

    def _sync(self):
        if self.syncing:return
        self.syncing=True
        try:
            pose=self.poses[self.selected]
            self.control.wxyz,self.control.position=_pose(pose)
            self.control.disable_rotations=self.selected=='table'
            values=[*pose[:3,3],*Rotation.from_matrix(pose[:3,:3]).as_euler('xyz',degrees=True)]
            self.fields[2].label='桌面顶高 Z (m)' if self.selected=='table' else 'Z (m)'
            if self.selected=='table':values[2]+=self.table_size[2]/2
            for field in self.fields[3:]:field.disabled=self.selected=='table'
            for field,value in zip(self.table_fields,self.table_size):field.value=float(value)
            for field,value in zip(self.fields,values):field.value=float(value)
            lower,upper=self.bounds[self.selected]
            outline=pose.copy();outline[:3,3]=pose[:3,3]+pose[:3,:3]@((lower+upper)/2)
            self.selection_box.dimensions=tuple(upper-lower+.004)
            self.selection_box.wxyz,self.selection_box.position=_pose(outline)
            self.info.content=(f'**当前物体：{self.selection.value}**  \n世界位置：'
                               f'{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f} m')
        finally:self.syncing=False

    def set_pose(self,name,pose):
        """拖拽和数值共用同一草稿；不推进物理，也不修改运行输入。"""
        with self.lock:
            pose=np.asarray(pose,dtype=float)
            # GUI 字段程序同步也会排队触发回调；相同 pose 不应把单击选中标成未保存。
            if np.allclose(self.poses[name],pose,atol=1e-10,rtol=0):return
            self.poses[name]=pose.copy()
            delta=self.poses[name]@np.linalg.inv(self.base[name])
            for handle,geom in self.geoms[name]:
                handle.wxyz,handle.position=_pose(delta@geom)
            self._mark_dirty()
            self._sync()

    def _mark_dirty(self):
        self.saved=None
        if not self.dirty:
            from gene_agent.feedback.live_record import atomic_json
            atomic_json(self.output/'draft_state.json',{'unsaved':True})
            self.dirty=True
        self.status.content='有未保存的场景摆位或桌面修改。保存后再检查分布或运行仿真。'

    def table_config(self):
        pose=self.poses['table']
        return {'center_xy_m':pose[:2,3].tolist(),'top_z_m':float(pose[2,3]+self.table_size[2]/2),
                'size_xy_m':self.table_size[:2].tolist(),'thickness_m':float(self.table_size[2])}

    def resize_table(self,size):
        from gene_agent.scene.table import validate_table
        with self.lock:
            size=np.asarray(size,float)
            if size.shape!=(3,):raise ValueError('桌面尺寸需要长、宽、厚三个值')
            validate_table({**self.table_config(),'size_xy_m':size[:2].tolist(),'thickness_m':float(size[2])})
            if np.allclose(size,self.table_size,atol=1e-10,rtol=0):return
            # 调厚度时保持顶面高度，不把物体支撑面上下挪动。
            top=self.table_config()['top_z_m']
            self.table_size=size.copy()
            self.bounds['table']=(-size/2,size/2)
            for handle,_ in self.geoms['table']:handle.dimensions=tuple(size)
            pose=self.poses['table'].copy();pose[2,3]=top-size[2]/2
            self.set_pose('table',pose);self._mark_dirty();self._sync()

    def _table_numbers(self,event):
        if self.syncing:return
        try:self.resize_table([field.value for field in self.table_fields])
        except ValueError as exc:
            self.status.content=str(exc);self._sync()

    async def _control_update(self,event):
        self._drag(event)

    def _mesh_drag(self,name,event):
        """以抓住点的高度为水平拖动平面，不随像素更新累计误差或改变 Z。"""
        key=(event.client_id,event.target.name)
        with self.lock:
            if event.phase=='start':
                self.select(name)
                camera=event.client.camera
                from types import SimpleNamespace
                frozen=SimpleNamespace(position=np.asarray(camera.position).copy(),
                    wxyz=np.asarray(camera.wxyz).copy(),fov=camera.fov,aspect=camera.aspect)
                z=float(event.start_position[2])
                try:origin=screen_to_xy_plane(frozen,event.start_screen_pos,z)
                except ValueError as exc:
                    self.status.content=str(exc);return
                self.mesh_drags[key]=(self.poses[name].copy(),frozen,origin,z)
            if key not in self.mesh_drags:return
            base,camera,origin,z=self.mesh_drags[key]
            try:
                delta=screen_to_xy_plane(camera,event.end_screen_pos,z)-origin
                pose=base.copy();pose[:2,3]+=delta[:2]
                self.set_pose(name,pose)
            except ValueError as exc:self.status.content=str(exc)
            finally:
                if event.phase=='end':
                    self.mesh_drags.pop(key,None)
                    print(json.dumps({'event':'object_drag_end','object':name,
                        'position_m':self.poses[name][:3,3].tolist()}),flush=True)

    def _drag(self,event):
        with self.lock:
            if self.syncing:return
            pose=np.eye(4)
            pose[:3,:3]=Rotation.from_quat(np.asarray(self.control.wxyz)[[1,2,3,0]]).as_matrix()
            pose[:3,3]=self.control.position
            self.set_pose(self.selected,pose)

    def _numbers(self,event):
        with self.lock:
            if self.syncing:return
            values=[float(f.value) for f in self.fields]
            pose=np.eye(4)
            pose[:3,3]=values[:3]
            pose[:3,:3]=Rotation.from_euler('xyz',values[3:],degrees=True).as_matrix()
            if self.selected=='table':pose[2,3]-=self.table_size[2]/2
            self.set_pose(self.selected,pose)

    def _reset_one(self,event):
        if self.selected=='table':self.resize_table(self.table_base_size)
        self.set_pose(self.selected,self.base[self.selected])

    def _reset_all(self,event):
        with self.lock:
            self.resize_table(self.table_base_size)
            for name in self.poses:self.set_pose(name,self.base[name])

    def payload(self):
        with self.lock:
            return {'task':self.case.task,'scene':str(self.case.scene_dir),'placement_rules':self.case.rules,
                    'table':self.table_config(),
                    'objects':[{'name':n,'T_world_object':p.tolist()} for n,p in self.poses.items() if n!='table']}

    def _save(self,event):
        self.save_button.disabled=True
        try:
            # 经同一个工作台 API 写版本和索引；子进程不并发修改 library.json。
            with urllib.request.urlopen(self.url+'/api/bootstrap',timeout=5) as response:
                bootstrap=json.load(response)
            if not bootstrap.get('table_editing'):
                raise ValueError('工作台后台仍是旧版本，不支持保存桌面；请重启工作台后重新打开场景编辑器')
            token=bootstrap['token']
            request=urllib.request.Request(self.url+'/api/scene',data=json.dumps(self.payload()).encode(),
                headers={'Content-Type':'application/json','X-Workbench-Token':token},method='POST')
            with urllib.request.urlopen(request,timeout=30) as response:result=json.load(response)
            self.saved=result;self.dirty=False
            from gene_agent.feedback.live_record import atomic_json
            atomic_json(self.output/'draft_state.json',{'unsaved':False})
            target=self.output/'saved_scene.json'
            temp=target.with_suffix('.tmp')
            temp.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            temp.replace(target)
            self.status.content=f"**已保存新场景**：{result['path']}  \n工作台将选用此摆位并关闭随机扰动。仍需正式仿真验证。"
        except Exception as exc:
            detail=exc.read().decode() if isinstance(exc,urllib.error.HTTPError) else str(exc)
            self.status.content=f'**保存失败**：{detail}'
            raise
        finally:self.save_button.disabled=False
