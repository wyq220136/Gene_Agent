"""保存现场的Viser双视图；相机和可见层只影响显示，不重算碰撞。"""
from pathlib import Path
import json
import math
import time
import numpy as np
from .motion_diagnostics import build_diagnostic_scene


def select_view(bundle, *, view='global', pair_index=0, azimuth=-55., elevation=25., distance=None, look_at=None):
    if view not in ('global','local'):raise ValueError('view必须为global/local')
    collision=bundle.get('collisions') or {};pairs=collision.get('pairs',[])
    if pairs and not 0<=pair_index<len(pairs):raise ValueError('碰撞对编号越界')
    pair=pairs[pair_index] if pairs else None
    spheres=[s for s in collision.get('spheres',[]) if s.get('effective_radius_m',s.get('radius_m',0))>0]
    selected=[]
    if pair:
        selected=[pair['first']]
        if pair['kind']=='self':selected.append(pair['second'])
    focus=selected if view=='local' and selected else spheres
    if focus:
        centers=np.array([s['center_world_m'] for s in focus]);radii=np.array([s.get('effective_radius_m',s.get('radius_m',.02)) for s in focus])[:,None]
        low=(centers-radii).min(0);high=(centers+radii).max(0)
        target=(low+high)/2;extent=max(.12,float(np.linalg.norm(high-low)))
    else:
        target=np.asarray(bundle.get('request',{}).get('target',{}).get('matrix',np.eye(4)))[:3,3];extent=.5
    focus=target.copy()
    if look_at is not None:
        target=np.asarray(look_at,float)
        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('look_at需三个有限world坐标（米）')
    distance=float(distance if distance is not None else extent*1.9)
    if not all(math.isfinite(v) for v in (distance,azimuth,elevation)) or distance<=0 or abs(elevation)>=89:
        raise ValueError('相机需有限数值，distance>0，abs(elevation)<89')
    a,e=np.deg2rad([azimuth,elevation]);position=target+distance*np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    labels=[]
    if pair:
        labels=[pair['first']['label'],pair['second'].get('label') or f"{pair['second'].get('object_id','unknown')} / {pair['second'].get('mesh_id','unknown')}"]
    return dict(view=view,pair_index=pair_index,pair=pair,labels=labels,target=target.tolist(),position=position.tolist(),
        distance=distance,azimuth=float(azimuth),elevation=float(elevation),extent=extent,
        focus=focus.tolist(),warnings=[] if pair else ['No saved collision pair. This view does not establish a collision.'])


# 固定角色配色；碰撞双方优先于机器人/持物/环境分类。
COLORS = dict(first=(235,65,55), second=(45,125,220), robot=(105,115,130),
              held=(235,165,45), environment=(195,205,190))


def nearby_bounds(bounds, focus, radius):
    bounds = np.asarray(bounds);focus = np.asarray(focus)
    delta = np.maximum(bounds[0]-focus, 0)+np.minimum(bounds[1]-focus, 0)
    return bool(np.linalg.norm(delta) <= radius)


def plain_render(rgb):
    """只合成透明背景，不在场景图上绘制文字。"""
    from PIL import Image
    raw = Image.fromarray(rgb).convert('RGBA')
    return Image.alpha_composite(Image.new('RGBA', raw.size, (238,238,238,255)), raw).convert('RGB')


class DiagnosticViewer:
    def __init__(self, bundle, *, port=8081, layer='collision', context_radius=.25, **options):
        import viser
        from .motion_diagnostic_visual import load_visual_scene
        if layer not in ('visual', 'collision'):
            raise ValueError('layer必须为visual/collision')
        if not math.isfinite(context_radius) or context_radius <= 0:
            raise ValueError('context_radius必须为正米制距离')
        select_view(bundle, **options)
        self.bundle=bundle;self.scene=build_diagnostic_scene(bundle);self.options=options
        self.visual=load_visual_scene(bundle);self.layer=layer;self.context_radius=context_radius
        self.server=viser.ViserServer(host='127.0.0.1',port=port)
        self.server.scene.set_up_direction('+z')
        self.server.gui.add_markdown('## Saved planning diagnostic\nNot an executed candidate. Display filtering does not change planning.')
        self.info=self.server.gui.add_markdown('')
        choice=self.server.gui.add_dropdown('View',options=('global','local'),initial_value=options.get('view','global'))
        layers=self.server.gui.add_dropdown('Geometry',options=('visual','collision'),initial_value=layer)
        pairs=(bundle.get('collisions') or {}).get('pairs',[])
        pair_choice=self.server.gui.add_dropdown('Collision pair',options=tuple(str(i) for i in range(len(pairs))) or ('0',),initial_value=str(options.get('pair_index',0)))
        @choice.on_update
        def change(_):self.set_view(view=choice.value)
        @layers.on_update
        def change_layer(_):self.set_view(layer=layers.value)
        @pair_choice.on_update
        def change_pair(_):self.set_view(pair_index=int(pair_choice.value))
        self.set_view(**options)
        @self.server.on_client_connect
        def connected(client):self.apply_camera(client)

    def apply_camera(self,client):
        client.camera.position=np.asarray(self.selection['position'])
        client.camera.look_at=np.asarray(self.selection['target'])
        client.camera.up_direction=np.array([0.,0.,1.])
        client.camera.fov=np.deg2rad(45.)
        client.flush()

    def set_view(self, *, layer=None, **options):
        import trimesh
        if layer is not None:
            if layer not in ('visual','collision'):raise ValueError('未知显示层')
            self.layer=layer
        self.options.update(options);v=self.selection=select_view(self.bundle,**self.options)
        v['warnings'].extend(self.scene['warnings'] if self.layer=='collision' else self.visual['warnings'])
        pair=v['pair'];local=v['view']=='local';focus=np.array(v['focus'])
        # 局部相机同时覆盖碰撞点周围空间，两层共用这一相机计算。
        if local and self.options.get('distance') is None:
            distance=max(v['distance'],self.context_radius*3.4)
            direction=np.asarray(v['position'])-v['target']
            v['position']=(np.asarray(v['target'])+direction/np.linalg.norm(direction)*distance).tolist()
            v['distance']=distance
        self.server.scene.reset()
        self.server.scene.set_background_image(np.full((8,8,3),238,dtype=np.uint8),format='png')
        visible=[]
        mesh_name=pair['second'].get('mesh_id') if pair and pair['kind']=='environment' else None
        rows=self.visual['meshes'] if self.layer=='visual' else self.scene['meshes']
        for i,row in enumerate(rows):
            name=row['name'];mesh=row['mesh'];is_pair=name==mesh_name
            if local and not is_pair and not nearby_bounds(mesh.bounds,focus,self.context_radius):continue
            # 只隐藏已登记的100m碰撞plane代理，不能按尺寸删掉原生visual。
            records=(self.bundle.get('snapshot') or {}).get('records',[])
            if self.layer=='collision' and any(r.get('obstacle_name')==name and r.get('approximation')=='plane_workspace_slab_100m' for r in records):
                v['warnings'].append(f'Display filter: plane proxy {name}')
                continue
            if self.layer=='visual':
                self.server.scene.add_mesh_trimesh('/visual/'+str(i),mesh)
            else:
                self.server.scene.add_mesh_simple('/environment/'+str(i),mesh.vertices,mesh.faces,
                    color=COLORS['second' if is_pair else row.get('role','environment')],opacity=1.,side='double')
            visible.append(name)
        if self.layer=='collision':
            first=pair['first']['sphere_index'] if pair else None
            second=pair['second']['sphere_index'] if pair and pair['kind']=='self' else None
            groups={name:[] for name in ('first','second','robot','held')}
            for sphere in self.scene['spheres']:
                center=np.array(sphere['center_world_m']);radius=sphere.get('effective_radius_m',sphere.get('radius_m',0))
                if radius<=0:continue
                index=sphere['sphere_index'];selected=index in (first,second)
                if local and not selected and np.linalg.norm(center-focus)-radius>self.context_radius:continue
                role='first' if index==first else 'second' if index==second else 'held' if sphere.get('role')=='held' else 'robot'
                mesh=trimesh.creation.icosphere(subdivisions=4 if selected else 2,radius=radius)
                mesh.apply_translation(center);groups[role].append(mesh);visible.append(sphere['label'])
            for name,meshes in groups.items():
                if meshes:
                    mesh=trimesh.util.concatenate(meshes)
                    self.server.scene.add_mesh_simple('/robot/'+name,mesh.vertices,mesh.faces,color=COLORS[name],opacity=1.)
        self.visible=visible
        self.info.content='**'+v['view'].upper()+' / '+self.layer+'**\n\n'+'\n\n'.join(f'{letter}: `{label}`' for letter,label in zip('AB',v['labels']))+'\n\n'+'\n'.join(v['warnings'])
        for client in self.server.get_clients().values():self.apply_camera(client)
        self.server.flush()

    def close(self):self.server.stop()


def export_viser_views(bundle,output,*,view='both',layer='both',pair_index=0,azimuth=-55.,elevation=25.,distance=None,look_at=None,context_radius=.25):
    """同一浏览器导出对齐四图；缺visual时明确跳过，不画替代图。"""
    from playwright.sync_api import sync_playwright
    from .motion_debug.capture import _free_port
    import hashlib,os
    if view not in ('both','global','local') or layer not in ('both','visual','collision'):
        raise ValueError('未知view/layer')
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    port=_free_port();app=DiagnosticViewer(bundle,port=port,view='global' if view=='both' else view,
        pair_index=pair_index,azimuth=azimuth,elevation=elevation,distance=distance,look_at=look_at,context_radius=context_radius)
    records=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=os.environ.get('GENE_DIAGNOSTIC_BROWSER'),headless=True,args=['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
            try:
                page=browser.new_page(viewport={'width':1400,'height':900});page.goto(f'http://127.0.0.1:{port}/')
                deadline=time.monotonic()+30
                while not app.server.get_clients():
                    if time.monotonic()>deadline:raise TimeoutError('Viser client未连接')
                    page.wait_for_timeout(100)
                client=next(iter(app.server.get_clients().values()))
                for name in (('global','local') if view=='both' else (view,)):
                    for geometry in (('visual','collision') if layer=='both' else (layer,)):
                        if geometry=='visual' and not app.visual['meshes']:
                            records.append(dict(view=name,layer=geometry,status='missing',image=None))
                            continue
                        app.set_view(view=name,layer=geometry)
                        client.get_render(height=8,width=8,transport_format='png',timeout=30)
                        page.wait_for_timeout(400)
                        rgb=client.get_render(height=800,width=1000,transport_format='png',timeout=30)
                        path=output/f'{name}_{geometry}.png';plain_render(rgb).save(path)
                        records.append(dict(view=name,layer=geometry,status=app.visual['status'] if geometry=='visual' else ('partial' if app.scene['warnings'] else 'available'),
                            warnings=app.selection['warnings'],
                            image=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            camera={k:app.selection[k] for k in ('target','position','distance','azimuth','elevation')},
                            visible_objects=app.visible,labels=app.selection['labels']))
            finally:browser.close()
    finally:app.close()
    report=dict(mode='planned_diagnostic',renderer='viser',bundle_sha256=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest(),
        warnings=app.scene['warnings']+app.visual['warnings'],visual_status=app.visual['status'],colors=COLORS,
        collision_pair=app.selection['pair'],nearest_checked=(bundle.get('candidates') or {}).get('nearest_checked'),
        scene_snapshot_sha256=(bundle.get('snapshot') or {}).get('snapshot_sha256'),configuration_identity=bundle.get('configuration_identity'),
        pair_index=pair_index,context_radius_m=context_radius,views=records,elapsed_s=time.perf_counter()-started)
    (output/'views.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# 保存现场的规划诊断','不是候选已实际执行的证据。','',
        '红：碰撞A；蓝：碰撞B；灰：机器人；橙：attachment；浅绿灰：其余环境。',
        'A/B: '+' ↔ '.join(app.selection['labels']),f'Visual状态：{app.visual["status"]}',
        f'局部邻域：{context_radius} m；按几何包围盒距离选择，未裁切mesh。','',
        '碰撞原值、单位、FK误差、配置身份及相机见 views.json。backend cost不是距离。']
    lines += ['- '+w for w in report['warnings']]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return report
