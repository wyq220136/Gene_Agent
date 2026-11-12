"""只读规划碰撞快照；不加载物理引擎，不重新求解或执行。"""
from pathlib import Path
import json
import numpy as np


def build_diagnostic_scene(bundle, configuration_ref=None):
    import trimesh
    snapshot=bundle.get('snapshot',{})
    world_from_base=np.linalg.inv(np.asarray(snapshot.get('base_from_world',np.eye(4)),float))
    meshes=[];warnings=[]
    robot_links=set()
    urdf=(bundle.get('effective_robot_config',{}).get('kinematics',{})).get('urdf_path')
    if urdf:
        try:
            import xml.etree.ElementTree as ET
            robot_links={link.get('name') for link in ET.parse(urdf).getroot().findall('link')}
        except Exception as exc:warnings.append(f'Robot mesh classification unavailable: {exc}')
    bodies={row['obstacle_name']:row.get('body_name') for row in snapshot.get('records',[]) if row.get('obstacle_name')}
    hashes={row['obstacle_name']:row.get('mesh_sha256') for row in snapshot.get('records',[]) if row.get('obstacle_name')}
    for kind,objects in snapshot.get('scene_model',{}).items():
        for name,obj in objects.items():
            try:
                if kind=='cuboid':mesh=trimesh.creation.box(extents=obj['dims'])
                elif kind=='mesh':
                    import hashlib
                    expected=hashes.get(name)
                    if expected and hashlib.sha256(Path(obj['file_path']).read_bytes()).hexdigest()!=expected:
                        raise ValueError('collision asset hash changed')
                    if not expected:warnings.append(f'{name}: saved collision asset hash missing; integrity unverified')
                    mesh=trimesh.load(obj['file_path'],force='mesh',process=False)
                    mesh.apply_scale(obj.get('scale',[1,1,1]))
                else:
                    warnings.append(f'Unsupported display geometry: {kind}/{name}');continue
                pose=obj['pose'];matrix=trimesh.transformations.quaternion_matrix(pose[3:7]);matrix[:3,3]=pose[:3]
                mesh.apply_transform(world_from_base@matrix)
                meshes.append(dict(name=name,mesh=mesh,role='robot' if bodies.get(name) in robot_links else 'environment'))
            except Exception as exc:warnings.append(f'{name}: {type(exc).__name__}: {exc}')
    collision=bundle.get('collisions') or {};spheres=collision.get('spheres',[])
    if not spheres:warnings.append('No saved sphere evidence; no spheres reconstructed from current model')
    candidate=(bundle.get('candidates') or {}).get('nearest_checked')
    return dict(mode='planned_diagnostic',configuration_ref=configuration_ref or bundle.get('configuration_ref'),
        meshes=meshes,spheres=spheres,pairs=collision.get('pairs',[]),candidate=candidate,
        requested_pose=(bundle.get('request',{}).get('target') or {}).get('matrix'),warnings=warnings)


def export_diagnostic_image(bundle, output):
    """离线导出一张总览；pair位置为球中心/中点，不冒充mesh接触点。"""
    import time
    started=time.perf_counter()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    scene=build_diagnostic_scene(bundle)
    fig=plt.figure(figsize=(12,8));ax=fig.add_subplot(111,projection='3d')
    hot={p['first']['sphere_index'] for p in scene['pairs'] if not p.get('allowed')}
    hot.update(p['second']['sphere_index'] for p in scene['pairs'] if p['kind']=='self' and not p.get('allowed'))
    for row in scene['meshes']:
        mesh=row['mesh'];faces=mesh.faces[::max(1,len(mesh.faces)//2000)]
        ax.add_collection3d(Poly3DCollection(mesh.vertices[faces],alpha=.08,facecolor='steelblue',edgecolor='none'))
    coords=[]
    for sphere in scene['spheres']:
        center=np.asarray(sphere['center_world_m']);radius=sphere.get('effective_radius_m',sphere.get('radius_m',0))
        if radius<=0:continue
        coords.append(center)
        color='crimson' if sphere['sphere_index'] in hot else 'darkorange' if sphere.get('role')=='held' else 'slategray'
        # 三个正交大圆保留真实米制半径，避免scatter像素大小冒充碰撞几何。
        t=np.linspace(0,2*np.pi,18)
        for plane in [(0,1),(0,2),(1,2)]:
            points=np.tile(center,(len(t),1));points[:,plane[0]]+=radius*np.cos(t);points[:,plane[1]]+=radius*np.sin(t)
            ax.plot(*points.T,color=color,alpha=.7 if sphere['sphere_index'] in hot else .18,linewidth=.6)
    poses=[]
    if scene['requested_pose'] is not None:poses.append(('target',scene['requested_pose']))
    if scene['candidate']:poses.append(('candidate FK',scene['candidate']['fk_tcp_pose_world']))
    for label,value in poses:
        matrix=np.asarray(value);coords.append(matrix[:3,3])
        for k,color in enumerate(['red','green','blue']):ax.quiver(*matrix[:3,3],*matrix[:3,k],length=.08,color=color)
        ax.text(*matrix[:3,3],label,fontsize=8)
    for pair in scene['pairs'][:5]:
        label=pair['first']['label']+' <-> '+pair['second'].get('label',pair['second'].get('object_id','unknown'))
        ax.text(*pair['position_world_m'],label,fontsize=6,color='crimson')
    if coords:
        points=np.array(coords);center=(points.min(0)+points.max(0))/2;span=max(.3,float(np.ptp(points,axis=0).max())*1.15)
        ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2);ax.set_zlim(center[2]-span/2,center[2]+span/2)
    ax.set_box_aspect([1,1,1]);ax.view_init(elev=24,azim=-55)
    ax.set(xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title='PLANNED DIAGNOSTIC — not actual execution')
    fig.tight_layout();output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=130);plt.close(fig)
    meta=dict(mode=scene['mode'],configuration_ref=scene['configuration_ref'],camera={'elevation_deg':24,'azimuth_deg':-55},
        warnings=scene['warnings'],render_seconds=time.perf_counter()-started,
        source_snapshot=(bundle.get('snapshot') or {}).get('snapshot_sha256'),mesh_display_face_limit=2000,
        contact_locations='sphere centers/midpoints, not exact mesh contact',actual_image=False)
    output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta


def serve_diagnostic_scene(bundle, port=8081):
    """兼容旧Python入口，交互显示统一交给Viser双视图实现。"""
    import time
    from .motion_diagnostic_viser import DiagnosticViewer
    app=DiagnosticViewer(bundle,port=port)
    print(f'Diagnostic viewer: http://localhost:{port}',flush=True)
    try:
        while True:time.sleep(1)
    finally:app.close()
