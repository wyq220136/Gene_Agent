"""冻结 RGB-D 的标定/桌面对照；只读，不连接设备、不运行仿真。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

BOWL = "Bowl/Bowl_0"
ROOT = Path(__file__).resolve().parents[2]


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def load_inputs(*, run=None, perception=None, camera_params=None):
    """优先读取原始 camera pose，绝不靠当前环境补换历史 mesh。"""
    warnings=[]
    if run:
        from gene_agent.realworld.scene_archive import verify_scene_archive
        directory=Path(run).resolve()
        bundle=directory if (directory/'manifest.json').is_file() else directory/'initial_scene'
        root,manifest=verify_scene_archive(bundle)
        entry=manifest['scenes'].get('initial')
        if not entry:raise ValueError('没有完整 initial 场景；请用 --perception 检查准备记录')
        scene=json.loads((root/entry['scene_ref']).read_text())
        mapping=manifest['file_map']
        def resolve(raw):
            if raw not in mapping:raise FileNotFoundError(f'包内未保存输入：{raw}')
            return root/mapping[raw]
        setups=[orig for orig in mapping if orig.endswith('/perception_setup.json')]
        if len(setups)!=1:raise ValueError('包内感知准备记录不唯一或缺失；请使用原 --perception 目录')
        setup=json.loads(resolve(setups[0]).read_text())
        provenance=scene['provenance']
        source=provenance['source_paths']
        camera_original=source['camera']
        # sources.camera 通常是 tracked/camera/camera_parameters_0.json，
        # RGB-D 位于 capture/。必须沿同帧 provenance 定位，不能猜同名图片。
        capture_provenance=str(Path(camera_original).parent.parent/'capture_provenance.json')
        if capture_provenance in mapping:
            origin=json.loads(resolve(capture_provenance).read_text())
            files=origin['source_files']
            camera_file=resolve(files['camera.json'])
            rgb_file=resolve(files['rgb.png'])
            depth_file=resolve(files['depth.png'])
        else:
            camera_file=resolve(camera_original)
            rgb_file=resolve(str(Path(camera_original).parent/'rgb.png'))
            depth_file=resolve(str(Path(camera_original).parent/'depth.png'))
        world_sdk=np.asarray(provenance['T_world_sdk'],float)
        table_sdk=np.linalg.inv(world_sdk)@np.asarray(provenance['table_pose_world'],float)
        thickness=float(provenance['accepted_table_thickness_m'])
        size=np.asarray(provenance['table_size_xy_m'],float)
        raw_pose=json.loads(resolve(source[BOWL]).read_text())
        pose_keys=[k for k in raw_pose if k.startswith('camera_parameters_')]
        if len(pose_keys)!=1:raise ValueError('原始碗 pose 的相机字段不唯一')
        capture=raw_pose.get('_capture',{})
        camera_pose=np.asarray(raw_pose[pose_keys[0]][0],float)
        mesh_file=resolve(setup['meshes'][BOWL])
        calibration=setup['camera']
        pose_source='recorded_raw_camera_pose'
    else:
        directory=Path(perception).resolve()
        setup_path=directory/'perception_setup.json'
        setup=json.loads(setup_path.read_text()) if setup_path.is_file() else None
        camera_file=directory/'capture/camera.json'
        rgb_file=directory/'capture/rgb.png';depth_file=directory/'capture/depth.png'
        if setup:
            calibration=setup['camera'];measurement=setup['table']
            raw_file=directory/'registrations/Bowl/Bowl_0.json'
            if raw_file.is_file():
                registration=json.loads(raw_file.read_text())
                camera_pose=np.asarray(registration['T_camera_object'],float)
                pose_source='recorded_registration_camera_pose'
            else:
                camera_pose=np.linalg.inv(np.asarray(calibration['T_sdk_camera']))@np.asarray(setup['closed_reference_poses_sdk'][BOWL])
                pose_source='reconstructed_from_recorded_sdk_pose'
                warnings.append('此旧准备记录未保存原始注册返回；相机 pose 由 SDK pose 反算，不能独立排除 pose 转换错误。运行日志 --run 优先使用原始 tracking pose。')
            mesh_file=Path(setup['meshes'][BOWL])
            if not mesh_file.is_file():raise FileNotFoundError('注册 mesh 原路径不可用；请使用带资产的 --run 场景包，不能换同名文件')
        else:
            if not camera_params:raise ValueError('准备失败、没有 setup 时，需 --camera-params 指定原标定文件')
            from gene_agent.realworld.microwave.perception_setup import load_camera_calibration,pose_mesh_path
            calibration=load_camera_calibration(camera_params)
            measurement=json.loads((directory/'table_measurement.json').read_text())
            raw=json.loads((directory/'registrations/Bowl/Bowl_0.json').read_text())
            camera_pose=np.asarray(raw['T_camera_object'],float)
            pose_source='recorded_registration_camera_pose'
            mesh_file=pose_mesh_path('Bowl',ROOT/'data/task_1/objs/Bowl/Bowl_0')
            warnings.append('失败准备记录没有完整 setup；mesh 使用本项目选取路径，报告保存当前 hash，需核对其与注册时一致。')
        table_sdk=np.asarray(measurement['table_pose_world'],float)
        size=np.asarray(measurement['table_size_xy_m'],float)
        thickness=float(measurement['accepted_table_thickness_m'])
        capture=(registration.get('_capture',{}) if setup and raw_file.is_file() else
                 raw.get('_capture',{}) if not setup else {})
    camera=json.loads(camera_file.read_text())
    if camera.get('serial')!=calibration['serial']:raise ValueError('相机序列号与标定不一致')
    if capture and capture.get('frame_number')!=camera.get('frame_number'):raise ValueError('碗 pose 与 RGB-D 不是同一帧')
    K=np.asarray(camera['intrinsic_matrix'],float)
    if not np.allclose(K,calibration['intrinsic_matrix'],atol=1e-6,rtol=0):raise ValueError('内参与标定不一致')
    rgb=np.asarray(Image.open(rgb_file).convert('RGB'))
    depth=np.asarray(Image.open(depth_file),float)
    scale={'millimeter':.001,'millimeters':.001,'mm':.001,'meter':1.,'meters':1.,'m':1.}.get(camera.get('depth_unit'))
    if scale is None:raise ValueError('深度单位必须明确为 m 或 mm')
    depth*=scale
    if rgb.shape[:2]!=depth.shape:raise ValueError('RGB-D 尺寸不匹配')
    v,u=np.indices(depth.shape)
    cloud=np.stack(((u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth),axis=-1)
    valid=np.isfinite(depth)&(depth>.1)&(depth<3.)
    if valid.sum()<100:raise ValueError('有效深度不足，不能据此判断标定或桌面')
    return dict(rgb=rgb,depth=depth,cloud=cloud,valid=valid,K=K,
        T_sdk_camera=np.asarray(calibration['T_sdk_camera'],float),T_camera_bowl=camera_pose,
        table_sdk=table_sdk,table_size=size,thickness=thickness,mesh_file=Path(mesh_file),
        source=dict(camera_file=str(camera_file),rgb_file=str(rgb_file),depth_file=str(depth_file),
                    frame_number=camera['frame_number'],serial=camera['serial'],pose_source=pose_source,
                    mesh_file=str(mesh_file),mesh_sha256=hashlib.sha256(Path(mesh_file).read_bytes()).hexdigest()),warnings=warnings)


def fit_camera_plane(points):
    """只用用户指认的桌面 ROI；不使用标定或 SDK 水平假设。"""
    points=np.asarray(points,float)
    if len(points)<100:raise ValueError('ROI 有效深度不足 100 点')
    sample=points[np.linspace(0,len(points)-1,min(len(points),6000),dtype=int)]
    rng=np.random.default_rng(0);best=None
    for _ in range(192):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-9:continue
        normal/=length
        inlier=np.abs((sample-a)@normal)<.005
        if best is None or inlier.sum()>best.sum():best=inlier
    if best is None or best.sum()<100 or best.mean()<.5:raise ValueError('ROI 中没有占多数的平面；请选干净桌面，排除物体、桌沿和背景')
    selected=sample[best];center=selected.mean(0)
    _,_,vt=np.linalg.svd(selected-center,full_matrices=False);normal=vt[-1]
    if normal@(-center)<0:normal=-normal
    residual=(selected-center)@normal
    axes=vt[:2];uv=(selected-center)@axes.T
    lo,hi=np.min(uv,axis=0),np.max(uv,axis=0)
    quad=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])@axes+center
    return center,normal,quad,dict(inlier_fraction=float(best.mean()),rms_m=float(np.sqrt(np.mean(residual**2))),point_count=len(sample))


def build_alignment_report(inputs, output, *, table_roi=None, anchors=()):
    import trimesh
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    data=inputs;out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('输出目录已有内容，请指定新目录')
    out.mkdir(parents=True,exist_ok=True)
    rgb,cloud,valid=data['rgb'],data['cloud'],data['valid'];h,w=valid.shape
    T=data['T_sdk_camera'];inverse=np.linalg.inv(T)
    mesh=trimesh.load(data['mesh_file'],force='mesh',process=False)
    vertices=transform(np.asarray(mesh.vertices),data['T_camera_bowl'])
    faces=np.asarray(mesh.faces)
    concept_path=ROOT/'assets/objects/bowl_concept.json'
    concept=json.loads(concept_path.read_text())['conceptualization'][0]['parameters']
    data['source']['concept_path']=str(concept_path)
    data['source']['concept_sha256']=hashlib.sha256(concept_path.read_bytes()).hexdigest()
    data['warnings'].append('mesh 使用原注册资产；碗沿/开口轴使用报告注明的当前 Concept。若 Concept 修改过，先对照原版本再判断坐标问题。')
    C=np.asarray(concept['pose_matrix'],float);rim_pose=data['T_camera_bowl']@C
    theta=np.linspace(0,2*np.pi,97);radius=np.mean(concept['size'][:2])
    rim=transform(np.column_stack((radius*np.cos(theta),radius*np.sin(theta),np.zeros(len(theta)))),rim_pose)
    opening=-rim_pose[:3,2];opening_line=np.stack((rim_pose[:3,3],rim_pose[:3,3]+opening*.08))
    half=data['table_size']/2
    table_local=np.array([[-half[0],-half[1],data['thickness']/2],[half[0],-half[1],data['thickness']/2],
                          [half[0],half[1],data['thickness']/2],[-half[0],half[1],data['thickness']/2]])
    configured_sdk=transform(table_local,data['table_sdk']);configured=transform(configured_sdk,inverse)
    fig=make_subplots(rows=2,cols=2,specs=[[{'type':'xy','colspan':2},None],[{'type':'scene'},{'type':'scene'}]],
        subplot_titles=('RGB + 原相机 pose / 橙色配置桌面（悬停读取 u,v）','原相机坐标：深度与模型','右臂 SDK：同一数据与桌面配置'),vertical_spacing=.07)
    fig.add_trace(go.Image(z=rgb,hovertemplate='u=%{x} v=%{y}<extra>RGB pixel</extra>'),row=1,col=1)
    def project(points):
        p=points@data['K'].T
        uv=p[:,:2]/np.maximum(p[:,2:3],1e-9)
        uv[points[:,2]<=0]=np.nan
        return uv
    def overlay(points,name,color):
        uv=project(points)
        fig.add_trace(go.Scatter(x=uv[:,0],y=uv[:,1],mode='lines',name=name,line=dict(color=color,width=2)),row=1,col=1)
    def line3(points,name,color,col):
        fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='lines',name=name,line=dict(color=color,width=6)),row=2,col=col)
    def plane(points,name,color,col):
        fig.add_trace(go.Mesh3d(x=points[:,0],y=points[:,1],z=points[:,2],i=[0,0],j=[1,2],k=[2,3],opacity=.45,color=color,name=name,showlegend=True),row=2,col=col)
    overlay(rim,'碗沿（原相机 pose）','cyan');overlay(opening_line,'碗开口方向','gold')
    overlay(np.vstack((configured,configured[:1])),'配置桌面顶面','orange')
    edges=np.asarray(mesh.edges_unique);edges=edges[np.linspace(0,len(edges)-1,min(600,len(edges)),dtype=int)]
    wire=np.full((len(edges)*3,3),np.nan);wire[0::3]=vertices[edges[:,0]];wire[1::3]=vertices[edges[:,1]]
    overlay(wire,'注册 mesh 轮廓抽样','deepskyblue')
    indices=np.flatnonzero(valid);indices=indices[np.linspace(0,len(indices)-1,min(18000,len(indices)),dtype=int)]
    points=cloud.reshape(-1,3)[indices];colors=['rgb(%d,%d,%d)'%tuple(c) for c in rgb.reshape(-1,3)[indices]]
    for col,pose in [(1,np.eye(4)),(2,T)]:
        q=transform(points,pose)
        fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode='markers',marker=dict(size=1,color=colors),name=f'RGB-D {col}'),row=2,col=col)
        line3(transform(rim,pose),'碗沿','cyan',col);line3(transform(opening_line,pose),'开口方向','gold',col)
        plane(transform(configured,pose),'配置桌面','orange',col)
        mesh_points=transform(vertices,pose)
        shown_faces=faces[np.linspace(0,len(faces)-1,min(4000,len(faces)),dtype=int)]
        fig.add_trace(go.Mesh3d(x=mesh_points[:,0],y=mesh_points[:,1],z=mesh_points[:,2],
            i=shown_faces[:,0],j=shown_faces[:,1],k=shown_faces[:,2],color='deepskyblue',opacity=.3,
            name='注册碗 mesh（显示面抽样）',showlegend=True),row=2,col=col)
        # SDK 原点与轴是独立的机器人坐标参照，不是相机位置。
        origin=np.zeros((2,3))
        for axis,color in enumerate(['red','green','blue']):
            origin[:]=0;origin[1,axis]=.12
            axis_points=transform(origin,inverse) if col==1 else origin.copy()
            line3(axis_points,'SDK '+['X','Y','Z'][axis],color,col)
    report=dict(schema_version=1,mode='read_only_frozen_rgbd_alignment',source=data['source'],
        T_sdk_camera=T.tolist(),configured_table_sdk=data['table_sdk'].tolist(),table_size_xy_m=data['table_size'].tolist(),
        table_roi_pixels=table_roi,warnings=list(data['warnings']),camera_plane=None,anchors=[],
        interpretation='无独立 SDK 参考点时，不能仅凭碗/桌面唯一证明外参正确。ROI 拟合不使用外参或 SDK 水平假设。')
    if table_roi:
        x0,y0,x1,y1=map(int,table_roi)
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('ROI 超出图像或顺序错误：u0 v0 u1 v1')
        mask=np.zeros(valid.shape,bool);mask[y0:y1,x0:x1]=True;mask&=valid
        center,normal,quad,quality=fit_camera_plane(cloud[mask])
        overlay(np.vstack((quad,quad[:1])),'ROI 实测平面','lime')
        fig.add_shape(type='rect',x0=x0,x1=x1,y0=y0,y1=y1,line=dict(color='lime',width=3),row=1,col=1)
        plane(quad,'ROI 实测平面','lime',1);plane(transform(quad,T),'ROI 实测平面','lime',2)
        for col,pose in [(1,np.eye(4)),(2,T)]:
            line3(transform(np.stack((center,center+normal*.10)),pose),'实测上表面法向（朝相机）','lime',col)
            configured_normal=data['table_sdk'][:3,2]
            sdk_center0=configured_sdk.mean(0)
            configured_arrow=np.stack((sdk_center0,sdk_center0+configured_normal*.10))
            line3(transform(configured_arrow,inverse) if col==1 else configured_arrow,'配置顶面 +Z 法向','orange',col)
        sdk_center=transform(center[None],T)[0];sdk_normal=T[:3,:3]@normal
        up=data['table_sdk'][:3,2];top=configured_sdk.mean(0)
        report['camera_plane']=dict(**quality,normal_orientation='toward_camera; select visible TOP of table',
            center_camera_m=center.tolist(),normal_camera=normal.tolist(),
            rim_height_above_plane_m=float((rim_pose[:3,3]-center)@normal),
            opening_axis_plane_dot=float(opening@normal),center_sdk_m=sdk_center.tolist(),normal_sdk=sdk_normal.tolist(),
            configured_normal_angle_deg=float(np.degrees(np.arccos(np.clip(abs(sdk_normal@up),0,1)))),
            configured_top_normal_alignment=float(sdk_normal@up),
            configured_oriented_normal_angle_deg=float(np.degrees(np.arccos(np.clip(sdk_normal@up,-1,1)))),
            measured_minus_configured_height_m=float((sdk_center-top)@up))
    for values in anchors:
        u,v=map(int,values[:2]);expected=np.asarray(values[2:],float)
        if not 0<=u<w or not 0<=v<h or not valid[v,u]:raise ValueError('参考点像素超界或没有有效深度')
        measured=transform(cloud[v,u][None],T)[0];residual=measured-expected
        report['anchors'].append(dict(pixel_uv=[u,v],expected_sdk_m=expected.tolist(),measured_sdk_m=measured.tolist(),
            residual_sdk_m=residual.tolist(),error_m=float(np.linalg.norm(residual))))
        line3(np.stack((expected,measured)),'参考点残差（SDK）','magenta',2)
        fig.add_trace(go.Scatter(x=[u],y=[v],mode='markers+text',text=[str(len(report['anchors']))],marker=dict(color='magenta',size=10),name='独立参考点'),row=1,col=1)
    expected=np.asarray([r['expected_sdk_m'] for r in report['anchors']])
    report['noncollinear_reference_coverage']=bool(len(expected)>=3 and np.linalg.matrix_rank(expected-expected.mean(0),tol=1e-6)>=2)
    fig.update_xaxes(range=[0,w],title_text='u (pixel)',row=1,col=1)
    fig.update_yaxes(range=[h,0],title_text='v (pixel)',scaleanchor='x',row=1,col=1)
    axis=dict(xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',aspectmode='data')
    fig.update_layout(height=1100,title='只读真实 RGB-D 检查：蓝/青=碗，黄=开口，橙=配置桌面，绿=ROI 实测平面；不是仿真执行',scene=axis,scene2=axis)
    roi_script="""
const gd=document.getElementById('{plot_id}');
const note=document.createElement('pre');note.style.cssText='position:sticky;bottom:0;background:#fff4cc;padding:12px;z-index:1000';
note.textContent='点击工具栏矩形框选图标，在 RGB 图中框选干净桌面；这里将显示 --table-roi 参数。';document.body.appendChild(note);
gd.on('plotly_relayout',function(e){const shapes=(gd.layout.shapes||[]).filter(s=>s.type==='rect');if(!shapes.length)return;
const s=shapes[shapes.length-1];const uv=[Math.floor(Math.min(s.x0,s.x1)),Math.floor(Math.min(s.y0,s.y1)),Math.ceil(Math.max(s.x0,s.x1)),Math.ceil(Math.max(s.y0,s.y1))];
note.textContent='复制此参数重新生成报告（框内仅保留桌面上表面）： --table-roi '+uv.join(' ');});
"""
    fig.write_html(out/'alignment.html',include_plotlyjs=True,full_html=True,post_script=roi_script,
                   config=dict(modeBarButtonsToAdd=['drawrect','eraseshape']))
    (out/'alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    Image.fromarray(rgb).save(out/'rgb.png')
    notes=['# 标定与桌面可视化检查','打开 alignment.html；RGB 悬停读取像素，3D 可旋转、缩放、点图例隐藏图层。',
        '先不选 ROI 看模型贴合；再选不含碗、柜体、桌沿的干净桌面矩形重跑。绿色仅表示该 ROI 的平面，不自动认定为桌面。',
        '相机图中 mesh 已不贴实物：先查 pose/mesh/深度内参，而非直接改外参。',
        '独立 SDK 参考点残差小而橙/绿不重合：查桌面高度、XY、水平假设。残差大：查外参方向、右/左臂、单位或 RGB-D 配准。',
        '没有独立参考点时无法唯一分离标定与桌面错误；相机平面法向朝向相机，必须选择可见桌面上表面。',
        '具体数值和输入来源见 alignment.json；工具不修改标定/桌面，也不调用 SDK、感知服务或模拟器。',json.dumps(report['camera_plane'],ensure_ascii=False),json.dumps(report['anchors'],ensure_ascii=False)]
    (out/'README.md').write_text('\n\n'.join(notes)+'\n')
    return out/'alignment.html'
