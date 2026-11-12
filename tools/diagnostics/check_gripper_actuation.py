"""纯MuJoCo固定手腕台架：开口测量与接触握持，不调用SDK或planner。"""
from pathlib import Path
import os,sys,json,copy,time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def _numbers(x):return ' '.join(format(float(v),'.16g') for v in x)


def _bench(output_dir,block=False):
    import mujoco
    from gene_agent.robots import build_robot
    metadata=build_robot(output_dir/'robot')
    source=ET.parse(metadata['xml_path']).getroot()
    wrist=copy.deepcopy(source.find(".//body[@name='right_link_7']"))
    if wrist is None:raise ValueError('缺少显式right_link_7 frame')
    # 固定腕只移除机械臂最后一轴；指关节保持真实的动态slide和控制器。
    for joint in wrist.findall('joint'):wrist.remove(joint)
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('x',np.pi).as_matrix();T[:3,3]=[0,0,.16]
    W=T@np.linalg.inv(np.array(metadata['transforms']['right']['link_from_tcp']))
    wrist.set('pos',_numbers(W[:3,3]));wrist.set('quat',_numbers(Rotation.from_matrix(W[:3,:3]).as_quat()[[3,0,1,2]]))
    root=ET.Element('mujoco',model='fixed_wrist_gripper_bench');ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep='.001',integrator='implicitfast',gravity='0 0 -9.81')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'headlight',ambient='.45 .45 .45',diffuse='.7 .7 .7',specular='.1 .1 .1');ET.SubElement(visual,'global',offwidth='800',offheight='600')
    root.append(copy.deepcopy(source.find('asset')));world=ET.SubElement(root,'worldbody');world.append(wrist)
    ET.SubElement(world,'light',name='scene_light',pos='.1 -.1 .6',dir='0 0 -1',diffuse='.9 .9 .9')
    ET.SubElement(world,'geom',name='bench_floor',type='plane',pos='0 0 .02',size='.3 .3 .02',rgba='.25 .29 .33 1')
    joint_names={j.get('name') for j in wrist.findall('.//joint')}
    if len(joint_names)!=2:raise ValueError(f'台架要求2个动态指关节，实际{joint_names}')
    actuators=ET.SubElement(root,'actuator')
    for actuator in source.find('actuator'):
        if actuator.get('joint') in joint_names:actuators.append(copy.deepcopy(actuator))
    equality=source.find('equality')
    if equality is not None:
        selected=ET.SubElement(root,'equality')
        for constraint in equality:
            if constraint.get('joint1') in joint_names and constraint.get('joint2') in joint_names:selected.append(copy.deepcopy(constraint))
    # 方块中心位于两pad中段。pad实际尺寸从模型读取，而非再次写一套配置。
    temporary=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));td=mujoco.MjData(temporary);mujoco.mj_forward(temporary,td)
    pad_ids=[i for i in range(temporary.ngeom) if 'pad' in (temporary.geom(i).name or '') and temporary.geom_contype[i]!=0]
    if len(pad_ids)!=2:raise ValueError(f'预期2个pad碰撞体，得到{pad_ids}')
    center=np.mean(td.geom_xpos[pad_ids],axis=0)
    block_center=center.copy();block_center[2]+=0.0005
    if block:
        body=ET.SubElement(world,'body',name='bench_block',pos=_numbers(block_center));ET.SubElement(body,'freejoint',name='bench_block_free')
        ET.SubElement(body,'geom',name='bench_block_geom',type='box',size='.01 .007 .007',mass='.02',rgba='.9 .45 .12 1',friction='.6 .005 .0001')
        support=ET.SubElement(world,'body',name='bench_support',mocap='true',pos=_numbers([0,0,block_center[2]-.0075-.005]))
        ET.SubElement(support,'geom',name='bench_support_geom',type='box',size='.018 .018 .005',rgba='.3 .45 .5 1')
    from gene_agent.visualization import apply_scene_style
    apply_scene_style(root)
    path=output_dir/('holding_bench.xml' if block else 'no_load_bench.xml');ET.indent(root);ET.ElementTree(root).write(path,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    sliders=[model.joint(name).id for name in sorted(joint_names)]
    for joint in sliders:data.qpos[model.jnt_qposadr[joint]]=.04
    for a in range(model.nu):data.ctrl[a]=.04
    mujoco.mj_forward(model,data)
    return model,data,metadata,sliders


def _width(model,data,metadata):
    import mujoco
    site=model.site(metadata['tcp_sites']['right']).id;R=data.site_xmat[site].reshape(3,3);x=R[:,0];origin=data.site_xpos[site]
    pads=[i for i in range(model.ngeom) if 'pad' in (model.geom(i).name or '') and model.geom_contype[i]!=0]
    values=[]
    for i in pads:
        center=float((data.geom_xpos[i]-origin)@x)
        half=float(np.abs(data.geom_xmat[i].reshape(3,3).T@x)@model.geom_size[i])
        values.append((center,half,model.geom(i).name))
    values.sort();left,right=values
    inner_left=left[0]+left[1];inner_right=right[0]-right[1]
    return {'actual_width_m':inner_right-inner_left,'symmetry_error_m':abs((inner_left+inner_right)/2),'tcp_position_m':origin.tolist(),'tcp_rotation':R.tolist(),'pad_geom_names':[v[2] for v in values]}


def _command(model,data,width,metadata):
    # 使用公开接口设置两指ctrl；几何验收另从pad box内表面独立计算，避免自证。
    from gene_agent.robots.gripper import set_gripper_width
    set_gripper_width(model,data,metadata,'right',width)


def _forces(model,data):
    import mujoco
    result=[]
    for i,contact in enumerate(data.contact):
        names=[model.geom(contact.geom1).name,model.geom(contact.geom2).name]
        if 'bench_block_geom' in names:
            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
            result.append({'geoms':names,'normal_force_n':float(wrench[0]),'force_norm_n':float(np.linalg.norm(wrench[:3]))})
    return result


def run(output_dir,render=True):
    """验收阈值预先固定；失败如实写报告，不改变场景/控制参数来凑结果。"""
    if render:os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    from PIL import Image,ImageDraw
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    frames=[];camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.19];camera.distance=.32;camera.azimuth=105;camera.elevation=-12
    model,data,meta,sliders=_bench(out,False);renderer=mujoco.Renderer(model,height=600,width=800) if render else None
    reference=_width(model,data,meta);targets=[];max_drift=0.;max_rotation=0.
    def capture(renderer,model,data,label):
        if renderer is None:return None
        option=mujoco.MjvOption();option.sitegroup[:]=0
        renderer.update_scene(data,camera=camera,scene_option=option);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle([0,0,800,38],fill='black');draw.text((12,12),label,fill='white');return frame
    try:
        for width in [0.,.007,.041,.08]:
            _command(model,data,width,meta)
            for i in range(800):
                mujoco.mj_step(model,data)
                measurement=_width(model,data,meta)
                max_drift=max(max_drift,float(np.linalg.norm(np.array(measurement['tcp_position_m'])-reference['tcp_position_m'])))
                max_rotation=max(max_rotation,float(np.max(np.abs(np.array(measurement['tcp_rotation'])-reference['tcp_rotation']))))
                if render and i%40==0:frames.append(capture(renderer,model,data,f'PARALLEL SLIDE PROXY | target {width*1000:.0f} mm | actual {measurement["actual_width_m"]*1000:.2f} mm'))
            mujoco.mj_forward(model,data);measurement=_width(model,data,meta)
            from gene_agent.robots.gripper import measure_gripper_width
            api_measurement=measure_gripper_width(model,data,meta,'right')
            measurement['api_width_m']=api_measurement['width_m'];measurement['api_geometry_error_m']=abs(api_measurement['width_m']-measurement['actual_width_m'])
            measurement.update(target_width_m=width,width_error_m=abs(measurement['actual_width_m']-width));targets.append(measurement)
            if render and width in {0.,.08}:frames[-1].save(out/('closed.png' if width==0 else 'open.png'))
    finally:
        if renderer:renderer.close()
    no_load={'targets':targets,'max_tcp_position_drift_m':max_drift,'max_tcp_rotation_element_drift':max_rotation,'pass':all(x['width_error_m']<=.0005 and x['symmetry_error_m']<=.00025 and x['api_geometry_error_m']<1e-9 for x in targets) and max_drift<1e-10 and max_rotation<1e-10}
    model,data,meta,sliders=_bench(out,True);renderer=mujoco.Renderer(model,height=600,width=800) if render else None
    block=model.body('bench_block').id;support=model.body('bench_support').mocapid[0];events=[];peak=0.;peak_total=0.;both=False;force_finite=True
    try:
        for _ in range(500):mujoco.mj_step(model,data)
        initial=data.xpos[block].copy()
        for i in range(800):
            # 显式0.8秒开口斜坡，单指参考速度.05m/s，避免ctrl阶跃撞击方块。
            _command(model,data,.08*(1-(i+1)/800),meta)
            mujoco.mj_step(model,data)
            closing_contacts=_forces(model,data)
            peak_total=max(peak_total,sum(c['force_norm_n'] for c in closing_contacts if any('pad' in (n or '') for n in c['geoms'])))
            for c in closing_contacts:
                if any('pad' in (n or '') for n in c['geoms']):
                    peak=max(peak,c['force_norm_n']);force_finite&=bool(np.isfinite(c['force_norm_n']))
            if render and i%40==0:frames.append(capture(renderer,model,data,'CONTACT GRASP | free block on support | no weld'))
        height_at_release=float(data.xpos[block,2]);start=float(data.time)
        data.mocap_pos[support,2]-=.08
        min_height=height_at_release
        for i in range(2500):
            mujoco.mj_step(model,data);min_height=min(min_height,float(data.xpos[block,2]));contacts=_forces(model,data)
            pad_contacts=[c for c in contacts if any('pad' in (n or '') for n in c['geoms'])]
            peak_total=max(peak_total,sum(c['force_norm_n'] for c in pad_contacts))
            pads={n for c in pad_contacts for n in c['geoms'] if 'pad' in (n or '')}
            both|=len(pads)==2
            for c in pad_contacts:peak=max(peak,c['force_norm_n']);force_finite&=bool(np.isfinite(c['force_norm_n']))
            if i%25==0:events.append({'time_s':float(data.time-start),'block_position_m':data.xpos[block].tolist(),'width_m':_width(model,data,meta)['actual_width_m'],'pad_contact_forces':pad_contacts,'actuator_force_n':data.actuator_force.tolist()})
            if render and i%50==0:frames.append(capture(renderer,model,data,f'CONTACT HOLD | support withdrawn | t={data.time-start:.2f}s | no object pose writes'))
        final=data.xpos[block].copy();retention=float(data.time-start);drop=height_at_release-min_height
        final_contacts=_forces(model,data)
        final_pad_names={n for c in final_contacts for n in c['geoms'] if 'pad' in (n or '')}
        unexpected_support=[c for c in final_contacts if not any('pad' in (n or '') for n in c['geoms'])]
        holding={'closing_command':{'type':'linear_width_ramp','from_width_m':.08,'to_width_m':0.,'duration_s':.8,'per_finger_reference_speed_m_per_s':.05},'initial_block_position_m':initial.tolist(),'height_at_support_removal_m':height_at_release,'final_block_position_m':final.tolist(),'maximum_drop_m':drop,'retention_duration_s':retention,'bilateral_contacts':both and len(final_pad_names)==2,'final_contacts':final_contacts,'unexpected_support_contacts':unexpected_support,'peak_contact_force_n':peak,'peak_contact_force_semantics':'maximum individual MuJoCo contact-point force norm','peak_total_pad_contact_force_norm_sum_n':peak_total,'finite_contact_forces':force_finite,'actual_width_m':_width(model,data,meta)['actual_width_m'],'object_pose_assignments_after_reset':0,'object_constraints':[],'trace':events,'pass':bool(drop<.004 and both and len(final_pad_names)==2 and not unexpected_support and force_finite and 0<peak<100 and retention>=2.)}
        if render:frames[-1].save(out/'holding.png')
        # 松爪负例：同一个自由方块必须在重力下掉落，排除隐藏绑定或遗漏约束。
        release_height=float(data.xpos[block,2]);_command(model,data,.08,meta)
        for i in range(750):
            mujoco.mj_step(model,data)
            if render and i%40==0:frames.append(capture(renderer,model,data,'NEGATIVE CONTROL | gripper released | block must fall'))
        release_drop=release_height-float(data.xpos[block,2])
        release={'drop_m':release_drop,'duration_s':.75,'final_position_m':data.xpos[block].tolist(),'pass':bool(release_drop>.02)}
        if render:frames[-1].save(out/'released.png')
    finally:
        if renderer:renderer.close()
    report={'mode':'parallel_slide_proxy_bench','gripper_metadata':meta['grippers']['right'],'thresholds':{'no_load_width_error_m':.0005,'symmetry_error_m':.00025,'retention_drop_m':.004,'min_retention_s':2.,'max_contact_force_n':100.},'no_load':no_load,'holding':holding,'release_negative_control':release,'pass':bool(no_load['pass'] and holding['pass'] and release['pass']),'limitations':['Fixed wrist, small known block; no arm motion or bowl grasp claim.','Slide motion, pad dimensions, mass, actuator gains and friction are uncalibrated proxy assumptions.','Only support mocap was withdrawn; free block not teleported or bound.']}
    (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if render and frames:frames[0].save(out/'actuation.gif',save_all=True,append_images=frames[1:],duration=50,loop=0)
    return report


def viewer(output_dir):
    import mujoco
    import mujoco.viewer
    model,data,meta,_=_bench(Path(output_dir),False)
    from gene_agent.visualization.viewer import open_viewer
    with open_viewer(model,data,camera_preset=None) as view:
        view.cam.lookat[:]=[0,0,.19];view.cam.distance=.32;view.cam.azimuth=105;view.cam.elevation=-12
        view.opt.sitegroup[:]=0
        start=time.monotonic()
        while view.is_running():
            tick=time.monotonic();width=.04*(1+np.sin((tick-start)*1.2));_command(model,data,width,meta)
            for _ in range(10):mujoco.mj_step(model,data)
            view.sync();time.sleep(max(0,.01-(time.monotonic()-tick)))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('runs/gripper-actuation'));parser.add_argument('--viewer',action='store_true');parser.add_argument('--no-render',action='store_true');args=parser.parse_args()
    if args.viewer:viewer(args.output_dir)
    else:
        result=run(args.output_dir,render=not args.no_render);print(json.dumps({'pass':result['pass'],'no_load':result['no_load']['pass'],'holding':result['holding']['pass'],'output_dir':str(args.output_dir)}));raise SystemExit(0 if result['pass'] else 1)
