"""从冻结URDF派生明确的双指平行滑动代理，保留原安装和末端mesh姿态。"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'assets/robots/realman'
CONFIG=ROOT/'config/robots/gripper_parallel_slide_proxy.json'


def _text(values):return ' '.join(format(float(x),'.17g') for x in values)


def _origin_matrix(element):
    result=np.eye(4)
    if element is not None:
        result[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        result[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return result


def _add_origin(parent, transform):
    return ET.SubElement(parent,'origin',xyz=_text(transform[:3,3]),rpy=_text(Rotation.from_matrix(transform[:3,:3]).as_euler('xyz')))


def generate_gripper_urdf(group: str, link_from_tcp: np.ndarray) -> tuple[Path,dict]:
    """生成独立可编译URDF，不覆盖源URDF或其provenance哈希。

    两个新prismatic都以TCP frame为零点，轴分别+X/-X；q为各侧内面到TCP的距离。
    正q等量，因此URDF mimic multiplier=+1。MuJoCo coupling另由构建器明确添加。
    """
    config=json.loads(CONFIG.read_text(encoding='utf-8'))
    source=ASSETS/'urdf'/f'realman_{group}.urdf'
    tree=ET.parse(source).getroot()
    links={link.get('name'):link for link in tree.findall('link')}
    transforms={'link_7':np.eye(4)}
    pending=list(tree.findall('joint'))
    # 只展开link_7后代，取得旧指端相对于TCP的真实静态变换，不假设掌部rpy精确等于pi。
    while True:
        additions=[]
        for joint in pending:
            parent=joint.find('parent').get('link')
            if parent in transforms:
                transforms[joint.find('child').get('link')]=transforms[parent]@_origin_matrix(joint.find('origin'))
                additions.append(joint)
        if not additions:break
        pending=[j for j in pending if j not in additions]
    prefix='l' if group=='left' else 'r'
    old_fingers={'positive':f'{prefix}_inspire_jaw_link11','negative':f'{prefix}_inspire_jaw_link22'}
    removed={f'{prefix}_inspire_jaw_link{suffix}' for suffix in ['1','2','11','22']}
    for element in list(tree):
        if element.tag=='link' and element.get('name') in removed:tree.remove(element)
        elif element.tag=='joint' and element.find('child').get('link') in removed:tree.remove(element)
    tcp_link=ET.SubElement(tree,'link',name='gripper_tcp_frame')
    tcp_joint=ET.SubElement(tree,'joint',name='gripper_tcp_fixed',type='fixed')
    ET.SubElement(tcp_joint,'parent',link='link_7');ET.SubElement(tcp_joint,'child',link='gripper_tcp_frame');_add_origin(tcp_joint,link_from_tcp)
    descriptors={}
    t=config['pad_thickness_m'];dy=config['pad_y_width_m'];dz=config['pad_z_length_m'];z=config['pad_tip_z_m']-dz/2
    mass=config['finger_mass_kg']
    for label,sign in [('positive',1),('negative',-1)]:
        name=f'gripper_{label}_finger';joint_name=f'gripper_{label}_slide'
        finger=ET.SubElement(tree,'link',name=name)
        pad_transform=np.eye(4);pad_transform[:3,3]=[sign*t/2,0,z]
        inertia_transform=np.eye(4);inertia_transform[:3,3]=config['finger_com_positive_tcp_m'];inertia_transform[0,3]*=sign
        inertia=ET.SubElement(finger,'inertial');_add_origin(inertia,inertia_transform)
        ix,iy,iz=config['finger_inertia_box_size_m']
        ET.SubElement(inertia,'mass',value=str(mass))
        ET.SubElement(inertia,'inertia',ixx=str(mass*(iy*iy+iz*iz)/12),iyy=str(mass*(ix*ix+iz*iz)/12),izz=str(mass*(ix*ix+iy*iy)/12),ixy='0',ixz='0',iyz='0')
        original=old_fingers[label]
        visual=copy.deepcopy(links[original].find('visual'))
        visual_transform=np.linalg.inv(link_from_tcp)@transforms[original]@_origin_matrix(visual.find('origin'))
        visual_transform[0,3]-=sign*config['reference_visual_width_m']/2
        origin=visual.find('origin')
        if origin is not None:visual.remove(origin)
        _add_origin(visual,visual_transform);finger.append(visual)
        # pad是独立接触代理，原末端mesh仅用于显示，避免把mesh凸包误作平整内接触垫。
        collision=ET.SubElement(finger,'collision');_add_origin(collision,pad_transform)
        ET.SubElement(ET.SubElement(collision,'geometry'),'box',size=_text([t,dy,dz]))
        # visual与collision分开：原连杆和指端保持静态装配关系共同平移。
        # 掌部端连接点随开口横移，这是未标定的视觉近似，不代表真实转动连杆。
        old_rod=f"{prefix}_inspire_jaw_link{'1' if label=='positive' else '2'}"
        rod_visual=copy.deepcopy(links[old_rod].find('visual'))
        rod_transform=np.linalg.inv(link_from_tcp)@transforms[old_rod]@_origin_matrix(rod_visual.find('origin'))
        rod_transform[0,3]-=sign*config['reference_visual_width_m']/2
        old_origin=rod_visual.find('origin')
        if old_origin is not None:rod_visual.remove(old_origin)
        _add_origin(rod_visual,rod_transform);finger.append(rod_visual)
        # 接触代理的尺寸、位置和顺序保持不变，不再重复生成可见box。
        for part,spec in config['moving_structure_boxes'].items():
            transform=np.eye(4);transform[:3,3]=spec['positive_center_tcp_m'];transform[0,3]*=sign
            element=ET.SubElement(finger,'collision',name=f'{label}_{part}_collision');_add_origin(element,transform)
            ET.SubElement(ET.SubElement(element,'geometry'),'box',size=_text(spec['size_m']))
        joint=ET.SubElement(tree,'joint',name=joint_name,type='prismatic')
        ET.SubElement(joint,'parent',link='gripper_tcp_frame');ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0');ET.SubElement(joint,'axis',xyz=f'{sign} 0 0')
        ET.SubElement(joint,'limit',lower='0',upper=str(config['width_range_m'][1]/2),effort=str(config['actuator_force_limit_N']),velocity=str(config['urdf_velocity_limit_m_per_s']))
        ET.SubElement(joint,'dynamics',damping=str(config['joint_damping_Ns_per_m']),friction='0')
        if label=='negative':ET.SubElement(joint,'mimic',joint='gripper_positive_slide',multiplier='1',offset='0')
        descriptors[label]={'joint_name':joint_name,'link_name':name,'axis_tcp':[sign,0,0],
                            'T_tcp_finger_closed':np.eye(4).tolist(),'T_finger_pad':pad_transform.tolist(),
                            'T_finger_visual_mesh':visual_transform.tolist(),'source_fixed_link_name':original,
                            'T_tcp_source_fixed_link':(np.linalg.inv(link_from_tcp)@transforms[original]).tolist(),
                            'inner_site_local_m':[0,0,z]}
    target=ASSETS/'urdf/generated'/f'realman_{group}_parallel_slide_proxy.urdf'
    target.parent.mkdir(parents=True,exist_ok=True)
    for mesh in tree.findall('.//mesh'):
        mesh.set('filename',os.path.relpath((source.parent/mesh.get('filename')).resolve(),target.parent))
    extension=ET.SubElement(tree,'mujoco')
    ET.SubElement(extension,'compiler',discardvisual='false',fusestatic='false',strippath='false')
    ET.indent(tree);content=ET.tostring(tree,encoding='unicode')
    # 多个场景可并行构建；用原子replace避免另一构建者读到半份URDF。
    if not target.exists() or target.read_text(encoding='utf-8')!=content:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=target.parent,delete=False) as stream:
            stream.write(content);temporary=Path(stream.name)
        temporary.replace(target)
    return target,{'mode':config['mode'],'config':config,'fingers':descriptors,'generated_urdf_path':str(target),
                   'generated_urdf_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                   'source_urdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest()}


def set_gripper_width(model,data,robot_metadata,group,width_m):
    """只设置两个位置执行器ctrl，不写qpos、不瞬移；米为真实两内面间隙。"""
    if group not in robot_metadata.get('grippers',{}):raise ValueError(f'未知夹爪组: {group}')
    gripper=robot_metadata['grippers'][group]
    if gripper['mode']=='pika_urdf':
        value=float(width_m);low,high=gripper['width_range_m']
        if not np.isfinite(value) or not low<=value<=high:raise ValueError(f'开口必须在[{low},{high}]米且有限')
        displacement=(high-value)/2
        left,right=gripper['actuator_names']
        data.ctrl[model.actuator(left).id]=-displacement
        data.ctrl[model.actuator(right).id]=displacement
        return
    if gripper['mode']!='parallel_slide_proxy':raise ValueError('fixed_urdf没有开合驱动')
    value=float(width_m);low,high=gripper['width_range_m']
    if not np.isfinite(value) or not low<=value<=high:raise ValueError(f'开口必须在[{low},{high}]米且有限')
    actuator_ids=[model.actuator(name).id for name in gripper['actuator_names']]
    data.ctrl[actuator_ids]=value/2


def measure_gripper_width(model,data,robot_metadata,group):
    """用真实内面site投影到当前TCP测量；副本forward保证step后缓存一致。"""
    import mujoco
    if group not in robot_metadata.get('grippers',{}):raise ValueError(f'未知夹爪组: {group}')
    gripper=robot_metadata['grippers'][group]
    if gripper['mode']=='pika_urdf':
        current=mujoco.MjData(model);mujoco.mj_copyData(current,model,data);mujoco.mj_forward(model,current)
        tcp=current.site(robot_metadata['tcp_sites'][group]);rotation=tcp.xmat.reshape(3,3)
        axis=np.asarray(gripper['closing_axis_tcp'],dtype=float)
        positive=current.site(gripper['positive_inner_site_name'])
        negative=current.site(gripper['negative_inner_site_name'])
        positive_tcp=(positive.xpos-tcp.xpos)@rotation
        negative_tcp=(negative.xpos-tcp.xpos)@rotation
        width=float((positive_tcp-negative_tcp)@axis)
        return {'width_m':width,'symmetry_error_m':float(abs((positive_tcp+negative_tcp)@axis)),
                'inner_surface_points_tcp_m':{'positive':positive_tcp.tolist(),'negative':negative_tcp.tolist()},
                'joint_positions_m':{name:float(current.qpos[model.joint(name).qposadr[0]]) for name in gripper['joint_names']},
                'actuator_forces_N':{name:float(current.actuator_force[model.actuator(name).id]) for name in gripper['actuator_names']},
                'tcp_position_world_m':tcp.xpos.tolist()}
    if gripper['mode']!='parallel_slide_proxy':raise ValueError('fixed_urdf没有代理内面site')
    current=mujoco.MjData(model);mujoco.mj_copyData(current,model,data);mujoco.mj_forward(model,current)
    tcp=current.site(robot_metadata['tcp_sites'][group]);rotation=tcp.xmat.reshape(3,3)
    points={label:((current.site(gripper['fingers'][label]['inner_site_name']).xpos-tcp.xpos)@rotation).tolist() for label in ['positive','negative']}
    width=points['positive'][0]-points['negative'][0]
    return {'width_m':float(width),'symmetry_error_m':float(abs(points['positive'][0]+points['negative'][0])),
            'inner_surface_points_tcp_m':points,'joint_positions_m':{name:float(current.qpos[model.joint(name).qposadr[0]]) for name in gripper['joint_names']},
            'actuator_forces_N':{name:float(current.actuator_force[model.actuator(name).id]) for name in gripper['actuator_names']},
            'tcp_position_world_m':tcp.xpos.tolist()}
