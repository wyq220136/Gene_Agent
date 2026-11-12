"""冻结URDF的窄转换器：保留关节/惯量/视觉与分墙碰撞，不用整机凸包。"""
from pathlib import Path
import xml.etree.ElementTree as ET
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .objects import _numbers, _pose

BODY_NAMES={'cabinet':'microwave','door':'microwave_door','power_knob':'knob_0','timer_knob':'knob_1'}
JOINT_NAMES={'door_hinge':'microwave_door_joint','power_knob_rotation':'knob_0_joint','timer_knob_rotation':'knob_1_joint'}


def urdf_pose(node):
    T=np.eye(4)
    if node is not None:
        T[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T


def add_microwave(root,asset,world,path:Path,T_world_root):
    """读取所有link、origin、材质、碰撞与关节；仅支持本资产用到的box/mesh。"""
    native=ET.parse(path.parent/'source_mjcf.xml').getroot()
    native_geom=native.find('default/geom').attrib
    urdf=ET.parse(path).getroot();links={x.get('name'):x for x in urdf.findall('link')};joints={j.find('child').get('link'):j for j in urdf.findall('joint')}
    bodies={};collision_geoms={};mapping={};mesh_names={}
    for name in ['cabinet','door','power_knob','timer_knob']:
        link=links[name];joint=joints.get(name)
        T=T_world_root if joint is None else urdf_pose(joint.find('origin'))
        parent=world if joint is None else bodies[joint.find('parent').get('link')]
        body=ET.SubElement(parent,'body',name=BODY_NAMES[name],**_pose(T));bodies[name]=body;collision_geoms[name]=[]
        if joint is not None:
            axis=joint.find('axis').get('xyz');lim=joint.find('limit');dyn=joint.find('dynamics')
            ET.SubElement(body,'joint',name=JOINT_NAMES[joint.get('name')],type='hinge',axis=axis,range=f"{lim.get('lower')} {lim.get('upper')}",limited='true',damping=dyn.get('damping'),frictionloss=dyn.get('friction'),armature=native.find(f".//joint[@name='{joint.get('name')}']").get('armature'))
            mapping[joint.get('name')]={'joint':JOINT_NAMES[joint.get('name')],'body':BODY_NAMES[name],'axis_link':np.fromstring(axis,sep=' ').tolist(),'T_parent_link':T.tolist(),'range_rad':[float(lim.get('lower')),float(lim.get('upper'))]}
        inertia=link.find('inertial');iv=inertia.find('inertia');I=np.array([[float(iv.get('ixx')),float(iv.get('ixy')),float(iv.get('ixz'))],[float(iv.get('ixy')),float(iv.get('iyy')),float(iv.get('iyz'))],[float(iv.get('ixz')),float(iv.get('iyz')),float(iv.get('izz'))]])
        Ip=urdf_pose(inertia.find('origin'));I=Ip[:3,:3]@I@Ip[:3,:3].T
        ET.SubElement(body,'inertial',pos=_numbers(Ip[:3,3]),mass=inertia.find('mass').get('value'),fullinertia=_numbers([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        for tag in ['visual','collision']:
            for entry in link.findall(tag):
                geom=entry.find('geometry');attrs={'name':'microwave_'+entry.get('name'),**_pose(urdf_pose(entry.find('origin')))}
                mesh=geom.find('mesh');box=geom.find('box')
                if mesh is not None:
                    filename=mesh.get('filename')
                    if tag=='visual' and entry.get('name') in {'power_knob_rim_visual','timer_knob_rim_visual'}:
                        filename=filename.replace('_visual.obj','_visual_repaired.obj')
                    meshname='microwave_'+Path(filename).stem
                    if meshname not in mesh_names:
                        ET.SubElement(asset,'mesh',name=meshname,file=str((path.parent/filename).resolve()),scale=mesh.get('scale','1 1 1'));mesh_names[meshname]=filename
                    attrs.update(type='mesh',mesh=meshname)
                elif box is not None:attrs.update(type='box',size=_numbers(np.fromstring(box.get('size'),sep=' ')/2))
                else:raise ValueError('未知URDF几何，不静默丢失')
                if tag=='visual':
                    color=entry.find('material/color');attrs.update(contype='0',conaffinity='0',group='2',density='0',rgba=color.get('rgba') if color is not None else '.7 .7 .7 1')
                else:
                    attrs.update(group='3',rgba='.3 .8 .4 0',**native_geom);collision_geoms[name].append(attrs['name'])
                ET.SubElement(body,'geom',**attrs)
    for native_body in native.findall('.//body'):
        for site in native_body.findall('site'):
            copied=deepcopy(site);copied.set('name','microwave_'+site.get('name'));bodies[native_body.get('name')].append(copied)
    # 显式检查门/旋钮与固定机身的父子碰撞，避免MuJoCo默认parent filtering漏检。
    contact=ET.SubElement(root,'contact')
    for name in ['door','power_knob','timer_knob']:
        for cabinet_geom in collision_geoms['cabinet']:
            for child_geom in collision_geoms[name]:ET.SubElement(contact,'pair',geom1=cabinet_geom,geom2=child_geom,solref=native_geom['solref'],solimp=native_geom['solimp'],margin=native_geom['margin'])
    return bodies,mapping
