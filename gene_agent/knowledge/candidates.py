"""由 Concept 局部几何与显式 GT pose 生成 TCP 候选，单位米/弧度。"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
SOURCE='AffordToolAct/configs/grasp/grasp_profiles.json'


def _pose(value):
    T=np.asarray(value,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        raise ValueError('pose必须为有限齐次4x4矩阵')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-6):
        raise ValueError('pose旋转必须为SO(3)')
    return T.copy()


def _concept(name,component=None):
    entries=json.loads((ROOT/f'assets/objects/{name}_concept.json').read_text())['conceptualization']
    return next(x['parameters'] for x in entries if component is None or x.get('component')==component)


def _record(identifier,primitive,object_pose,world_pose,parameters,source,raw_primitive):
    return {'candidate_id':identifier,'T_primitive_tcp':primitive.tolist(),'T_primitive_tcp_raw':raw_primitive.tolist(),'T_object_tcp':object_pose.tolist(),'T_world_tcp':world_pose.tolist(),'parameters':parameters,'parameter_source':source,'reachable':None,'collision_checked':False}


def bowl_grasp_candidates(T_world_bowl, *, pose_source='mujoco_gt', parameters=None):
    """保持旧task_1 bowl profile的bottom方向、中心向外倾角顺序和世界偏移。

    X为开口轴，Z为接近轴。Concept本身Z朝物体-Y，因此bottom方向对应从碗上方进入。
    世界偏移在完整变换后应用，再反算T_object_tcp；不能混成局部Z偏移。
    """
    world=_pose(T_world_bowl); concept=_concept('bowl'); C=np.array(concept['pose_matrix'])
    p={'azimuths_rad':np.deg2rad(np.arange(0,360,45)).tolist(),'tip_y_rad':np.deg2rad([0,10,-10,20,-20]).tolist(),'max_abs_tip_y_rad':np.pi/3,'radial_offset_m':0.0,'tilt_pivot_tcp_m':[0,0,0],'depth_m':.015,'clearance_m':.001,'tcp_world_offset_m':[0,0,-.015],'pregrasp_opening_width_m':.041,'max_opening_width_m':.08,'finger_length_m':.06}
    overrides=parameters or {}
    if set(overrides)-set(p):raise ValueError('未知bowl参数')
    p.update(overrides)
    ro,ri,height=concept['size']; depth=float(p['depth_m']);clearance=float(p['clearance_m'])
    if not np.isfinite([depth,clearance,p['finger_length_m'],p['max_opening_width_m'],p['pregrasp_opening_width_m'],p['radial_offset_m']]).all():raise ValueError('参数必须有限')
    tip_angles=np.asarray(p['tip_y_rad'],dtype=float);tip_limit=float(p['max_abs_tip_y_rad'])
    if tip_angles.ndim!=1 or not np.isfinite(tip_angles).all():raise ValueError('tip_y_rad 必须是一维有限角度列表')
    if not np.isfinite(tip_limit) or not 0<tip_limit<np.pi/2:raise ValueError('max_abs_tip_y_rad 必须位于 (0, pi/2)')
    opening=ro-ri+2*clearance
    if clearance<=0 or depth<min(.03,height/2) or depth+clearance>p['finger_length_m']:raise ValueError('非法碗插入深度或clearance')
    if not opening<=p['pregrasp_opening_width_m']<=p['max_opening_width_m']:raise ValueError('夹爪开口不足')
    offset=np.asarray(p['tcp_world_offset_m'],dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('偏移必须是有限三维向量')
    pivot=np.asarray(p['tilt_pivot_tcp_m'],dtype=float)
    if pivot.shape!=(3,) or not np.isfinite(pivot).all():raise ValueError('倾斜支点必须是 TCP frame 中的有限三维向量')
    out=[]
    for i,a in enumerate(p['azimuths_rad']):
        for j,b in enumerate(tip_angles):
            if not np.isfinite([a,b]).all() or abs(b)>tip_limit:raise ValueError('方位或倾角非法')
            x=np.array([np.cos(a),np.sin(a),0.]);z=np.array([0.,0.,1.]);y=np.cross(z,x)
            L=np.eye(4);L[:3,:3]=np.column_stack([x,y,z])@Rotation.from_euler('y',b).as_matrix();L[:3,3]=x*(ro+ri)/2+[0,0,-height/2+depth]
            # 绕实际指垫中心倾斜，避免只旋转 TCP 后将接触区扫入碗壁。
            L[:3,3]+=np.column_stack([x,y,z])@pivot-L[:3,:3]@pivot
            radial=np.array([np.cos(a),np.sin(a),0.])
            L[:3,3]+=radial*float(p['radial_offset_m'])
            W=world@C@L;W[:3,3]+=offset;O=np.linalg.inv(world)@W
            params={**p,'azimuth_rad':float(a),'tip_y_rad_value':float(b),'opening_width_m':float(opening),'direction':'bottom','explicit_overrides':list(overrides)}
            c=_record(f'bowl_ring_{i:02d}_{j:02d}',np.linalg.inv(C)@O,O,W,params,SOURCE+'#task_1_bowl_0',L);c['pose_source']=pose_source;out.append(c)
    return out


def knob_grasp_candidates(T_world_microwave, *, pose_source='mujoco_gt', azimuths_rad=None, tcp_local_offset_m=(0,0,-.015), concept=None):
    """旧默认run_microwave_knob脚本选用task_1_knob_0，而非任务文件fallback的top方向。"""
    world=_pose(T_world_microwave);cp=_concept('microwave','knob_0') if concept is None else concept;C=_pose(cp['pose_matrix'])
    direction=cp.get('grasp_direction','bottom');sign=-1. if direction=='top' else 1.
    if direction not in {'top','bottom'}:raise ValueError('未知grasp方向')
    angles=np.deg2rad([55,36,41,31,46,26]) if azimuths_rad is None else azimuths_rad
    offset=np.asarray(tcp_local_offset_m,dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('偏移必须是有限三维向量')
    out=[]
    for i,a in enumerate(angles):
        if not np.isfinite(a):raise ValueError('方位必须有限')
        L=np.eye(4);L[:3,:3]=Rotation.from_euler('z',a).as_matrix()@np.diag([1,sign,sign]);L[2,3]=sign*(-cp['size'][2]/2+.005)
        shifted=L.copy();shifted[:3,3]+=L[:3,:3]@offset
        params={'azimuth_rad':float(a),'direction':direction,'depth_m':.005,'tcp_local_offset_m':offset.tolist(),'opening_width_m':float(2*np.hypot(cp['size'][0]*np.cos(a),cp['size'][1]*np.sin(a))+.002),'pregrasp_opening_width_m':.08}
        c=_record(f'knob_0_axial_{i:02d}',shifted,C@shifted,world@C@shifted,params,cp.get('source',SOURCE+'#task_1_knob_0'),L);c['pose_source']=pose_source;out.append(c)
    return out


def rotation_about_axis(start_pose, axis_point, axis_direction, rotation_rad, step_rad):
    """绕世界中的指定轴旋转TCP（原点也跟随旋转），包含起点和精确终点。"""
    start=_pose(start_pose);point=np.asarray(axis_point,float);axis=np.asarray(axis_direction,float)
    if point.shape!=(3,) or axis.shape!=(3,) or not np.isfinite([point,axis]).all() or np.linalg.norm(axis)<1e-12:raise ValueError('旋转轴非法')
    if not np.isfinite([rotation_rad,step_rad]).all() or step_rad<=0:raise ValueError('角度有限且步长必须正')
    axis=axis/np.linalg.norm(axis);n=int(np.floor(abs(rotation_rad)/step_rad));sign=1 if rotation_rad>=0 else -1
    angles=[sign*step_rad*i for i in range(n+1)]
    if not np.isclose(angles[-1],rotation_rad):angles.append(float(rotation_rad))
    else:angles[-1]=float(rotation_rad)
    out=[]
    for angle in angles:
        R=Rotation.from_rotvec(axis*angle).as_matrix();W=start.copy();W[:3,:3]=R@start[:3,:3];W[:3,3]=R@(start[:3,3]-point)+point
        out.append({'angle_rad':float(angle),'T_world_tcp':W.tolist()})
    return out


def knob_rotate_keyposes(T_world_microwave, *, rotation_rad=np.pi/4, step_rad=np.pi/18, azimuth_rad=np.deg2rad(55), pose_source='mujoco_gt', concept=None):
    """旋转轴取当前GT父pose乘Concept局部轴；不替代IK或执行成功检查。"""
    world=_pose(T_world_microwave);cp=_concept('microwave','knob_0') if concept is None else concept;C=_pose(cp['pose_matrix']);K=world@C
    grasp=knob_grasp_candidates(world,azimuths_rad=[azimuth_rad],pose_source=pose_source,concept=cp)[0]
    return {'mode':'geometric_kinematic','pose_source':pose_source,'grasp':grasp,'axis_point_world':K[:3,3].tolist(),'axis_direction_world':K[:3,2].tolist(),'rotation_rad':float(rotation_rad),'step_rad':float(step_rad),'waypoints':rotation_about_axis(grasp['T_world_tcp'],K[:3,3],K[:3,2],rotation_rad,step_rad),'source':'AffordToolAct/tamp/skills/rotate_knob.py: generate_pose_rotation_about_axis_waypoints','reachable':None}
