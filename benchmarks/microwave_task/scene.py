"""微波炉任务的初始摆位解析；预览与正式 Task 共用。"""
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.scene.placement import sample_placement
from gene_agent.simulation import build_scene


def environment_settings(settings):
    """从兼容的旧复合输入投影题目条件，抓取/倾倒等解法参数不进入场景。"""
    keys = ('schema_version', 'seed', 'microwave_pose_world', 'bowl_pose_world',
            'turntable_height_offset_m', 'microwave_floor_height_offset_m',
            'simulation_attachment_pairs', 'placement_mode', 'placement_rules',
            'placement_record', 'door_arm_kp_scale', 'extract_arm_kp_scale', 'table')
    return {key: deepcopy(settings[key]) for key in keys if key in settings}


def resolve_settings(settings,*,seed=None,randomize=None):
    result=deepcopy(settings)
    if result.get('schema_version')!=1:raise ValueError('微波炉任务配置版本不支持')
    seed=result.get('seed',123) if seed is None else seed
    if isinstance(seed,bool) or int(seed)!=seed or not 0<=int(seed)<2**31:
        raise ValueError('seed 应为 0–2147483647 的整数')
    result['seed']=int(seed)
    if randomize is not None:result['placement_mode']='random' if randomize else 'fixed'
    mode=result.get('placement_mode','fixed')
    if mode not in {'fixed','random'}:raise ValueError('摆位模式应为 fixed/random')
    if 'placement_rules' in result:
        nominal={n:np.asarray(result[n+'_pose_world']) for n in ('microwave','bowl')}
        cfg=json.loads((Path(__file__).resolve().parents[2]/'config/scenes/task_1.json').read_text())
        proxy=cfg['bowl_proxy'];radius=proxy['rim_radius_m']
        footprints={'microwave':[*cfg['microwave_support']['size_xy_m'],cfg['microwave_support']['height_m']],
                    'bowl':[radius*2,proxy['top_y_m']-proxy['bottom_y_m'],radius*2]}
        record=sample_placement(nominal,result['placement_rules'],seed=int(seed),
                                randomize=mode=='random',footprints=footprints)
        result['placement_record']=record
        for name,pose in record['sampled_poses_world'].items():result[name+'_pose_world']=pose
    elif mode=='random':raise ValueError('随机测试需要先保存物体摆位规则')
    return result


def inspect_initial(model,data):
    """只检查初态及门运动包络的几何，不将此结果冒称 IK 可达。"""
    from gene_agent.scene.rigid_geometry import collision_geom_bounds,collision_geoms
    reasons=[]
    table_id=model.geom('scene_table').id
    table=collision_geom_bounds(model,data,table_id,0)
    top=table[1,2]
    for name in ('bowl','microwave_support'):
        boxes=[collision_geom_bounds(model,data,g,0) for g in collision_geoms(model,name)]
        if not boxes:
            reasons.append({'reason':'missing_support_geometry','object':name});continue
        lower=np.min([b[0] for b in boxes],axis=0);upper=np.max([b[1] for b in boxes],axis=0)
        if lower[2]<top-.003:
            reasons.append({'reason':'below_or_penetrating_table','object':name,'bottom_z_m':float(lower[2]),'table_top_z_m':float(top)})
        if lower[2]>top+.03:
            reasons.append({'reason':'unsupported_above_table','object':name,'gap_m':float(lower[2]-top)})
        if np.any(lower[:2]<table[0,:2]-.003) or np.any(upper[:2]>table[1,:2]+.003):
            reasons.append({'reason':'outside_table_support','object':name})
    for i in range(data.ncon):
        c=data.contact[i]
        if c.dist>=-.003:continue
        names=[model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
        def category(n):
            if n=='bowl':return 'bowl'
            if n.startswith(('microwave','knob')):return 'microwave'
            return 'environment'
        groups=[category(n) for n in names]
        if groups[0]!=groups[1] and any(g!='environment' for g in groups):
            reasons.append({'reason':'initial_penetration','bodies':names,'depth_m':float(-c.dist)})
    probe=mujoco.MjData(model);probe.qpos[:]=data.qpos
    joint=model.joint('microwave_door_joint')
    address=int(joint.qposadr[0])
    upper=float(model.jnt_range[joint.id,1])
    for angle in np.linspace(0,min(upper,np.pi/2),7):
        probe.qpos[address]=angle
        mujoco.mj_forward(model,probe)
        for i in range(probe.ncon):
            c=probe.contact[i]
            if c.dist>=-.002:continue
            names={model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)}
            if names=={'bowl','microwave_door'}:
                reasons.append({'reason':'door_sweep_hits_bowl','door_angle_rad':float(angle),
                                'depth_m':float(-c.dist)})
                break
    return {'valid':not reasons,'invalid_reasons':reasons,
            'scope':'table support/footprint, object-environment penetration and sampled door sweep; no IK/path check'}


def build_initial_scene(settings,output):
    poses={n:np.asarray(settings[n+'_pose_world'],dtype=float) for n in ('microwave','bowl')}
    model,data,metadata=build_scene(Path(output),seed=int(settings['seed']),pose_overrides=poses,
        turntable_height_offset_m=float(settings['turntable_height_offset_m']),
        microwave_floor_height_offset_m=float(settings['microwave_floor_height_offset_m']),
        simulation_attachment_pairs=dict(settings['simulation_attachment_pairs']),table_config=settings.get('table'))
    record=deepcopy(settings.get('placement_record',{'mode':'fixed','seed':settings['seed'],
        'nominal_poses_world':{n:p.tolist() for n,p in poses.items()},
        'sampled_poses_world':{n:p.tolist() for n,p in poses.items()},'valid':True,'ik_filter':False}))
    checks=inspect_initial(model,data)
    record['initial_check']=checks
    record['valid']=record['valid'] and checks['valid']
    record['initial_poses_world']={}
    for name in poses:
        body=data.body(name);p=np.eye(4);p[:3,:3]=body.xmat.reshape(3,3);p[:3,3]=body.xpos
        record['initial_poses_world'][name]=p.tolist()
    record['settled_poses_world']=None
    record['settling_note']='与完整任务相同的构建初态；后续物理状态由正式 Runtime 记录，预览不提前推进机器人'
    Path(output).mkdir(parents=True,exist_ok=True)
    (Path(output)/'placement.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return model,data,metadata,record

