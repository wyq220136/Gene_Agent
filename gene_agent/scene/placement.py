"""可复现的场景摆位规则。坐标为 T_world_object，米；随机角度配置使用度。

只检查声明的初态几何约束，不筛选 IK 或轨迹，也不把有效初态当作任务成功。
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose


def validate_rules(rules,names):
    if not isinstance(rules,dict) or rules.get('schema_version')!=1:
        raise ValueError('摆位规则需要 schema_version=1')
    objects=rules.get('objects',{})
    if set(objects)!=set(names):raise ValueError('摆位规则物体集合与场景不一致')
    for name,row in objects.items():
        if row.get('mode') not in {'fixed','random'}:raise ValueError('物体模式应为 fixed/random')
        frame=row.get('frame','world')
        if frame not in {'world',*names} or frame==name:raise ValueError('未知或自引用的摆位参考系')
        extent=np.asarray(row.get('translation_half_range_m',[0,0,0]),dtype=float)
        yaw=float(row.get('yaw_half_range_deg',0))
        if (extent.shape!=(3,) or not np.isfinite(extent).all() or np.any(extent<0)
            or np.any(extent>1) or not np.isfinite(yaw) or not 0<=yaw<=180):
            raise ValueError('随机范围需为有限非负数，平移 ≤1 m，yaw ≤180°')
        seen={name};parent=frame
        while parent!='world':
            if parent in seen:raise ValueError('摆位参考系不能循环依赖')
            seen.add(parent)
            if parent not in objects:raise ValueError('未知摆位参考系')
            parent=objects[parent].get('frame','world')
    attempts=rules.get('max_attempts',100)
    if isinstance(attempts,bool) or not isinstance(attempts,int) or not 1<=attempts<=500:
        raise ValueError('初态采样次数范围为 1–500')
    constraints=rules.get('constraints',{})
    if 'bounds_xy_m' in constraints:
        b=np.asarray(constraints['bounds_xy_m'],dtype=float)
        if b.shape!=(4,) or not np.isfinite(b).all() or b[0]>=b[1] or b[2]>=b[3]:
            raise ValueError('桌面边界应为 xmin,xmax,ymin,ymax')
    clearance=float(constraints.get('min_clearance_m',0))
    if not np.isfinite(clearance) or clearance<0 or clearance>.5:raise ValueError('间距范围无效')
    for pair in constraints.get('min_distances',[]):
        if len(pair.get('objects',[]))!=2 or not set(pair['objects'])<=set(names):
            raise ValueError('距离约束物体无效')
        distance=float(pair.get('distance_m',0))
        if not np.isfinite(distance) or distance<0:raise ValueError('距离约束无效')
    for region in constraints.get('excluded_bounds_xy_m',[]):
        if region.get('object') not in names:
            raise ValueError('排除区域物体无效')
        bounds=np.asarray(region.get('bounds_xy_m'),dtype=float)
        if (bounds.shape!=(4,) or not np.isfinite(bounds).all()
                or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]):
            raise ValueError('排除区域应为 xmin,xmax,ymin,ymax')
    return deepcopy(rules)


def _invalid(poses,rules,footprints):
    cfg=rules.get('constraints',{});reasons=[];boxes={}
    for name,pose in poses.items():
        x,y=pose[:2,3]
        half=np.zeros(2)
        if name in footprints:
            dimensions=np.asarray(footprints[name],dtype=float)
            half=np.abs(pose[:2,:len(dimensions)])@(dimensions/2)
        box=np.array([x-half[0],x+half[0],y-half[1],y+half[1]])
        boxes[name]=box
        if 'bounds_xy_m' in cfg:
            b=cfg['bounds_xy_m']
            if box[0]<b[0] or box[1]>b[1] or box[2]<b[2] or box[3]>b[3]:
                reasons.append({'reason':'table_boundary','object':name})
    if footprints and 'min_clearance_m' in cfg:
        names=list(poses)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if a not in footprints or b not in footprints:continue
                one,two=boxes[a],boxes[b]
                gap=np.array([max(0,one[0]-two[1],two[0]-one[1]),
                              max(0,one[2]-two[3],two[2]-one[3])])
                if np.linalg.norm(gap)<cfg['min_clearance_m']:
                    reasons.append({'reason':'object_clearance','objects':[a,b]})
    for row in cfg.get('min_distances',[]):
        a,b=row['objects'];distance=float(np.linalg.norm(poses[a][:2,3]-poses[b][:2,3]))
        if distance<row['distance_m']:
            reasons.append({'reason':'min_distance','objects':[a,b],'actual_m':distance})
    for row in cfg.get('excluded_bounds_xy_m',[]):
        name=row['object'];excluded=np.asarray(row['bounds_xy_m'],dtype=float)
        box=boxes[name]
        overlap=(box[1]>excluded[0] and box[0]<excluded[1]
                 and box[3]>excluded[2] and box[2]<excluded[3])
        if overlap:
            reasons.append({'reason':'excluded_region','object':name,
                            'region':row.get('label'),'bounds_xy_m':excluded.tolist()})
    return reasons


def sample_placement(nominal,rules,*,seed,randomize,footprints=None):
    nominal={name:validate_pose(value,field_name=name).copy() for name,value in nominal.items()}
    rules=validate_rules(rules,tuple(nominal))
    rng=np.random.default_rng(int(seed));rejected=[];sampled={};deltas={}
    for attempt in range(rules.get('max_attempts',100) if randomize else 1):
        sampled={};deltas={}
        def resolve(name):
            if name in sampled:return sampled[name]
            row=rules['objects'][name];frame=row.get('frame','world')
            parent=np.eye(4) if frame=='world' else resolve(frame)
            base=nominal[name] if frame=='world' else np.linalg.inv(nominal[frame])@nominal[name]
            if randomize and row['mode']=='random':
                delta=rng.uniform(-np.asarray(row.get('translation_half_range_m',[0,0,0])),
                                  np.asarray(row.get('translation_half_range_m',[0,0,0])))
                yaw=float(rng.uniform(-row.get('yaw_half_range_deg',0),row.get('yaw_half_range_deg',0)))
            else:delta=np.zeros(3);yaw=0.
            value=base.copy()
            value[:3,3]+=delta
            value[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@value[:3,:3]
            sampled[name]=parent@value if randomize else nominal[name].copy()
            deltas[name]={'translation_m':delta.tolist(),'yaw_deg':yaw,'frame':frame}
            return sampled[name]
        for name in nominal:resolve(name)
        reasons=_invalid(sampled,rules,footprints or {})
        if not reasons:break
        rejected.append({'attempt':attempt+1,'reasons':reasons,
                         'poses_world':{n:p.tolist() for n,p in sampled.items()}})
    return {'schema_version':1,'mode':'random' if randomize else 'fixed','seed':int(seed),
            'sampler_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rules':rules,'rules_sha256':hashlib.sha256(json.dumps(rules,sort_keys=True).encode()).hexdigest(),
            'distribution':'independent uniform proposals with declared geometric rejection','units':'m; yaw ranges deg; T_world_object',
            'nominal_poses_world':{n:p.tolist() for n,p in nominal.items()},
            'sampled_poses_world':{n:p.tolist() for n,p in sampled.items()},
            'deltas':deltas,'valid':not reasons,'invalid_reasons':reasons,
            'rejected_attempts':rejected,'ik_filter':False}
