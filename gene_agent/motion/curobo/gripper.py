"""动态夹爪的冻结几何快照；只派生规划碰撞球，不改原URDF或自碰撞忽略表。"""
from __future__ import annotations
import hashlib
import itertools
import json
import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _stable_numbers(value, digits=12):
    """消除世界 FK 往返产生的 1e-15 级抖动，保留精确可哈希的局部几何。

    规划碰撞容差以毫米计；这里保留到 1e-12 m/rad。规范化发生在快照和
    robot_config 生成之前，因此 snapshot hash 与 solver 实际使用的球仍
    保持严格一致，不在 planner pool 中做模糊等价判断。
    """
    if isinstance(value, float):
        rounded = round(value, digits)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {key: _stable_numbers(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_numbers(item, digits) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_numbers(item, digits) for item in value)
    return value


def validate_gripper_snapshot(snapshot):
    """拒绝内容已改变却沿用旧标识的快照，不把width标量当成完整几何身份。"""
    payload={key:value for key,value in snapshot.items() if key!='snapshot_id'}
    if snapshot.get('snapshot_id') != _digest(payload):
        raise ValueError('Gripper snapshot content hash mismatch')
    return snapshot


def cover_bounds(bounds,max_cell_m=.012):
    """把三维AABB分成等尺寸小格，每格取外接球，完整覆盖格内所有点。

    这是保守解析覆盖：半径为格子半对角线，不能把几个抽样点在球内当成全体覆盖。
    返回球坐标与bounds相同frame；不改变物理接触几何或额外放宽碰撞。
    """
    bounds=np.asarray(bounds,dtype=float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<bounds[0]) or not np.isfinite(max_cell_m) or max_cell_m<=0:
        raise ValueError('Invalid finite AABB or cell size')
    counts=np.maximum(1,np.ceil((bounds[1]-bounds[0])/max_cell_m).astype(int))
    sizes=(bounds[1]-bounds[0])/counts
    radius=float(np.linalg.norm(sizes)/2+1e-9)
    return [{'center':(bounds[0]+(np.array(index)+.5)*sizes).tolist(),'radius':radius}
            for index in itertools.product(*(range(int(n)) for n in counts))]


def _transform(position,rotation):
    matrix=np.eye(4);matrix[:3,:3]=np.asarray(rotation).reshape(3,3);matrix[:3,3]=position
    return matrix


def _source_config(group):
    from pathlib import Path
    import yaml
    path=Path(__file__).resolve().parents[3]/f'assets/robots/realman/config/realman_{group}.yml'
    return yaml.safe_load(path.read_text()),hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_tool_spheres_tcp(group,robot_metadata,spheres):
    """覆盖审计包括原末端所有球，不能只检查某根手指就断言整个旧配置漏检。

    只读冻结URDF的link_7固定后代，变换来源仍为原资产与已有link_from_tcp；
    该链只用于旧球覆盖审计，新增pad球frame来自机器人metadata。
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    source=Path(__file__).resolve().parents[3]/f'assets/robots/realman/urdf/realman_{group}.urdf'
    joints=list(ET.parse(source).getroot().findall('joint'))
    transforms={'link_7':np.linalg.inv(robot_metadata['transforms'][group]['link_from_tcp'])}
    while True:
        ready=[joint for joint in joints if joint.find('parent').get('link') in transforms]
        if not ready:break
        for joint in ready:
            if joint.get('type')!='fixed':raise ValueError('Frozen planning tool unexpectedly contains moving joints')
            origin=joint.find('origin');local=np.eye(4)
            if origin is not None:
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            transforms[joint.find('child').get('link')]=transforms[joint.find('parent').get('link')]@local
            joints.remove(joint)
    return np.array([[*(transform[:3,:3]@ball['center']+transform[:3,3]),ball['radius']]
                     for link,transform in transforms.items() for ball in spheres.get(link,[]) if ball['radius']>0])


def validate_snapshot_gripper_opening(width_m: float, low_m: float, high_m: float) -> None:
    """允许物理执行器在端点处的 0.2 mm 数值弹性，但不截断实际几何。

    此容差只决定能否从当前真实状态生成规划快照；导出的指垫位置仍
    取自 MuJoCo 实际几何。超过容差时报告实际值，便于定位执行器问题。
    """
    if not np.isfinite(width_m) or not low_m - 2e-4 <= width_m <= high_m + 2e-4:
        raise ValueError(
            f"夹爪实际开口 {width_m:.6f} m 超出声明范围 "
            f"[{low_m:.6f}, {high_m:.6f}] m（含 0.0002 m 数值容差）"
        )


def export_gripper_snapshot(model,data,robot_metadata,group):
    """从MuJoCo副本导出当前pad几何，保守球挂回原fixed finger link以保留ignore语义。

    仅当前开口有效；后续手指运动需要重建。实际碰撞geom来自model/data，
    TCP到原规划finger frame来自机器人生成器可追溯metadata，不复制末端经验补偿。
    """
    import mujoco
    from gene_agent.robots.gripper import measure_gripper_width
    gripper=robot_metadata['grippers'][group]
    if gripper['mode']!='parallel_slide_proxy':
        raise ValueError('Dynamic gripper snapshot requires parallel_slide_proxy')
    current=mujoco.MjData(model);mujoco.mj_copyData(current,model,data);mujoco.mj_forward(model,current)
    measured=measure_gripper_width(model,current,robot_metadata,group)
    low,high=gripper['width_range_m']
    validate_snapshot_gripper_opening(measured['width_m'], low, high)
    tcp=current.site(robot_metadata['tcp_sites'][group])
    world_tcp=_transform(tcp.xpos,tcp.xmat)
    config,config_hash=_source_config(group)
    original_spheres=config['kinematics']['collision_spheres']
    legacy_tool=_legacy_tool_spheres_tcp(group,robot_metadata,original_spheres)
    references={finger['source_fixed_link_name']:world_tcp@np.asarray(finger['T_tcp_source_fixed_link'])
                for finger in gripper['fingers'].values()}
    geometries=[];uncovered=0;probe_count=0;worst_clearance=-np.inf
    for geom_name in gripper['collision_geom_names']:
        geom_id=model.geom(geom_name).id
        body_name=model.body(int(model.geom_bodyid[geom_id])).name
        planning_link=gripper['planning_link_by_body'][body_name]
        if model.geom_contype[geom_id]==0 and model.geom_conaffinity[geom_id]==0:
            raise ValueError(f'Gripper collision metadata refers to disabled geom: {geom_name}')
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise NotImplementedError('Gripper snapshot currently supports actual box pads only')
        if model.geom_margin[geom_id] or model.geom_gap[geom_id]:
            raise NotImplementedError('Nonzero pad margin/gap requires explicit sphere expansion')
        world_link=references[planning_link]
        link_geom=np.linalg.inv(world_link)@_transform(current.geom_xpos[geom_id],current.geom_xmat[geom_id])
        size=model.geom_size[geom_id]
        corners=np.array(list(itertools.product(*[[-value,value] for value in size])))
        points=corners@link_geom[:3,:3].T+link_geom[:3,3]
        bounds=np.array([points.min(axis=0),points.max(axis=0)])
        added=cover_bounds(bounds)
        tcp_link=np.linalg.inv(world_tcp)@world_link
        probes_tcp=points@tcp_link[:3,:3].T+tcp_link[:3,3]
        clearance=np.min(np.linalg.norm(probes_tcp[:,None]-legacy_tool[None,:,:3],axis=-1)-legacy_tool[None,:,3],axis=1)
        uncovered+=int(np.count_nonzero(clearance>1e-8));probe_count+=len(points)
        worst_clearance=max(worst_clearance,float(clearance.max()))
        geometries.append({'geom_name':geom_name,'body_name':body_name,'planning_link':planning_link,
                           'geom_type':'box','geom_half_sizes_m':size.tolist(),'T_planning_link_geom':link_geom.tolist(),
                           'bounds_planning_link_m':bounds.tolist(),'added_spheres':added,
                           'coverage':'Every AABB cell is completely contained in its circumscribed sphere',
                           'legacy_uncovered_corner_count':int(np.count_nonzero(clearance>1e-8))})
    geometries=_stable_numbers(geometries)
    additions={}
    for geom in geometries:
        additions.setdefault(geom['planning_link'],[]).extend(geom['added_spheres'])
    import copy
    merged=copy.deepcopy(original_spheres)
    for link,balls in additions.items():merged[link].extend(balls)
    snapshot={'schema_version':1,'group':group,'mode':gripper['mode'],
              'width_m':_stable_numbers(float(measured['width_m'])),
              'joint_positions_m':_stable_numbers(measured['joint_positions_m']),
              'symmetry_error_m':_stable_numbers(float(measured['symmetry_error_m'])),
              'source_hashes':copy.deepcopy(gripper['source_hashes']),'source_robot_config_sha256':config_hash,
              'geometries':geometries,'added_spheres_by_link':additions,'merged_sphere_sha256':_digest(merged),
              'legacy_coverage':{'scope':'pad corners versus all original fixed tool-descendant spheres, expressed in TCP',
                                 'probe_count':probe_count,'uncovered_probe_count':uncovered,
                                 'max_uncovered_distance_m':_stable_numbers(worst_clearance)},
              'limitations':['Only the current opening is synchronized; re-export after finger motion.',
                             'Added balls cover physical box pads; moving visual-only finger meshes and decorative rail are not validated collision solids.',
                             'Original fixed finger spheres are retained conservatively; arm spheres and self-ignore are unchanged.',
                             'This is geometry synchronization, not execution or grasp validation.']}
    snapshot['snapshot_id']=_digest(snapshot)
    return snapshot


def apply_gripper_snapshot(config,snapshot,gripper_metadata,group):
    """只在内存副本追加pad球，校验源版本；原资产文件不写入。"""
    import copy
    validate_gripper_snapshot(snapshot)
    _,source_hash=_source_config(group)
    if snapshot['group']!=group or snapshot['source_hashes']!=gripper_metadata['source_hashes'] or snapshot['source_robot_config_sha256']!=source_hash:
        raise ValueError('Gripper snapshot source or group mismatch')
    derived=copy.deepcopy(config)
    for link,balls in snapshot['added_spheres_by_link'].items():
        derived['kinematics']['collision_spheres'][link].extend(copy.deepcopy(balls))
    if _digest(derived['kinematics']['collision_spheres']) != snapshot['merged_sphere_sha256']:
        raise ValueError('Gripper snapshot source sphere configuration mismatch')
    return derived


def gripper_result_metadata(snapshot):
    if snapshot is None:
        return {'gripper_geometry_synchronized':False,'gripper_snapshot_id':None,
                'gripper_limitations':['Diagnostic arm result only: moving gripper opening has not been synchronized.',
                                       'Do not execute this result as a dynamic-gripper collision-checked trajectory.']}
    validate_gripper_snapshot(snapshot)
    return {'gripper_geometry_synchronized':True,'gripper_snapshot_id':snapshot['snapshot_id'],
            'gripper_width_m':snapshot['width_m'],'gripper_source_hashes':snapshot['source_hashes'],
            'gripper_limitations':snapshot['limitations']}
