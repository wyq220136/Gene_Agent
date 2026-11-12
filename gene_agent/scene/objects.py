"""采集场景的 MuJoCo 资产；显示、Concept 和接触几何分别保存。"""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def _numbers(values):
    return ' '.join(format(float(x), '.12g') for x in values)


def _pose(matrix):
    """矩阵是 T_parent_body；MuJoCo 使用 wxyz，不重复应用采集pose。"""
    q = Rotation.from_matrix(matrix[:3,:3]).as_quat()
    return {'pos':_numbers(matrix[:3,3]),'quat':_numbers(q[[3,0,1,2]])}


def _wall_parts(cfg):
    """沿物体 +Y 开口；分层切向小盒子覆盖碗壁，保留内部凹腔。"""
    n,layers=cfg['segments'],cfg['layers']
    lo,hi=cfg['bottom_wall_y_m'],cfg['top_y_m']
    dh=(hi-lo)/layers
    for layer in range(layers):
        r=cfg['bottom_radius_m']+(cfg['rim_radius_m']-cfg['bottom_radius_m'])*(layer+.5)/layers
        for i in range(n):
            angle=2*math.pi*i/n
            rot=Rotation.from_euler('y',-angle).as_matrix()
            center=np.array([r*math.cos(angle),lo+(layer+.5)*dh,r*math.sin(angle)])
            half=np.array([cfg['wall_half_thickness_m'],dh/2+.0005,r*math.tan(math.pi/n)+.0005])
            yield layer,i,center,rot,half


def shift_turntable_geoms(root: ET.Element, offset_m: float) -> tuple[str, ...]:
    """同步平移微波炉转盘显示与碰撞几何，保留炉体及操作知识。

    转盘各 geom 的 `pos` 都处在各自父 link 局部坐标系；当前微波炉
    姿态只有绕世界 Z 的旋转，所以局部 Z 偏移仍对应垂直高度。
    调整只用于物理场景摆位，绝不改变门、放碗或取碗的 TCP keypose。
    """
    delta = float(offset_m)
    if not np.isfinite(delta) or abs(delta) > .03:
        raise ValueError("转盘高度偏移必须是 ±30 mm 内的有限值")
    names = []
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if "turntable" not in name:
            continue
        position = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        if position.shape != (3,):
            raise ValueError(f"转盘几何 {name} 的局部位置不是三维")
        position[2] += delta
        geom.set("pos", _numbers(position))
        names.append(name)
    if "microwave_turntable_collision" not in names:
        raise ValueError("场景缺少微波炉转盘碰撞代理")
    return tuple(names)


def shift_microwave_floor_geoms(root: ET.Element, offset_m: float) -> tuple[str, ...]:
    """同步移动炉腔地板显示与碰撞代理，避免调低转盘后产生虚假支撑。

    只移动经过核对的两个几何，不改变炉壁、门或操作目标。转盘是否
    真正支撑 Bowl 仍须在实际 MuJoCo 接触报告中单独确认。
    """
    delta = float(offset_m)
    if not np.isfinite(delta) or abs(delta) > .03:
        raise ValueError("炉腔地板高度偏移必须是 ±30 mm 内的有限值")
    targets = {"microwave_cabinet_cavity_floor_visual",
               "microwave_floor_collision"}
    changed = set()
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name not in targets:
            continue
        position = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        if position.shape != (3,):
            raise ValueError(f"炉腔地板几何 {name} 的局部位置不是三维")
        position[2] += delta
        geom.set("pos", _numbers(position))
        changed.add(name)
    if changed != targets:
        raise ValueError(f"炉腔地板显示/碰撞代理缺失：{sorted(targets - changed)}")
    return tuple(sorted(changed))


def build_objects(output_dir: Path, seed: int = 0, perturb: bool = False,
                  pose_overrides: dict[str, np.ndarray] | None = None,
                  table_size_xy_m: tuple[float, float] | None = None,
                  turntable_height_offset_m: float = 0.,
                  microwave_floor_height_offset_m: float = 0.,
                  world_frame: str | None = None,
                  table_config: dict | None = None) -> dict:
    """生成可独立加载或 include 的 objects.xml，并保存完整采样元数据。

    扰动公式 p'=p+[dx,dy,0], R'=Rz(dyaw)R，yaw 绕各物体原点。
    不做 IK 或拒绝采样；微波炉子旋钮随父 frame 一起移动。
    """
    height_offset = float(turntable_height_offset_m)
    floor_offset = float(microwave_floor_height_offset_m)
    if not np.isfinite(height_offset) or abs(height_offset) > .03:
        raise ValueError("转盘高度偏移必须是 ±30 mm 内的有限值")
    if not np.isfinite(floor_offset) or abs(floor_offset) > .03:
        raise ValueError("炉腔地板高度偏移必须是 ±30 mm 内的有限值")
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/scenes/task_1.json').read_text(encoding='utf-8'))
    source_world_frame=cfg['world_frame']
    if world_frame is not None:
        cfg['world_frame']=world_frame
    nominal={name:np.array(obj['simulation_nominal_pose']) for name,obj in cfg['objects'].items()}
    captured={name:obj['captured_pose'] for name,obj in cfg['objects'].items()}
    sample={name:T.copy() for name,T in nominal.items()}; deltas={}
    rng=np.random.default_rng(seed); ranges=cfg['perturbation']
    for name,T in sample.items():
        dx,dy=rng.uniform(-ranges['xy_half_range_m'],ranges['xy_half_range_m'],2) if perturb else [0.,0.]
        yaw=float(rng.uniform(-ranges['yaw_half_range_rad'],ranges['yaw_half_range_rad'])) if perturb else 0.
        T[:3,3]+=np.array([dx,dy,0]); T[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@T[:3,:3]
        deltas[name]={'dx_m':float(dx),'dy_m':float(dy),'yaw_rad':yaw}
    table=dict(cfg['table'])
    if table_config is not None:
        from gene_agent.scene.table import validate_table
        table=validate_table(table_config,table)
    if table_size_xy_m is not None:
        dimensions = np.asarray(table_size_xy_m, dtype=float)
        if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError('桌面长宽必须是两个正的有限米数值')
        table['size_xy_m'] = dimensions.tolist()
    cfg['table']=table.copy()
    # 桌面位姿以盒体中心为原点；标称配置仍用桌面顶高，方便和旧场景数据对照。
    table_pose=np.eye(4)
    table_pose[:3,3]=[*table['center_xy_m'],table['top_z_m']-table['thickness_m']/2]
    nominal['table']=table_pose.copy()
    sample['table']=table_pose.copy()
    # 编辑器传入的是 world 中完整刚体位姿。此处作为最终采样位姿使用，
    # 微波炉支撑、旋钮与碰撞随后都由同一矩阵生成，避免只移动 visual。
    for name, value in (pose_overrides or {}).items():
        if name not in sample:
            raise ValueError(f'未知物体位姿覆盖: {name}')
        pose=np.asarray(value,dtype=float)
        if (pose.shape!=(4,4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3],[0,0,0,1],atol=1e-9)
                or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-6)):
            raise ValueError(f'{name} 覆盖位姿必须是有限刚体变换')
        sample[name]=pose.copy()
    bc=cfg['bowl_proxy']; parts=list(_wall_parts(bc)); T=nominal['bowl']
    root=ET.Element('mujoco',model='scene_objects'); ET.SubElement(root,'compiler',angle='radian'); asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'mesh',name='bowl_visual_mesh',file=str(ROOT/'assets/objects/bowl_visual.obj'))
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'geom',name='scene_table',type='box',size=_numbers([* (np.array(table['size_xy_m'])/2),table['thickness_m']/2]),**_pose(sample['table']),rgba='.45 .38 .29 1')
    support_cfg=dict(cfg['microwave_support'])
    support_local=np.eye(4)
    support_local[:3,3]=[*support_cfg['center_xy_local_m'],-support_cfg['height_m']/2]
    support_world=sample['microwave']@support_local
    support=ET.SubElement(world,'body',name='microwave_support',**_pose(support_world))
    support_half=[support_cfg['size_xy_m'][0]/2,support_cfg['size_xy_m'][1]/2,support_cfg['height_m']/2]
    ET.SubElement(support,'geom',name='microwave_support_visual',type='box',size=_numbers(support_half),contype='0',conaffinity='0',group='2',density='0',rgba='.48 .29 .14 1')
    ET.SubElement(support,'geom',name='microwave_support_collision',type='box',size=_numbers(support_half),group='3',rgba='.15 .55 .95 0',friction='.8 .01 .001',condim='4')
    bowl=ET.SubElement(world,'body',name='bowl',**_pose(sample['bowl']))
    ET.SubElement(bowl,'freejoint',name='bowl_free')
    ET.SubElement(bowl,'inertial',pos='0 0 0',mass=str(bc['mass_kg']),diaginertia='.0003 .0005 .0003')
    ET.SubElement(bowl,'geom',name='bowl_visual',type='mesh',mesh='bowl_visual_mesh',contype='0',conaffinity='0',group='2',rgba='.82 .86 .9 1')
    contact={'group':'3','rgba':'.5 .7 .9 0','friction':'.8 .01 .001','condim':'4'}
    ET.SubElement(bowl,'geom',name='bowl_bottom',type='cylinder',pos=_numbers([0,bc['bottom_y_m'],0]),quat='.707106781187 .707106781187 0 0',size=_numbers([bc['bottom_radius_m'],bc['bottom_half_thickness_m']]),**contact)
    for layer,i,c,r,h in parts:
        W=np.eye(4); W[:3,:3]=r; W[:3,3]=c
        ET.SubElement(bowl,'geom',name=f'bowl_wall_{layer}_{i}',type='box',size=_numbers(h),**_pose(W),**contact)
    from .microwave import add_microwave
    bodies,mapping=add_microwave(root,asset,world,ROOT/cfg['microwave_asset']['urdf'],sample['microwave'])
    moved_turntable_geoms = shift_turntable_geoms(root, height_offset)
    moved_floor_geoms = shift_microwave_floor_geoms(root, floor_offset)
    concept=json.loads((ROOT/cfg['microwave_asset']['concept']).read_text(encoding='utf-8'))
    knob=next(x['parameters'] for x in concept['conceptualization'] if x['component']=='knob_0')
    local=np.array(knob['pose_matrix'])
    joint_origin=np.array(mapping['power_knob_rotation']['T_parent_link'])
    site_local=np.linalg.inv(joint_origin)@local
    ET.SubElement(bodies['power_knob'],'site',name='knob_0_site',size='.001',rgba='1 .3 .1 0',**_pose(site_local))
    path=out/'objects.xml'; ET.indent(root); ET.ElementTree(root).write(path,encoding='unicode')
    limitations=['仿真摆放由用户授权调整，与captured_pose分开；桌高与16cm微波炉垫材为仿真配置。','碗碰撞、惯量和摩擦为接触代理，允许gravity settling。','微波炉真实标定visual与分墙collision独立，质量/惯量/摩擦仍是源模型先验。','微波炉固定基座、脚位于垫材顶面，未模拟倾覆；旋钮业务演示仍为运动学，无操作可达性声明。']
    if cfg['world_frame']!=source_world_frame:
        limitations.append('task_1物体数值布局暂按新机器人base坐标解释；不是重新估计的单臂实物场景。')
    result={'xml_path':str(path),'bowl_body':'bowl','bowl_joint':'bowl_free','knob_joint':'knob_0_joint','knob_site':'knob_0_site','frame':cfg['frame'],'world_frame':cfg['world_frame'],'seed':seed,'perturb':perturb,'captured_poses':captured,'simulation_nominal_poses':{n:T.tolist() for n,T in nominal.items()},'knob_concept':knob,'microwave_joint_mapping':mapping,'door_joint':'microwave_door_joint','nominal_poses':{n:T.tolist() for n,T in nominal.items()},'sample_poses':{n:T.tolist() for n,T in sample.items()},'pose_overrides':list((pose_overrides or {}).keys()),'perturbations':deltas,'T_microwave_knob_0':local.tolist(),'T_world_knob_0':(sample['microwave']@local).tolist(),'table':table|{'T_world_table':sample['table'].tolist()},'microwave_support':support_cfg|{'T_world_support':support_world.tolist()},'turntable_height_offset_m':height_offset,'moved_turntable_geoms':list(moved_turntable_geoms),'microwave_floor_height_offset_m':floor_offset,'moved_floor_geoms':list(moved_floor_geoms),'limitations':limitations,'resolved_config':cfg,'settled_poses':None}
    result['source_world_frame']=source_world_frame
    (out/'scene_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
