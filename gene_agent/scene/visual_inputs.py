"""纯视觉输入：冻结资产、URDF FK 与 robot_only 规划配置；不加载物理引擎。

只适配已存在的清扫场景与 task_1 微波炉。所有 pose 为 T_world_object，
距离为米，关节为弧度；预测开合/夹持不能充当接触或任务成功证据。
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.curobo.gripper import (
    _digest, _source_config, _stable_numbers, cover_bounds,
    validate_snapshot_gripper_opening,
)

ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, Path): return str(item)
        raise TypeError(type(item).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, default=convert, ensure_ascii=False,
                                    indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _rigid(value):
    """安装文件有限精度旋转按现有 MuJoCo 模型的四元数转换投影一次。"""
    result = np.asarray(value, float).copy()
    result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
    return result


def _pose_list(transform):
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return [*transform[:3, 3].tolist(), *q.tolist()]


def _mesh_path(urdf_path, filename):
    path = Path(filename)
    if not path.is_absolute(): path = Path(urdf_path).parent/path
    return path.resolve()


class VisualRobot:
    def __init__(self, metadata, group, urdf_path, start):
        self.metadata = metadata
        self.group = group
        self.urdf_path = Path(urdf_path).resolve()
        self.joint_names = tuple(metadata['joint_groups'][group])
        self.start_positions_rad = np.asarray(start, float)
        transforms = metadata['transforms'][group]
        self.world_from_base = _rigid(transforms['world_from_urdf_base'])
        self.link_from_tcp = np.asarray(transforms['link_from_tcp'], float)
        self.urdf = URDF.load(str(self.urdf_path), load_meshes=True,
                              load_collision_meshes=False, build_scene_graph=True)
        self.urdf_joint_names = tuple(f'joint_{i}' for i in range(1, 8))
        self.home_width_m = float(metadata['grippers'][group]['home_width_m'])
        self.joint_configuration = {}
        self.set_state(self.start_positions_rad, self.home_width_m)

    def gripper_configuration(self, width):
        gripper = self.metadata['grippers'][self.group]
        width = float(width)
        lo, hi = gripper['width_range_m']
        validate_snapshot_gripper_opening(width, lo, hi)
        if gripper['mode'] == 'pika_urdf':
            return dict(zip(gripper['joint_names'], [-(hi-width)/2, (hi-width)/2]))
        return {'gripper_positive_slide': width/2, 'gripper_negative_slide': width/2}

    def set_state(self, q, width=None):
        q = np.asarray(q, float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError('机器人关节必须为七个有限弧度值')
        cfg = dict(zip(self.urdf_joint_names, q.tolist()))
        cfg.update(self.gripper_configuration(self.home_width_m if width is None else width))
        self.urdf.update_cfg(cfg)
        self.joint_configuration = cfg
        return cfg

    def fk(self, q):
        """返回 world TCP，不改变显示用机器人状态。"""
        saved = dict(self.joint_configuration)
        self.set_state(q)
        result = self.world_from_base @ self.urdf.get_transform('link_7') @ self.link_from_tcp
        self.urdf.update_cfg(saved)
        self.joint_configuration = saved
        return result

    def gripper_snapshot(self, width):
        descriptor = self.metadata['grippers'][self.group]
        joint_cfg = self.gripper_configuration(width)
        if descriptor['mode'] == 'pika_urdf':
            from gene_agent.motion.curobo.profiles.pika import _pika_config, SOURCE_CONFIG
            config = _pika_config(self.metadata)
            result = dict(schema_version=1, group=self.group, mode='pika_urdf',
                width_m=float(width), joint_positions_m=joint_cfg,
                source_hashes=copy.deepcopy(self.metadata['source_hashes']),
                source_robot_config_sha256=hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest(),
                merged_sphere_sha256=_digest(config['kinematics']['collision_spheres']),
                limitations=['Pika 指关节按预测开口冻结；采用现有保守碰撞球，未验证物理接触。'])
        else:
            # 直接读取与仿真相同的派生 URDF box，转换回原固定 finger link。
            # 复用原球并追加全部 pad/shank/carriage 的保守球，ignore 表不改。
            config, source_hash = _source_config(self.group)
            merged = copy.deepcopy(config['kinematics']['collision_spheres'])
            saved = dict(self.joint_configuration)
            self.set_state(np.zeros(7), width)
            tcp = self.urdf.get_transform('link_7') @ self.link_from_tcp
            additions, geometries = {}, []
            for label, finger in descriptor['fingers'].items():
                planning_link = finger['source_fixed_link_name']
                reference = tcp @ np.asarray(finger['T_tcp_source_fixed_link'])
                link = self.urdf.link_map[finger['link_name']]
                for index, collision in enumerate(link.collisions):
                    box = collision.geometry.box
                    if box is None: raise ValueError('当前标准夹爪只支持已冻结 box 代理')
                    local = np.linalg.inv(reference) @ self.urdf.get_transform(link.name) @ collision.origin
                    half = np.asarray(box.size)/2
                    points = np.array(list(itertools.product(*[[-v, v] for v in half])))
                    points = points @ local[:3, :3].T + local[:3, 3]
                    bounds = np.array([points.min(0), points.max(0)])
                    balls = cover_bounds(bounds)
                    additions.setdefault(planning_link, []).extend(balls)
                    geometries.append(dict(geom_name=f'{self.group}_{label}_{index}',
                        planning_link=planning_link, bounds_planning_link_m=bounds.tolist(),
                        geom_half_sizes_m=half.tolist(), T_planning_link_geom=local.tolist(),
                        added_spheres=balls, coverage='URDF collision box conservative AABB cells'))
            self.urdf.update_cfg(saved)
            self.joint_configuration = saved
            additions = _stable_numbers(additions)
            for name, balls in additions.items(): merged[name].extend(balls)
            result = dict(schema_version=1, group=self.group, mode='parallel_slide_proxy',
                width_m=float(width), joint_positions_m={f'{self.group}_{n}':v for n,v in joint_cfg.items()},
                symmetry_error_m=0., source_hashes=descriptor['source_hashes'],
                source_robot_config_sha256=source_hash, geometries=_stable_numbers(geometries),
                added_spheres_by_link=additions, merged_sphere_sha256=_digest(merged),
                limitations=['按预测开口同步派生 URDF 全部 box 代理；不表示实际夹持。'])
        result['snapshot_id'] = _digest(result)
        return result


def _dual_robots():
    """复用冻结安装变换和夹爪 URDF 生成器，避免调用物理模型构建器。"""
    from gene_agent.robots.gripper import generate_gripper_urdf
    assets = ROOT/'assets/robots/realman'
    config = _read(ROOT/'config/robots/realman.json')
    installation = _read(assets/'installation.json')[config['installation']]
    standard = _read(ROOT/'config/robots/standard_gripper.json')
    world_body = np.linalg.inv(installation['body_from_sdk']['right'])
    flange_tcp = np.eye(4)
    flange_tcp[:3, :3] = Rotation.from_euler('z', standard['flange_to_tcp_rotation_z_deg'], degrees=True).as_matrix()
    flange_tcp[2, 3] = standard['flange_to_tcp_z_m']
    metadata = dict(profile='realman_dual_standard', world_frame='right_sdk_base',
        world_from_body=world_body.tolist(), joint_groups={}, transforms={}, grippers={},
        group_root_bodies={}, collision_protection_body_names=['robot_body'],
        mounting_contact_exceptions={'robot_links':['base_link','link_1']})
    paths, starts = {}, {}
    for group in ('left', 'right'):
        link_tcp = np.linalg.inv(installation['sdk_flange_to_curobo_frame'][group]) @ flange_tcp
        path, derived = generate_gripper_urdf(group, link_tcp)
        world_base = world_body @ installation['body_from_sdk'][group] @ np.linalg.inv(installation['sdk_to_curobo'][group])
        metadata['joint_groups'][group] = [f'{group}_joint_{i}' for i in range(1,8)]
        metadata['transforms'][group] = dict(world_from_urdf_base=_rigid(world_base).tolist(), link_from_tcp=link_tcp.tolist())
        metadata['group_root_bodies'][group] = f'{group}_mount'
        metadata['grippers'][group] = dict(mode='parallel_slide_proxy',
            width_range_m=derived['config']['width_range_m'], home_width_m=derived['config']['home_width_m'],
            fingers=derived['fingers'], source_hashes={k:derived[k] for k in
                ('generated_urdf_sha256','source_urdf_sha256','config_sha256')})
        paths[group] = path
        starts[group] = _read(ROOT/f'config/robots/realman_{group}_default_pose.json')['joints_rad']
    return {g:VisualRobot(metadata,g,paths[g],starts[g]) for g in paths}


def _geometry_rows(urdf, path, *, collisions=False):
    for link in urdf.robot.links:
        for index, item in enumerate(link.collisions if collisions else link.visuals):
            transform = urdf.get_transform(link.name) @ (np.eye(4) if item.origin is None else item.origin)
            geometry = item.geometry
            row = dict(name=f'{link.name}_{index}', visual_name=item.name, link_name=link.name,
                       T_object_visual=transform, color=[.65,.69,.74,1.])
            if not collisions and item.material is not None and item.material.color is not None:
                row['color'] = list(item.material.color.rgba)
            if geometry.mesh is not None:
                mesh_path = _mesh_path(path, geometry.mesh.filename)
                # 与现有场景构建器一致，旋钮外圈显示使用修复后的冻结 mesh。
                if not collisions and item.name in {'power_knob_rim_visual','timer_knob_rim_visual'}:
                    mesh_path = mesh_path.with_name(mesh_path.stem+'_repaired'+mesh_path.suffix)
                row.update(mesh_path=mesh_path,
                           scale=np.ones(3) if geometry.mesh.scale is None else np.asarray(geometry.mesh.scale))
            elif geometry.box is not None:
                row['size_xyz_m'] = np.asarray(geometry.box.size)
            elif collisions:
                raise ValueError('冻结机器人保护目前只支持 mesh/box')
            else:
                raise ValueError('冻结 visual 资产只支持 mesh/box')
            yield row


class VisualInputs:
    def __init__(self, task, source, output, config):
        self.task, self.source = task, Path(source).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.objects, self.robots = {}, {}
        self.object_urdfs = {}
        self.object_joint_positions = {}
        self.static_visuals = []
        self.placement = None

    def set_object_joints(self, name, values):
        """仅更新 URDF 运动学显示；关节名使用源 URDF 名，不隐含接触。"""
        urdf, path = self.object_urdfs[name]
        unknown = set(values)-set(urdf.joint_map)
        if unknown: raise ValueError(f'未知物体关节: {sorted(unknown)}')
        self.object_joint_positions.setdefault(name, {}).update(values)
        urdf.update_cfg(self.object_joint_positions[name])
        self.objects[name]['visuals'] = list(_geometry_rows(urdf, path))
        self._microwave_visual_offsets()

    def object_link_pose(self, name, link):
        return self.objects[name]['T_world_object'] @ self.object_urdfs[name][0].get_transform(link)

    def _microwave_visual_offsets(self):
        for row in self.objects.get('microwave', {}).get('visuals', []):
            filename = str(row.get('mesh_path',''))
            if 'turntable' in filename:
                row['T_object_visual'][2,3] += self.config.get('turntable_height_offset_m',0.)
            if 'cavity_floor' in filename:
                row['T_object_visual'][2,3] += self.config.get('microwave_floor_height_offset_m',0.)

    def prepare_planner(self, group, width, *, collision_mode="robot_only", held_objects=()):
        """只准备机器人保护；pool 生命周期与 CUDA Graph 策略由调用方持有。"""
        from gene_agent.motion.curobo.core import load_realman_motion_config
        if collision_mode not in {'full','robot_only'}:raise ValueError('未知碰撞模式')
        robot = self.robots[group]
        gripper = robot.gripper_snapshot(width)
        settings = load_realman_motion_config(group, robot.metadata, gripper_snapshot=gripper)
        scene, records = {'cuboid':{}, 'mesh':{}}, []
        base_world = np.linalg.inv(robot.world_from_base)
        body_names = []
        if len(self.robots)>1:
            metadata = robot.metadata
            offset = np.eye(4)
            offset[2,3] = _read(ROOT/'config/robots/realman.json')['body_asset_z_offset_m']
            world_body = _rigid(metadata['world_from_body']) @ offset
            for box in _read(ROOT/'assets/robots/realman/collision/robot_body.json')['cuboid']:
                local = np.eye(4)
                local[:3,3] = box['pose'][:3]
                local[:3,:3] = Rotation.from_quat(np.asarray(box['pose'][3:])[[1,2,3,0]]).as_matrix()
                name = box['name']
                scene['cuboid'][name] = dict(dims=box['dims'], pose=_pose_list(base_world@world_body@local))
                records.append(dict(obstacle_name=name,body_name='robot_body',export_kind='cuboid'))
                body_names.append(name)
        for other, frozen in self.robots.items():
            if other == group: continue
            import trimesh
            for row in _geometry_rows(frozen.urdf, frozen.urdf_path, collisions=True):
                transform = base_world @ frozen.world_from_base @ row['T_object_visual']
                if 'mesh_path' in row:
                    mesh = trimesh.load_mesh(row['mesh_path'],process=False)
                    vertices = np.asarray(mesh.vertices)*row['scale']
                    lower, upper = vertices.min(0), vertices.max(0)
                    transform[:3,3] += transform[:3,:3] @ ((lower+upper)/2)
                    dims = upper-lower
                else: dims = row['size_xyz_m']
                name = f'{other}_{row["name"]}'
                scene['cuboid'][name] = dict(dims=np.maximum(dims,1e-8).tolist(),pose=_pose_list(transform))
                records.append(dict(obstacle_name=name,body_name=f'{other}_{row["link_name"]}',
                                    export_kind='cuboid',approximation='conservative_URDF_local_box'))
        # 纯视觉版本直接从 URDF/重建资产导出环境，不借助物理引擎。
        # 已夹持物不重复当静态障碍；本适配器尚不生成持物碰撞球。
        if collision_mode == 'full':
            import trimesh
            environment=[]
            for name,obj in self.objects.items():
                if name in held_objects:continue
                for index,row in enumerate(obj['visuals']):
                    environment.append((f'object_{name}_{index}',row,obj['T_world_object']@row['T_object_visual']))
            for index,row in enumerate(self.static_visuals):
                if row.get('name')=='robot_body':continue  # 已由精简本体碰撞代理覆盖。
                transform=(self.objects[row['parent']]['T_world_object']@row['T_object_visual']
                           if row.get('parent') else np.asarray(row['T_world_visual']))
                environment.append((f'environment_{index}',row,transform))
            for name,row,world in environment:
                transform=base_world@world
                if row.get('mesh_path'):
                    mesh=trimesh.load_mesh(row['mesh_path'],process=False)
                    if isinstance(mesh,trimesh.Scene):mesh=mesh.to_geometry()
                    mesh=mesh.copy();mesh.vertices*=np.asarray(row.get('scale',[1,1,1]))
                    digest=_digest(dict(path=str(row['mesh_path']),scale=row.get('scale',[1,1,1])))[:16]
                    path=self.output/'collision_meshes'/f'{digest}.obj'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    if not path.exists():mesh.export(path)
                    scene['mesh'][name]=dict(file_path=str(path),pose=_pose_list(transform))
                    kind='mesh'
                else:
                    scene['cuboid'][name]=dict(dims=row['size_xyz_m'],pose=_pose_list(transform));kind='cuboid'
                records.append(dict(obstacle_name=name,body_name=name,export_kind=kind,geometry_source='visual_asset'))
        frozen_groups = sorted(set(self.robots)-{group})
        snapshot = dict(scene_model=scene,gripper_snapshot=gripper,collision_mode=collision_mode,
            scope='robot_only：机器人/夹爪自碰撞 + 本体及冻结另一臂；无场景物体/attachment',
            records=records,excluded=[],active_group=group,frozen_groups=frozen_groups,
            other_arm_frozen=bool(frozen_groups),environment_geometry_exported=collision_mode=='full',
            base_from_world=base_world.tolist(),
            mounting_contact_policy={'robot_links':['base_link','link_1'] if body_names else [],'obstacle_names':body_names},
            robot_joints={g:dict(r.joint_configuration) for g,r in self.robots.items()},
            evidence_mode='kinematic_plan_preview',physics_stepped=False)
        if collision_mode=='full':snapshot['scope']='full：机器人保护 + 场景视觉资产；持物无碰撞球，非物理接触验收'
        snapshot['snapshot_sha256'] = _digest(snapshot)
        return settings, snapshot


def load_visual_inputs(task, source, output, seed=123, randomize=False):
    source = Path(source).resolve()
    frozen = _read(source) if source.is_file() and source.name == 'inputs.json' else None
    if frozen is not None:
        task, source, randomize = frozen['task'], Path(frozen['source']), False
    path = source/('scene.json' if task=='sweep' else 'task.json') if source.is_dir() else source
    config = copy.deepcopy(frozen['config']) if frozen is not None else _read(path)
    scene = VisualInputs(task, source, output, config)
    if task == 'sweep':
        directory = path.parent
        metadata = _read(directory/config['robot']['metadata_path'])
        # 保存的派生 URDF 可缺失时，从同一个原始资产组装；不运行仿真构建器。
        urdf_path = directory/'robot'/Path(metadata['planning_urdf_path']).name
        if not urdf_path.is_file():
            from gene_agent.robots.realman_pika import assemble_single_pika_urdf
            target = scene.output/'robot';target.mkdir(exist_ok=True)
            urdf_path = assemble_single_pika_urdf(target,_read(ROOT/'config/robots/realman_single_pika.json'))
        metadata['planning_urdf_path'] = str(urdf_path)
        group = next(iter(metadata['joint_groups']))
        metadata['transforms'][group]['world_from_urdf_base'] = config['robot']['T_world_base']
        scene.robots = {group:VisualRobot(metadata,group,urdf_path,config['robot']['standby_joint_positions_rad'])}
        scene.concepts_dir = directory/'concepts'
        for name, row in config['objects'].items():
            asset = 'cube' if name.startswith('cube_') else name
            mesh = directory/row.get('mesh_path',f'assets/{asset}.obj')
            concept = directory/row.get('concept_path',f'concepts/{asset}.json')
            scene.objects[name] = dict(T_world_object=validate_pose(row['T_world_object']),
                mesh_path=mesh,concept_path=concept,
                visuals=[dict(mesh_path=mesh,T_object_visual=np.eye(4),color=[.6,.65,.7,1.])])
        scene.table_top_z_m = float(config['table']['top_z_m'])
    elif task == 'microwave':
        asset_config = _read(ROOT/'config/scenes/task_1.json')
        from gene_agent.scene.table import validate_table
        asset_config['table']=validate_table(config.get('table',{}),asset_config['table'])
        scene.robots = _dual_robots()
        scene.concepts_dir = scene.output/'concepts';scene.concepts_dir.mkdir(exist_ok=True)
        for name in ('bowl','microwave'):
            concept = ROOT/(asset_config['microwave_asset']['concept'] if name=='microwave' else asset_config['objects'][name]['concept'])
            mesh = ROOT/'assets/objects/bowl_visual.obj' if name=='bowl' else None
            scene.objects[name] = dict(T_world_object=validate_pose(config[name+'_pose_world']),
                mesh_path=mesh,concept_path=concept,visuals=[] if mesh is None else
                [dict(mesh_path=mesh,T_object_visual=np.eye(4),color=[.82,.86,.9,1.])])
            _write(scene.concepts_dir/f'{name}.json',_read(concept))
        urdf_path = ROOT/asset_config['microwave_asset']['urdf']
        scene.object_urdfs['microwave'] = (URDF.load(str(urdf_path),load_meshes=False),urdf_path)
        scene.set_object_joints('microwave',{'door_hinge':0.,'power_knob_rotation':0.,'timer_knob_rotation':0.})
        import trimesh
        from gene_agent.scene.microwave_frame import estimate_legacy_microwave_frame_from_meshes
        alignment_inputs = {}
        for visual_name in ('door_door_frame_visual','door_handle_visual'):
            row = next(r for r in scene.objects['microwave']['visuals'] if r['visual_name']==visual_name)
            mesh = trimesh.load_mesh(row['mesh_path'],process=False)
            mesh.vertices = np.asarray(mesh.vertices)*row['scale']
            mesh.apply_transform(row['T_object_visual'])
            alignment_inputs[visual_name] = mesh
        scene.legacy_frame_alignment = estimate_legacy_microwave_frame_from_meshes(ROOT/'data/task_1',
            alignment_inputs['door_door_frame_visual'].vertices,alignment_inputs['door_handle_visual'].centroid)
        scene.legacy_microwave_transform = scene.legacy_frame_alignment.T_calibrated_legacy
        scene.table_top_z_m = float(asset_config['table']['top_z_m'])
        # 台面、垫台和机器人本体只显示，不进入场景碰撞。
        support = asset_config['microwave_support'];transform=np.eye(4)
        transform[:3,3]=[*support['center_xy_local_m'],-support['height_m']/2]
        scene.static_visuals.append(dict(name='microwave_support',parent='microwave',T_object_visual=transform,
            size_xyz_m=[*support['size_xy_m'],support['height_m']],color=[.48,.29,.14,1.]))
        body=np.eye(4);body[2,3]=_read(ROOT/'config/robots/realman.json')['body_asset_z_offset_m']
        scene.static_visuals.append(dict(name='robot_body',mesh_path=ROOT/'assets/robots/realman/meshes/body/realman_body_without_arms.obj',
            T_world_visual=_rigid(scene.robots['right'].metadata['world_from_body'])@body,color=[.65,.69,.74,1.]))
    else: raise ValueError(f'纯视觉输入尚不支持任务 {task}')
    if frozen is not None:
        for name, pose in frozen['object_poses_world'].items():scene.objects[name]['T_world_object']=validate_pose(pose)
        for group,q in frozen.get('robot_start_positions_rad',{}).items():
            scene.robots[group].start_positions_rad=np.asarray(q,float);scene.robots[group].set_state(q)
        for name,values in frozen.get('object_joint_positions',{}).items():scene.set_object_joints(name,values)
        scene.placement=frozen.get('placement')
    elif randomize:
        from gene_agent.scene.placement import sample_placement
        rules=config.get('placement_rules')
        footprints={n:r['extents_xyz_m'][:2] for n,r in config.get('objects',{}).items() if 'extents_xyz_m' in r}
        if rules is None and task=='sweep' and 'randomization' in config:
            from gene_agent.scene.tabletop_randomization import sample_tabletop_layout
            nominal={n:(float(p['T_world_object'][0,3]),float(p['T_world_object'][1,3]),
                        float(np.arctan2(p['T_world_object'][1,0],p['T_world_object'][0,0])))
                     for n,p in scene.objects.items()}
            cfg=config['randomization']
            record=sample_tabletop_layout(nominal,cfg['ranges_xyyaw_rad'],footprints,
                table_bounds_xy=cfg['table_bounds_xy_m'],seed=seed,
                min_clearance_m=cfg['min_clearance_m'],max_attempts=cfg['max_attempts'])
            poses={}
            for name,(x,y,yaw) in record['poses_xyyaw_rad'].items():
                pose=scene.objects[name]['T_world_object'].copy()
                pose[:2,3]=[x,y]
                # 保留名义 roll/pitch，只对声明的世界 Z yaw 施加差值。
                pose[:3,:3]=Rotation.from_euler('z',yaw-nominal[name][2]).as_matrix()@pose[:3,:3]
                poses[name]=pose.tolist()
            scene.placement={**record,'sampled_poses_world':poses,'physics_settling_checked':False}
        elif rules is not None:
            scene.placement=sample_placement({n:r['T_world_object'] for n,r in scene.objects.items()},
                rules,seed=seed,randomize=True,footprints=footprints)
        else: raise ValueError('纯视觉随机采样需要显式 placement_rules；不能隐式改用其他分布')
        for name,pose in scene.placement['sampled_poses_world'].items():scene.objects[name]['T_world_object']=np.asarray(pose)
        if not scene.placement['valid']:raise ValueError(f'摆位不满足声明约束: {scene.placement["invalid_reasons"]}')
    table = config['table'] if task=='sweep' else asset_config['table']
    thickness=float(table.get('thickness_m',.04));transform=np.eye(4)
    transform[:3,3]=[*table['center_xy_m'],scene.table_top_z_m-thickness/2]
    scene.static_visuals.append(dict(name='table',T_world_visual=transform,size_xyz_m=[*table['size_xy_m'],thickness],color=[.45,.38,.29,1.]))
    _write(scene.output/'inputs.json',dict(schema_version=1,task=task,source=str(source),config=config,
        evidence_mode='kinematic_plan_preview',physics_stepped=False,seed=seed,
        object_poses_world={n:r['T_world_object'] for n,r in scene.objects.items()},
        object_joint_positions=scene.object_joint_positions,
        robot_start_positions_rad={g:r.start_positions_rad for g,r in scene.robots.items()},
        placement=scene.placement))
    return scene
