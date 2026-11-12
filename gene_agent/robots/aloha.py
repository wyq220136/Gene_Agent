"""从 RoboTwin 的 ALOHA-AgileX 源资产建立规划描述，不创建仿真对象。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from .model import RobotDescription


def digest(value):
    """规范内容签名，拒绝非有限数；用于模型与动态夹爪身份。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_aloha_metadata(*, embodiment_dir: Path, resolved_config: Mapping, world_from_bases: Mapping, link_from_ee_frames: Mapping) -> dict:
    """base 必须由实际原生 link pose 提供；不照抄模板 frame_bias。"""
    root = Path(embodiment_dir).resolve()
    cfg = deepcopy(dict(resolved_config))
    urdf = (root / cfg['urdf_path']).resolve()
    joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
    result = dict(profile='aloha_agilex', embodiment_id='aloha-agilex', world_frame='world',
                  joint_groups={}, planner_joint_names={}, joint_types={}, joint_limits={},
                  tool_links={}, transforms={}, grippers={}, planning_assets={}, source_hashes={})
    files = {root/'config.yml', urdf}
    for index, group in enumerate(('left', 'right')):
        path = root / f'curobo_{group}_tmp.yml'
        if not path.is_file():
            raise FileNotFoundError(f'缺少可追溯的规划模板: {path}')
        template = yaml.safe_load(path.read_text())
        kin = template['robot_cfg']['kinematics']
        names = list(cfg['arm_joints_name'][index])
        result['joint_groups'][group] = names
        result['planner_joint_names'][group] = names.copy()
        result['joint_types'][group] = [joints[n].get('type') for n in names]
        result['joint_limits'][group] = [[float(joints[n].find('limit').get(k)) for k in ('lower','upper')] for n in names]
        result['tool_links'][group] = cfg['move_group'][index]
        tcp = np.eye(4)
        tcp[:3,:3] = np.asarray(cfg['global_trans_matrix']) @ np.asarray(cfg['delta_matrix'])
        tcp[:3,3] = tcp[:3,:3] @ [float(cfg['gripper_bias']),0,0]
        # native TCP基于joint.global_pose；URDF规划使用child link，必须包含pose_in_child。
        tcp = np.asarray(link_from_ee_frames[group],dtype=float) @ tcp
        result['transforms'][group] = dict(world_from_urdf_base=deepcopy(world_from_bases[group]),link_from_tcp=tcp.tolist(),link_from_native_ee=deepcopy(link_from_ee_frames[group]))
        gripper = cfg['gripper_name'][index]
        low, high = map(float, cfg['gripper_scale'])
        if not low < high or high <= 0:
            raise ValueError('ALOHA 夹爪归一化范围无效')
        result['grippers'][group] = dict(mode='aloha_parallel', joint_names=[gripper['base'],*[row[0] for row in gripper['mimic']]],
            base_joint=gripper['base'], mimic=deepcopy(gripper['mimic']), native_range=[low,high],
            approach_axis_tcp=[1.,0.,0.],opening_axis_tcp=[0.,1.,0.],
            width_range_m=[0.,2*high], width_mapping='width=q7+q8; zero command keeps native closing preload')
        # 接触白名单来自原URDF的指关节child link，不允许任务任意豁免机械臂link。
        result['grippers'][group]['contact_links'] = [joints[n].find('child').get('link') for n in result['grippers'][group]['joint_names']]
        sphere_source = kin.get('collision_spheres')
        if isinstance(sphere_source, str):
            sphere_path = root / Path(sphere_source).name
            files.add(sphere_path)
            kin['collision_spheres'] = yaml.safe_load(sphere_path.read_text())['collision_spheres']
        kin['urdf_path'] = str(urdf)
        kin['asset_root_path'] = str(urdf.parent)
        result['planning_assets'][group] = dict(robot_config=template['robot_cfg'],base_link=kin['base_link'],template=str(path))
        files.add(path)
    result['source_hashes'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    RobotDescription.from_metadata(result)
    return result


def make_aloha_gripper_snapshot(robot_metadata: dict, group: str, joint_positions_m: Mapping) -> dict:
    """保存真实指关节，锁住的几何随开口变化；不套用 Pika 球重定位算法。"""
    descriptor = robot_metadata['grippers'][group]
    positions = {name:float(joint_positions_m[name]) for name in descriptor['joint_names']}
    if not np.isfinite(list(positions.values())).all():
        raise ValueError('夹爪关节必须有限')
    if any(q < -.0002 or q > .04785 for q in positions.values()):
        raise ValueError('ALOHA 实际指关节越过 URDF 限位')
    payload = dict(schema_version=1, group=group, mode='aloha_parallel', joint_positions_m=positions,
                   width_m=sum(positions.values()), source_hashes=robot_metadata['source_hashes'],
                   merged_sphere_sha256=digest(robot_metadata['planning_assets'][group]['robot_config']['kinematics']['collision_spheres']),
                   source_robot_config_sha256=digest(robot_metadata['planning_assets'][group]['robot_config']),
                   joint_names=list(descriptor['joint_names']),
                   added_spheres_by_link={},limitations=['源指球保持，实际指关节锁定；兼容开口变化精确同步，不兼容结构重建。'])
    return {**payload,'snapshot_id':digest(payload)}
