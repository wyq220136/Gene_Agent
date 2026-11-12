"""已有任务 Knowledge 的纯几何装配；不筛 IK，不产生物理成功结论。"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.knowledge.sweep import make_brush_grasp_pose, make_sweep_layout
from gene_agent.robots.pika_geometry import pika_grasp_center_tcp
from gene_agent.knowledge import bowl_grasp_candidates
from gene_agent.knowledge.geometry import load_concept
from gene_agent.knowledge.grasp import load_grasp_profiles, generate_grasps, parse_grasp_profiles
from gene_agent.knowledge.operation import (
    AxisSpec, generate_handle_pull_candidates, generate_handle_pull_path,
    generate_rotation_path,
)
from gene_agent.knowledge.operation.door_push import (
    generate_mode_door_push_candidate_sets, make_door_push_precontact,
)

ROOT=Path(__file__).resolve().parents[3]
KNOWLEDGE={'sweep':{'brush_grasp':'刷柄抓取','sweep_contact':'清扫方向与起止点'},
           'microwave':{'bowl_grasp':'碗沿抓取','handle_grasp':'门把手抓取','knob_grasp':'旋钮抓取',
                        'door_push':'门面推门路径','handle_pull':'把手拉门路径','knob_turn':'旋钮旋转路径'}}
SKILLS={'sweep':{'pick_brush':'抓起刷子','sweep_cube':'清扫方块'},
        'microwave':{'pick_bowl':'抓碗','open_door':'拉把手开门','close_door':'拉把手关门',
                     'rotate_knob':'旋转旋钮','insert_bowl':'放入碗','extract_bowl':'取出碗'}}


def proposals(inputs, selection, *, tilt=-20., parameters=None):
    """返回既有候选字段；操作路径额外携带可序列化的阶段/物体关节元数据。"""
    if selection in {'door_push','handle_pull','knob_turn'}:
        return _operation_proposals(inputs, selection, dict(parameters or {}))
    if parameters:
        raise ValueError('该 Knowledge 不支持操作路径参数')
    objects=inputs.objects
    if selection=='brush_grasp':
        center=pika_grasp_center_tcp(inputs.robots['left'].metadata,'left')
        pose=make_brush_grasp_pose(objects['brush']['T_world_object'],inputs.concepts_dir/'brush.json',tilt_deg=tilt,
                                   grasp_center_tcp_m=center)
        return [dict(name='brush_grasp',group='left',object='brush',pose=pose,width_m=.04,
                     parameters={'grasp_tilt_deg':tilt,'grasp_center_tcp_m':center.tolist()},source='knowledge.sweep.make_brush_grasp_pose')]
    if selection=='sweep_contact':
        rows=[];poses={n:o['T_world_object'] for n,o in objects.items()}
        for name in ('cube','cube_red','cube_blue'):
            if name not in poses:continue
            layout=make_sweep_layout(poses,inputs.concepts_dir,target_name=name)
            for kind,xyz in [('start',layout.brush_start_world_xyz),('end',layout.brush_end_world_xyz),('goal',layout.cube_goal_world_xyz)]:
                pose=np.eye(4);pose[:3,3]=xyz
                rows.append(dict(name=f'{name}/{kind}',group='left',object=name,pose=pose,
                                 source='knowledge.sweep.make_sweep_layout',parameters={'direction_xy':layout.direction_world_xy.tolist()}))
        return rows
    if selection=='bowl_grasp':
        return [dict(name=r['candidate_id'],group='right',object='bowl',pose=np.asarray(r['T_world_tcp']),
                     width_m=r['parameters']['pregrasp_opening_width_m'],parameters=r['parameters'],
                     source='knowledge.candidates.bowl_grasp_candidates') for r in bowl_grasp_candidates(
                         objects['bowl']['T_world_object'],pose_source='pure_visual_nominal')]
    if selection in {'handle_grasp','knob_grasp'}:
        if selection=='handle_grasp':
            path=ROOT/'assets/objects/microwave_calibrated/handle_concept.json'
            profiles=load_grasp_profiles(ROOT/'data/task_1/knowledge/grasp_profiles.microwave_door.json')
            profile=next(iter(profiles.values()))
            part=load_concept(path,part_name=profile.part_name).concept
            parent=inputs.object_link_pose('microwave','door')@np.linalg.inv(inputs.object_urdfs['microwave'][0].joint_map['door_hinge'].origin)
            group='left'
        else:
            path=ROOT/'assets/objects/microwave_calibrated/concept.json'
            profiles=load_grasp_profiles(ROOT/'data/task_1/knowledge/grasp_profiles.microwave_calibrated.json')
            profile=profiles['microwave_knob_0'];part=load_concept(path,part_name='knob_0').concept
            parent=objects['microwave']['T_world_object'];group='right'
        generated=generate_grasps(part,parent,profile)
        return [dict(name=p.candidate_id,group=group,object='microwave',pose=p.tcp_pose_world,
                     width_m=.08,parameters=dict(p.parameters),source=str(path)) for p in generated.proposals]
    raise ValueError(f'未知 Knowledge: {selection}')


def _joint_axis(inputs, name):
    urdf = inputs.object_urdfs['microwave'][0]
    joint = urdf.joint_map[name]
    world = inputs.objects['microwave']['T_world_object']
    origin = world @ urdf.get_transform(joint.parent) @ joint.origin
    return joint, AxisSpec(origin[:3, 3], origin[:3, :3] @ joint.axis,
                           'world', f'calibrated_urdf:{name}')


def _angle(joint, value):
    value = float(value)
    if not np.isfinite(value) or not joint.limit.lower - 1e-5 <= value <= joint.limit.upper + 1e-5:
        raise ValueError(f'{joint.name} 角度超出 URDF 限位')
    return float(np.clip(value, joint.limit.lower, joint.limit.upper))


def _turn(axis, angle):
    """绕 world 轴的主动旋转；轴已含 URDF 正方向，不能额外翻转符号。"""
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(axis.direction * angle).as_matrix()
    transform[:3, 3] = axis.point - transform[:3, :3] @ axis.point
    return transform


def _operation_row(inputs, kind, group, axis, joint, pose, angle, width,
                   index, label, parameters, normal, contact, metadata):
    # 保留其他关节的实际可视状态；调用方选中阶段时再显式更新显示。
    joints = dict(inputs.object_joint_positions.get('microwave', {}))
    joints[joint.name] = float(angle)
    return dict(name=f'{kind}/{index:02d}_{label}', group=group, object='microwave',
        pose=np.asarray(pose), width_m=float(width), parameters=parameters,
        source=f'knowledge.operation.{kind}', path_id=kind, waypoint_index=index,
        waypoint_label=label, angle_rad=float(angle),
        axis_point_world=axis.point.tolist(), axis_direction_world=axis.direction.tolist(),
        surface_normal_world=np.asarray(normal).tolist(), contact_points_world=[np.asarray(contact).tolist()],
        object_joint_positions={'microwave': joints}, metadata=metadata)


def _operation_proposals(inputs, selection, tune):
    settings = json.loads((ROOT/'data/task_1/knowledge/microwave_knowledge_view.json').read_text())
    common = {'target_angle_rad', 'candidate_rank'}
    specific = {
        'door_push': {'mode', 'start_angle_rad', 'step_rad', 'pre_contact_distance_m',
            *settings['door_push']},
        'handle_pull': {'pregrasp_distance_m', 'retreat_distance_m', 'retreat_up_m',
            'open_width_m', 'close_width_m', 'rotation_steps', 'tip_rotation_x_scale'},
        'knob_turn': {'step_rad'},
    }
    if set(tune) - common - specific[selection]:
        raise ValueError(f'未知操作参数: {sorted(set(tune) - common - specific[selection])}')
    rank = tune.get('candidate_rank', 0)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError('candidate_rank 必须为非负整数')
    joint_name = 'power_knob_rotation' if selection == 'knob_turn' else 'door_hinge'
    joint, axis = _joint_axis(inputs, joint_name)
    current = _angle(joint, inputs.object_joint_positions.get('microwave', {}).get(joint_name, 0.))
    metadata = dict(evidence_mode='geometry_only', ik_checked=False, collision_checked=False,
        physics_stepped=False, frame='world', units='m, rad',
        geometry_source='calibrated_urdf', axis_source=axis.source,
        profile_source='data/task_1/knowledge/microwave_knowledge_view.json',
        contact_points_meaning='Concept 表面参考点；未验证指垫接触')
    # 旧重建 frame 只为叠图存档；当前 URDF 几何已经在 calibrated frame，不能再乘一次。
    if hasattr(inputs, 'legacy_microwave_transform'):
        metadata['T_world_legacy_preview'] = (inputs.objects['microwave']['T_world_object'] @
                                             inputs.legacy_microwave_transform).tolist()
        metadata['legacy_alignment_accepted_for_planning'] = False
    if selection == 'door_push':
        return _push_rows(inputs, settings, tune, rank, joint, axis, metadata)

    world = inputs.objects['microwave']['T_world_object']
    raw = json.loads((ROOT/'data/task_1/knowledge/grasp_profiles.json').read_text())
    for name, values in settings['profile_overrides'].items():
        raw['profiles'][name]['parameters'].update(values)
    profiles = parse_grasp_profiles(raw)
    if selection == 'handle_pull':
        # 与旧 Knowledge viewer 同源：保留旧 Concept 尺寸/参数，将 grip 中心
        # 对齐校准 handle visual 质心。门角在 closed reference 之后只应用一次。
        import trimesh
        urdf, urdf_path = inputs.object_urdfs['microwave']
        visual = next(v for v in urdf.link_map[joint.child].visuals if v.name == 'door_handle_visual')
        mesh = trimesh.load_mesh(urdf_path.parent/visual.geometry.mesh.filename, process=False)
        scale = visual.geometry.mesh.scale
        if scale is not None:
            mesh.vertices *= np.asarray(scale)
        center = (world @ joint.origin @ visual.origin @ np.r_[mesh.center_mass, 1.])[:3]
        part = load_concept(ROOT/'assets/objects/microwave_concept.json', part_name='handle').concept
        handle = np.eye(4)
        handle[:3, :3] = np.column_stack((world[:3, 0], world[:3, 2], -world[:3, 1]))
        handle[:3, 3] = center - handle[:3, 2] * part.grip_segment.pose_in_trifold[2, 3]
        part = replace(part, pose_in_object=np.linalg.inv(world) @ handle)
        grasp = generate_grasps(part, world, profiles[settings['handle_profile']], part_name='handle').proposals[0]
        motion = _turn(axis, current)
        handle = motion @ handle
        candidates = generate_handle_pull_candidates(motion @ grasp.tcp_pose_world, -handle[:3, 2])
        if rank >= len(candidates):
            raise ValueError('把手候选序号超出范围')
        chosen = candidates[rank]
        target = _angle(joint, tune.get('target_angle_rad', np.deg2rad(25)))
        path, labels, widths = generate_handle_pull_path(chosen.grasp_tcp_pose_world, axis,
            start_angle_rad=current, target_angle_rad=target,
            rotation_steps=tune.get('rotation_steps', 3),
            tip_rotation_x_scale=tune.get('tip_rotation_x_scale', 1.),
            pregrasp_distance_m=tune.get('pregrasp_distance_m', .03),
            retreat_distance_m=tune.get('retreat_distance_m', .015),
            retreat_up_m=tune.get('retreat_up_m', 0.),
            open_width_m=tune.get('open_width_m', .08), close_width_m=tune.get('close_width_m', .015))
        local_contact = part.grip_segment.pose_in_trifold[:3, 3].copy()
        local_contact += [0., grasp.parameters['vertical_offset_m'], part.grip_segment.size_xyz_m[2]/2]
        contact = (handle @ np.r_[local_contact, 1.])[:3]
        normal = handle[:3, 2]
        params = {**grasp.to_dict()['parameters'], **chosen.parameters,
                  'start_angle_rad': current, 'target_angle_rad': target,
                  'rotation_steps': 3, 'tip_rotation_x_scale': 1.,
                  'pregrasp_distance_m': .03, 'retreat_distance_m': .015, 'retreat_up_m': 0.,
                  'open_width_m': .08, 'close_width_m': .015, **tune,
                  'candidate_id': chosen.candidate_id}
        group = 'left'
    else:
        part = load_concept(ROOT/'assets/objects/microwave_calibrated/concept.json', part_name='knob_0').concept
        candidates = generate_grasps(part, world, profiles[settings['knob_profile']], part_name='knob_0').proposals
        if rank >= len(candidates):
            raise ValueError('旋钮候选序号超出范围')
        grasp = candidates[rank]
        target = _angle(joint, tune.get('target_angle_rad', settings['knob_target_angle_rad']))
        path = generate_rotation_path('knob', _turn(axis, current) @ grasp.tcp_pose_world, axis,
            start_angle_rad=current, target_angle_rad=target,
            step_rad=tune.get('step_rad', settings['waypoint_step_rad']))
        labels = tuple(f'turn_{i:02d}' for i in range(len(path.waypoints)))
        widths = (grasp.closing_width_m,) * len(path.waypoints)
        knob = _turn(axis, current) @ world @ part.pose_in_object
        normal = knob[:3, 2]
        contact = knob[:3, 3] + normal * part.height_m / 2
        params = {**grasp.to_dict()['parameters'], 'start_angle_rad': current,
                  'target_angle_rad': target, 'step_rad': settings['waypoint_step_rad'],
                  **tune, 'candidate_id': grasp.candidate_id}
        group = 'right'
    metadata.update(candidate_count=len(candidates), candidate_rank=rank)
    rows = []
    for index, (point, label, width) in enumerate(zip(path.waypoints, labels, widths, strict=True)):
        delta = _turn(axis, point.angle_rad - current)
        rows.append(_operation_row(inputs, selection, group, axis, joint,
            point.tcp_pose_world, point.angle_rad, width, index, label, dict(params),
            delta[:3, :3] @ normal, (delta @ np.r_[contact, 1.])[:3], dict(metadata)))
    return rows


def _push_rows(inputs, settings, tune, rank, joint, axis, metadata):
    mode = tune.get('mode', settings['door_push_mode'])
    start = _angle(joint, tune.get('start_angle_rad', settings['door_start_angle_rad']))
    target = _angle(joint, tune.get('target_angle_rad', settings['door_target_angle_rad']))
    if (mode.endswith('_open') and start >= target) or (mode.endswith('_close') and start <= target):
        raise ValueError('推门起终角与开关门方向不一致')
    urdf = inputs.object_urdfs['microwave'][0]
    panel = next(c for c in urdf.link_map[joint.child].collisions if c.name == 'door_panel_collision')
    # 源 joint.origin 是 closed reference；不能用当前 door FK 再叠加绝对门角。
    closed = inputs.objects['microwave']['T_world_object'] @ urdf.get_transform(joint.parent) @ joint.origin @ panel.origin
    size = panel.geometry.box.size
    group = mode.split('_')[0]
    overrides = {**settings['door_push'], **{k: v for k, v in tune.items() if k in settings['door_push']}}
    sets = generate_mode_door_push_candidate_sets(closed, panel_width_m=size[0],
        panel_height_m=size[2], panel_thickness_m=size[1], axis=axis, mode=mode,
        start_angle_rad=start, target_angle_rad=target,
        step_rad=tune.get('step_rad', settings['door_step_rad']), overrides=overrides,
        reference_origin_world=inputs.robots[group].world_from_base[:3, 3])
    if rank >= len(sets[0].candidates):
        raise ValueError('推门起始候选序号超出范围')
    chosen = [s.candidates[rank if i == 0 else 0] for i, s in enumerate(sets)]
    rows = []
    for index, candidate in enumerate(chosen):
        normal = candidate.surface_normal_world
        contact = candidate.tcp_pose_world[:3, 3] - candidate.parameters['standoff_m'] * normal
        params = {**candidate.parameters, 'mode': mode, 'start_angle_rad': start,
                  'target_angle_rad': target, 'step_rad': tune.get('step_rad', settings['door_step_rad']),
                  'pre_contact_distance_m': tune.get('pre_contact_distance_m', .05),
                  'candidate_id': candidate.candidate_id}
        details = {**metadata, 'candidate_count': len(sets[index].candidates),
                   'candidate_rank': candidate.preference_rank, 'candidate_source': candidate.source,
                   'panel_size_xyz_m': np.asarray(size).tolist(),
                   'closed_panel_pose_world': closed.tolist()}
        if index == 0:
            rows.append(_operation_row(inputs, 'door_push', group, axis, joint,
                make_door_push_precontact(candidate, distance_m=tune.get('pre_contact_distance_m', .05)),
                candidate.door_angle_rad, .06, 0, 'pre_contact', dict(params), normal, contact, details))
        rows.append(_operation_row(inputs, 'door_push', group, axis, joint,
            candidate.tcp_pose_world, candidate.door_angle_rad, .06, index+1,
            'contact' if index == 0 else f'push_{index:02d}', params, normal, contact, details))
    return rows
