"""MuJoCo GT静态碰撞快照：GPU探针正负例与真实候选IK对照。"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
import torch
from gene_agent.motion import SynchronousPlanner, load_realman_motion_config
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.knowledge import bowl_grasp_candidates, knob_grasp_candidates
from gene_agent.simulation import build_scene
from tools.diagnostics.check_reachability import body_pose, screen, summarize


def collision_probe(snapshot, points_in_base, radii):
    """使用固定提交当前真实GPU检查器；输出的是碰撞代价，不冒充完整SDF距离。"""
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import SceneCfg
    from curobo.types import DeviceCfg
    device = DeviceCfg()
    scene = SceneCollision.from_config(SceneCollisionCfg(device_cfg=device, scene_model=SceneCfg.create(snapshot['scene_model'])))
    values = np.column_stack([points_in_base, radii])
    spheres = torch.tensor(values[None, None], device='cuda', dtype=torch.float32)
    buffer = CollisionBuffer.from_shape(spheres.shape, device)
    cost = scene.get_sphere_distance_raw(spheres, buffer, torch.tensor([1.], device='cuda'), torch.tensor([0.], device='cuda'))
    torch.cuda.synchronize()
    return cost.detach().cpu().numpy().reshape(-1).tolist()


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    report = {'code_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), Path(__file__).resolve().parents[2]/'gene_agent/motion/curobo/scene.py', Path(__file__).resolve().parents[2]/'gene_agent/motion/curobo/core.py']}, 'mode': 'static_full_contact_scene_snapshot', 'cases': {}, 'limitations': ['另一臂与碗只冻结当前状态，不验证双臂协同。', '原仿真禁用knob接触，因此本快照也不包含knob碰撞，逐geom排除清单可查。', '活动组继续采用原cuRobo碰撞球及ignore，不宣称与MuJoCo mesh完全等价。']}
    for label, microwave_dx in [('nominal', 0.), ('suggested_knob_x_minus_010', -.10)]:
        model, data, metadata = build_scene(out/label/'source_scene')
        if microwave_dx:
            # 固定预算建议pose只在本实验模型移动；不改默认配置，也不宣称支撑有效。
            model.body_pos[model.body('microwave').id, 0] += microwave_dx
            mujoco.mj_forward(model, data)
        for _ in range(round(3/model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        snapshot = export_collision_scene(model, data, metadata['robot'], 'right', out/label/'collision')
        records = snapshot['records']
        points_world = [np.asarray(next(r for r in records if r['geom_name']=='scene_table')['T_world_geom'])[:3, 3], np.array([5.,5.,5.])]
        bowl_pose = body_pose(model, data, 'bowl')
        points_world.append((bowl_pose@np.array([0., .01, 0., 1.]))[:3])
        points_world.append(np.asarray(next(r for r in records if r['geom_name'].startswith('bowl_wall_'))['T_world_geom'])[:3, 3])
        transform = np.asarray(snapshot['base_from_world'])
        points_base = np.asarray(points_world)@transform[:3, :3].T+transform[:3, 3]
        costs = collision_probe(snapshot, points_base, [.001]*len(points_base))
        probe_names = ['inside_table', 'far_clearance', 'inside_bowl_cavity', 'inside_transparent_bowl_wall']
        case = {'snapshot_sha256': snapshot['snapshot_sha256'], 'collision_scope': snapshot['scope'], 'other_arm_frozen': snapshot['other_arm_frozen'],
                'obstacle_counts': {k: len(v) for k,v in snapshot['scene_model'].items()}, 'microwave_world_dx_m': microwave_dx,
                'probes': [{'name': name, 'point_world_m': point.tolist(), 'point_base_m': base.tolist(), 'collision_cost': cost} for name, point, base, cost in zip(probe_names, points_world, points_base, costs)],
                'probe_pass': costs[0]>0 and costs[1]==0 and costs[2]==0 and costs[3]>0,
                'GT_poses': {name: body_pose(model,data,name).tolist() for name in ['bowl','microwave']}}
        planner = SynchronousPlanner(**load_realman_motion_config('right', metadata['robot'], gripper_snapshot=snapshot.get('gripper_snapshot')), scene_model=snapshot['scene_model'],
                                     scene_scope=snapshot['scope'], scene_snapshot_sha256=snapshot['snapshot_sha256'], mounting_contact_policy=snapshot['mounting_contact_policy'])
        names = metadata['robot']['joint_groups']['right']
        q = [float(data.qpos[model.joint(name).qposadr[0]]) for name in names]
        case.update(joint_names=names, start_joint_positions_rad=q)
        # 检查当前活动组球是否已与快照障碍相交；可解释静态起点问题但不改碰撞。
        state = planner.kinematics.compute_kinematics(planner._state(names,q))
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1,4)
        valid = spheres[:,3]>0
        start_costs = collision_probe(snapshot,spheres[valid,:3],spheres[valid,3])
        case['scoped_mounting_start_validation'] = planner._validator.check([q], planner.planner_joint_names) if planner._validator else None
        case['start_sphere_collision'] = {'enabled_sphere_count': int(valid.sum()), 'colliding_sphere_count': int(np.count_nonzero(start_costs)), 'max_cost': max(start_costs, default=0)}
        # 逐障碍独立查询只用于解释碰撞代价来源，不把子集传给IK；真正planner始终
        # 使用上面的完整snapshot。保存每个球的link和base/world中心，便于视觉核对。
        params = planner.kinematics.config.kinematics_config
        sphere_link_ids = params.link_sphere_idx_map.detach().cpu().numpy()
        link_names = {value: key for key,value in params.link_name_to_idx_map.items()}
        indices = np.flatnonzero(valid)
        world_base = np.linalg.inv(np.asarray(snapshot['base_from_world']))
        contributions = []
        for record in records:
            kind, obstacle = record['export_kind'], record['obstacle_name']
            isolated = {'scene_model': {kind: {obstacle: snapshot['scene_model'][kind][obstacle]}}}
            costs_one = np.asarray(collision_probe(isolated, spheres[valid,:3], spheres[valid,3]))
            if np.any(costs_one>0):
                contacts = []
                for idx, cost in zip(indices[costs_one>0], costs_one[costs_one>0]):
                    center = spheres[idx,:3]
                    contacts.append({'sphere_index': int(idx), 'robot_link': link_names[int(sphere_link_ids[idx])],
                                     'center_base_m': center.tolist(), 'center_world_m': (world_base[:3,:3]@center+world_base[:3,3]).tolist(),
                                     'radius_m': float(spheres[idx,3]), 'collision_cost': float(cost)})
                contributions.append({'source_geom_id': record['geom_id'], 'source_geom_name': record['geom_name'],
                                      'source_body': record['body_name'], 'contacts': contacts})
        case['start_collision_by_obstacle'] = {'mode': 'isolated_obstacle_cost_diagnosis_only_no_planning', 'contributions': contributions}

        for name,candidates in [('bowl',bowl_grasp_candidates(body_pose(model,data,'bowl'))),('knob',knob_grasp_candidates(body_pose(model,data,'microwave'),concept=metadata['objects']['knob_concept']))]:
            case[name]=screen(planner,names,q,candidates)
            for candidate in case[name]:
                candidate.update(collision_checked=True, full_scene_collision_checked=True, collision_scope='static_enabled_contact_snapshot', collision_snapshot_sha256=snapshot['snapshot_sha256'])
        case['summary']={name:summarize(case[name]) for name in ['bowl','knob']}
        report['cases'][label]=case
        print(label,case['obstacle_counts'],case['probe_pass'],case['start_sphere_collision'],case['summary'],flush=True)
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['completed']=True
    report['collision_probe_pass']=all(case['probe_pass'] for case in report['cases'].values())
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report['collision_probe_pass']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/scene-collision'));args=parser.parse_args()
    try:
        raise SystemExit(0 if run(args.output) else 1)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'raw_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
