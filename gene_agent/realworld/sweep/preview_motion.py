"""清扫前瞻的阶段规划。预测夹持仅用于碰撞代理与显示，不是物理夹持证据。"""
from __future__ import annotations

import copy
from pathlib import Path
import mujoco
import numpy as np

from gene_agent.motion import load_realman_motion_config
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode, collision_mode_metadata
from gene_agent.motion.curobo.core import CuroboPlannerAdapter
from gene_agent.motion.curobo.scene import export_collision_scene, filter_collision_snapshot
from gene_agent.motion.model import JointState, MotionRequest, MotionOperation, PoseTarget, PlanningSceneRef
from gene_agent.robots.pika_frames import pose_from_mujoco, pose_error
from gene_agent.scene.rigid_geometry import collision_spheres_object
from .planning import _set_prediction, check_world_contacts, write_json


def plan_stage(model, data, robot, pool, stage_dir, stage, target, path_kind,
               allowed_contacts, grip, *, base_table_mount=False):
    """单一阶段入口：严格 cuRobo，完整几何 FK 后检查，再更新预测状态。"""
    stage_dir = Path(stage_dir)
    names = robot['joint_groups']['left']
    start = [float(data.qpos[model.joint(n).qposadr[0]]) for n in names]
    transferred = {}
    attachment = None
    robot_only = resolve_collision_mode() is CollisionMode.ROBOT_ONLY
    if grip is not None and not robot_only:
        spheres = collision_spheres_object(model, data, 'brush', max_cell_m=.012)
        brush = pose_from_mujoco(data.body('brush'))
        world = spheres.copy();world[:, :3] = spheres[:, :3] @ brush[:3, :3].T + brush[:3, 3]
        transferred = {'brush': world.tolist()}
        link_grip = np.asarray(robot['transforms']['left']['link_from_tcp']) @ grip
        attachment = spheres.copy()
        attachment[:, :3] = spheres[:, :3] @ link_grip[:3, :3].T + link_grip[:3, 3]
    full = export_collision_scene(model, data, robot, 'left', stage_dir/'full_collision',
                                   transferred_objects=transferred)
    rejected_bodies = {b for a, b in allowed_contacts if a == 'brush' and b in {'cube', 'dustpan'}}
    if grip is None and any('brush' in p and any(n.startswith('pika_gripper_') for n in p) for p in allowed_contacts):
        rejected_bodies.add('brush')
    rejected_geoms = {'scene_table'} if ('brush', 'table') in allowed_contacts and grip is not None else set()
    snapshot = filter_collision_snapshot(full, stage_dir/'stage_collision', keep_body_names=None,
        rejected_body_names=rejected_bodies, rejected_geom_names=rejected_geoms,
        attached_object_id='brush' if grip is not None else None,
        scope='explicit preview stage contacts; full world FK postcheck follows')
    settings = load_realman_motion_config('left', robot, gripper_snapshot=snapshot['gripper_snapshot'])
    if attachment is not None:
        settings['attachment_spheres_tool'] = attachment.tolist()
    lease = pool.acquire(settings, snapshot, random_seed=123)
    adapter = CuroboPlannerAdapter(lease.planner, group='left', scene_revision=0)
    operation = MotionOperation.LINEAR_TCP if path_kind == 'linear_tcp' else MotionOperation.PLAN
    request = MotionRequest(stage, operation, 'left', JointState(tuple(names), tuple(start)),
        PoseTarget('world_tcp', target), PlanningSceneRef(0, snapshot['snapshot_sha256'], snapshot['scope']), seed=123)
    print(f'cuRobo 规划：{stage} ({path_kind})', flush=True)
    result = adapter.plan(request)
    row = {'name': stage, 'path_kind': path_kind, 'target_tcp_world': target.tolist(),
           **collision_mode_metadata(),
           'request': request.to_dict(), 'result': result.to_dict(), 'success': bool(result.success),
           'allowed_contact_pairs': [list(p) for p in allowed_contacts],
           'attachment': 'hypothetical_grasp' if attachment is not None else None,
           'full_collision_snapshot': str(stage_dir/'full_collision/collision_snapshot.json'),
           'stage_collision_snapshot': str(stage_dir/'stage_collision/collision_snapshot.json')}
    raw = copy.deepcopy(getattr(lease.planner, 'last_raw', None))
    if raw is not None:
        write_json(stage_dir/'raw_planner_return.json', raw)
    frames = []
    if result.success:
        if result.trajectory is None:
            raise RuntimeError('cuRobo 成功但未返回轨迹')
        positions = result.trajectory.positions_rad
        post = check_world_contacts(model, data, robot, positions, grip, allowed_contacts,
                                     base_table_mount=base_table_mount)
        row['full_world_postcheck'] = post
        row['success'] = bool(post['success'])
        if not post['success']:
            rejected = mujoco.MjData(model);mujoco.mj_copyData(rejected, model, data)
            rejected_frames = []
            for q in positions:
                _set_prediction(model, rejected, robot, q, grip)
                rejected_frames.append({'stage': stage, 'qpos': rejected.qpos.tolist(),
                                        'predicted': True, 'accepted': False})
            row['rejected_preview_frames_path'] = str(stage_dir/'rejected_preview_frames.json')
            write_json(stage_dir/'rejected_preview_frames.json', rejected_frames)
        if post['success']:
            for q in positions:
                _set_prediction(model, data, robot, q, grip)
                frames.append({'stage': stage, 'qpos': data.qpos.tolist(), 'predicted': True})
            row['achieved_tcp_world'] = pose_from_mujoco(data.site('pika_tcp')).tolist()
            row['target_error'] = pose_error(target, np.asarray(row['achieved_tcp_world']))
    write_json(stage_dir/'motion_result.json', row)
    return row, frames
