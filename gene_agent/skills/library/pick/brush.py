"""工具抓取参数与几何适配，通用 Pick 阶段保持独立。"""
from pathlib import Path
import json
import math
from gene_agent.skills.runtime.profiles import resolve_operation_profile


def brush_pick_source_hashes():
    """记录共同流程实际代码及配置版本，不沿用输入历史run的源码hash。"""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    paths = ('gene_agent/skills/library/pick/brush.py', 'gene_agent/skills/library/pick/skill.py', 'gene_agent/skills/contracts.py',
             'gene_agent/skills/runtime/runner.py', 'gene_agent/skills/runtime/profiles.py', 'gene_agent/skills/library/pick/model.py',
             'gene_agent/skills/runtime/operation.py',
             'gene_agent/skills/library/pick/planning.py', 'gene_agent/simulation/pick_backend.py',
             'gene_agent/knowledge/sweep.py', 'gene_agent/robots/pika_geometry.py',
             'config/tasks/sweep/operation.json', 'config/tasks/sweep/contacts.json',
             'gene_agent/skills/library/sweep/model.py', 'gene_agent/skills/library/sweep/skill.py',
             'gene_agent/skills/runtime/planning.py', 'gene_agent/skills/runtime/poses.py')
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def brush_pick_targets(brush, concept, profile, *, grasp_center_tcp_m=None):
    """几何来自共享 Knowledge；预抓方向明确沿世界 Z，不能套通用 TCP -Z。"""
    from gene_agent.knowledge.sweep import make_brush_grasp_pose
    p = profile.values
    grasp = make_brush_grasp_pose(brush, concept, tilt_deg=p['grasp_tilt_deg'],
                                  tcp_above_handle_m=p['tcp_above_handle_m'],
                                  grasp_center_tcp_m=grasp_center_tcp_m)
    pregrasp = grasp.copy()
    pregrasp[2, 3] += p['approach_distance_m']
    return {'pregrasp': pregrasp, 'approach': grasp}


def brush_grasp_cross_section(brush, mesh_path, grasp, grasp_center_tcp_m):
    """沿闭合轴查询夹持点的局部外壳宽度；不能代替完整手指的首次接触开口。"""
    import numpy as np
    import trimesh
    point = np.asarray(grasp_center_tcp_m,dtype=float)
    world = np.asarray(grasp,dtype=float)
    object_world = np.linalg.inv(np.asarray(brush,dtype=float))
    center = (object_world @ world @ np.r_[point,1.])[:3]
    axis = object_world[:3,:3] @ world[:3,1]
    mesh = trimesh.load_mesh(mesh_path,process=False)
    limits = []
    for direction in (axis,-axis):
        hits,_,_ = mesh.ray.intersects_location([center],[direction],multiple_hits=True)
        distances = (hits-center) @ direction
        distances = distances[(distances>0)&(distances<.025)]
        if not len(distances):
            raise ValueError('实际夹持中心未落在刷柄双侧表面之间，不能假定抓住物体')
        # SAM3D资产存在内部重叠壳；闭爪首先接触外表面，不能取内部最近交点。
        limits.append(float(distances.max()))
    return {'predicted_width_m':sum(limits), 'surface_distances_m':limits,
            'grasp_center_object_m':center.tolist(), 'closing_axis_object':axis.tolist(),
            'source':'mesh_outer_surface_cross_section', 'physical_hold_verified':False}


def brush_lift_target(actual_tcp, pregrasp, profile):
    from gene_agent.skills.library.pick.planning import make_postpick_candidates
    if profile.values['lift_policy'] == 'pregrasp_direction':
        return make_postpick_candidates(actual_tcp, pregrasp, profile.values['lift_distance_m'])[0]
    target = actual_tcp.copy()
    target[2, 3] += profile.values['lift_distance_m']
    return target


def load_brush_pick_profile(environment, *, overrides=None, object_profile=None, path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[4]/'config/tasks/sweep/operation.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1 or config.get('units') != 'meter_degree_second':
        raise ValueError('刷子操作配置版本或单位错误')
    if environment not in config['environments']:
        raise ValueError('未知运行环境')
    profile = resolve_operation_profile(config['defaults'], object_profile,
                                        config['environments'][environment], overrides)
    for key, value in profile.values.items():
        if key == 'lift_policy':
            if value not in {'world_z', 'pregrasp_direction'}: raise ValueError('未知抬升策略')
        elif key == 'verify_hold_for_pick':
            if not isinstance(value, bool): raise ValueError('稳定检查策略必须是布尔值')
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f'{key} 必须为有限数')
        elif key == 'grasp_tilt_deg':
            if abs(value) > 45: raise ValueError('抓取倾角超出±45度')
        elif value < 0 or (key in {'approach_distance_m', 'open_width_m'} and value == 0):
            raise ValueError(f'{key} 不允许该负值/零值')
    if environment == 'simulation' and not profile.values['verify_hold_for_pick']:
        raise ValueError('当前仿真夹持验收需要 hold 检查，不支持关闭')
    if profile.values['open_width_m'] > .09:
        raise ValueError('当前刷子 Pika 配置最大开口为 0.09 m')
    return profile
