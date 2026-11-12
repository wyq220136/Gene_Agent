"""独立visual资产读取与保存候选的纯FK显示；不启动仿真/规划。"""
from pathlib import Path
import hashlib
import numpy as np


def load_visual_scene(bundle):
    import trimesh
    snapshot = bundle.get('snapshot') or {}
    manifest = snapshot.get('visual_snapshot')
    if not manifest:
        return dict(meshes=[], status='missing', warnings=['Saved visual assets/state missing; collision is not a visual substitute.'])
    warnings = list(manifest.get('warnings', []))
    transforms = {}
    robot = manifest.get('active_robot')
    # 环境沿用保存pose；活动臂按诊断实际选中的q重放，而不是误用仿真起点。
    if robot:
        try:
            from yourdfpy import URDF
            path = Path(robot['urdf_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != robot['urdf_sha256']:
                raise ValueError('URDF hash changed')
            urdf = URDF.load(str(path), load_meshes=False, build_scene_graph=True)
            cfg = dict(robot['joint_positions'])
            names = bundle['replay_config']['joint_names']
            q = bundle['configuration_joint_positions_rad']
            if len(names) != len(q):
                raise ValueError('candidate joint names/positions mismatch')
            cfg.update(zip(names, q))
            required = set(robot.get('required_joints', [j.name for j in urdf.actuated_joints]))
            if required - cfg.keys():
                raise ValueError(f'missing joint state: {sorted(required-cfg.keys())}')
            actuated={j.name for j in urdf.actuated_joints}
            urdf.update_cfg({k:v for k,v in cfg.items() if k in actuated})
            base = np.asarray(robot['world_from_base'])
            for link in robot['links']:
                transforms[link] = base @ urdf.get_transform(link,frame_from=robot.get('base_link',urdf.base_link))
            held = manifest.get('attachment')
            if held:
                transforms[held['body_name']] = base @ urdf.get_transform(held['link'],frame_from=robot.get('base_link',urdf.base_link)) @ np.asarray(held['T_link_object'])
        except Exception as exc:
            warnings.append(f'Candidate visual FK unavailable: {type(exc).__name__}: {exc}')
    elif bundle.get('configuration_ref') not in ('start', 'start_state', 'snapshot'):
        # 外部adapter也可以直接保存当前诊断配置下的visual，须绑定配置身份。
        if manifest.get('configuration_identity') != bundle.get('configuration_identity') or not manifest.get('configuration_identity'):
            return dict(meshes=[], status='missing', warnings=warnings+['Visual configuration is not bound to this candidate.'])
    rows = []
    dynamic = set(robot['links']) if robot else set()
    if manifest.get('attachment'):
        dynamic.add(manifest['attachment']['body_name'])
    for row in manifest.get('objects', []):
        try:
            body = row['body_name']
            if body in dynamic and body not in transforms:
                raise ValueError('candidate transform unavailable')
            path = Path(row['file_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('visual asset hash changed')
            asset = trimesh.load(str(path), force='scene', process=False)
            world_from_body = transforms.get(body, np.asarray(row['world_from_body']))
            for node in asset.graph.nodes_geometry:
                local, key = asset.graph[node]
                mesh = asset.geometry[key].copy()
                mesh.apply_transform(world_from_body @ np.asarray(row['body_from_visual']) @ local)
                rows.append(dict(name=row['name']+'/'+str(node), body_name=body, mesh=mesh))
        except Exception as exc:
            warnings.append(f"{row.get('name', 'visual')}: {type(exc).__name__}: {exc}")
    return dict(meshes=rows, status='partial' if warnings else 'available', warnings=warnings)
