"""同几何世界的位姿同步；只处理已加载的障碍，不改变接触策略。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


_SELF_CONTAINED_MESH_SUFFIXES = {'.obj', '.ply', '.stl'}


def _mesh_format(path):
    try:
        suffix = Path(path).suffix.lower()
    except (TypeError, ValueError):
        return None
    return suffix if suffix in _SELF_CONTAINED_MESH_SUFFIXES else None


def mesh_content_hashes(scene: dict) -> dict[str, str | None]:
    """逐次记录加载的mesh内容；外部引用格式或缺文件时不能证明几何身份。"""
    from gene_agent.feedback.runtime_timing import measure_runtime
    meshes = scene.get('mesh', {})
    with measure_runtime("mesh_file_read", "hash_mesh_files",
                         metadata={"mesh_count": len(meshes)}):
        result = {}
        for name, row in meshes.items():
            try:
                path = row['file_path']
                if _mesh_format(path) is None:
                    result[name] = None
                    continue
                digest = hashlib.sha256()
                with open(path, 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                result[name] = digest.hexdigest()
            except (OSError, KeyError, TypeError, ValueError):
                result[name] = None
        return result


def pose_updates(previous: dict, current: dict, *, previous_mesh_hashes=None,
                 current_mesh_hashes=None) -> dict | None:
    """返回改变的world pose（米、wxyz）；无法证明已加载几何相同时完整重载。

    mesh必须对照加载时保存的hash和本次新读的hash；重读旧路径不能代替加载身份。
    glTF等文件可引用外部几何，因此即使主文件相同也不复用其GPU几何。
    """
    if previous.keys() != current.keys():
        return None
    updates = {}
    for kind, objects in current.items():
        if objects.keys() != previous[kind].keys():
            return None
        for name, row in objects.items():
            old = previous[kind][name]
            geometry = {k: v for k, v in row.items() if k != 'pose'}
            previous_geometry = {k: v for k, v in old.items() if k != 'pose'}
            if kind == 'mesh':
                mesh_format = _mesh_format(row.get('file_path'))
                if mesh_format is None or mesh_format != _mesh_format(old.get('file_path')):
                    return None
                if previous_mesh_hashes is None or current_mesh_hashes is None:
                    return None
                loaded_hash = previous_mesh_hashes.get(name)
                if loaded_hash is None or loaded_hash != current_mesh_hashes.get(name):
                    return None
                geometry['file_path'] = previous_geometry['file_path']
            if geometry != previous_geometry:
                return None
            pose = np.asarray(row.get('pose'), dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5):
                raise ValueError(f'障碍 {name} pose 必须是有限 xyz + 单位 wxyz')
            if row['pose'] != old.get('pose'):
                updates[name] = row['pose']
    return updates


def apply_pose_updates(worlds, updates: dict, make_pose) -> None:
    """每个 world 配明确障碍集合；先核对完整集合，再更新，避免静默漏碰撞。"""
    unique = {}
    for world, expected_names in worlds:
        names = set(world.get_obstacle_names())
        if missing := set(expected_names) - names:
            raise ValueError(f'碰撞 world 缺少障碍: {sorted(missing)}')
        unique[id(world)] = (world, names)
    poses = {name: make_pose(pose) for name, pose in updates.items()}
    for world, names in unique.values():
        for name, pose in poses.items():
            if name in names:
                world.update_obstacle_pose(name, pose)
