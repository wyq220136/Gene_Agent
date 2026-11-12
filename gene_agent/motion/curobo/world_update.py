"""同几何世界的位姿同步；只处理已加载的障碍，不改变接触策略。"""
from __future__ import annotations

import numpy as np


def pose_updates(previous: dict, current: dict) -> dict | None:
    """返回改变的 world pose（米、wxyz）；几何/路径变化返回 None，需完整重载。"""
    if previous.keys() != current.keys():
        return None
    updates = {}
    for kind, objects in current.items():
        if objects.keys() != previous[kind].keys():
            return None
        for name, row in objects.items():
            old = previous[kind][name]
            if {k: v for k, v in row.items() if k != 'pose'} != {
                k: v for k, v in old.items() if k != 'pose'
            }:
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
