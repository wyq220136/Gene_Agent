"""少量共享 TCP 位姿偏移；只处理几何，不作可达性推断。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion.model import PoseTarget


def offset_along_tcp_z(target: PoseTarget, distance_m: float) -> PoseTarget:
    """在 world 中沿目标 TCP 的局部 Z 偏移，保持原旋转不变。"""
    matrix = np.array(target.matrix, copy=True)
    matrix[:3, 3] += matrix[:3, 2] * distance_m
    return PoseTarget(target.frame, matrix)


def legacy_local_xy_pose_candidates(
    target: PoseTarget, *, limit_deg: float = 10., step_deg: float = 2.5,
) -> tuple[tuple[float, float, PoseTarget], ...]:
    """按旧 Skill 的优先级旋转局部 X/Y，TCP 原点始终不动。

    旧版使用 ``R_target @ Rx @ Ry``；这里故意不复用门 Skill 的
    ``Rotation.from_euler('xy')``，因为同时旋转两轴时两者顺序不同。
    """
    limit, step = float(limit_deg), float(step_deg)
    if not np.isfinite(limit) or limit < 0 or not np.isfinite(step) or step <= 0:
        raise ValueError("局部姿态 fallback 需要非负有限角度上限和正步长")
    count = int(np.floor(limit / step + 1e-9))
    angles = {0.}
    for index in range(1, count + 1):
        angle = float(min(limit, index * step))
        angles.update((angle, -angle))
    pairs = sorted(
        ((x, y) for x in angles for y in angles),
        key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[0]),
                          abs(pair[1]), pair[0], pair[1]),
    )
    result = []
    for x_deg, y_deg in pairs:
        rotated = target.matrix.copy()
        rotation = (Rotation.from_euler("x", x_deg, degrees=True).as_matrix()
                    @ Rotation.from_euler("y", y_deg, degrees=True).as_matrix())
        rotated[:3, :3] = target.matrix[:3, :3] @ rotation
        result.append((x_deg, y_deg, PoseTarget(target.frame, rotated)))
    return tuple(result)
