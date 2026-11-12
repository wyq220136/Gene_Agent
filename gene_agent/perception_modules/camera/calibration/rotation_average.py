"""相机标定多帧旋转平均；旋转向量不能直接逐元素求算术均值。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def mean_rodrigues_rotation(vectors: list[np.ndarray]) -> np.ndarray:
    """以 SO(3) 上的弦距离均值处理跨 ±π 的 Rodrigues 表示。"""
    if not vectors:
        raise ValueError("至少需要一个旋转向量")
    stacked = np.asarray(vectors, dtype=float).reshape(-1, 3)
    if not np.isfinite(stacked).all():
        raise ValueError("旋转向量必须是有限数")
    return Rotation.from_rotvec(stacked).mean().as_matrix()
