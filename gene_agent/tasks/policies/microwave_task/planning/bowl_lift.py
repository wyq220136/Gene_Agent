"""Task1 在世界坐标系中生成抓碗后的抬升目标。"""
import numpy as np

from gene_agent.skills.library.pick.planning import make_postpick_candidates


def make_bowl_lift_candidates(actual_tcp, lift_distance_m, base_position_world):
    """各目标抬升相同净高度，并允许手臂向实际肩部位置收回。"""
    tcp = np.asarray(actual_tcp, dtype=float)
    base = np.asarray(base_position_world, dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("肩部位置必须是世界坐标系下有限的三维向量")
    vertical = make_postpick_candidates(
        tcp, tcp, lift_distance_m, translation_direction_world=(0., 0., 1.),
    )
    inward_xy = base[:2] - tcp[:2, 3]
    radius = np.linalg.norm(inward_xy)
    if radius < 1e-9:
        return vertical
    result = [vertical[0]]
    for ratio in (0.5, 1., 1.5):
        travel = min(lift_distance_m * ratio, radius)
        target = vertical[0].copy()
        target[:2, 3] += inward_xy * (travel / radius)
        if not np.array_equal(target, result[-1]):
            result.append(target)
    rotation_order = (1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 5, 6, 7, 8)
    result.extend(vertical[index] for index in rotation_order[:len(vertical) - len(result)])
    return tuple(result)
