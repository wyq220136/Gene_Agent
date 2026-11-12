"""三段式把手抓取；局部 +X 为开合轴，+Z 指向把手内部。"""
from __future__ import annotations
import itertools
import numpy as np
from gene_agent.knowledge.geometry import TrifoldConcept
from gene_agent.knowledge.grasp.model import GraspProfile
from .common import LocalGraspCandidate, LocalGraspRejection, rotation_xyz, values


def generate_trifold_handle_local(concept: TrifoldConcept, profile: GraspProfile):
    if not isinstance(concept, TrifoldConcept):
        raise TypeError("trifold_handle generator 要求 TrifoldConcept")
    names = ("vertical_offset_m", "tip_x_rad", "tip_y_rad", "depth_m")
    sequences = tuple(values(profile.parameters, name, (0.0,)) for name in names)
    grip = concept.grip_segment
    size = np.asarray(grip.size_xyz_m)
    clearance = profile.closing_width.clearance_m or 0.0
    # 三段结构的连接段沿 Y 分布，抓持段离连接段的方向为 +Z。
    # 这个方向唯一确定接近面；与旧数值相同的零倾角基准为 diag(1,-1,-1)。
    base = np.diag([1.0, -1.0, -1.0])
    output = []
    for indices in itertools.product(*(range(len(sequence)) for sequence in sequences)):
        vertical, tip_x, tip_y, depth = (sequence[index] for sequence, index in zip(sequences, indices))
        vertical, tip_x, tip_y, depth = map(float, (vertical, tip_x, tip_y, depth))
        chosen = dict(zip(names, indices))
        if abs(vertical) + clearance > size[1] / 2.0 or depth < 0.0:
            output.append(LocalGraspRejection("HANDLE_OUT_OF_RANGE", "把手偏移超出长条或 depth 为负", {"vertical_offset_m": vertical, "depth_m": depth}, chosen))
            continue
        rotation = base @ rotation_xyz((0.0, tip_y, 0.0)) @ rotation_xyz((tip_x, 0.0, 0.0))
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = grip.pose_in_trifold[:3, 3] + [0.0, vertical, size[2] / 2.0 - depth]
        width = float(np.dot(size, np.abs(rotation[:, 0])) + 2.0 * clearance)
        output.append(LocalGraspCandidate(pose, width, dict(zip(names, (vertical, tip_x, tip_y, depth))), chosen))
    return tuple(output)
