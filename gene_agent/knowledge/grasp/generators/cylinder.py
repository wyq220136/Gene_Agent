"""椭圆柱侧面抓取，不含夹爪长度或运动规划约束。"""
from __future__ import annotations
import itertools
import numpy as np
from gene_agent.knowledge.geometry import CylinderConcept
from gene_agent.knowledge.grasp.model import GraspProfile
from .common import LocalGraspCandidate, LocalGraspRejection, rotation_xyz, values


def generate_cylinder_local(concept: CylinderConcept, profile: GraspProfile):
    if not isinstance(concept, CylinderConcept):
        raise TypeError("cylinder generator 要求 CylinderConcept")
    names = ("vertical_offset_m", "depth_m", "cylinder_rotation_rad", "tip_x_rad")
    sequences = (values(profile.parameters, names[0], (0.0,)), values(profile.parameters, names[1], (None,)), values(profile.parameters, names[2], (0.0,)), values(profile.parameters, names[3], (0.0,)))
    rx, ry = concept.radius_xy_m
    clearance = profile.closing_width.clearance_m or 0.0
    output = []
    for indices in itertools.product(*(range(len(sequence)) for sequence in sequences)):
        vertical, depth, angle, tip_x = (sequence[index] for sequence, index in zip(sequences, indices))
        vertical, angle, tip_x = float(vertical), float(angle), float(tip_x)
        chosen = dict(zip(names, indices))
        if abs(vertical) + clearance > concept.height_m / 2.0:
            output.append(LocalGraspRejection("CYLINDER_VERTICAL_OUT_OF_RANGE", "圆柱高度偏移超出截面", {"vertical_offset_m": vertical}, chosen)); continue
        a = angle % (2.0 * np.pi)
        point = np.array([rx * np.cos(a), ry * np.sin(a), vertical])
        outward = np.array([np.cos(a) / rx, np.sin(a) / ry, 0.0]); outward /= np.linalg.norm(outward)
        tcp_z = -outward
        tangent = np.array([-outward[1], outward[0], 0.0])
        old_x = np.cross(tangent, tcp_z)
        base = np.column_stack((old_x, tangent, tcp_z)) @ rotation_xyz((0.0, 0.0, np.pi / 2.0))
        rotation = base @ rotation_xyz((tip_x, 0.0, 0.0))
        # 不再依赖 gripper finger_length；显式深度仍需位于圆柱内部。
        maximum = 2.0 * float(np.dot(point[:2], outward[:2]))
        # 旧默认值把指尖放到对侧表面前的 clearance；不再按 finger_length 截断。
        penetration = float(depth) if depth is not None else maximum - clearance
        if penetration < 0.0 or penetration > maximum:
            output.append(LocalGraspRejection("CYLINDER_DEPTH_OUT_OF_RANGE", "depth 超出圆柱径向穿透范围", {"depth_m": penetration, "maximum_m": maximum}, chosen)); continue
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = point + tcp_z * penetration
        width = float(2.0 * np.sqrt((rx * rotation[0, 0]) ** 2 + (ry * rotation[1, 0]) ** 2) + 2.0 * clearance)
        output.append(LocalGraspCandidate(pose, width, dict(zip(names, (vertical, penetration, angle, tip_x))), chosen))
    return tuple(output)
