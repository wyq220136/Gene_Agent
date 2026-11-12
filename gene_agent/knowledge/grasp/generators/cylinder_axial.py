"""Cylinder 轴向端面抓取公式。"""
from __future__ import annotations
import itertools
import numpy as np
from gene_agent.knowledge.geometry import CylinderConcept
from gene_agent.knowledge.grasp.model import GraspProfile
from .common import LocalGraspCandidate, rotation_xyz, values

def generate_cylinder_axial_local(concept: CylinderConcept, profile: GraspProfile) -> tuple[LocalGraspCandidate, ...]:
    """从端面生成 part-local TCP，opening 为 +X，approach 为 +Z。"""
    if not isinstance(concept, CylinderConcept):
        raise TypeError("cylinder_axial generator 要求 CylinderConcept")
    names = ("surface", "depth_m", "azimuth_rad", "tip_x_rad")
    sequences = (values(profile.parameters, names[0], ("top",)), values(profile.parameters, names[1], (concept.height_m / 2.0,)), values(profile.parameters, names[2], (0.0,)), values(profile.parameters, names[3], (0.0,)))
    clearance = profile.closing_width.clearance_m or 0.0
    output = []
    for indices in itertools.product(*(range(len(sequence)) for sequence in sequences)):
        surface, depth, azimuth, tip_x = (sequence[index] for sequence, index in zip(sequences, indices))
        if surface not in {"top", "bottom"}:
            raise ValueError("cylinder_axial surface 只支持 top 或 bottom")
        depth = float(depth)
        if depth < 0.0:
            raise ValueError("cylinder_axial depth_m 必须是非负数")
        angle = float(azimuth) % (2.0 * np.pi)
        tcp_x = np.array([np.cos(angle), np.sin(angle), 0.0])
        tcp_z = np.array([0.0, 0.0, -1.0 if surface == "top" else 1.0])
        rotation = np.column_stack((tcp_x, np.cross(tcp_z, tcp_x), tcp_z)) @ rotation_xyz((float(tip_x), 0.0, 0.0))
        pose = np.eye(4); pose[:3, :3] = rotation
        face_z = concept.height_m / 2.0 if surface == "top" else -concept.height_m / 2.0
        pose[:3, 3] = np.array([0.0, 0.0, face_z]) + tcp_z * depth
        rx, ry = concept.radius_xy_m
        width = 2.0 * np.sqrt((rx * tcp_x[0]) ** 2 + (ry * tcp_x[1]) ** 2) + 2.0 * clearance
        parameter_values = (surface, depth, float(azimuth), float(tip_x))
        output.append(LocalGraspCandidate(pose, float(width), dict(zip(names, parameter_values)), dict(zip(names, indices))))
    return tuple(output)

__all__ = ["generate_cylinder_axial_local"]
