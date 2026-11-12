"""六面长方体抓取，保持旧面朝向与 90 度局部旋转语义。"""
from __future__ import annotations
import itertools
import numpy as np
from gene_agent.knowledge.geometry import CuboidConcept
from gene_agent.knowledge.grasp.model import GraspProfile
from .common import LocalGraspCandidate, LocalGraspRejection, rotation_xyz, values

NORMALS = {"front": (1,0,0), "back": (-1,0,0), "right": (0,1,0), "left": (0,-1,0), "top": (0,0,1), "bottom": (0,0,-1)}
TANGENTS = {"front": (0,1,0), "back": (0,1,0), "right": (1,0,0), "left": (1,0,0), "top": (0,1,0), "bottom": (0,1,0)}


def generate_cuboidal_block_local(concept: CuboidConcept, profile: GraspProfile):
    if not isinstance(concept, CuboidConcept):
        raise TypeError("cuboidal_block generator 要求 CuboidConcept")
    names = ("face", "rotate_z_90", "surface_parallel_offset_m", "depth_m", "tip_x_rad")
    sequences = (values(profile.parameters, names[0], ("front",)), values(profile.parameters, names[1], (False,)), values(profile.parameters, names[2], (0.0,)), values(profile.parameters, names[3], (None,)), values(profile.parameters, names[4], (0.0,)))
    size = np.asarray(concept.size_xyz_m)
    clearance = profile.closing_width.clearance_m or 0.0
    output = []
    for indices in itertools.product(*(range(len(sequence)) for sequence in sequences)):
        face, rotate, offset, depth, tip_x = (sequence[index] for sequence, index in zip(sequences, indices))
        chosen = dict(zip(names, indices))
        normal, tangent = np.asarray(NORMALS[face], float), np.asarray(TANGENTS[face], float)
        tcp_z = -normal
        base = np.column_stack((np.cross(tangent, tcp_z), tangent, tcp_z))
        if rotate: base = base @ rotation_xyz((0.0, 0.0, np.pi / 2.0))
        base = base @ rotation_xyz((0.0, 0.0, np.pi / 2.0))
        face_thickness = float(np.dot(size, np.abs(normal)))
        penetration = float(depth) if depth is not None else min(face_thickness / 2.0, 0.02)
        offset = float(offset); tip_x = float(tip_x)
        extent = float(np.dot(size, np.abs(base[:, 0])))
        if abs(offset) > extent / 2.0 - clearance or penetration < 0.0 or penetration > face_thickness:
            output.append(LocalGraspRejection("CUBOID_OUT_OF_RANGE", "面内偏移或 depth 超出几何范围", {"offset_m": offset, "depth_m": penetration}, chosen)); continue
        rotation = base @ rotation_xyz((tip_x, 0.0, 0.0))
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = normal * face_thickness / 2.0 + base[:, 0] * offset + tcp_z * penetration
        width = float(np.dot(size, np.abs(rotation[:, 0])) + 2.0 * clearance)
        output.append(LocalGraspCandidate(pose, width, dict(zip(names, (face, bool(rotate), offset, penetration, tip_x))), chosen))
    return tuple(output)
