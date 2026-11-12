"""清扫任务的持刷规划碰撞代理；由 MuJoCo 编译碰撞几何保守生成。"""

from __future__ import annotations

import mujoco
import numpy as np

from gene_agent.scene.rigid_geometry import (collision_geom_bounds, collision_spheres_object,
                                            collision_vertices_object)
from gene_agent.knowledge.sweep import correct_sequence_table_clearance


def _collision_geom_bounds_in_brush(model, data, geom_id, brush_id):
    """返回 mesh/box/cylinder 在刷体坐标的保守 AABB。"""
    return collision_geom_bounds(model, data, geom_id, brush_id)


def brush_collision_spheres_object(model, data, *, max_cell_m: float = .04) -> np.ndarray:
    """用局部 AABB 网格球保守覆盖刷子 mesh、box 和 cylinder。

    返回刷子 body frame 的 [x,y,z,r]；每个球外接其网格单元，
    覆盖编译后的所有碰撞顶点及凸包内部。视觉 mesh 不参与规划。
    """
    return collision_spheres_object(model, data, "brush", max_cell_m=max_cell_m)


def check_brush_pan_walls(planner, positions, grip, *, max_penetration_m=.001):
    """托盘接触白名单不包含穿入侧壁；只做 FK/碰撞检查，不求解 IK。"""
    if grip is None:
        return []
    preview = planner._copy()
    preview.eq_active[:] = 0
    invalid = []
    for index, q in enumerate(positions):
        preview.qpos[planner.qpos_addresses] = q
        mujoco.mj_forward(planner.model, preview)
        planner._set_attached(preview, grip, "brush")
        mujoco.mj_forward(planner.model, preview)
        invalid.extend(_pan_wall_contacts(planner.model, preview, index, max_penetration_m))
        if len(invalid) >= 12:
            return invalid
    return invalid


def _pan_wall_contacts(model, data, index, max_penetration_m):
    invalid = []
    for contact in data.contact:
        names = [model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
        if (any(n.startswith("brush_collision") for n in names)
                and any(n.startswith("dustpan_wall_") or n == "dustpan_back_collision" for n in names)
                and contact.dist < -max_penetration_m):
            invalid.append({"waypoint": index, "geom_names": names,
                "penetration_m": -float(contact.dist),
                "reason": "brush_penetrates_rigid_dustpan_wall"})
    return invalid


def check_observed_brush_pan_walls(planner, samples, *, max_penetration_m=.001):
    """复核实际回放；不能只检查理想刚性 attachment 的规划路径。"""
    preview = planner._copy()
    preview.eq_active[:] = 0
    invalid = []
    for index, sample in enumerate(samples):
        preview.qpos[:] = sample["qpos"]
        mujoco.mj_forward(planner.model, preview)
        invalid.extend(_pan_wall_contacts(planner.model, preview, index, max_penetration_m))
        if len(invalid) >= 12:
            break
    return invalid


def check_cartesian_brush_pan_walls(planner, start_tcp, end_tcp, grip):
    """扫动前的廉价几何排除；只适用同朝向直线，不代表IK或路径成功。"""
    from scipy.spatial.transform import Rotation
    start, end = np.asarray(start_tcp), np.asarray(end_tcp)
    if not np.allclose(start[:3, :3], end[:3, :3], atol=1e-7):
        raise ValueError("刷子几何扫掠预检要求固定朝向")
    preview = planner._copy()
    preview.eq_active[:] = 0
    addr = int(planner.model.joint("brush_free").qposadr[0])
    count = max(2, int(np.ceil(np.linalg.norm(end[:3, 3] - start[:3, 3]) / .005)) + 1)
    invalid = []
    for index, fraction in enumerate(np.linspace(0., 1., count)):
        tcp = start.copy()
        tcp[:3, 3] = (1. - fraction) * start[:3, 3] + fraction * end[:3, 3]
        brush = tcp @ grip
        preview.qpos[addr:addr + 3] = brush[:3, 3]
        preview.qpos[addr + 3:addr + 7] = Rotation.from_matrix(brush[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(planner.model, preview)
        invalid.extend(_pan_wall_contacts(planner.model, preview, index, .001))
        if len(invalid) >= 12:
            break
    return invalid


def brush_collision_vertices_object(model, data):
    """读取刷子碰撞代理的保守局部 AABB 角点，用于桌面净空。"""
    return collision_vertices_object(model, data, "brush")


def push_normal_alignment(normals, direction):
    """防侧滑看水平法向；另限制竖直分量，不能把下倾误当成侧推。"""
    normals = np.asarray(normals, dtype=float)
    direction = np.asarray(direction, dtype=float)
    horizontal_lengths = np.linalg.norm(normals[:, :2], axis=1)
    cosines = normals[:, :2] @ direction[:2] / np.maximum(horizontal_lengths, 1e-12)
    horizontal_cosine = float(np.mean(cosines))
    max_vertical = float(np.max(np.abs(normals[:, 2])))
    return {"success": bool(horizontal_cosine >= .97 and max_vertical <= .5
                            and np.all(horizontal_lengths > 1e-6)),
            "alignment_cosine": float(np.mean(normals @ direction)),
            "horizontal_alignment_cosine": horizontal_cosine,
            "max_abs_vertical_normal": max_vertical,
            "horizontal_cosine_min": .97, "vertical_normal_abs_max": .5}


def inspect_push_contact(planner, contact_tcp, end_tcp, grip, target_name):
    """沿预定推向探测首次真实凸包接触法向，不推进物理状态或求IK。"""
    from scipy.spatial.transform import Rotation
    start, end = np.asarray(contact_tcp), np.asarray(end_tcp)
    direction = end[:3, 3] - start[:3, 3]
    direction[2] = 0.
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return {"success": False, "reason": "zero_sweep_distance"}
    direction /= norm
    preview = planner._copy()
    preview.eq_active[:] = 0
    addr = int(planner.model.joint("brush_free").qposadr[0])
    probe_length = min(.03, float(norm))
    for offset in np.linspace(0., probe_length, max(2, int(np.ceil(probe_length / .001)) + 1)):
        tcp = start.copy(); tcp[:3, 3] += direction * offset
        brush = tcp @ grip
        preview.qpos[addr:addr + 3] = brush[:3, 3]
        preview.qpos[addr + 3:addr + 7] = Rotation.from_matrix(brush[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(planner.model, preview)
        normals = []
        for contact in preview.contact:
            names = [planner.model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
            if f"{target_name}_collision" not in names or not any(n.startswith("brush_collision") for n in names):
                continue
            if contact.dist > .0005:
                continue
            normal = np.array(contact.frame[:3])
            if names[1].startswith("brush_collision"):
                normal *= -1
            normals.append(normal)
        if normals:
            alignment = push_normal_alignment(normals, direction)
            has_push_distance = float(norm - offset) >= .002
            accepted = alignment["success"] and has_push_distance
            return {**alignment, "success": accepted,
                    "remaining_push_distance_m": float(norm - offset),
                    "probe_translation_m": float(offset),
                    "contact_normals_world": [n.tolist() for n in normals],
                    "reason": (None if accepted else "insufficient_push_distance" if not has_push_distance
                               else "push_face_not_aligned")}
    return {"success": False, "reason": "no_brush_cube_contact_before_probe_end"}
