"""从 MuJoCo 实际状态判断方块是否落在簸箕托盘内。"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from gene_agent.knowledge.geometry import CuboidConcept, load_concept


def evaluate_cube_in_dustpan(model, data, concept_dir: str | Path,
                             *, margin_m: float = .003,
                             max_speed_m_s: float = .025,
                             object_name: str = "cube") -> dict:
    """要求方块足迹在托盘内、底面由托盘支撑且运动基本停止。"""
    cube = load_concept(Path(concept_dir) / "cube.json", part_name="body").concept
    if not isinstance(cube, CuboidConcept):
        raise ValueError("方块 body 必须是 Cuboid Concept")
    pan = data.body("dustpan")
    obj = data.body(object_name)
    object_rotation = obj.xmat.reshape(3, 3)
    center_world = obj.xpos + object_rotation @ cube.pose_in_object[:3, 3]
    pan_rotation = pan.xmat.reshape(3, 3)
    center_local = pan_rotation.T @ (center_world - pan.xpos)
    tray = model.geom("dustpan_tray_collision")
    # 旋转后的方块足迹大于未旋转半边长，必须检查所有角点的投影包络。
    rotation_local = pan_rotation.T @ object_rotation @ cube.pose_in_object[:3, :3]
    cube_half_xy = np.abs(rotation_local[:2, :]) @ (cube.size_xyz_m / 2)
    inside = bool(np.all(np.abs(center_local[:2]) + cube_half_xy + margin_m
                         < tray.size[:2]))
    cube_joint = model.joint(f"{object_name}_free")
    velocity = np.asarray(data.qvel[cube_joint.dofadr[0]:cube_joint.dofadr[0] + 6])
    speed = float(np.linalg.norm(velocity[:3]))
    angular_speed = float(np.linalg.norm(velocity[3:]))
    tray_contact = False
    contact_rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name}
        if f"{object_name}_collision" in pair and "dustpan_tray_collision" in pair:
            tray_contact = True
            contact_rows.append({"dist_m": float(contact.dist),
                                 "position_world_m": contact.pos.tolist()})
    resting = speed <= max_speed_m_s and angular_speed <= .6
    return {"object_name": object_name,
            "success": bool(inside and tray_contact and resting),
            "projected_half_extents_dustpan_xy_m": cube_half_xy.tolist(),
            "inside_xy": inside, "tray_contact": tray_contact, "resting": resting,
            "cube_center_dustpan_m": center_local.tolist(),
            "cube_center_world_m": center_world.tolist(),
            "cube_linear_speed_m_s": speed, "cube_angular_speed_rad_s": angular_speed,
            "tray_contacts": contact_rows, "margin_m": margin_m,
            "max_speed_m_s": max_speed_m_s}
