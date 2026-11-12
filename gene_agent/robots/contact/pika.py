"""Pika 原始指端 mesh 在内侧 site 邻域的双侧实际接触检查。"""

from __future__ import annotations

import numpy as np


FINGER_BODIES = {"left": "pika_gripper_left_link", "right": "pika_gripper_right_link"}


def classify_pika_brush_grasp(contacts: list[dict], inner_sites_world: dict[str, np.ndarray],
                              handle_center_world, *, site_radius_m: float = .03,
                              handle_radius_m: float = .045,
                              object_name: str = "brush") -> dict:
    """只接受靠近指端内侧 site 且落在刷柄邻域的双指 mesh 接触。"""
    handle = np.asarray(handle_center_world, dtype=float)
    sites = {side: np.asarray(inner_sites_world[side], dtype=float)
             for side in FINGER_BODIES}
    if handle.shape != (3,) or not np.isfinite(handle).all() or any(
            site.shape != (3,) or not np.isfinite(site).all() for site in sites.values()):
        raise ValueError("抓取接触所用 site 和刷柄中心必须是有限世界坐标")
    sides = {side: False for side in FINGER_BODIES}
    relevant, undesired = [], []
    for row in contacts:
        body_names = set(row.get("body_names", ()))
        if object_name not in body_names:
            continue
        finger_side = next((side for side, name in FINGER_BODIES.items()
                            if name in body_names), None)
        if finger_side is None:
            if any(name.startswith("pika_") for name in body_names):
                undesired.append(row)
            continue
        relevant.append(row)
        position = np.asarray(row.get("position_world_m"), dtype=float)
        if (position.shape != (3,) or not np.isfinite(position).all()
                or np.linalg.norm(position - sites[finger_side]) > site_radius_m
                or np.linalg.norm(position - handle) > handle_radius_m
                or float(row.get("dist_m", 1.)) > .001):
            undesired.append(row)
        else:
            sides[finger_side] = True
    bilateral = all(sides.values())
    return {"success": bilateral and not undesired,
            "bilateral_pad_contact": bilateral and not undesired,
            "bilateral_inner_region_contact": bilateral,
            "contact_mode": "pika_inner_finger_mesh_region",
            "site_radius_m": site_radius_m, "handle_radius_m": handle_radius_m,
            "pad_sides": {"positive": sides["left"], "negative": sides["right"]},
            "contact_count": len(relevant),
            "undesired_gripper_contacts": undesired, "contacts": relevant}
