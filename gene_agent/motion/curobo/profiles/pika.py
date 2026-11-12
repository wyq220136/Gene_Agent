"""Pika 当前开口的 cuRobo 冻结配置；物理 mesh 与规划球的近似需单独审计。"""

from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import trimesh
import numpy as np
import yaml

from ..gripper import _digest, _stable_numbers, cover_bounds, validate_gripper_snapshot


ROOT = Path(__file__).resolve().parents[4]
SOURCE_CONFIG = ROOT / "assets/robots/realman/config/realman_left.yml"
MESH_ROOT = ROOT / "assets/robots/pika/meshes"
ARM_LINKS = ("base_link", *(f"link_{index}" for index in range(1, 8)))
PIKA_LINK_MESH = {
    "pika_gripper_base_link": "gripper_base_link.STL",
    "pika_gripper_left_link": "gripper_left_link.STL",
    "pika_gripper_right_link": "gripper_right_link.STL",
}


def cover_convex_mesh(mesh, *, max_cell_m=.02):
    """外接球覆盖凸包：只删除被半空间证明整格在凸包外的格子。

    每个保留格子的外接球覆盖整格，因此覆盖凸包全部内部与表面，
    并非仅覆盖抽样顶点。坐标/单位沿用 STL 的 link frame / 米。
    """
    spheres = cover_bounds(mesh.bounds, max_cell_m=max_cell_m)
    hull = mesh.convex_hull
    normals = np.asarray(hull.face_normals)
    offsets = -np.einsum('ij,ij->i', normals, hull.triangles[:, 0])
    extents = np.diff(mesh.bounds, axis=0)[0]
    counts = np.maximum(1, np.ceil(extents / max_cell_m).astype(int))
    halfsize = extents / counts / 2
    projected_halfsize = np.abs(normals) @ halfsize
    return [sphere for sphere in spheres
            if np.all(normals @ np.asarray(sphere['center']) + offsets
                      <= projected_halfsize + 1e-9)]


def _pika_config(robot_metadata: dict) -> dict:
    # 静态源内容作为 key：文件改变立即失效；每次返回副本，锁关节/SDK状态不缓存。
    meshes = tuple((link, (MESH_ROOT / filename).read_bytes()) for link, filename in PIKA_LINK_MESH.items())
    return copy.deepcopy(_cached_pika_config(str(robot_metadata["planning_urdf_path"]),
                                           SOURCE_CONFIG.read_bytes(), meshes))


@lru_cache(maxsize=4)
def _cached_pika_config(urdf_path: str, config_bytes: bytes, meshes: tuple) -> dict:
    config = yaml.safe_load(config_bytes)
    kin = config["kinematics"]
    original = kin["collision_spheres"]
    spheres = {name: copy.deepcopy(original[name]) for name in ARM_LINKS}
    for link, content in meshes:
        mesh = trimesh.load_mesh(BytesIO(content), file_type="stl", process=False)
        # 底座25 mm AABB球曾在真实mesh相隔30.7 mm时误报自碰撞。
        # 20 mm网格加保守凸包裁剪缩小空域占用；两侧手指维持既有覆盖。
        spheres[link] = (cover_convex_mesh(mesh, max_cell_m=.02)
                         if link == "pika_gripper_base_link"
                         else cover_bounds(mesh.bounds, max_cell_m=.04))
    kin["collision_spheres"] = spheres
    kin["collision_link_names"] = list(spheres)
    kin["mesh_link_names"] = list(spheres)
    kin["self_collision_buffer"] = {name: 0.0 for name in spheres}
    kin["self_collision_ignore"] = {
        name: [other for other in kin["self_collision_ignore"].get(name, []) if other in ARM_LINKS]
        for name in ARM_LINKS
    }
    # 原夹爪与腕部安装区的相邻接触需要排除；环境碰撞始终保留。
    for name in ("link_7", "pika_gripper_base_link", "pika_gripper_left_link", "pika_gripper_right_link"):
        kin["self_collision_ignore"].setdefault(name, [])
    for finger in ("pika_gripper_left_link", "pika_gripper_right_link"):
        for neighbor in ("link_7", "pika_gripper_base_link", "pika_gripper_left_link", "pika_gripper_right_link"):
            if neighbor != finger and neighbor not in kin["self_collision_ignore"][finger]:
                kin["self_collision_ignore"][finger].append(neighbor)
    for neighbor in ("pika_gripper_base_link", "pika_gripper_left_link", "pika_gripper_right_link"):
        if neighbor not in kin["self_collision_ignore"]["link_7"]:
            kin["self_collision_ignore"]["link_7"].append(neighbor)
    kin["self_collision_ignore"]["pika_gripper_base_link"].append("link_7")
    kin["urdf_path"] = urdf_path
    kin["asset_root_path"] = str(Path(kin["urdf_path"]).parent)
    return config


def export_pika_gripper_snapshot(model, data, robot_metadata: dict, group: str) -> dict:
    """记录当前 Pika 指关节；移动后必须按新关节位置重新锁定 planner。"""
    from gene_agent.robots.gripper import measure_gripper_width
    if robot_metadata.get("profile") != "realman_single_pika" or group != "left":
        raise ValueError("当前 Pika 规划快照只对应已接入的 Realman 单臂 left 配置")
    gripper = robot_metadata["grippers"][group]
    measured = measure_gripper_width(model, data, robot_metadata, group)
    config = _pika_config(robot_metadata)
    snapshot = {
        "schema_version": 1, "group": group, "mode": "pika_urdf",
        "width_m": _stable_numbers(float(measured["width_m"])),
        "joint_positions_m": _stable_numbers(measured["joint_positions_m"]),
        "source_hashes": copy.deepcopy(robot_metadata["source_hashes"]),
        "source_robot_config_sha256": hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest(),
        "merged_sphere_sha256": _digest(config["kinematics"]["collision_spheres"]),
        "limitations": [
            "Pika 底座为凸包保守网格球、手指为 AABB 网格球；局部净空已审计，非全工作空间保证。",
            "finger joints 固定为当前 MuJoCo 值，开口变化后必须重建规划快照。",
            "Pika 法兰平移、TCP 与实际接触参数尚未实物标定。",
        ],
    }
    snapshot["snapshot_id"] = _digest(snapshot)
    return snapshot


def apply_pika_gripper_snapshot(config: dict, snapshot: dict, robot_metadata: dict, group: str) -> dict:
    validate_gripper_snapshot(snapshot)
    if (snapshot["mode"] != "pika_urdf" or snapshot["group"] != group
            or snapshot["source_hashes"] != robot_metadata["source_hashes"]
            or snapshot["source_robot_config_sha256"] != hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest()):
        raise ValueError("Pika 规划快照与源资产/机器人组不一致")
    derived = _pika_config(robot_metadata)
    if _digest(derived["kinematics"]["collision_spheres"]) != snapshot["merged_sphere_sha256"]:
        raise ValueError("Pika 规划球与快照不一致")
    joints = snapshot["joint_positions_m"]
    if set(joints) != set(robot_metadata["grippers"][group]["joint_names"]):
        raise ValueError("Pika 快照缺少指关节")
    derived["kinematics"]["lock_joints"] = dict(joints)
    return derived
