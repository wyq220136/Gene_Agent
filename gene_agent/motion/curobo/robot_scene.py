"""快速模式只准备机器人保护代理；不转换普通场景几何或生成 mesh 文件。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np


def export_robot_only_snapshot(model, data, robot: dict, group: str, output_dir: Path) -> dict:
    from .scene import _descendant, _pose_list
    started = time.perf_counter()
    if group not in robot["joint_groups"]:
        raise ValueError(f"未知 active_group: {group}")
    base_world = np.linalg.inv(np.asarray(robot["transforms"][group]["world_from_urdf_base"]))
    other_groups = sorted(set(robot["joint_groups"]) - {group})
    # 模型构建器显式声明本体；不按任意物体名称猜测机器人所属关系。
    roots = [*robot.get("collision_protection_body_names", ()),
             *(robot["group_root_bodies"][g] for g in other_groups)]
    root_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in roots]
    if any(body < 0 for body in root_ids):
        raise ValueError("机器人保护 body 缺失，不能静默略过")
    scene, records = {"cuboid": {}, "mesh": {}}, []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if not any(_descendant(model, body, root) for root in root_ids):
            continue
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        kind = int(model.geom_type[geom])
        size = model.geom_size[geom]
        center = np.zeros(3)
        approximation = "robot_protection_conservative_local_box"
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[first:first+count]
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            center, half = (lo+hi)/2, (hi-lo)/2
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            half, approximation = size.copy(), "original_robot_box"
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.full(3, size[0])
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            half = np.array([size[0], size[0], size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half = size.copy()
        else:
            raise ValueError(f"不支持的机器人保护几何：{kind}")
        margin = float(model.geom_margin[geom])
        if margin < 0 or model.geom_gap[geom] != 0:
            raise ValueError("机器人保护代理不支持负 margin 或非零 gap")
        # 显式 pair 的正 margin 也必须保留。
        for pair in range(model.npair):
            if geom in (model.pair_geom1[pair], model.pair_geom2[pair]):
                if model.pair_gap[pair] != 0 or model.pair_margin[pair] < 0:
                    raise ValueError("机器人保护 contact pair 含不支持的 gap/margin")
                margin = max(margin, float(model.pair_margin[pair]))
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[geom].reshape(3,3)
        transform[:3, 3] = data.geom_xpos[geom] + transform[:3, :3] @ center
        name = f"robot_geom_{geom}"
        scene["cuboid"][name] = {"dims": (2*np.maximum(half+margin, 1e-8)).tolist(),
                                   "pose": _pose_list(base_world @ transform)}
        records.append({"obstacle_name": name, "geom_id": geom,
                        "geom_name": model.geom(geom).name, "body_name": model.body(body).name,
                        "export_kind": "cuboid", "approximation": approximation})
    gripper = None
    gripper_mode = robot.get("grippers", {}).get(group, {}).get("mode")
    if gripper_mode == "parallel_slide_proxy":
        from .gripper import export_gripper_snapshot
        gripper = export_gripper_snapshot(model, data, robot, group)
    elif gripper_mode == "pika_urdf":
        from .profiles.pika import export_pika_gripper_snapshot
        gripper = export_pika_gripper_snapshot(model, data, robot, group)
    exception = robot.get("mounting_contact_exceptions", {})
    names = [r["obstacle_name"] for r in records if r["geom_name"] in exception.get("source_geom_names", ())]
    policy = {"robot_links": exception.get("robot_links", []) if names else [], "obstacle_names": names}
    scope = "robot_only：原机器人/夹爪自碰撞 + 本体及冻结另一臂保护；无场景障碍和规划 attachment"
    payload = {"scene_model": scene, "gripper_snapshot": gripper,
               "mounting_contact_policy": policy, "collision_mode": "robot_only",
               "base_from_world": base_world.tolist(),
               "robot_joints": {g: [float(data.qpos[model.joint(n).qposadr[0]]) for n in js]
                                for g, js in robot["joint_groups"].items()}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = {**payload, "snapshot_sha256": digest, "scope": scope, "records": records,
              "excluded": [], "transferred_objects": {}, "active_group": group,
              "active_group_root_body": robot['group_root_bodies'][group],
              "disabled_body_names": [], "moving_body_names": [], "static_internal_contact_pairs": [],
              "world_frame": robot["world_frame"], "other_arm_frozen": True, "frozen_groups": other_groups,
              "qpos_snapshot": data.qpos.tolist(), "simulation_time_s": float(data.time),
              "robot_protection_roots": roots, "environment_geometry_exported": False,
              "limitations": ["另一臂非 box 几何使用保守局部包围盒，可能比 mesh 更早拒绝目标；不检查桌面/物体。"],
              "preparation_duration_s": time.perf_counter()-started}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"collision_snapshot.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result
