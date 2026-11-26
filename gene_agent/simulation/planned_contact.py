"""用 MuJoCo 实际几何复核规划阶段明确允许的目标物接触。"""

from __future__ import annotations

from gene_agent._compat import strict_zip

from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from gene_agent.motion.model import JointTrajectory


def classify_target_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_metadata: Mapping,
    *,
    group: str,
    object_body: str,
    allowed_geom_names: Sequence[str],
) -> dict:
    """Classify current active-arm/target contacts without advancing physics."""
    active_root = model.body(robot_metadata["group_root_bodies"][group]).id
    target_root = model.body(object_body).id
    allowed = frozenset(str(name) for name in allowed_geom_names)

    def descendant(body_id: int, root_id: int) -> bool:
        while body_id:
            if body_id == root_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return root_id == 0

    permitted = []
    forbidden = []
    for contact in data.contact:
        geom_ids = (int(contact.geom1), int(contact.geom2))
        body_ids = tuple(int(model.geom_bodyid[geom_id]) for geom_id in geom_ids)
        if descendant(body_ids[0], target_root) and descendant(body_ids[1], active_root):
            object_index, robot_index = 0, 1
        elif descendant(body_ids[1], target_root) and descendant(body_ids[0], active_root):
            object_index, robot_index = 1, 0
        else:
            continue
        robot_geom = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_ids[robot_index],
        )
        row = {
            "robot_geom_name": robot_geom,
            "object_geom_name": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_ids[object_index],
            ),
            "penetration_m": max(0.0, -float(contact.dist)),
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
        }
        (permitted if robot_geom in allowed else forbidden).append(row)
    return {
        "success": not forbidden,
        "allowed_robot_geom_names": sorted(allowed),
        "allowed_contacts": permitted,
        "forbidden_contacts": forbidden,
        "source": "mujoco_actual_contact_geometry_during_execution",
    }


def validate_planned_target_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_metadata: Mapping,
    *,
    group: str,
    object_body: str,
    trajectory: JointTrajectory,
    gripper_joint_positions_m: Mapping[str, float],
    allowed_geom_names: Sequence[str],
) -> dict:
    """逐个规划路点检查目标物与活动机器人的接触部位。

    cuRobo 目前不能按 link×物体对放行碰撞，因此接触阶段会临时从
    规划世界排除目标物。本函数在执行前用原 MuJoCo 碰撞几何独立复核：
    只有显式列出的指垫 geom 可以碰目标，掌部、指节、腕部和机械臂都
    不可碰。检查使用 data 副本，不改变在线仿真状态；这是离散路点检查，
    不替代真实执行中的连续接触与力判定。
    """
    if not isinstance(trajectory, JointTrajectory):
        raise TypeError("trajectory 必须是 JointTrajectory")
    active_root = model.body(robot_metadata["group_root_bodies"][group]).id
    target_root = model.body(object_body).id
    allowed = frozenset(str(name) for name in allowed_geom_names)
    if not allowed:
        raise ValueError("目标接触阶段必须显式指定允许的机器人 geom")
    frozen = mujoco.MjData(model)
    mujoco.mj_copyData(frozen, model, data)
    for name, raw in gripper_joint_positions_m.items():
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"夹爪关节 {name} 位置必须有限")
        frozen.qpos[model.joint(name).qposadr[0]] = value

    def descendant(body_id: int, root_id: int) -> bool:
        while body_id:
            if body_id == root_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return root_id == 0

    allowed_count = 0
    forbidden_count = 0
    forbidden: list[dict] = []
    max_allowed_penetration = 0.0
    for index, positions in enumerate(trajectory.positions_rad):
        for name, value in strict_zip(trajectory.joint_names, positions):
            frozen.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, frozen)
        for contact in frozen.contact:
            geom_ids = (int(contact.geom1), int(contact.geom2))
            body_ids = tuple(int(model.geom_bodyid[geom_id]) for geom_id in geom_ids)
            if descendant(body_ids[0], target_root) and descendant(body_ids[1], active_root):
                object_index, robot_index = 0, 1
            elif descendant(body_ids[1], target_root) and descendant(body_ids[0], active_root):
                object_index, robot_index = 1, 0
            else:
                continue
            robot_geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_ids[robot_index])
            object_geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_ids[object_index])
            penetration = max(0.0, -float(contact.dist))
            if robot_geom in allowed:
                allowed_count += 1
                max_allowed_penetration = max(max_allowed_penetration, penetration)
            else:
                forbidden_count += 1
                if len(forbidden) < 30:
                    forbidden.append({
                        "waypoint_index": index,
                        "robot_geom_name": robot_geom,
                        "object_geom_name": object_geom,
                        "penetration_m": penetration,
                    })
    return {
        "success": forbidden_count == 0,
        "checked_waypoints": len(trajectory.positions_rad),
        "allowed_robot_geom_names": sorted(allowed),
        "allowed_contact_count": allowed_count,
        "max_allowed_penetration_m": max_allowed_penetration,
        "forbidden_contact_count": forbidden_count,
        "forbidden_contacts": forbidden,
        "source": "mujoco_contact_geometry_at_planned_waypoints",
        "limitation": "离散路点复核；执行时仍需检查实际接触、运动连续段和夹持力",
    }


def validate_planned_active_arm_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, robot_metadata: Mapping,
    *, group: str, trajectory: JointTrajectory,
    allowed_target_body: str | None = None,
    allowed_target_geom_names: Sequence[str] = (),
) -> dict:
    """用实际已启用 geom 拒绝活动臂路径穿入，保留明确允许的接触对。"""
    if not isinstance(trajectory, JointTrajectory):
        raise TypeError("trajectory 必须是 JointTrajectory")
    active_root = model.body(robot_metadata["group_root_bodies"][group]).id

    def descendant(body_id, root_id):
        while body_id:
            if body_id == root_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return root_id == 0

    active = {i for i in range(model.nbody) if descendant(i, active_root)}
    mounting = robot_metadata.get("mounting_contact_exceptions", {})
    mount_bodies = {model.body(f"{group}_{link}").id
                    for link in mounting.get("robot_links", ())}
    if not mount_bodies <= active:
        raise ValueError("安装接触允许body必须属于活动机械臂")
    mount_geoms = {model.geom(name).id for name in mounting.get("source_geom_names", ())}
    pads = {model.geom(name).id for name in allowed_target_geom_names}
    if pads and allowed_target_body is None:
        raise ValueError("允许指垫接触必须明确指定目标body")
    if any(int(model.geom_bodyid[g]) not in active for g in pads):
        raise ValueError("允许接触的指垫必须属于活动机械臂")
    target_root = model.body(allowed_target_body).id if allowed_target_body is not None else None
    frozen = mujoco.MjData(model)
    mujoco.mj_copyData(frozen, model, data)
    qaddrs = [int(model.joint(name).qposadr[0]) for name in trajectory.joint_names]
    forbidden, forbidden_count, allowed_count, margin_count = [], 0, 0, 0
    for index, positions in enumerate(trajectory.positions_rad):
        frozen.qpos[qaddrs] = positions
        mujoco.mj_forward(model, frozen)
        for contact in frozen.contact:
            g0, g1 = int(contact.geom1), int(contact.geom2)
            b0, b1 = int(model.geom_bodyid[g0]), int(model.geom_bodyid[g1])
            if b0 not in active and b1 not in active:
                continue
            distance = float(contact.dist)
            if distance >= 0.:
                margin_count += 1
                continue
            mount_allowed = ((b0 in mount_bodies and g1 in mount_geoms)
                             or (b1 in mount_bodies and g0 in mount_geoms))
            pad_allowed = target_root is not None and (
                (g0 in pads and descendant(b1, target_root))
                or (g1 in pads and descendant(b0, target_root)))
            if mount_allowed or pad_allowed:
                allowed_count += 1
                continue
            forbidden_count += 1
            if len(forbidden) < 30:
                forbidden.append({
                    "waypoint_index": index,
                    "geom_names": [model.geom(g0).name, model.geom(g1).name],
                    "body_names": [model.body(b0).name, model.body(b1).name],
                    "distance_m": distance, "penetration_m": -distance,
                    "joint_positions_rad": list(positions),
                })
    return {
        "success": forbidden_count == 0,
        "checked_waypoints": len(trajectory.positions_rad),
        "forbidden_contact_count": forbidden_count,
        "forbidden_contacts": forbidden,
        "allowed_contact_count": allowed_count,
        "nonpenetrating_margin_contact_count": margin_count,
        "mounting_contact_policy": dict(mounting),
        "allowed_target_body": allowed_target_body,
        "allowed_target_geom_names": list(allowed_target_geom_names),
        "source": "mujoco_enabled_active_arm_geometry_at_planned_waypoints",
        "rule": "negative contact.dist only; explicit mounting and target-pad pairs",
        "limitation": "离散路点几何复核；不代表连续扫掠、实际跟踪或夹持成功",
    }
