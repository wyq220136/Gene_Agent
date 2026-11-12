"""用 MuJoCo 实际几何复核规划阶段明确允许的目标物接触。"""

from __future__ import annotations

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
        for name, value in zip(trajectory.joint_names, positions, strict=True):
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
