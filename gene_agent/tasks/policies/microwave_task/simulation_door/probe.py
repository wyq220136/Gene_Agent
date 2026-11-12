"""在 MuJoCo 副本上复核规划轨迹与活动门的名义几何接触。"""

from __future__ import annotations

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.simulation_door.execution import classify_door_contact


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def probe_planned_door_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_metadata: dict,
    *,
    group: str,
    joint_names: tuple[str, ...],
    positions_rad: np.ndarray,
    door_start_rad: float,
    door_target_rad: float,
    allowed_door_geom_names: tuple[str, ...] = (),
    allowed_robot_geom_names: tuple[str, ...] = (),
) -> dict:
    """只检查名义几何，不修改在线状态，也不推断真实力或物理成功。

    活动门按轨迹采样比例从阶段实际起始角线性插值到 Knowledge 目标角。
    真实门会有延迟、反弹及接触动力学，所以这个快速探针只能提供
    规划轨迹潜在碰撞的证据，不能代替 MuJoCo actuator 执行。
    """
    path = np.asarray(positions_rad, dtype=float)
    if path.ndim != 2 or len(path) < 2 or path.shape[1] != len(joint_names) or \
            not np.isfinite(path).all():
        raise ValueError("规划轨迹须为至少两帧、维度与关节名一致的有限矩阵")
    if not np.isfinite([door_start_rad, door_target_rad]).all():
        raise ValueError("名义门角必须有限")
    robot_root = model.body(robot_metadata["group_root_bodies"][group]).id
    door_root = model.body("microwave_door").id
    pad_names = set(robot_metadata["grippers"][group]["pad_geom_names"])
    allowed_names = set(allowed_door_geom_names)
    allowed_robot = set(allowed_robot_geom_names)
    joint_addresses = [model.joint(name).qposadr[0] for name in joint_names]
    door_address = model.joint("microwave_door_joint").qposadr[0]
    copied = mujoco.MjData(model)
    mujoco.mj_copyData(copied, model, data)
    pairs: dict[str, dict] = {}
    forbidden_samples: set[int] = set()
    pad_samples: set[int] = set()
    allowed_samples: set[int] = set()
    for sample_index, positions in enumerate(path):
        copied.qpos[joint_addresses] = positions
        fraction = sample_index / (len(path) - 1)
        copied.qpos[door_address] = ((1. - fraction) * door_start_rad
                                    + fraction * door_target_rad)
        mujoco.mj_forward(model, copied)
        for index in range(copied.ncon):
            contact = copied.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            first_body = int(model.geom_bodyid[first])
            second_body = int(model.geom_bodyid[second])
            if (_descends_from(model, first_body, robot_root)
                    and _descends_from(model, second_body, door_root)):
                robot_geom, door_geom = first, second
            elif (_descends_from(model, second_body, robot_root)
                  and _descends_from(model, first_body, door_root)):
                robot_geom, door_geom = second, first
            else:
                continue
            robot_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            door_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, door_geom)
            classification = classify_door_contact(
                robot_name, door_name, pad_names,
                tuple(allowed_names), allowed_robot,
            )
            allowed = classification != "forbidden"
            key = f"{robot_name} ↔ {door_name}"
            row = pairs.setdefault(key, {
                "classification": classification,
                "contact_point_count": 0,
                "first_sample_index": sample_index,
                "first_contact_world_m": contact.pos.tolist(),
                "max_penetration_depth_m": 0.0,
            })
            row["contact_point_count"] += 1
            row["last_sample_index"] = sample_index
            row["max_penetration_depth_m"] = max(
                row["max_penetration_depth_m"], max(0., -float(contact.dist))
            )
            if allowed:
                allowed_samples.add(sample_index)
                if classification in {"pad_target", "pad_handle"}:
                    pad_samples.add(sample_index)
            else:
                forbidden_samples.add(sample_index)
    first_pad = min(pad_samples) if pad_samples else None
    first_allowed = min(allowed_samples) if allowed_samples else None
    first_forbidden = min(forbidden_samples) if forbidden_samples else None
    allowed_kind = "pad" if first_allowed == first_pad else "gripper"
    if first_allowed is None and first_forbidden is None:
        order = "no_contact"
    elif first_allowed is None:
        order = "forbidden_only"
    elif first_forbidden is None:
        order = f"{allowed_kind}_only"
    elif first_allowed < first_forbidden:
        order = f"{allowed_kind}_before_forbidden"
    elif first_forbidden < first_allowed:
        order = f"forbidden_before_{allowed_kind}"
    else:
        order = "simultaneous"
    return {
        "status": "sampled_nominal_geometry",
        "assumption": ("门角固定在阶段实际起始角；无物理时间、控制误差或接触力"
                       if door_start_rad == door_target_rad else
                       "门角按轨迹采样比例插值；无物理时间、控制误差或接触力"),
        "sample_count": len(path),
        "door_start_rad": float(door_start_rad),
        "door_target_rad": float(door_target_rad),
        "allowed_door_geom_names": sorted(allowed_names),
        "allowed_robot_geom_names": sorted(allowed_robot),
        "pad_target_sample_count": len(pad_samples),
        "allowed_target_sample_count": len(allowed_samples),
        "forbidden_sample_count": len(forbidden_samples),
        "first_pad_target_sample_index": first_pad,
        "first_allowed_target_sample_index": first_allowed,
        "first_forbidden_sample_index": first_forbidden,
        "first_contact_order": order,
        "pairs": pairs,
    }


__all__ = ["probe_planned_door_contacts"]
