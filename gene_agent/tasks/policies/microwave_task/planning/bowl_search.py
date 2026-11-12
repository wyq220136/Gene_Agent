"""按实际抓碗后的 TCP→碗关系，筛选微波炉内放置位姿。

这里只做候选目标的纯 IK 与 MuJoCo 静态碰撞检查；连续路径、夹持、
松爪落稳和门关闭仍须由各 Skill 的真实执行验收。旧版 keypose 不直接
复用，因为它使用的夹爪和 Bowl frame 与当前仿真不一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.feedback.state import restore_state
from gene_agent.visualization.grasp_tuner.pose_ik import MuJoCoPoseIK


def _pose(position, rotation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = position
    return result


def _body_child(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _contacts(model: mujoco.MjModel, data: mujoco.MjData, *,
              first_root: int, second_root: int,
              allowed_second_geoms: set[str] = frozenset()) -> dict:
    collisions = []
    for index in range(data.ncon):
        item = data.contact[index]
        a, b = int(item.geom1), int(item.geom2)
        ab, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        if _body_child(model, ab, first_root) and _body_child(model, bb, second_root):
            first_geom, second_geom = a, b
        elif _body_child(model, bb, first_root) and _body_child(model, ab, second_root):
            first_geom, second_geom = b, a
        else:
            continue
        second_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second_geom)
        if second_name in allowed_second_geoms:
            continue
        collisions.append({
            "first_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first_geom),
            "second_geom": second_name,
            "penetration_depth_m": max(0.0, -float(item.dist)),
        })
    return {"count": len(collisions),
            "max_penetration_depth_m": max((x["penetration_depth_m"] for x in collisions),
                                           default=0.0),
            "pairs": collisions}


def search_bowl_placement(
    *,
    scene_xml: str | Path,
    robot_metadata_json: str | Path,
    picked_run: str | Path,
    bowl_center_xy_world: list[tuple[float, float]],
    bowl_yaw_deg: list[float],
    bowl_tilt_y_deg: list[float] | None = None,
    bowl_center_z_world: float = 0.121,
    bowl_radius_m: float = 0.072,
    door_angle_deg: float = 85.0,
    group: str = "right",
) -> dict:
    """生成固定门角的放碗目标筛选表，所有几何长度单位均为米。"""
    source = Path(picked_run).resolve()
    source_model = mujoco.MjModel.from_xml_path(str(source / "model/scene.xml"))
    source_data = mujoco.MjData(source_model)
    restore_state(source_model, source_data, source / "stages/held/state.npz")
    mujoco.mj_forward(source_model, source_data)
    source_robot = json.loads((source / "model/robot/robot_metadata.json").read_text(
        encoding="utf-8"))
    held_tcp = source_data.site(source_robot["tcp_sites"][group])
    held_bowl = source_data.body("bowl")
    tcp_from_bowl = np.linalg.inv(_pose(held_tcp.xpos, held_tcp.xmat)) @ _pose(
        held_bowl.xpos, held_bowl.xmat,
    )
    model = mujoco.MjModel.from_xml_path(str(Path(scene_xml).resolve()))
    data = mujoco.MjData(model)
    robot = json.loads(Path(robot_metadata_json).read_text(encoding="utf-8"))
    if tuple(robot["joint_groups"][group]) != tuple(source_robot["joint_groups"][group]):
        raise ValueError("放置场景与抓碗记录的机器人关节组不一致")
    door = model.joint("microwave_door_joint")
    angle = float(np.deg2rad(door_angle_deg))
    if not door.range[0] <= angle <= door.range[1]:
        raise ValueError("测试门角超出 MuJoCo 限位")
    # 用闭门门板的外表面作为入口平面。碗半径全部跨过该平面，
    # 才能称作放进微波炉；只看碗中心会误收悬在门口的候选。
    mujoco.mj_forward(model, data)
    panel = data.geom("microwave_door_panel_collision")
    panel_center = np.asarray(panel.xpos).copy()
    inside = np.asarray(panel.xmat).reshape(3, 3)[:, 1].copy()
    cabinet_center = np.asarray(data.body("microwave").xpos).copy()
    if np.dot(inside, cabinet_center - panel_center) < 0:
        inside = -inside
    data.qpos[door.qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    microwave_pose = _pose(data.body("microwave").xpos,
                           data.body("microwave").xmat)
    upright_rotation = data.body("bowl").xmat.reshape(3, 3).copy()
    solver = MuJoCoPoseIK(model, data, tuple(robot["joint_groups"][group]),
                          robot["tcp_sites"][group])
    arm_root = model.body(robot["group_root_bodies"][group]).id
    microwave_root = model.body("microwave").id
    bowl_root = model.body("bowl").id
    bowl_qpos = model.joint("bowl_free").qposadr[0]
    rows = []
    tilts = [0.0] if bowl_tilt_y_deg is None else list(bowl_tilt_y_deg)
    if not tilts or any(abs(float(value)) > 15.0 for value in tilts):
        raise ValueError("碗倾角候选不能为空，且碗口偏离朝上方向不得超过 15°")
    for x, y in bowl_center_xy_world:
        for yaw in bowl_yaw_deg:
            for tilt in tilts:
                desired_bowl = np.eye(4)
                desired_bowl[:3, :3] = (Rotation.from_euler("y", tilt, degrees=True).as_matrix()
                                       @ Rotation.from_euler("z", yaw, degrees=True).as_matrix()
                                       @ upright_rotation)
                desired_bowl[:3, 3] = [x, y, bowl_center_z_world]
                desired_tcp = desired_bowl @ np.linalg.inv(tcp_from_bowl)
                ik = solver.solve(desired_tcp)
                row = {
                    "bowl_center_world_m": [x, y, bowl_center_z_world],
                    "bowl_yaw_deg": yaw,
                    "bowl_tilt_y_deg": tilt,
                    "T_microwave_bowl": (np.linalg.inv(microwave_pose) @ desired_bowl).tolist(),
                    "T_microwave_tcp": (np.linalg.inv(microwave_pose) @ desired_tcp).tolist(),
                    "bowl_up_world": desired_bowl[:3, 1].tolist(),
                    "pure_ik_reachable": ik.reachable,
                    "ik_position_error_m": ik.position_error_m,
                    "ik_rotation_error_deg": ik.rotation_error_deg,
                    "joint_positions_rad": ik.joint_positions,
                }
                check_data = mujoco.MjData(model)
                mujoco.mj_copyData(check_data, model, data)
                if ik.joint_positions is not None:
                    for name, value in zip(robot["joint_groups"][group], ik.joint_positions):
                        check_data.qpos[model.joint(name).qposadr[0]] = value
                check_data.qpos[bowl_qpos:bowl_qpos + 3] = desired_bowl[:3, 3]
                q = Rotation.from_matrix(desired_bowl[:3, :3]).as_quat()
                check_data.qpos[bowl_qpos + 3:bowl_qpos + 7] = q[[3, 0, 1, 2]]
                mujoco.mj_forward(model, check_data)
                row["robot_microwave_collision"] = _contacts(
                    model, check_data, first_root=arm_root, second_root=microwave_root,
                )
                row["bowl_microwave_collision_except_turntable"] = _contacts(
                    model, check_data, first_root=bowl_root, second_root=microwave_root,
                    allowed_second_geoms={"microwave_turntable_collision"},
                )
                # 根据倾斜后的碗口平面投影计算前沿；直接减整碗半径会误判倾斜候选。
                rim_span = bowl_radius_m * np.linalg.norm([
                    np.dot(inside, desired_bowl[:3, 0]),
                    np.dot(inside, desired_bowl[:3, 2]),
                ])
                row["front_clearance_m"] = float(
                    np.dot(desired_bowl[:3, 3] - panel_center, inside) - rim_span
                )
                row["fully_inside_front"] = row["front_clearance_m"] >= 0.005
                row["static_target_pass"] = bool(
                    row["fully_inside_front"] and ik.reachable
                    and row["robot_microwave_collision"]["max_penetration_depth_m"] <= 0.002
                    and row["bowl_microwave_collision_except_turntable"]["count"] == 0
                )
                rows.append(row)
    return {
        "schema_version": 1, "claim": "目标纯 IK 与静态碰撞筛选；不是可执行放碗 Skill",
        "source_picked_run": str(source), "scene_xml": str(Path(scene_xml).resolve()),
        "door_angle_deg": door_angle_deg,
        "bowl_radius_m": bowl_radius_m,
        "closed_door_front_plane_center_world": panel_center.tolist(),
        "closed_door_inside_normal_world": inside.tolist(),
        "T_tcp_bowl_from_actual_hold": tcp_from_bowl.tolist(),
        "tested": len(rows), "static_target_pass_count": sum(x["static_target_pass"] for x in rows),
        "candidates": rows,
    }


__all__ = ["search_bowl_placement"]
