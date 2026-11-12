"""由功能 Concept 和实际物体 pose 生成刷子抓取、推扫几何。"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import CuboidConcept, CylinderConcept, load_concept


@dataclass(frozen=True)
class SweepLayout:
    direction_world_xy: np.ndarray
    brush_start_world_xyz: np.ndarray
    brush_end_world_xyz: np.ndarray
    cube_goal_world_xyz: np.ndarray


def correct_sequence_table_clearance(candidate, grip, vertices, table_z, *, clearance_m=.001):
    """按调用方给出的物体边界向上修正目标，不运行碰撞或路径规划。

    仿真与真机只规划共用此纯几何修正，避免 Concept 尺寸低估时压入桌面。
    """
    row = copy.deepcopy(candidate)
    brush = np.asarray(row["targets"]["lower_to_contact"]) @ grip
    low = float((vertices @ brush[:3, :3].T + brush[:3, 3])[:, 2].min())
    lift = max(0., table_z + clearance_m - low)
    for target in row["targets"].values():
        target[2][3] += lift
    row["mesh_height_adjustment_m"] = lift
    row["collision_low_z_m"] = low + lift
    row["face_low_z_m"] += lift
    row["target_tcp_world"] = row["targets"]["lower_to_contact"]
    row["face_center_world_xyz"][2] += lift
    return row


def make_brush_grasp_pose(T_world_brush: np.ndarray, concept_path: str | Path,
                          *, tilt_deg: float = -20., tcp_above_handle_m: float = .009,
                          grasp_center_tcp_m=None) -> np.ndarray:
    """Pika 从上方夹持刷柄：局部 X 指向下方，局部 Y 沿闭合轴。"""
    part = load_concept(concept_path, part_name="grip_handle").concept
    if not isinstance(part, CylinderConcept):
        raise ValueError("刷柄必须是 Cylinder Concept")
    brush = np.asarray(T_world_brush, dtype=float)
    if brush.shape != (4, 4) or not np.isfinite(brush).all():
        raise ValueError("T_world_brush 必须是有限 4x4 pose")
    part_world = brush @ part.pose_in_object
    local_top_down = np.column_stack(([0., 0., -1.], [0., 1., 0.], [1., 0., 0.]))
    result = np.eye(4)
    result[:3, :3] = brush[:3, :3] @ Rotation.from_euler(
        "y", tilt_deg, degrees=True).as_matrix() @ local_top_down
    if grasp_center_tcp_m is None:
        # 保留历史几何探针的数值入口；实际机器人调用应显式传夹持中心。
        result[:3, 3] = part_world[:3, 3] + brush[:3, :3] @ [0., 0., tcp_above_handle_m]
    else:
        center = np.asarray(grasp_center_tcp_m,dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('夹持中心必须为有限 TCP 坐标三维向量')
        # T_world_TCP * p_TCP_grasp = p_world_handle；不能把操作TCP直接放到刷柄上。
        result[:3,3] = part_world[:3,3] - result[:3,:3] @ center
    return result


def make_sweep_layout(object_poses: Mapping[str, np.ndarray], concept_dir: str | Path,
                      *, target_name: str = "cube",
                      contact_gap_m: float = .005,
                      recovery_inside_pan: bool = False,
                      cube_goal_world_xyz=None,
                      goal_inset_extra_m: float = 0.) -> SweepLayout:
    """从 cube 朝向簸箕内腔求推扫轴，并使刷面在方块后方开始。"""
    concepts = Path(concept_dir)
    poses = {name: np.asarray(object_poses[name], dtype=float)
             for name in ("brush", target_name, "dustpan")}
    if any(pose.shape != (4, 4) or not np.isfinite(pose).all() for pose in poses.values()):
        raise ValueError("刷子、方块、簸箕须提供有限 4x4 pose")
    face = load_concept(concepts / "brush.json", part_name="bristle_face").concept
    cube = load_concept(concepts / "cube.json", part_name="body").concept
    tray = load_concept(concepts / "dustpan.json", part_name="receiving_tray").concept
    if not all(isinstance(part, CuboidConcept) for part in (face, cube, tray)):
        raise ValueError("刷面、方块和接收托盘必须是 Cuboid Concept")
    brush_pose, cube_pose, pan_pose = (poses[name] for name in ("brush", target_name, "dustpan"))
    cube_center = (cube_pose @ cube.pose_in_object)[:3, 3]
    tray_center = (pan_pose @ tray.pose_in_object)[:3, 3]
    opening = pan_pose[:2, 0]
    # 从入口向内留出方块实际投影半径及 16 mm 余量。
    # 推到托盘中心会让斜置的整把刷子撞侧壁；任务只要求方块完整入盘。
    cube_rotation = cube_pose[:3, :3] @ cube.pose_in_object[:3, :3]
    opening_radius = .5 * np.sum(np.abs(opening @ cube_rotation[:2, :]) * cube.size_xyz_m)
    if not np.isfinite(goal_inset_extra_m) or not 0. <= goal_inset_extra_m <= .02:
        raise ValueError("额外入盘深度须为 0–20 mm")
    tray_center[:2] += opening * (float(tray.size_xyz_m[0]) / 2 - opening_radius
                                 - .016 - goal_inset_extra_m)
    if cube_goal_world_xyz is not None:
        tray_center = np.asarray(cube_goal_world_xyz, dtype=float).copy()
        if tray_center.shape != (3,) or not np.isfinite(tray_center).all():
            raise ValueError("中间推扫目标必须是有限世界坐标 xyz")
    if recovery_inside_pan and cube_goal_world_xyz is None:
        # 已进托盘边缘的方块只沿托盘内侧轴推进，保留它当前的横向位置。
        direction = -opening / np.linalg.norm(opening)
        tray_center[:2] = (cube_center[:2] + direction *
                           np.dot(tray_center[:2] - cube_center[:2], direction))
    else:
        direction = tray_center[:2] - cube_center[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            raise ValueError("方块与簸箕托盘中心重合，无法确定扫动方向")
        direction /= norm
        # 允许斜向入盘；45° 的旧经验阈值会直接丢弃用户要求的侧向样本。
        # 是否穿侧壁由完整刷子碰撞扫掠及 cuRobo 轨迹复核决定。
        if cube_goal_world_xyz is None and np.dot(direction, -opening) <= 0.:
            raise ValueError("簸箕开口没有朝向方块")
    lateral = np.array([-direction[1], direction[0]])
    face_rotation = brush_pose[:3, :3] @ face.pose_in_object[:3, :3]
    cube_rotation = cube_pose[:3, :3] @ cube.pose_in_object[:3, :3]
    face_radius = .5 * np.sum(np.abs(direction @ face_rotation[:2, :]) * face.size_xyz_m)
    cube_radius = .5 * np.sum(np.abs(direction @ cube_rotation[:2, :]) * cube.size_xyz_m)
    face_offset = brush_pose[:3, :3] @ face.pose_in_object[:3, 3]
    lateral_offset = np.dot(face_offset[:2], lateral) * lateral
    forward_offset = float(np.dot(face_offset[:2], direction) + face_radius + cube_radius)
    start = np.asarray(brush_pose[:3, 3], dtype=float).copy()
    end = start.copy()
    start[:2] = cube_center[:2] - lateral_offset - direction * (forward_offset + contact_gap_m)
    end[:2] = tray_center[:2] - lateral_offset - direction * forward_offset
    return SweepLayout(direction.copy(), start, end, tray_center.copy())


def align_sweep_contact_candidates(candidates, current_tcp_world, object_poses,
                                   concept_dir, *, target_name="cube",
                                   recovery_inside_pan=False, cube_goal_world_xyz=None,
                                   goal_inset_extra_m=0.):
    """倾角变化后重算刷面投影半径和后方间距；保留候选离桌高度。"""
    grip = np.linalg.inv(np.asarray(current_tcp_world)) @ np.asarray(object_poses["brush"])
    face = load_concept(Path(concept_dir) / "brush.json", part_name="bristle_face").concept
    result = []
    for candidate in candidates:
        tcp = np.asarray(candidate["target_tcp_world"], dtype=float).copy()
        brush = tcp @ grip
        poses = {**object_poses, "brush": brush}
        layout = make_sweep_layout(poses, concept_dir, target_name=target_name,
                                   recovery_inside_pan=recovery_inside_pan,
                                   cube_goal_world_xyz=cube_goal_world_xyz,
                                   goal_inset_extra_m=goal_inset_extra_m)
        tcp[:2, 3] += layout.brush_start_world_xyz[:2] - brush[:2, 3]
        world_face = tcp @ grip @ face.pose_in_object
        result.append({**candidate, "target_tcp_world": tcp.tolist(),
                       "face_center_world_xyz": world_face[:3, 3].tolist(),
                       "brush_end_world_xyz": layout.brush_end_world_xyz.tolist(),
                       "contact_gap_m": .005, "xy_recomputed_after_rotation": True})
    return result


def make_sweep_pose_sequences(current_tcp_world, object_poses, concept_dir, *,
                             target_name="cube", recovery_inside_pan=False,
                             table_top_z_m=0., initial_brush_z=0., orientation_policy="grip_relative",
                             retreat_lift_m=.05, reference_brush_rotation=None,
                             cube_goal_world_xyz=None, goal_inset_extra_m=0.):
    """把手朝向与倾角作为整段操作候选；所有目标共享同一个夹持关系。"""
    tcp = np.asarray(current_tcp_world)
    brush = np.asarray(object_poses["brush"])
    grip = np.linalg.inv(tcp) @ brush
    layout = make_sweep_layout(object_poses, concept_dir, target_name=target_name,
                               recovery_inside_pan=recovery_inside_pan,
                               cube_goal_world_xyz=cube_goal_world_xyz,
                               goal_inset_extra_m=goal_inset_extra_m)
    base = tcp.copy()
    desired = layout.brush_start_world_xyz.copy(); desired[2] = initial_brush_z
    base[:3, 3] += desired - brush[:3, 3]
    if reference_brush_rotation is not None:
        # 固定抓取后参考朝向，避免每扫一块都在上一段倾角上继续叠加。
        base[:3, :3] = np.asarray(reference_brush_rotation) @ grip[:3, :3].T
    face = load_concept(Path(concept_dir) / "brush.json", part_name="bristle_face").concept
    direction = layout.direction_world_xy
    # 刷面长轴横跨扫动方向；从功能 frame 求 TCP，不继承上次接触后的倾斜。
    face_rotation = np.column_stack(([direction[1], -direction[0], 0.],
                                     [direction[0], direction[1], 0.], [0., 0., 1.]))
    result = []
    if orientation_policy not in ("grip_relative", "functional"):
        raise ValueError("未知刷面候选策略")
    yaws = (0., 180., 90., -90.) if orientation_policy == "grip_relative" else (0., 180.)
    if orientation_policy == "grip_relative":
        # 对齐矩形刷头的宽/窄推面，而不是让方块沿斜刷面的法向侧滑。
        reference_face = base[:3, :3] @ grip[:3, :3] @ face.pose_in_object[:3, :3]
        aligned = []
        for axis in (1, 0):
            angle = np.rad2deg(np.arctan2(direction[1], direction[0])
                              - np.arctan2(reference_face[1, axis], reference_face[0, axis]))
            aligned.extend((float(angle), float(angle + 180.)))
        yaws = (*aligned, *yaws)
    for yaw in yaws:
        rotated = base.copy()
        yaw_rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        if orientation_policy == "functional":
            desired_face = yaw_rotation @ face_rotation
            rotated[:3, :3] = desired_face @ face.pose_in_object[:3, :3].T @ grip[:3, :3].T
        else:
            rotated[:3, :3] = yaw_rotation @ base[:3, :3]
        candidates = make_sweep_contact_pose_candidates(
            rotated, tcp, brush, Path(concept_dir) / "brush.json",
            sweep_direction_world_xy=direction,
            table_top_z_m=table_top_z_m, tilt_degrees=(0., 10., 20.))
        if orientation_policy == "functional":
            candidates = [row for row in candidates
                          if row["tilt_axis"] in ("height_corrected", "sweep_lateral")]
        elif yaw != 0.:
            candidates = candidates[1:]
        candidates = align_sweep_contact_candidates(candidates, tcp, object_poses,
            concept_dir, target_name=target_name, recovery_inside_pan=recovery_inside_pan,
            cube_goal_world_xyz=cube_goal_world_xyz,
            goal_inset_extra_m=goal_inset_extra_m)
        for row in candidates:
            contact = np.asarray(row["target_tcp_world"])
            pre = contact.copy(); pre[2, 3] += .055
            end = contact.copy()
            end[:2, 3] += np.asarray(row["brush_end_world_xyz"])[:2] - (contact @ grip)[:2, 3]
            # 向铲口外侧斜上撤离，避免在最深端保持伸腕姿态纯竖直抬升。
            retreat_delta = np.r_[-layout.direction_world_xy * .05, retreat_lift_m]
            retreat = end.copy(); retreat[:3, 3] += retreat_delta
            result.append({**row, "yaw_delta_deg": yaw, "orientation_policy": orientation_policy,
                "retreat_lift_m": float(retreat_lift_m),
                "retreat_delta_world_m": retreat_delta.tolist(),
                "targets": {"move_behind_cube": pre.tolist(), "lower_to_contact": contact.tolist(),
                            "sweep": end.tolist(), "retreat": retreat.tolist()}})
    return result


def make_sweep_contact_pose_candidates(
    base_tcp_world: np.ndarray,
    current_tcp_world: np.ndarray,
    current_brush_world: np.ndarray,
    brush_concept_path: str | Path,
    *,
    sweep_direction_world_xy: np.ndarray,
    table_top_z_m: float,
    face_clearance_m: float = .002,
    tilt_degrees: tuple[float, ...] = (0., 5., 10., 15., 20.),
) -> list[dict]:
    """沿扫动方向的横轴轻倾刷子，同时让功能刷面最低点贴近桌面。

    返回候选几何，不宣称 IK、碰撞或实际接触可行；筛选属于 Skill。
    `base_tcp_world` 是原始下降目标，`T_tcp_brush` 来自持刷的实际状态。
    """
    poses = [np.asarray(item, dtype=float) for item in
             (base_tcp_world, current_tcp_world, current_brush_world)]
    if any(item.shape != (4, 4) or not np.isfinite(item).all()
           for item in poses):
        raise ValueError("TCP 与 brush pose 必须是有限 4x4 矩阵")
    direction = np.asarray(sweep_direction_world_xy, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("扫动方向必须是有限 XY 向量")
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8 or not np.isfinite([table_top_z_m, face_clearance_m]).all():
        raise ValueError("扫动方向和桌面高度必须有效")
    direction /= norm
    tilt_axes = {"world_x": np.array([1., 0., 0.]),
                 "world_y": np.array([0., 1., 0.]),
                 "sweep_lateral": np.array([direction[1], -direction[0], 0.])}
    face = load_concept(brush_concept_path, part_name="bristle_face").concept
    if not isinstance(face, CuboidConcept):
        raise ValueError("刷面必须是 Cuboid Concept")
    T_tcp_brush = np.linalg.inv(poses[1]) @ poses[2]
    half = .5 * np.asarray(face.size_xyz_m, dtype=float)
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1])
                        for z in (-half[2], half[2])], dtype=float)
    candidates = []
    original_face = poses[0] @ T_tcp_brush @ face.pose_in_object
    original_corners = (original_face[:3, :3] @ corners.T).T + original_face[:3, 3]
    candidates.append({
        "tilt_deg": 0., "tilt_axis": "original",
        "target_tcp_world": poses[0].tolist(),
        "face_low_z_m": float(original_corners[:, 2].min()),
        "face_center_world_xyz": original_face[:3, 3].tolist(),
        "face_normal_world": (original_face[:3, :3] @ [0., 0., -1.]).tolist(),
    })
    combinations = [("height_corrected", 0.)] + [
        (axis, float(angle)) for axis in tilt_axes
        for angle in tilt_degrees if angle != 0.]
    for axis_name, angle in combinations:
        if not np.isfinite(angle) or not 0. <= angle <= 25.:
            raise ValueError("刷面倾角须在 0–25 度内")
        target = poses[0].copy()
        tilt_axis = tilt_axes.get(axis_name, tilt_axes["world_x"])
        target[:3, :3] = (Rotation.from_rotvec(
            tilt_axis * np.deg2rad(angle)).as_matrix() @ target[:3, :3])
        T_world_face = target @ T_tcp_brush @ face.pose_in_object
        corners_world = (T_world_face[:3, :3] @ corners.T).T + T_world_face[:3, 3]
        target[2, 3] += table_top_z_m + face_clearance_m - float(corners_world[:, 2].min())
        T_world_face = target @ T_tcp_brush @ face.pose_in_object
        candidates.append({
            "tilt_deg": float(angle),
            "tilt_axis": axis_name,
            "target_tcp_world": target.tolist(),
            "face_low_z_m": float(table_top_z_m + face_clearance_m),
            "face_center_world_xyz": T_world_face[:3, 3].tolist(),
            "face_normal_world": (T_world_face[:3, :3] @ [0., 0., -1.]).tolist(),
        })
    return candidates
