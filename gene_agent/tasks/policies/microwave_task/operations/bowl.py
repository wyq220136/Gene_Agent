"""迁移旧微波炉放入/取出任务的固定 TCP 关键位姿。

这里只做旧版 ``microwave_fixed.py`` 中的矩阵组合，不选 IK、不规划轨迹。
传入的微波炉 frame 必须是 profile 记录时使用的旧资产 frame；当前校准
URDF 的 root 与它不同，须经单独几何对齐后才能用于当前资产的执行。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from scipy.spatial.transform import Rotation
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.model import PoseTarget
from gene_agent.skills.library.transfer.planning.extraction import ExtractionTargets
from gene_agent.tasks.checks import CheckResult


PROFILE_NAME = "knowledge/microwave_object_manipulation.json"


@dataclass(frozen=True)
class FixedPlaceTargets:
    pre_place_tcp: np.ndarray
    place_tcp: np.ndarray
    post_place_1_tcp: np.ndarray
    post_place_2_tcp: np.ndarray
    placed_object: np.ndarray


def object_relative_insert_targets(T_world_object: np.ndarray,
                                   T_tcp_object: np.ndarray,
                                   inward_world: np.ndarray, *,
                                   approach_distance_m: float = .16,
                                   entry_clearance_m: float = .04,
                                   finger_clearance_m: float = .06) -> dict:
    """由物体终态反算 TCP；沿通道轴拔出手指，再退到炉口外。

    抓取改变时固定 TCP 会把碗翻转。这里保持物体目标独立于抓取，
    preplace→insert 的姿态一致，避免在狭窄入口内转腕。
    """
    goal = validate_pose(T_world_object, field_name="T_world_object")
    attachment = validate_pose(T_tcp_object, field_name="T_tcp_object")
    inward = np.asarray(inward_world, dtype=float)
    if inward.shape != (3,) or not np.isfinite(inward).all() \
            or not np.isclose(np.linalg.norm(inward), 1.) \
            or not np.isfinite([approach_distance_m, entry_clearance_m, finger_clearance_m]).all() \
            or min(approach_distance_m, finger_clearance_m) <= 0 or entry_clearance_m < 0:
        raise ValueError("插入需要单位进炉方向和正有限的接近/手指退出距离")
    insert = goal @ np.linalg.inv(attachment)
    preplace, retreat_1 = insert.copy(), insert.copy()
    preplace[:3, 3] -= inward * approach_distance_m
    preplace[2, 3] += entry_clearance_m
    # 名义第一段沿炉口退出；松爪后的实际开指几何会在执行编排中按
    # 碗沿净空抬高。不能在闭指/持碗状态提前猜测开指后的指垫最低点。
    retreat_1[:3, 3] -= inward * finger_clearance_m
    # 返回已经到达过的炉外准备位置，不把拔指的水平分量再叠加一次。
    # 多退这段会改变腕部/肘部构型，可能使终点落入较大稳态跟踪偏差区。
    retreat_2 = preplace.copy()
    return dict(preplace=preplace, insert=insert,
                retreat_1=retreat_1, retreat_2=retreat_2)


def bounded_insert_tilt_for_tcp_clearance(
    T_world_object: np.ndarray,
    T_tcp_object: np.ndarray,
    inward_world: np.ndarray,
    *,
    max_tcp_table_angle_deg: float = 60.0,
    max_object_tilt_deg: float = 12.0,
) -> tuple[np.ndarray, dict]:
    """在炉外小幅预倾斜碗，使持碗夹爪进入炉腔时不过分竖直。

    Bowl 的局部 +Y 是中轴线。函数只绕 ``桌面法向 x 进炉方向`` 的
    水平轴旋转目标物体，不改变目标位置和进炉航向；因此不会在狭窄
    入口内临时转腕。倾角上限必须小于任务的 15° 稳定性判据，后者
    仍由执行检查独立保留。
    """
    goal = validate_pose(T_world_object, field_name="T_world_object")
    attachment = validate_pose(T_tcp_object, field_name="T_tcp_object")
    inward = np.asarray(inward_world, dtype=float)
    limits = np.asarray([max_tcp_table_angle_deg, max_object_tilt_deg], dtype=float)
    if (inward.shape != (3,) or not np.isfinite(inward).all()
            or not np.isclose(np.linalg.norm(inward), 1.)
            or not np.isfinite(limits).all()
            or not 0. < max_tcp_table_angle_deg < 90.
            or not 0. <= max_object_tilt_deg < 15.):
        raise ValueError("插入压平策略需要单位进炉方向、0~90° TCP 上限和小于15°的碗倾角")

    up = np.array([0., 0., 1.])
    side = np.cross(up, inward)
    side_norm = float(np.linalg.norm(side))
    if side_norm <= 1e-9:
        raise ValueError("进炉方向不能与桌面法向平行")
    side /= side_norm

    def tcp_table_angle(candidate: np.ndarray) -> float:
        tcp = candidate @ np.linalg.inv(attachment)
        return float(np.rad2deg(np.arcsin(np.clip(abs(tcp[2, 2]), 0., 1.))))

    nominal_angle = tcp_table_angle(goal)
    candidates = [(0., nominal_angle, goal.copy())]
    # 0.25° 分辨率足以表达操作知识，同时避免把求解器噪声当作目标变化。
    for magnitude in np.linspace(.25, max_object_tilt_deg,
                                 max(1, int(np.ceil(max_object_tilt_deg * 4)))):
        for sign in (-1., 1.):
            angle = float(sign * magnitude)
            candidate = goal.copy()
            candidate[:3, :3] = (
                Rotation.from_rotvec(side * np.deg2rad(angle)).as_matrix()
                @ goal[:3, :3]
            )
            candidates.append((angle, tcp_table_angle(candidate), candidate))

    # 0.25°离散采样及旋转矩阵往返会产生约1e-5°数值误差；这里只给
    # 目标生成的角度标签留1e-3°数值容差，不改变碰撞或任务判据。
    satisfying = [row for row in candidates if row[1] <= max_tcp_table_angle_deg + 1e-3]
    if satisfying:
        selected = min(satisfying, key=lambda row: (abs(row[0]), row[1]))
    else:
        selected = min(candidates, key=lambda row: (row[1], abs(row[0])))
    applied_angle, final_angle, adjusted = selected
    evidence = {
        "policy": "bounded_object_tilt_before_narrow_passage",
        "tcp_axis_table_angle_definition": "acute angle(TCP +Z, table plane)",
        "nominal_tcp_table_angle_deg": nominal_angle,
        "final_tcp_table_angle_deg": final_angle,
        "maximum_tcp_table_angle_deg": float(max_tcp_table_angle_deg),
        "applied_object_tilt_deg": applied_angle,
        "maximum_object_tilt_deg": float(max_object_tilt_deg),
        "target_met": bool(final_angle <= max_tcp_table_angle_deg + 1e-3),
    }
    return adjusted, evidence


def load_object_operation_profile(task_dir: str | Path, object_id: str) -> dict:
    """读取 task 目录内原样复制的旧操作参数，不混入 Skill 默认值。"""
    path = Path(task_dir) / PROFILE_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    objects = data.get("objects")
    if not isinstance(objects, dict) or object_id not in objects:
        raise KeyError(f"{path} 中不存在物体操作参数：{object_id}")
    profile = objects[object_id]
    if not isinstance(profile, dict):
        raise ValueError(f"{object_id} 操作参数必须是 JSON 对象")
    return profile


def _world_target(
    T_world_profile_microwave: np.ndarray,
    profile: Mapping,
    key: str,
) -> np.ndarray:
    """按旧代码的 ``microwave_pose @ microwave_from_tcp`` 组合一次。"""
    parent = validate_pose(
        T_world_profile_microwave, field_name="T_world_profile_microwave"
    )
    local = validate_pose(profile[key], field_name=key)
    return validate_pose(parent @ local, field_name=f"T_world_{key}")


def fixed_place_targets(
    T_world_profile_microwave: np.ndarray,
    profile: Mapping,
    T_tcp_object: np.ndarray,
) -> FixedPlaceTargets:
    """复现旧 ``make_microwave_insertion_plan`` 的固定四点分支。

    ``T_tcp_object`` 是抓取后由实际抓取位姿计算的 attachment；物体目标
    因此跟着本次选择的 grasp 走，不能提前写死为 profile 中的炉内物体 pose。
    """
    if not profile.get("calibration", {}).get("place", False):
        raise ValueError("place TCP 尚未标定")
    names = (
        "microwave_from_pre_place_tcp",
        "microwave_from_place_tcp",
        "microwave_from_post_place_1_tcp",
        "microwave_from_post_place_2_tcp",
    )
    if any(profile.get(name) is None for name in names):
        raise ValueError("缺少完整的固定 place keypose")
    pre, place, post_1, post_2 = (
        _world_target(T_world_profile_microwave, profile, name) for name in names
    )
    attachment = validate_pose(T_tcp_object, field_name="T_tcp_object")
    return FixedPlaceTargets(
        pre_place_tcp=pre,
        place_tcp=place,
        post_place_1_tcp=post_1,
        post_place_2_tcp=post_2,
        placed_object=validate_pose(place @ attachment, field_name="placed_object"),
    )


def fixed_extract_targets(
    T_world_profile_microwave: np.ndarray,
    profile: Mapping,
) -> tuple[np.ndarray, ...]:
    """复现旧 ``fixed_extract_waypoints`` 的五点顺序。"""
    if not profile.get("calibration", {}).get("extract", False):
        raise ValueError("extract TCP 尚未标定")
    names = (
        "microwave_from_pre_extract_1_tcp",
        "microwave_from_pre_extract_2_tcp",
        "microwave_from_extract_tcp",
        "microwave_from_post_extract_1_tcp",
        "microwave_from_post_extract_2_tcp",
    )
    if any(profile.get(name) is None for name in names):
        raise ValueError("缺少完整的固定 extract keypose")
    return tuple(
        _world_target(T_world_profile_microwave, profile, name) for name in names
    )


__all__ = [
    "FixedPlaceTargets", "fixed_extract_targets", "fixed_place_targets",
    "load_object_operation_profile", "bounded_insert_tilt_for_tcp_clearance",
]


def offset_insert_keyposes(targets: dict, T_world_microwave: np.ndarray,
                           depth_offset_m: float) -> dict:
    """沿炉体局部 +Y 调整插入深度，不改四点动作顺序或姿态。"""
    depth = float(depth_offset_m)
    microwave = np.asarray(T_world_microwave, dtype=float)
    if not np.isfinite(depth) or abs(depth) > .05 or microwave.shape != (4, 4) \
            or not np.isfinite(microwave).all():
        raise ValueError("插入深度偏移须在 ±50 mm 内，且微波炉 pose 有效")
    delta = microwave[:3, 1] * depth
    result = {}
    for name, pose in targets.items():
        shifted = np.asarray(pose, dtype=float).copy()
        if shifted.shape != (4, 4) or not np.isfinite(shifted).all():
            raise ValueError(f"无效的 {name} keypose")
        # 四点共同平移才保持旧版各段的相对位移；漏移最后的撤离点会让
        # gripper 沿新的斜向路线扫过刚放下的碗。
        shifted[:3, 3] += delta
        result[name] = shifted
    return result


def rotate_retreat_keyposes(targets: dict, rotations_local_deg: dict) -> dict:
    """仅调整撤离目标的 TCP 局部姿态，保留旧版阶段顺序和目标位置。"""
    if set(rotations_local_deg) - {"retreat_1", "retreat_2"}:
        raise ValueError("局部旋转只支持两段撤离目标")
    result = {}
    for name, value in targets.items():
        pose = np.asarray(value, dtype=float).copy()
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"无效的 {name} keypose")
        if name in rotations_local_deg:
            angles = np.asarray(rotations_local_deg[name], dtype=float)
            if angles.shape != (3,) or not np.isfinite(angles).all() \
                    or np.any(np.abs(angles) > 15.):
                raise ValueError("撤离局部旋转的各轴角度必须在 ±15° 内")
            if np.any(angles):
                pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler(
                    "xyz", angles, degrees=True).as_matrix()
        result[name] = pose
    return result


def align_table_tcp_to_surface(
    original: PoseTarget, *, current_top_z_m: float,
    reference_top_z_m: float, height_bias_m: float,
) -> PoseTarget:
    """保留旧世界 XY 和夹爪朝向，只按桌面顶高与显式物理偏差修正 Z。

    旧虚拟 Place 桌面和当前大桌面并非同一个 XY 锚点：直接套用桌面
    中心的刚体位移会把 Bowl 目标平移到微波炉附近。这里只把桌面
    **表面高度**当成物理事实，抓取和 Place Skill 的阶段逻辑不变。
    """
    values = (current_top_z_m, reference_top_z_m, height_bias_m)
    if original.frame != "world_tcp" or not all(np.isfinite(values)):
        raise ValueError("桌面高度对齐要求 world_tcp 和有限的高度参数")
    target = original.matrix.copy()
    target[2, 3] += current_top_z_m - reference_top_z_m + height_bias_m
    return PoseTarget("world_tcp", target)


def extract_profile_targets(
    world_legacy_microwave: np.ndarray, profile: dict,
) -> tuple[ExtractionTargets, PoseTarget]:
    """炉内五点乘旧 microwave frame，桌面 TCP 保持 profile 的世界坐标。"""
    keyposes = fixed_extract_targets(world_legacy_microwave, profile)
    table = validate_pose(profile["table_tcp_pose"], field_name="table_tcp_pose")
    return (
        ExtractionTargets(*(PoseTarget("world_tcp", pose) for pose in keyposes)),
        PoseTarget("world_tcp", table),
    )


LEGACY_TABLE_CONFIG = Path(__file__).resolve().parents[5] / "data/task_1/knowledge/legacy_extract_visual_table.json"


def adjust_insert_targets(geometry: FixedPlaceTargets, T_world_microwave: np.ndarray, *,
                          depth_offset_m: float = 0.,
                          retreat_rotation_local_deg=(0., 0., 0.)) -> tuple[dict, np.ndarray]:
    """仿真与真机共用放入目标修正；每次只对原始四点应用一次。"""
    original = {
        "preplace": geometry.pre_place_tcp, "insert": geometry.place_tcp,
        "retreat_1": geometry.post_place_1_tcp, "retreat_2": geometry.post_place_2_tcp,
    }
    shifted = offset_insert_keyposes(original, T_world_microwave, depth_offset_m)
    angles = np.asarray(retreat_rotation_local_deg, dtype=float)
    targets = rotate_retreat_keyposes(shifted, {"retreat_1": angles, "retreat_2": angles})
    object_goal = geometry.placed_object.copy()
    object_goal[:3, 3] += np.asarray(T_world_microwave)[:3, 1] * float(depth_offset_m)
    return targets, object_goal


def check_bowl_access(door_angle_rad: float | None, *, target_angle_rad: float,
                      tolerance_rad: float = np.deg2rad(5.)) -> CheckResult:
    """Task 层已打开范围检查；具体碰撞/接触继续由 Skill 检查。

    目标由调用方的场景配置提供。GT 和实测门角用相同规则；未知不通过。
    """
    if not np.isfinite(target_angle_rad) or not np.isfinite(tolerance_rad) or tolerance_rad < 0:
        raise ValueError("开门目标和容差必须有限，容差非负")
    evidence = {"actual_angle_rad": door_angle_rad, "target_angle_rad": float(target_angle_rad),
                "tolerance_rad": float(tolerance_rad)}
    if door_angle_rad is None or not np.isfinite(door_angle_rad):
        evidence["actual_angle_rad"] = None
        return CheckResult("unknown", "current_door_angle", evidence, "当前门角未知")
    passed = abs(door_angle_rad - target_angle_rad) <= tolerance_rad
    return CheckResult("pass" if passed else "fail", "current_door_angle", evidence,
                       "" if passed else "门未处于已打开的操作范围")
