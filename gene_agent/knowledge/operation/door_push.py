"""由门板几何和用户参数生成推门接触 TCP；不包含 IK 或运动规划。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from .rotation import AxisSpec, OperationPath, OperationWaypoint, generate_rotation_path


DOOR_PUSH_MODES = ("left_open", "left_close", "right_close", "right_open")
TASK1_DOOR_PUSH_SOURCE = (
    "AffordToolAct:visualization/knowledge/"
    "microwave_door_cycle_visualizer.py::_push_stage_candidate_sets"
)


@dataclass(frozen=True, slots=True)
class DoorPushCandidate:
    """一个 Task1 同源门面接触候选；全部几何量位于 world frame。"""

    candidate_id: str
    tcp_pose_world: np.ndarray
    surface_normal_world: np.ndarray
    door_angle_rad: float
    parameters: dict
    preference_rank: int
    source: str = TASK1_DOOR_PUSH_SOURCE

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id 不能为空")
        pose = validate_pose(self.tcp_pose_world, field_name="tcp_pose_world").copy()
        normal = np.asarray(self.surface_normal_world, dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("surface_normal_world 必须是有限三维向量")
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            raise ValueError("surface_normal_world 不能是零向量")
        angle = float(self.door_angle_rad)
        if not np.isfinite(angle):
            raise ValueError("door_angle_rad 必须是有限数")
        rank = int(self.preference_rank)
        if rank < 0:
            raise ValueError("preference_rank 不能为负数")
        parameters = dict(self.parameters)
        pose.setflags(write=False)
        normal = normal / length
        normal.setflags(write=False)
        object.__setattr__(self, "tcp_pose_world", pose)
        object.__setattr__(self, "surface_normal_world", normal)
        object.__setattr__(self, "door_angle_rad", angle)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "preference_rank", rank)
        object.__setattr__(self, "source", str(self.source))

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "tcp_pose_world": self.tcp_pose_world.tolist(),
            "surface_normal_world": self.surface_normal_world.tolist(),
            "door_angle_rad": self.door_angle_rad,
            "parameters": dict(self.parameters),
            "preference_rank": self.preference_rank,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class DoorPushCandidateSet:
    """同一个绝对门角下按 Task1 偏好排序的接触候选。"""

    door_angle_rad: float
    candidates: tuple[DoorPushCandidate, ...]
    axis_point_world: np.ndarray
    axis_direction_world: np.ndarray
    axis_source: str

    def __post_init__(self) -> None:
        angle = float(self.door_angle_rad)
        candidates = tuple(self.candidates)
        if not np.isfinite(angle) or not candidates:
            raise ValueError("候选集合必须包含有限门角和至少一个候选")
        if any(not isinstance(item, DoorPushCandidate) for item in candidates):
            raise TypeError("candidates 必须是 DoorPushCandidate")
        if any(not np.isclose(item.door_angle_rad, angle, atol=1e-12) for item in candidates):
            raise ValueError("候选门角必须与集合门角一致")
        ids = tuple(item.candidate_id for item in candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("同一门角下 candidate_id 必须唯一")
        point = np.asarray(self.axis_point_world, dtype=float)
        direction = np.asarray(self.axis_direction_world, dtype=float)
        if point.shape != (3,) or direction.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(direction).all():
            raise ValueError("候选集合门轴必须是有限三维向量")
        length = float(np.linalg.norm(direction))
        if length <= 1e-12 or not self.axis_source:
            raise ValueError("候选集合门轴方向不能为零且 source 不能为空")
        point = point.copy()
        direction = direction / length
        point.setflags(write=False)
        direction.setflags(write=False)
        object.__setattr__(self, "door_angle_rad", angle)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "axis_point_world", point)
        object.__setattr__(self, "axis_direction_world", direction)
        object.__setattr__(self, "axis_source", str(self.axis_source))

    def to_dict(self) -> dict:
        return {
            "door_angle_rad": self.door_angle_rad,
            "axis_point_world": self.axis_point_world.tolist(),
            "axis_direction_world": self.axis_direction_world.tolist(),
            "axis_source": self.axis_source,
            "candidates": [item.to_dict() for item in self.candidates],
        }


def _door_angles(start_angle_rad: float, target_angle_rad: float, step_rad: float) -> tuple[float, ...]:
    """返回包含起点和精确终点的绝对门角。"""
    start = float(start_angle_rad)
    target = float(target_angle_rad)
    step = float(step_rad)
    if not np.isfinite([start, target, step]).all() or step <= 0.:
        raise ValueError("推门角度必须有限，step_rad 必须为正数")
    delta = target - start
    count = int(math.floor(abs(delta) / step))
    sign = 1. if delta >= 0. else -1.
    values = [start + sign * step * index for index in range(count + 1)]
    if not math.isclose(values[-1], target, abs_tol=1e-12):
        values.append(target)
    else:
        values[-1] = target
    return tuple(float(value) for value in values)


def _panel_pose_at_angle(
    closed_panel_pose_world: np.ndarray,
    axis: AxisSpec,
    angle_rad: float,
) -> np.ndarray:
    """从闭门参考按绝对门角旋转一次，得到当前门板 world pose。"""
    rotation = Rotation.from_rotvec(np.asarray(axis.direction) * float(angle_rad)).as_matrix()
    panel = np.eye(4)
    panel[:3, :3] = rotation @ closed_panel_pose_world[:3, :3]
    panel[:3, 3] = np.asarray(axis.point) + rotation @ (
        closed_panel_pose_world[:3, 3] - np.asarray(axis.point)
    )
    return validate_pose(panel, field_name="panel_pose_world")


def _surface_normal_world(panel_pose_world: np.ndarray, surface: str) -> np.ndarray:
    """返回所选门面的朝外单位法向；门板局部 +Y 指向机身内部。"""
    if surface not in {"inside", "outside"}:
        raise ValueError("surface 必须为 inside 或 outside")
    panel_y = np.asarray(panel_pose_world[:3, 1], dtype=float)
    normal = panel_y if surface == "inside" else -panel_y
    return normal / np.linalg.norm(normal)


def _tip_rotation_xy_search_order(
    center_x_degrees: float,
    center_y_degrees: float,
    *,
    x_search_radius_degrees: float,
    y_search_radius_degrees: float,
    step_degrees: float = 10.,
    minimum_x_degrees: float = -60.,
    maximum_x_degrees: float = 60.,
    minimum_y_degrees: float = -90.,
    maximum_y_degrees: float = 90.,
) -> tuple[tuple[float, float], ...]:
    """复刻 Task1 tuner 的 tip X/Y 网格和确定性优先顺序。"""
    values = np.asarray([
        center_x_degrees, center_y_degrees,
        x_search_radius_degrees, y_search_radius_degrees, step_degrees,
        minimum_x_degrees, maximum_x_degrees,
        minimum_y_degrees, maximum_y_degrees,
    ], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("tip rotation 搜索参数必须有限")
    center_x, center_y, x_radius, y_radius, step, min_x, max_x, min_y, max_y = values
    if min_x >= max_x or min_y >= max_y or x_radius < 0. or y_radius < 0. or step <= 0.:
        raise ValueError("tip rotation 搜索范围无效")
    if not min_x <= center_x <= max_x or not min_y <= center_y <= max_y:
        raise ValueError("tip rotation 中心超出搜索范围")

    def axis_values(center: float, radius: float, minimum: float, maximum: float) -> tuple[float, ...]:
        first = int(np.ceil(minimum / step - 1e-12))
        last = int(np.floor(maximum / step + 1e-12))
        grid = [float(index * step) for index in range(first, last + 1)]
        samples = [float(center)]
        samples.extend(
            sample for sample in grid
            if abs(sample - center) <= radius + 1e-9
            and not np.isclose(sample, center, atol=1e-9)
        )
        return tuple(sorted(samples, key=lambda sample: (
            abs(sample - center),
            0 if sample > center else 1,
        )))

    x_values = axis_values(center_x, x_radius, min_x, max_x)
    y_values = axis_values(center_y, y_radius, min_y, max_y)
    pairs = [(float(center_x), float(center_y))]
    pairs.extend(
        (x_value, y_value)
        for x_value in x_values
        for y_value in y_values
        if not (
            np.isclose(x_value, center_x, atol=1e-9)
            and np.isclose(y_value, center_y, atol=1e-9)
        )
    )

    def key(pair: tuple[float, float]) -> tuple[float, int, int, int]:
        x_value, y_value = pair
        x_equal = np.isclose(x_value, center_x, atol=1e-9)
        y_equal = np.isclose(y_value, center_y, atol=1e-9)
        change_kind = 0 if y_equal else 1 if x_equal else 2
        return (
            abs(x_value - center_x) + abs(y_value - center_y),
            change_kind,
            0 if x_value > center_x else 1,
            0 if y_value > center_y else 1,
        )

    pairs[1:] = sorted(pairs[1:], key=key)
    return tuple((float(x_value), float(y_value)) for x_value, y_value in pairs)


def generate_mode_door_push_candidate_sets(
    closed_panel_pose_world: np.ndarray,
    *,
    panel_width_m: float,
    panel_height_m: float,
    panel_thickness_m: float,
    axis: AxisSpec,
    mode: str,
    start_angle_rad: float,
    target_angle_rad: float,
    step_rad: float,
    reference_origin_world: np.ndarray,
    overrides: dict | None = None,
) -> tuple[DoorPushCandidateSet, ...]:
    """按 Task1 的实际搜索范围生成每个门角的完整有序候选集合。

    本函数只生成 world pose 和门面法向，不做 IK、碰撞或连续轨迹筛选。
    ``closed_panel_pose_world`` 必须是 0 rad 闭门参考，不能传入已经转过门角
    的 pose；每个绝对门角只在本函数内部应用一次。
    """
    if axis.frame != "world":
        raise ValueError("Task1 门候选要求 world 门轴")
    closed = validate_pose(closed_panel_pose_world, field_name="closed_panel_pose_world")
    if mode not in DOOR_PUSH_MODES:
        raise ValueError(f"未知推门模式：{mode}")
    origin = np.asarray(reference_origin_world, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("reference_origin_world 必须是有限三维点")
    tune = dict(overrides or {})
    allowed = {
        "hinge_fraction_offset", "top_down_distance_offset_m",
        "standoff_offset_m", "tip_x_offset_deg", "tip_y_offset_deg", "surface",
    }
    if set(tune) - allowed:
        raise ValueError(f"未知推门偏移参数：{sorted(set(tune) - allowed)}")
    if tune.get("surface", "auto") not in {"auto", "inside", "outside"}:
        raise ValueError("推门表面只能为 auto、inside 或 outside")

    result = []
    # 旧 Task1 关门阶段即使最终目标是 0°/5°，候选的逐角位置公式仍固定
    # 以 10° 为偏好终点；实际角度序列继续保留调用者给出的精确终点。
    preference_target_angle = (
        float(target_angle_rad)
        if mode.endswith("_open")
        else float(np.deg2rad(10.))
    )
    for angle in _door_angles(start_angle_rad, target_angle_rad, step_rad):
        panel = _panel_pose_at_angle(closed, axis, angle)
        state = legacy_mode_parameters(
            mode,
            angle_rad=angle,
            target_angle_rad=preference_target_angle,
        )
        surface = state["surface"] if tune.get("surface", "auto") == "auto" else tune["surface"]
        top = state["top_down_distance_m"] + float(tune.get("top_down_distance_offset_m", 0.))
        standoff = state["standoff_m"] + float(tune.get("standoff_offset_m", 0.))
        if not 0. <= top <= panel_height_m or standoff < 0.:
            raise ValueError("推门高度或离门距离偏移超出当前门板范围")
        fractions = tuple(
            float(np.clip(value + float(tune.get("hinge_fraction_offset", 0.)), 0., 1.))
            for value in state["hinge_fractions"]
        )
        center_y = state["tip_y_deg"] + float(tune.get("tip_y_offset_deg", 0.))
        base = door_push_pose(
            panel,
            panel_width_m=panel_width_m,
            panel_height_m=panel_height_m,
            panel_thickness_m=panel_thickness_m,
            hinge_fraction=fractions[0],
            top_down_distance_m=top,
            standoff_m=standoff,
            tip_x_deg=0.,
            tip_y_deg=center_y,
            surface=surface,
            panel_tilt_degrees=state["panel_tilt_degrees"],
        )
        center_x = tip_rotation_x_away_from_reference(
            base,
            origin,
            maximum_degrees=40.,
        ) + float(tune.get("tip_x_offset_deg", 0.))
        tip_pairs = _tip_rotation_xy_search_order(
            center_x,
            center_y,
            x_search_radius_degrees=50.,
            y_search_radius_degrees=70. if mode.endswith("_open") else 50.,
        )
        normal = _surface_normal_world(panel, surface)
        angle_degrees = float(np.rad2deg(angle))
        candidates = []
        for tip_x, tip_y in tip_pairs:
            for fraction in fractions:
                parameters = {
                    "surface": surface,
                    "hinge_fraction": fraction,
                    "top_down_distance_m": top,
                    "standoff_m": standoff,
                    "tip_x_deg": tip_x,
                    "tip_y_deg": tip_y,
                    "panel_tilt_degrees": state["panel_tilt_degrees"],
                }
                pose = door_push_pose(
                    panel,
                    panel_width_m=panel_width_m,
                    panel_height_m=panel_height_m,
                    panel_thickness_m=panel_thickness_m,
                    **parameters,
                )
                rank = len(candidates)
                candidate_id = (
                    f"{mode}:angle_{angle_degrees:07.3f}:candidate_{rank:04d}"
                )
                candidates.append(DoorPushCandidate(
                    candidate_id=candidate_id,
                    tcp_pose_world=pose,
                    surface_normal_world=normal,
                    door_angle_rad=angle,
                    parameters=parameters,
                    preference_rank=rank,
                ))
        result.append(DoorPushCandidateSet(
            angle,
            tuple(candidates),
            np.asarray(axis.point, dtype=float),
            np.asarray(axis.direction, dtype=float),
            axis.source,
        ))
    return tuple(result)


def generate_profile_door_push_candidate_sets(
    closed_panel_pose_world: np.ndarray,
    *,
    panel_width_m: float,
    panel_height_m: float,
    panel_thickness_m: float,
    axis: AxisSpec,
    mode: str,
    start_angle_rad: float,
    target_angle_rad: float,
    surface: str,
    waypoints: tuple[dict, ...],
    source: str,
) -> tuple[DoorPushCandidateSet, ...]:
    """把已经调好的逐角门面参数还原为 Knowledge 候选集合。

    首点的参数来自 profile，但门角使用实际观测的起始角；这样把手
    拉门结束后的少量门角误差不会让推门从虚构状态开始。后续各点
    保持 profile 的绝对目标角。此入口只生成几何目标，不规划或
    放宽碰撞；原有自动候选生成器完全保留。
    """
    closed = validate_pose(closed_panel_pose_world, field_name="closed_panel_pose_world")
    if axis.frame != "world" or mode not in DOOR_PUSH_MODES:
        raise ValueError("逐角推门 profile 要求 world 门轴与有效操作模式")
    if surface not in {"inside", "outside"} or not source:
        raise ValueError("逐角推门 profile 要求明确门面和来源")
    rows = tuple(waypoints)
    required = {
        "angle_deg", "hinge_fraction", "top_down_distance_m", "standoff_m",
        "tip_x_deg", "tip_y_deg", "panel_tilt_degrees",
    }
    if len(rows) < 2 or any(not isinstance(row, dict) or set(row) != required
                            for row in rows):
        raise ValueError("逐角推门 profile 至少需要两个完整参数点")
    nominal_angles = np.deg2rad([float(row["angle_deg"]) for row in rows])
    start, target = float(start_angle_rad), float(target_angle_rad)
    if not np.isfinite([start, target, *nominal_angles]).all():
        raise ValueError("逐角推门角度必须有限")
    direction = 1. if mode.endswith("_open") else -1.
    actual_angles = np.asarray([start, *nominal_angles[1:]], dtype=float)
    if (abs(float(nominal_angles[0] - start)) > np.deg2rad(10.)
            or not np.isclose(actual_angles[-1], target, atol=1e-9)
            or np.any(direction * np.diff(actual_angles) <= 0.)):
        raise ValueError("逐角推门 profile 必须从实测起点附近单调走到目标角")
    result = []
    for index, (angle, row) in enumerate(zip(actual_angles, rows, strict=True)):
        panel = _panel_pose_at_angle(closed, axis, float(angle))
        parameters = {key: float(value) for key, value in row.items()
                      if key != "angle_deg"}
        parameters["surface"] = surface
        pose = door_push_pose(
            panel, panel_width_m=panel_width_m,
            panel_height_m=panel_height_m,
            panel_thickness_m=panel_thickness_m,
            **parameters,
        )
        candidate = DoorPushCandidate(
            candidate_id=f"{mode}:profile_{index:02d}:angle_{np.rad2deg(angle):07.3f}",
            tcp_pose_world=pose,
            surface_normal_world=_surface_normal_world(panel, surface),
            door_angle_rad=float(angle),
            parameters=parameters,
            preference_rank=0,
            source=source,
        )
        result.append(DoorPushCandidateSet(
            float(angle), (candidate,),
            np.asarray(axis.point, dtype=float),
            np.asarray(axis.direction, dtype=float), axis.source,
        ))
    return tuple(result)


def select_door_push_path(
    candidate_sets: tuple[DoorPushCandidateSet, ...],
    selected_candidate_ids: tuple[str, ...] | None = None,
) -> tuple[OperationPath, tuple[dict, ...]]:
    """把每角已选候选组装成接触路径；不对选择作 IK 或碰撞判断。"""
    sets = tuple(candidate_sets)
    if not sets:
        raise ValueError("candidate_sets 不能为空")
    if selected_candidate_ids is None:
        selected_ids = tuple(item.candidates[0].candidate_id for item in sets)
    else:
        selected_ids = tuple(str(value) for value in selected_candidate_ids)
        if len(selected_ids) != len(sets):
            raise ValueError("selected_candidate_ids 必须与候选集合数量一致")
    selected = []
    for candidate_set, candidate_id in zip(sets, selected_ids, strict=True):
        match = next(
            (item for item in candidate_set.candidates if item.candidate_id == candidate_id),
            None,
        )
        if match is None:
            raise ValueError(
                f"候选 {candidate_id!r} 不属于门角 {candidate_set.door_angle_rad:.6f}"
            )
        selected.append(match)
    first, final = sets[0], sets[-1]
    if any(
        item.axis_source != first.axis_source
        or not np.allclose(item.axis_point_world, first.axis_point_world, atol=1e-12)
        or not np.allclose(item.axis_direction_world, first.axis_direction_world, atol=1e-12)
        for item in sets[1:]
    ):
        raise ValueError("candidate_sets 必须共享同一条 world 门轴")
    waypoints = tuple(
        OperationWaypoint(item.door_angle_rad, item.tcp_pose_world)
        for item in selected
    )
    parameters = tuple({
        **item.parameters,
        "candidate_id": item.candidate_id,
        "preference_rank": item.preference_rank,
        "surface_normal_world": item.surface_normal_world.tolist(),
        "candidate_source": item.source,
    } for item in selected)
    return OperationPath(
        "door",
        first.axis_point_world,
        first.axis_direction_world,
        first.axis_source,
        first.door_angle_rad,
        final.door_angle_rad,
        waypoints,
    ), parameters


def make_door_push_precontact(
    selected_candidate: DoorPushCandidate,
    *,
    distance_m: float = .05,
) -> np.ndarray:
    """沿已选 candidate 的门面朝外法向生成 pre-contact，姿态保持不变。"""
    if not isinstance(selected_candidate, DoorPushCandidate):
        raise TypeError("selected_candidate 必须是 DoorPushCandidate")
    distance = float(distance_m)
    if not np.isfinite(distance) or distance <= 0.:
        raise ValueError("distance_m 必须是有限正数")
    pose = selected_candidate.tcp_pose_world.copy()
    pose[:3, 3] += distance * selected_candidate.surface_normal_world
    return validate_pose(pose, field_name="precontact_pose_world")


def add_door_push_transitions(
    contact_path: OperationPath,
    contact_parameters: tuple[dict, ...],
    *, pre_contact_distance_m: float = .02,
    retract_up_m: float = .02,
    retract_local_z_m: float = .03,
) -> tuple[OperationPath, tuple[str, ...], tuple[dict, ...]]:
    """按旧任务脚本的 pre-contact 和两段撤离规则包装逐角接触路径。

    接触前沿 TCP 局部 Y 轴中朝世界上方的一侧退开；末端先沿世界 +Z，
    再沿最后一个 TCP 的局部 +Z 退开。过渡点门角保持接触段的起/终角。
    """
    distances = (pre_contact_distance_m, retract_up_m, retract_local_z_m)
    if not np.isfinite(distances).all() or any(value < 0 for value in distances):
        raise ValueError("推门预接触及撤离距离必须为有限非负数")
    if len(contact_path.waypoints) != len(contact_parameters) or not contact_path.waypoints:
        raise ValueError("接触路径与逐角参数数量必须相同且非空")
    first = contact_path.waypoints[0]
    final = contact_path.waypoints[-1]
    pre = first.tcp_pose_world.copy()
    local_y = pre[:3, 1]
    pre[:3, 3] += pre_contact_distance_m * (local_y if local_y[2] >= 0 else -local_y)
    up = final.tcp_pose_world.copy()
    up[2, 3] += retract_up_m
    out = up.copy()
    out[:3, 3] += retract_local_z_m * final.tcp_pose_world[:3, 2]
    waypoints = (OperationWaypoint(first.angle_rad, pre), *contact_path.waypoints,
                 OperationWaypoint(final.angle_rad, up), OperationWaypoint(final.angle_rad, out))
    labels = ("pre_contact", "contact", *(f"push_{index:02d}" for index in range(1, len(contact_path.waypoints))),
              "retract_up", "retract_local_z")
    parameters = ({**contact_parameters[0], "phase": "pre_contact"},
                  *({**value, "phase": label} for value, label in zip(contact_parameters, labels[1:-2])),
                  {**contact_parameters[-1], "phase": "retract_up"},
                  {**contact_parameters[-1], "phase": "retract_local_z"})
    path = OperationPath(contact_path.kind, contact_path.axis_point_world,
                         contact_path.axis_direction_world, contact_path.axis_source,
                         contact_path.start_angle_rad, contact_path.target_angle_rad, waypoints)
    return path, labels, parameters


def tip_rotation_x_away_from_reference(
    base_tcp_pose_world: np.ndarray, reference_origin_world: np.ndarray,
    *, maximum_degrees: float = 40.,
) -> float:
    """继承旧 tuner 的倾角规则：让指尖局部 +Z 的水平分量远离参考原点。

    参考原点由调用者显式给出。它通常是机器人 SDK base，但纯几何
    Knowledge 不读取机器人模型，也不假定 world 原点等于 SDK base。
    """
    pose = validate_pose(base_tcp_pose_world, field_name="base_tcp_pose_world")
    origin = np.asarray(reference_origin_world, dtype=float)
    maximum = float(maximum_degrees)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("reference_origin_world 必须是有限三维点")
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("maximum_degrees 必须为有限非负数")
    away = pose[:3, 3] - origin
    away[2] = 0.
    length = float(np.linalg.norm(away))
    if length <= 1e-10 or maximum == 0:
        return 0.
    away /= length
    direction = -pose[:3, 1].copy()
    direction[2] = 0.
    return maximum * float(np.clip(direction @ away, -1., 1.))


def legacy_mode_parameters(mode: str, *, angle_rad: float, target_angle_rad: float) -> dict:
    """继承旧 door_push_mode_state 的四种策略和逐角度数值。

    这里仅返回纯几何参数；机器人侧、IK 和轨迹连续性由调用方检查。
    旧版的 progress、门宽候选、预接触退让和末段倾角均保持原公式。
    """
    if mode not in DOOR_PUSH_MODES:
        raise ValueError(f"未知推门模式：{mode}")
    if not np.isfinite([angle_rad, target_angle_rad]).all():
        raise ValueError("门角度必须为有限数")
    if mode.endswith("_open"):
        progress = float(np.clip(angle_rad / max(abs(target_angle_rad), 1e-9), 0, 1))
    else:
        progress = float(np.clip((np.pi / 2 - angle_rad) / max(np.pi / 2 - target_angle_rad, 1e-9), 0, 1))
    common = {"standoff_m": .04 * (1 - progress),
              "top_down_distance_m": .03 + .02 * progress,
              "tip_y_deg": 0., "panel_tilt_degrees": 0.}
    if mode == "left_open":
        return common | {"surface": "inside", "hinge_fractions": (.75, .875, 1.),
                         "tip_x_deg": 10., "tip_y_deg": 10.}
    if mode == "left_close":
        return common | {"surface": "outside",
                         "hinge_fractions": tuple(a + progress * (b - a) for a, b in zip((.5, .625, .75), (.375, .5, .625))),
                         "tip_x_deg": 0.}
    if mode == "right_close":
        return common | {"surface": "outside", "hinge_fractions": (.75, .875, 1.),
                         "tip_x_deg": 40.}
    fractions = (.75, .875, 1.) if progress < .5 else (.5, .625, .75)
    panel_tilt = 0. if angle_rad <= target_angle_rad * .5 else 25. * float(np.clip(
        (angle_rad - target_angle_rad * .5) / (target_angle_rad * .5), 0, 1))
    return common | {"surface": "inside", "hinge_fractions": fractions,
                     "tip_x_deg": -35. * (1 - progress),
                     "panel_tilt_degrees": panel_tilt}


def door_push_pose(
    panel_pose_world: np.ndarray,
    *,
    panel_width_m: float,
    panel_height_m: float,
    panel_thickness_m: float,
    hinge_fraction: float,
    top_down_distance_m: float,
    standoff_m: float,
    tip_x_deg: float,
    tip_y_deg: float,
    surface: str,
    panel_tilt_degrees: float = 0.,
) -> np.ndarray:
    """在门板上选一个推压点，并给出从上方接近的 TCP 世界位姿。

    当前微波炉门板的局部 +X 从门轴指向把手、+Y 指向机身内部、
    +Z 向上。`hinge_fraction` 为从门轴侧到把手侧的宽度比例；
    `top_down_distance_m` 从门板上缘向下量。接触点先到门板表面，
    `standoff_m` 再沿所选表面法向退开，以便检查预接触姿态。
    夹爪局部 +Z 默认竖直向下，局部 +X 沿门宽；两个 tip 角度
    仅在该基准姿态上旋转，不改变接触点位置。
    """
    panel = validate_pose(panel_pose_world, field_name="panel_pose_world")
    values = (panel_width_m, panel_height_m, panel_thickness_m,
              hinge_fraction, top_down_distance_m, standoff_m, tip_x_deg, tip_y_deg, panel_tilt_degrees)
    if not np.isfinite(values).all():
        raise ValueError("推门参数必须为有限数")
    if panel_width_m <= 0 or panel_height_m <= 0 or panel_thickness_m <= 0:
        raise ValueError("门板尺寸必须为正")
    if not 0 <= hinge_fraction <= 1:
        raise ValueError("hinge_fraction 必须在 [0, 1] 内")
    if not 0 <= top_down_distance_m <= panel_height_m:
        raise ValueError("top_down_distance_m 必须在门板高度内")
    if standoff_m < 0:
        raise ValueError("standoff_m 不能为负")
    if surface not in {"outside", "inside"}:
        raise ValueError("surface 必须为 outside 或 inside")

    lateral = panel[:3, 0]
    outward = -panel[:3, 1] if surface == "outside" else panel[:3, 1]
    up = panel[:3, 2]
    position = (panel[:3, 3]
                + lateral * (hinge_fraction - .5) * panel_width_m
                + up * (panel_height_m / 2 - top_down_distance_m)
                + outward * (panel_thickness_m / 2 + standoff_m))
    tcp_x = lateral
    tcp_z = np.array([0., 0., -1.])
    if panel_tilt_degrees:
        tilt = np.deg2rad(panel_tilt_degrees)
        tcp_z = np.cos(tilt) * tcp_z - np.sin(tilt) * outward
    if np.linalg.norm(tcp_z - tcp_x * np.dot(tcp_z, tcp_x)) <= 1e-10:
        # 旧版在门宽方向几乎竖直时，改用门轴方向作 TCP +X，
        # 仍保持指尖朝世界下方，避免产生退化的旋转矩阵。
        tcp_x = up - tcp_z * np.dot(up, tcp_z)
        tcp_x /= np.linalg.norm(tcp_x)
    tcp_z -= tcp_x * np.dot(tcp_z, tcp_x)
    tcp_z /= np.linalg.norm(tcp_z)
    tcp_y = np.cross(tcp_z, tcp_x)
    rotation = np.column_stack((tcp_x, tcp_y, tcp_z))
    rotation = rotation @ Rotation.from_euler("xy", [tip_x_deg, tip_y_deg], degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def generate_mode_door_push_path(
    closed_panel_pose_world: np.ndarray, *, panel_width_m: float,
    panel_height_m: float, panel_thickness_m: float, axis: AxisSpec,
    mode: str, start_angle_rad: float, target_angle_rad: float,
    step_rad: float, overrides: dict | None = None,
    selection_style: str = "top_down",
    reference_origin_world: np.ndarray | None = None,
) -> tuple[OperationPath, tuple[dict, ...]]:
    """逐门角重建旧版推门候选，并按指定的旧界面策略选中一个。

    门角取旧任务的离散步长；每帧先将门板绕显式门轴更新到当前角度，
    再按旧 mode_state 重新采样门宽位置与局部 X 倾角。``legacy_viser``
    继承旧版右臂 Viser 的动态倾角及候选顺序；``top_down`` 保留之前
    新界面的选择行为。纯 Knowledge
    不进行 IK 筛选；外部检查器可对返回的每个 TCP 单独检查。
    """
    if axis.frame != "world":
        raise ValueError("推门路径要求 world 门轴")
    closed = validate_pose(closed_panel_pose_world, field_name="closed_panel_pose_world")
    if mode not in DOOR_PUSH_MODES:
        raise ValueError(f"未知推门模式：{mode}")
    if selection_style not in {"top_down", "legacy_viser"}:
        raise ValueError("selection_style 只能为 top_down 或 legacy_viser")
    if selection_style == "legacy_viser" and reference_origin_world is None:
        raise ValueError("legacy_viser 需要显式 reference_origin_world")
    if step_rad <= 0 or not np.isfinite([start_angle_rad, target_angle_rad, step_rad]).all():
        raise ValueError("推门角度和步长必须有限，且步长为正")
    tune = dict(overrides or {})
    allowed = {"hinge_fraction_offset", "top_down_distance_offset_m", "standoff_offset_m",
               "tip_x_offset_deg", "tip_y_offset_deg", "surface"}
    if set(tune) - allowed:
        raise ValueError(f"未知推门偏移参数：{sorted(set(tune) - allowed)}")
    if tune.get("surface", "auto") not in {"auto", "inside", "outside"}:
        raise ValueError("推门表面只能为 auto、inside 或 outside")
    def at_angle(angle: float) -> tuple[np.ndarray, dict]:
        rotation = Rotation.from_rotvec(np.asarray(axis.direction) * angle).as_matrix()
        panel = np.eye(4)
        panel[:3, :3] = rotation @ closed[:3, :3]
        panel[:3, 3] = np.asarray(axis.point) + rotation @ (closed[:3, 3] - np.asarray(axis.point))
        state = legacy_mode_parameters(mode, angle_rad=angle, target_angle_rad=target_angle_rad)
        surface = state["surface"] if tune.get("surface", "auto") == "auto" else tune["surface"]
        top = state["top_down_distance_m"] + float(tune.get("top_down_distance_offset_m", 0))
        stand = state["standoff_m"] + float(tune.get("standoff_offset_m", 0))
        if not 0 <= top <= panel_height_m or stand < 0:
            raise ValueError("推门高度或离门距离偏移超出当前门板范围")
        tip_y = state["tip_y_deg"] + float(tune.get("tip_y_offset_deg", 0))
        if selection_style == "legacy_viser":
            # 旧 Viser 先用门宽中心候选和零 X 倾角求参考姿态，
            # 再按实际 TCP 相对 SDK base 的方向决定本角度的默认 X 倾角。
            # 它在右臂模式默认关闭 prefer_top_down，因此保留生成顺序。
            base = door_push_pose(
                panel, panel_width_m=panel_width_m, panel_height_m=panel_height_m,
                panel_thickness_m=panel_thickness_m,
                hinge_fraction=float(state["hinge_fractions"][1]),
                top_down_distance_m=top, standoff_m=stand, tip_x_deg=0.,
                tip_y_deg=tip_y, surface=surface,
                panel_tilt_degrees=state["panel_tilt_degrees"],
            )
            center_x = tip_rotation_x_away_from_reference(base, reference_origin_world)
            if mode in {"left_close", "right_open"}:
                fractions = state["hinge_fractions"]
            else:
                center = float(state["hinge_fractions"][1])
                fractions = tuple(sorted({float(np.clip(center + offset, 0., 1.))
                                          for offset in (-.15, 0., .15)}))
            prefer_top_down = mode in {"left_open", "left_close"}
        else:
            center_x = state["tip_x_deg"]
            fractions = state["hinge_fractions"]
            prefer_top_down = True
        candidates = []
        for fraction in fractions:
            # 偏移是在旧候选集合上做人工微调。边缘候选到门边时停住，
            # 避免用户把中心候选右移 1cm 就让整个场景进入错误状态。
            fraction = float(np.clip(fraction + float(tune.get("hinge_fraction_offset", 0)), 0., 1.))
            for tip_offset in (0., -10., 10.):
                params = {"surface": surface, "hinge_fraction": fraction,
                          "top_down_distance_m": top, "standoff_m": stand,
                          "tip_x_deg": center_x + tip_offset + float(tune.get("tip_x_offset_deg", 0)),
                          "tip_y_deg": tip_y,
                          "panel_tilt_degrees": state["panel_tilt_degrees"]}
                pose = door_push_pose(panel, panel_width_m=panel_width_m,
                                      panel_height_m=panel_height_m,
                                      panel_thickness_m=panel_thickness_m, **params)
                # 旧版 prefer_top_down=True：只按 TCP +Z 与世界向下的夹角
                # 排序，同分时保持门宽位置和 tip offset 的生成顺序。
                score = float(pose[:3, 2] @ [0., 0., -1.])
                candidates.append((score if prefer_top_down else 0., pose, params))
        _, pose, params = max(candidates, key=lambda item: item[0])
        return pose, params
    first, _ = at_angle(start_angle_rad)
    skeleton = generate_rotation_path("door", first, axis, start_angle_rad=start_angle_rad,
                                      target_angle_rad=target_angle_rad, step_rad=step_rad)
    waypoints = []
    parameters = []
    for item in skeleton.waypoints:
        pose, params = at_angle(item.angle_rad)
        waypoints.append(OperationWaypoint(item.angle_rad, pose))
        parameters.append(params)
    return OperationPath("door", skeleton.axis_point_world, skeleton.axis_direction_world,
                         skeleton.axis_source, start_angle_rad, target_angle_rad,
                         tuple(waypoints)), tuple(parameters)


__all__ = [
    "DOOR_PUSH_MODES",
    "DoorPushCandidate",
    "DoorPushCandidateSet",
    "TASK1_DOOR_PUSH_SOURCE",
    "add_door_push_transitions",
    "door_push_pose",
    "generate_mode_door_push_candidate_sets",
    "generate_profile_door_push_candidate_sets",
    "generate_mode_door_push_path",
    "legacy_mode_parameters",
    "make_door_push_precontact",
    "select_door_push_path",
    "tip_rotation_x_away_from_reference",
]
