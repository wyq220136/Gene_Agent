"""门把手从预抓取到拉门的纯几何目标序列。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose

from .rotation import AxisSpec, OperationPath, OperationWaypoint


TASK1_HANDLE_PULL_SOURCE = (
    "AffordToolAct:visualization/knowledge/"
    "microwave_door_cycle_visualizer.py::_handle_grasp_pose_candidates"
)
_HANDLE_LOCAL_Y_ROTATION_DEGREES = (0., 5., -5., 10., -10., 15., -15.)
_HANDLE_LOCAL_X_ROTATION_DEGREES = (0., 10., -10., 20., -20., 30., -30.)


@dataclass(frozen=True, slots=True)
class HandlePullCandidate:
    """Task1 把手路线的一个局部姿态候选。"""

    candidate_id: str
    grasp_tcp_pose_world: np.ndarray
    parameters: dict
    preference_rank: int
    source: str = TASK1_HANDLE_PULL_SOURCE

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id 不能为空")
        pose = validate_pose(
            self.grasp_tcp_pose_world,
            field_name="grasp_tcp_pose_world",
        ).copy()
        rank = int(self.preference_rank)
        if rank < 0:
            raise ValueError("preference_rank 不能为负数")
        pose.setflags(write=False)
        object.__setattr__(self, "grasp_tcp_pose_world", pose)
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "preference_rank", rank)
        object.__setattr__(self, "source", str(self.source))

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "grasp_tcp_pose_world": self.grasp_tcp_pose_world.tolist(),
            "parameters": dict(self.parameters),
            "preference_rank": self.preference_rank,
            "source": self.source,
        }


def generate_handle_pull_candidates(
    grasp_tcp_pose_world: np.ndarray,
    toward_panel_direction_world: np.ndarray,
) -> tuple[HandlePullCandidate, ...]:
    """生成 Task1 把手 grasp 的有序局部 X/Y 旋转候选。

    只绕 TCP 原点改变姿态，不改变位置。对每个局部 Y 基准，若局部 X
    旋转让 TCP +Z 更朝向门板，则丢弃该候选。本函数不做 IK、碰撞
    或整条拉门路线的可达性筛选。
    """
    base = validate_pose(
        grasp_tcp_pose_world,
        field_name="grasp_tcp_pose_world",
    )
    toward_panel = np.asarray(toward_panel_direction_world, dtype=float)
    if toward_panel.shape != (3,) or not np.isfinite(toward_panel).all():
        raise ValueError("toward_panel_direction_world 必须是有限三维向量")
    length = float(np.linalg.norm(toward_panel))
    if length <= 1e-12:
        raise ValueError("toward_panel_direction_world 不能是零向量")
    toward_panel /= length
    pairs = [
        (x_degrees, y_degrees)
        for y_degrees in _HANDLE_LOCAL_Y_ROTATION_DEGREES
        for x_degrees in _HANDLE_LOCAL_X_ROTATION_DEGREES
    ]
    pairs.sort(key=lambda pair: (
        abs(pair[0]) + abs(pair[1]),
        0 if pair[0] == 0. else 1,
        abs(pair[1]),
        abs(pair[0]),
    ))
    candidates = []
    for x_degrees, y_degrees in pairs:
        y_pose = base.copy()
        y_pose[:3, :3] = base[:3, :3] @ Rotation.from_euler(
            "y", y_degrees, degrees=True,
        ).as_matrix()
        y_depth = float(y_pose[:3, 2] @ toward_panel)
        pose = y_pose.copy()
        pose[:3, :3] = y_pose[:3, :3] @ Rotation.from_euler(
            "x", x_degrees, degrees=True,
        ).as_matrix()
        if x_degrees != 0. and float(pose[:3, 2] @ toward_panel) > y_depth + 1e-9:
            continue
        rank = len(candidates)
        candidates.append(HandlePullCandidate(
            candidate_id=f"handle_pull:candidate_{rank:04d}",
            grasp_tcp_pose_world=pose,
            parameters={
                "local_x_rotation_deg": float(x_degrees),
                "local_y_rotation_deg": float(y_degrees),
            },
            preference_rank=rank,
        ))
    return tuple(candidates)


def generate_handle_pull_path(
    grasp_tcp_pose_world: np.ndarray,
    axis: AxisSpec,
    *,
    start_angle_rad: float,
    target_angle_rad: float,
    rotation_steps: int = 3,
    tip_rotation_x_scale: float = 1.,
    pregrasp_distance_m: float,
    retreat_distance_m: float = .015,
    retreat_up_m: float = 0.,
    open_width_m: float,
    close_width_m: float,
) -> tuple[OperationPath, tuple[str, ...], tuple[float, ...]]:
    """返回目标 TCP、阶段名与夹爪开口；不表示规划或物理执行成功。

    pregrasp 沿 TCP 局部 -Z 从抓取位姿后退。抓取点与闭爪点的 TCP
    相同，夹爪开口不同。随后按 Task1 的实际阶段一逻辑，把 TCP 等分为
    ``rotation_steps`` 个门轴旋转点；每点再绕当前 TCP 局部 X 轴补偿
    ``-门增量角 * tip_rotation_x_scale``。角度是门关节绝对角（rad），
    位置是 world frame（m）。这里的负号对应旧实现 ``clockwise=True``；
    当前 world 门轴本身已经按 MuJoCo 门关节正方向定义，不能再反转门轴旋转。
    """
    if axis.frame != "world":
        raise ValueError("把手拉门要求 world frame 门轴")
    values = (start_angle_rad, target_angle_rad, tip_rotation_x_scale,
              pregrasp_distance_m,
              retreat_distance_m, retreat_up_m,
              open_width_m, close_width_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("把手拉门参数必须有限")
    if not start_angle_rad < target_angle_rad or pregrasp_distance_m <= 0:
        raise ValueError("拉门目标角必须大于起始角，预抓取距离必须为正")
    if isinstance(rotation_steps, bool) or not isinstance(rotation_steps, int) or rotation_steps <= 0:
        raise ValueError("rotation_steps 必须是正整数")
    if open_width_m <= 0 or not 0 <= close_width_m < open_width_m:
        raise ValueError("夹爪开口必须满足 0 ≤ 闭爪宽度 < 预抓取宽度")
    if retreat_distance_m < 0 or retreat_up_m < 0:
        raise ValueError("撤离距离不能为负")
    grasp = validate_pose(grasp_tcp_pose_world, field_name="grasp_tcp_pose_world")
    swing_waypoints = [OperationWaypoint(float(start_angle_rad), grasp.copy())]
    total_delta = float(target_angle_rad - start_angle_rad)
    for index in range(1, rotation_steps + 1):
        delta = total_delta * index / rotation_steps
        angle = float(start_angle_rad + delta)
        world_rotation = Rotation.from_rotvec(axis.direction * delta).as_matrix()
        pose = grasp.copy()
        pose[:3, :3] = world_rotation @ grasp[:3, :3]
        pose[:3, 3] = axis.point + world_rotation @ (grasp[:3, 3] - axis.point)
        # 旧 Task1 的 clockwise=True 使局部 X 补偿使用负的门增量角。
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler(
            "x", -delta * float(tip_rotation_x_scale),
        ).as_matrix()
        pose.setflags(write=False)
        swing_waypoints.append(OperationWaypoint(angle, pose))
    swing = OperationPath(
        "door",
        np.array(axis.point, copy=True),
        np.array(axis.direction, copy=True),
        axis.source,
        float(start_angle_rad),
        float(target_angle_rad),
        tuple(swing_waypoints),
    )
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= pregrasp_distance_m * grasp[:3, 2]
    release_pose = swing.waypoints[-1].tcp_pose_world.copy()
    retreat_pose = release_pose.copy()
    # 先松爪，随后沿末端局部 -Z 离开把手，同时世界 +Z 略微抬高。
    # 这里只生成目标 pose；直线运动的碰撞、接触和门是否保持打开由 Skill 检查。
    retreat_pose[:3, 3] -= retreat_distance_m * release_pose[:3, 2]
    retreat_pose[2, 3] += retreat_up_m
    waypoints = (
        OperationWaypoint(float(start_angle_rad), pregrasp),
        OperationWaypoint(float(start_angle_rad), grasp.copy()),
        OperationWaypoint(float(start_angle_rad), grasp.copy()),
        *swing.waypoints[1:],
        OperationWaypoint(float(target_angle_rad), release_pose),
        OperationWaypoint(float(target_angle_rad), retreat_pose),
    )
    labels = ("pregrasp", "grasp", "close", *(
        f"pull_{round(math.degrees(point.angle_rad - start_angle_rad)):02d}"
        for point in swing.waypoints[1:]
    ), "release", "retreat")
    widths = (float(open_width_m), float(open_width_m),
              *(float(close_width_m) for _ in range(len(swing.waypoints))),
              float(open_width_m), float(open_width_m))
    return (OperationPath(
        "handle_pull", swing.axis_point_world, swing.axis_direction_world,
        swing.axis_source, float(start_angle_rad), float(target_angle_rad),
        waypoints,
    ), labels, widths)


__all__ = [
    "HandlePullCandidate",
    "generate_handle_pull_candidates",
    "generate_handle_pull_path",
]
