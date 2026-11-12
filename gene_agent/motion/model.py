"""与具体 planner 无关的同步运动规划数据契约。

本模块不能导入 cuRobo、MuJoCo 或显示工具。它只定义一次规划请求和结果必须
携带的事实，使 Skill、反馈记录和不同 motion backend 使用同一套语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping, Sequence

import numpy as np


class PlannerMode(str, Enum):
    """默认开启CUDA Graph；development供显式诊断使用，保留旧枚举名兼容记录。"""

    DEVELOPMENT = "development"
    CUDA_GRAPH_BENCHMARK = "cuda_graph_benchmark"


class MotionOperation(str, Enum):
    """一次 motion 请求的语义。"""

    IK = "ik"
    PLAN = "plan"
    LINEAR_TCP = "linear_tcp"
    PIVOT_TCP = "pivot_tcp"


class MotionFailureKind(str, Enum):
    """跨后端稳定的失败类别；原始 backend 状态另行保留。"""

    INVALID_REQUEST = "invalid_request"
    START_COLLISION = "start_collision"
    GOAL_IK_NO_SOLUTION = "goal_ik_no_solution"
    GOAL_COLLISION = "goal_collision"
    PATH_NOT_FOUND = "path_not_found"
    PATH_COLLISION = "path_collision"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT = "timeout"
    BACKEND_ERROR = "backend_error"


def _nonempty(value: str, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} 不能为空")
    return result


def _joint_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(_nonempty(value, name="joint name") for value in values)
    if not names:
        raise ValueError("joint_names 不能为空")
    if len(names) != len(set(names)):
        raise ValueError("joint_names 不能重复")
    return names


def _finite_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是一维有限数组")
    return tuple(float(value) for value in array)


def _rigid_transform(value: Any, *, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须是有限 4x4 矩阵")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} 最后一行必须为 [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} 旋转矩阵必须正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} 旋转矩阵行列式必须为 1")
    matrix.setflags(write=False)
    return matrix


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须可安全写入 JSON") from error
    return result


@dataclass(frozen=True)
class JointState:
    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        names = _joint_names(self.joint_names)
        positions = _finite_vector(self.positions_rad, name="positions_rad")
        if len(names) != len(positions):
            raise ValueError("joint_names 与 positions_rad 数量必须一致")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions_rad", positions)

    def to_dict(self) -> dict[str, Any]:
        return {"joint_names": list(self.joint_names), "positions_rad": list(self.positions_rad)}


@dataclass(frozen=True)
class PoseTarget:
    frame: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _nonempty(self.frame, name="target frame"))
        object.__setattr__(self, "matrix", _rigid_transform(self.matrix, name="target pose"))

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "matrix": self.matrix.tolist()}


@dataclass(frozen=True)
class PlanningSceneRef:
    revision: int
    snapshot_sha256: str
    scope: str

    def __post_init__(self) -> None:
        revision = int(self.revision)
        if revision < 0:
            raise ValueError("scene revision 必须是非负整数")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(
            self, "snapshot_sha256", _nonempty(self.snapshot_sha256, name="scene snapshot hash")
        )
        object.__setattr__(self, "scope", _nonempty(self.scope, name="scene scope"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "snapshot_sha256": self.snapshot_sha256,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class CollisionAllowance:
    """某个 Skill 阶段允许的一对预期接触。"""

    first: str
    second: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _nonempty(self.first, name="碰撞允许项 first"))
        object.__setattr__(self, "second", _nonempty(self.second, name="碰撞允许项 second"))
        object.__setattr__(self, "reason", _nonempty(self.reason, name="碰撞允许原因"))
        if self.first == self.second:
            raise ValueError("碰撞允许项必须指向两个不同对象或 link")

    def to_dict(self) -> dict[str, str]:
        return {"first": self.first, "second": self.second, "reason": self.reason}


@dataclass(frozen=True)
class CollisionPolicy:
    policy_id: str = "strict"
    allowances: tuple[CollisionAllowance, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _nonempty(self.policy_id, name="collision policy ID"))
        allowances = tuple(self.allowances)
        if any(not isinstance(value, CollisionAllowance) for value in allowances):
            raise TypeError("allowances 必须由 CollisionAllowance 组成")
        object.__setattr__(self, "allowances", allowances)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "allowances": [value.to_dict() for value in self.allowances],
        }


@dataclass(frozen=True)
class PivotLimits:
    """限制TCP从起点的最大偏移和整条路径长度，单位米。"""

    max_tcp_excursion_m: float
    max_tcp_path_length_m: float

    def __post_init__(self) -> None:
        for key in ("max_tcp_excursion_m", "max_tcp_path_length_m"):
            value = float(getattr(self, key))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("pivot限制必须是正有限米数")
            object.__setattr__(self, key, value)

    def to_dict(self) -> dict[str, float]:
        return {"max_tcp_excursion_m": self.max_tcp_excursion_m,
                "max_tcp_path_length_m": self.max_tcp_path_length_m}


@dataclass(frozen=True)
class MotionRequest:
    request_id: str
    operation: MotionOperation
    group: str
    start: JointState
    target: PoseTarget
    scene: PlanningSceneRef
    collision_policy: CollisionPolicy = field(default_factory=CollisionPolicy)
    seed: int = 0
    planner_mode: PlannerMode = PlannerMode.CUDA_GRAPH_BENCHMARK
    approximate_ik_budget: Mapping[str, float] | None = None
    pivot_limits: PivotLimits | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _nonempty(self.request_id, name="request ID"))
        object.__setattr__(self, "group", _nonempty(self.group, name="robot group"))
        object.__setattr__(self, "operation", MotionOperation(self.operation))
        object.__setattr__(self, "planner_mode", PlannerMode(self.planner_mode))
        object.__setattr__(self, "seed", int(self.seed))
        if not isinstance(self.start, JointState):
            raise TypeError("start 必须是 JointState")
        if not isinstance(self.target, PoseTarget):
            raise TypeError("target 必须是 PoseTarget")
        if not isinstance(self.scene, PlanningSceneRef):
            raise TypeError("scene 必须是 PlanningSceneRef")
        if not isinstance(self.collision_policy, CollisionPolicy):
            raise TypeError("collision_policy 必须是 CollisionPolicy")
        if self.operation is MotionOperation.PIVOT_TCP:
            if not isinstance(self.pivot_limits, PivotLimits):
                raise ValueError("pivot_tcp必须显式提供PivotLimits")
        elif self.pivot_limits is not None:
            raise ValueError("pivot_limits只适用于pivot_tcp")
        if self.approximate_ik_budget is not None:
            if self.operation not in {MotionOperation.PLAN, MotionOperation.LINEAR_TCP}:
                raise ValueError("近似 IK 预算只适用于规划请求")
            budget = dict(self.approximate_ik_budget)
            keys = {"max_position_error_m", "max_rotation_error_rad"}
            if set(budget) != keys or any(not np.isfinite(float(budget[key])) or float(budget[key]) <= 0 for key in keys):
                raise ValueError("近似 IK 预算须包含正的有限平移/转角上限")
            object.__setattr__(self, "approximate_ik_budget", {key: float(budget[key]) for key in keys})

    def to_dict(self) -> dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "operation": self.operation.value,
            "group": self.group,
            "start": self.start.to_dict(),
            "target": self.target.to_dict(),
            "scene": self.scene.to_dict(),
            "collision_policy": self.collision_policy.to_dict(),
            "seed": self.seed,
            "planner_mode": self.planner_mode.value,
            "approximate_ik_budget": None if self.approximate_ik_budget is None else dict(self.approximate_ik_budget),
        }
        if self.pivot_limits is not None:
            result["pivot_limits"] = self.pivot_limits.to_dict()
        return result


@dataclass(frozen=True)
class JointTrajectory:
    joint_names: tuple[str, ...]
    times_s: tuple[float, ...]
    positions_rad: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        names = _joint_names(self.joint_names)
        times = _finite_vector(self.times_s, name="times_s")
        positions_array = np.asarray(self.positions_rad, dtype=float)
        if positions_array.shape != (len(times), len(names)) or not np.all(
            np.isfinite(positions_array)
        ):
            raise ValueError("positions_rad 必须是 waypoint × joint 的有限矩阵")
        if len(times) < 1 or times[0] < 0.0:
            raise ValueError("轨迹至少包含一个非负时间 waypoint")
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError("times_s 必须严格递增")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(
            self,
            "positions_rad",
            tuple(tuple(float(value) for value in row) for row in positions_array),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "times_s": list(self.times_s),
            "positions_rad": [list(row) for row in self.positions_rad],
        }


@dataclass(frozen=True)
class CollisionEvidence:
    first: str
    second: str
    position_world_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _nonempty(self.first, name="碰撞对象 first"))
        object.__setattr__(self, "second", _nonempty(self.second, name="碰撞对象 second"))
        if self.position_world_m is not None:
            position = _finite_vector(self.position_world_m, name="collision position")
            if len(position) != 3:
                raise ValueError("collision position 必须是三维坐标")
            object.__setattr__(self, "position_world_m", position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            "position_world_m": (
                None if self.position_world_m is None else list(self.position_world_m)
            ),
        }


@dataclass(frozen=True)
class MotionFailure:
    kind: MotionFailureKind
    summary: str
    backend_status: str | None = None
    collision_evidence: tuple[CollisionEvidence, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MotionFailureKind(self.kind))
        object.__setattr__(self, "summary", _nonempty(self.summary, name="failure summary"))
        if self.backend_status is not None:
            object.__setattr__(self, "backend_status", str(self.backend_status))
        evidence = tuple(self.collision_evidence)
        if any(not isinstance(value, CollisionEvidence) for value in evidence):
            raise TypeError("collision_evidence 必须由 CollisionEvidence 组成")
        object.__setattr__(self, "collision_evidence", evidence)
        object.__setattr__(self, "details", _json_mapping(self.details, name="failure details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "summary": self.summary,
            "backend_status": self.backend_status,
            "collision_evidence": [value.to_dict() for value in self.collision_evidence],
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MotionResult:
    request_id: str
    operation: MotionOperation
    success: bool
    backend: str
    planner_mode: PlannerMode
    duration_s: float
    goal_state: JointState | None = None
    trajectory: JointTrajectory | None = None
    failure: MotionFailure | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _nonempty(self.request_id, name="request ID"))
        object.__setattr__(self, "operation", MotionOperation(self.operation))
        object.__setattr__(self, "backend", _nonempty(self.backend, name="backend"))
        object.__setattr__(self, "planner_mode", PlannerMode(self.planner_mode))
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s 必须是非负有限数")
        object.__setattr__(self, "duration_s", duration)
        if bool(self.success) and self.failure is not None:
            raise ValueError("成功结果不能包含 failure")
        if not bool(self.success) and self.failure is None:
            raise ValueError("失败结果必须包含 failure")
        if self.goal_state is not None and not isinstance(self.goal_state, JointState):
            raise TypeError("goal_state 必须是 JointState")
        if self.trajectory is not None and not isinstance(self.trajectory, JointTrajectory):
            raise TypeError("trajectory 必须是 JointTrajectory")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "metrics", _json_mapping(self.metrics, name="motion metrics"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "operation": self.operation.value,
            "success": self.success,
            "backend": self.backend,
            "planner_mode": self.planner_mode.value,
            "duration_s": self.duration_s,
            "goal_state": None if self.goal_state is None else self.goal_state.to_dict(),
            "trajectory": None if self.trajectory is None else self.trajectory.to_dict(),
            "failure": None if self.failure is None else self.failure.to_dict(),
            "metrics": dict(self.metrics),
        }


def motion_request_from_dict(value: Mapping[str, Any]) -> MotionRequest:
    """从已保存的公共 JSON 契约恢复请求。"""
    start = value["start"]
    target = value["target"]
    scene = value["scene"]
    policy = value["collision_policy"]
    return MotionRequest(
        request_id=value["request_id"],
        operation=MotionOperation(value["operation"]),
        group=value["group"],
        start=JointState(tuple(start["joint_names"]), tuple(start["positions_rad"])),
        target=PoseTarget(target["frame"], target["matrix"]),
        scene=PlanningSceneRef(
            scene["revision"], scene["snapshot_sha256"], scene["scope"]
        ),
        collision_policy=CollisionPolicy(
            policy_id=policy["policy_id"],
            allowances=tuple(
                CollisionAllowance(item["first"], item["second"], item["reason"])
                for item in policy.get("allowances", ())
            ),
        ),
        seed=value["seed"],
        planner_mode=PlannerMode(value["planner_mode"]),
        approximate_ik_budget=value.get("approximate_ik_budget"),
        pivot_limits=(None if value.get("pivot_limits") is None
                      else PivotLimits(**value["pivot_limits"])),
    )


def motion_result_from_dict(value: Mapping[str, Any]) -> MotionResult:
    """从已保存的公共 JSON 契约恢复结果。"""
    goal = value.get("goal_state")
    trajectory = value.get("trajectory")
    failure = value.get("failure")
    return MotionResult(
        request_id=value["request_id"],
        operation=MotionOperation(value["operation"]),
        success=value["success"],
        backend=value["backend"],
        planner_mode=PlannerMode(value["planner_mode"]),
        duration_s=value["duration_s"],
        goal_state=(
            None
            if goal is None
            else JointState(tuple(goal["joint_names"]), tuple(goal["positions_rad"]))
        ),
        trajectory=(
            None
            if trajectory is None
            else JointTrajectory(
                tuple(trajectory["joint_names"]),
                tuple(trajectory["times_s"]),
                tuple(tuple(row) for row in trajectory["positions_rad"]),
            )
        ),
        failure=(
            None
            if failure is None
            else MotionFailure(
                kind=MotionFailureKind(failure["kind"]),
                summary=failure["summary"],
                backend_status=failure.get("backend_status"),
                collision_evidence=tuple(
                    CollisionEvidence(
                        item["first"],
                        item["second"],
                        (
                            None
                            if item.get("position_world_m") is None
                            else tuple(item["position_world_m"])
                        ),
                    )
                    for item in failure.get("collision_evidence", ())
                ),
                details=failure.get("details", {}),
            )
        ),
        metrics=value.get("metrics", {}),
    )


__all__ = [
    "PlannerMode",
    "MotionOperation",
    "MotionFailureKind",
    "JointState",
    "PoseTarget",
    "PlanningSceneRef",
    "CollisionAllowance",
    "CollisionPolicy",
    "MotionRequest",
    "PivotLimits",
    "JointTrajectory",
    "CollisionEvidence",
    "MotionFailure",
    "MotionResult",
    "motion_request_from_dict",
    "motion_result_from_dict",
]
