"""在接近起点先对齐腕部，再直线平移；统一复核完整 approach 合同。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion.model import (
    CollisionPolicy, JointState, JointTrajectory, MotionFailure, MotionFailureKind,
    MotionOperation, MotionResult, PivotLimits, PoseTarget,
)
from gene_agent.motion.path_metrics import measure_linear_tcp_path


def plan_aligned_approach(strict_planner, contact_planner, request, strict_scene):
    """返回严格原位对齐＋允许指垫接触直线平移组成的 approach。"""
    start_pose = np.asarray(strict_planner.fk(request.start), dtype=float)
    angle = float(Rotation.from_matrix(
        start_pose[:3, :3].T @ request.target.matrix[:3, :3]
    ).magnitude())
    details = {
        "composition": "strict_pivot_then_contact_linear",
        "operation": "linear_tcp",
        "contract_start": request.start.to_dict(),
        "contract_target": request.target.to_dict(),
        "initial_rotation_error_rad": angle,
        "segments": [],
    }
    attempts = []

    def record(phase, child_request, result):
        if (result.request_id != child_request.request_id
                or result.operation is not child_request.operation):
            raise RuntimeError("接近子段返回与真实请求不匹配")
        row = {"phase": phase, "request": child_request.to_dict(),
               "result": result.to_dict()}
        details["segments"].append(row)
        attempts.append({"stage": phase, "request": row["request"],
                         "result": row["result"]})

    def failure(summary, *, child=None, metrics=None):
        return MotionResult(
            request.request_id + ":composite", MotionOperation.LINEAR_TCP, False,
            "composite", request.planner_mode,
            sum(row["result"]["duration_s"] for row in details["segments"]),
            failure=(child.failure if child is not None and child.failure is not None else
                     MotionFailure(MotionFailureKind.CONSTRAINT_VIOLATION, summary)),
            metrics={"composition": details["composition"], **(metrics or {})},
        ), details, attempts

    pivot = None
    if angle > 1e-5:
        aligned = start_pose.copy()
        aligned[:3, :3] = request.target.matrix[:3, :3]
        pivot_request = replace(
            request, request_id=request.request_id + ":align",
            operation=MotionOperation.PIVOT_TCP,
            target=PoseTarget("world_tcp", aligned),
            scene=strict_scene, collision_policy=CollisionPolicy(),
            pivot_limits=PivotLimits(.005, .05),
        )
        pivot = strict_planner.plan(pivot_request)
        record("approach_align", pivot_request, pivot)
        if not pivot.success:
            return failure("接近起点原位对齐失败", child=pivot)
        if pivot.goal_state is None or pivot.trajectory is None:
            raise RuntimeError("原位对齐成功结果必须包含关节轨迹和终态")
    translation_request = replace(
        request, request_id=request.request_id + ":translate",
        start=pivot.goal_state if pivot is not None else request.start,
    )
    translation = contact_planner.plan(translation_request)
    record("approach_translate", translation_request, translation)
    if not translation.success:
        return failure("对齐后的直线接近失败", child=translation)
    if translation.trajectory is None or translation.goal_state is None:
        raise RuntimeError("直线接近成功结果必须包含关节轨迹和终态")

    trajectories = ([pivot.trajectory] if pivot is not None else []) + [translation.trajectory]
    names = request.start.joint_names
    if any(trajectory.joint_names != names for trajectory in trajectories):
        return failure("接近子段关节顺序不一致")
    positions = np.asarray(trajectories[0].positions_rad, dtype=float)
    times = np.asarray(trajectories[0].times_s, dtype=float)
    if len(times) < 2:
        return failure("接近子段缺少时间采样")
    dt = float(times[1] - times[0])
    if dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-9, rtol=1e-6):
        return failure("接近子段必须使用均匀时间采样")
    times = times - times[0]
    junction_error = 0.
    for segment in trajectories[1:]:
        q = np.asarray(segment.positions_rad, dtype=float)
        ts = np.asarray(segment.times_s, dtype=float)
        if len(ts) < 2 or not np.allclose(np.diff(ts), dt, atol=1e-9, rtol=1e-6):
            return failure("接近子段时间采样不一致")
        junction_error = max(
            junction_error, float(np.max(np.abs(q[0] - positions[-1]))),
        )
        if junction_error > 1e-5:
            return failure("接近子段关节终点与下一段起点不连续",
                           metrics={"junction_max_joint_error_rad": junction_error})
        times = np.concatenate((times, times[-1] + ts[1:] - ts[0]))
        positions = np.concatenate((positions, q[1:]), axis=0)

    poses = np.asarray([
        strict_planner.fk(JointState(names, tuple(q))) for q in positions
    ])
    metrics = measure_linear_tcp_path(
        poses[:, :3, 3], start_pose[:3, 3], request.target.matrix[:3, 3],
    )
    rotation_error = float(Rotation.from_matrix(
        request.target.matrix[:3, :3].T @ poses[-1, :3, :3]
    ).magnitude())
    checks = {
        "composition": details["composition"],
        "junction_max_joint_error_rad": junction_error,
        "linear_path_metrics": {**metrics.to_dict(),
                                "goal_orientation_error_rad": rotation_error},
        "linear_path_limits": {"lateral_m": .005, "backward_step_m": .001,
                               "cumulative_backward_m": .002, "overshoot_m": .002,
                               "endpoint_m": .005, "rotation_rad": .05},
        "segment_scene_sha256": [
            row["request"]["scene"]["snapshot_sha256"]
            for row in details["segments"]
        ],
    }
    if (not metrics.satisfies(lateral_m=.005, backward_step_m=.001,
                              cumulative_backward_m=.002, overshoot_m=.002,
                              endpoint_m=.005)
            or rotation_error > .05):
        return failure("完整接近轨迹未通过原直线、单向性或终点合同", metrics=checks)
    trajectory = JointTrajectory(
        names, tuple(times), tuple(tuple(q) for q in positions),
    )
    return MotionResult(
        request.request_id + ":composite", MotionOperation.LINEAR_TCP, True,
        "composite", request.planner_mode,
        sum(row["result"]["duration_s"] for row in details["segments"]),
        goal_state=translation.goal_state, trajectory=trajectory, metrics=checks,
    ), details, attempts
