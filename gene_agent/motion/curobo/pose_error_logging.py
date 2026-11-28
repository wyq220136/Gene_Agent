"""物理 TCP 误差日志；不改求解器成功标志、候选排序或准入策略。"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def backend_error_metrics(position, rotation):
    """保留后端原值。多 tool 聚合值不能直接解释为 TCP 米数/弧度。"""
    def finite(value):
        return None if value is None or not np.isfinite(value) else float(value)
    return {"position_error": finite(position), "rotation_error": finite(rotation),
            "source": "curobo_returned_aggregate",
            "units": "backend_native; not independently verified world TCP errors"}


def _pose(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-4, rtol=0)
            or np.linalg.det(value[:3, :3]) <= 0):
        raise ValueError(f"{name}不是有效的world TCP刚体变换")
    return value


def annotate_tcp_pose_errors(diagnostic, target, joint_names, *, fk_batch=None, fk_poses=None):
    """用一次批量FK核对同一目标下的候选，原始后端判定全部保留。

    fk_batch须返回world TCP，不能返回URDF base的link_7。批筛调用方可传
    已计算的fk_poses复用一次批量FK。FK失败只使误差未知，不替换原规划失败。
    """
    records = diagnostic.get("seeds") or [diagnostic]
    for row in records:
        if "backend_error_metrics" not in row:
            row["backend_error_metrics"] = backend_error_metrics(
                row.get("position_error_m"), row.get("rotation_error_rad"))
        row.update(position_error_m=None, rotation_error_rad=None,
                   tcp_pose_converged=None, pose_error_source="world_tcp_fk_unavailable")
    try:
        goal = _pose(target, "目标")
        values = np.asarray([row["joint_positions_rad"] for row in records], dtype=float)
        if values.shape != (len(records), len(joint_names)) or not np.isfinite(values).all():
            raise ValueError("诊断候选与活动关节顺序/维度不一致或包含非有限值")
        if fk_poses is None:
            if fk_batch is None:
                raise ValueError("没有可用的world TCP FK")
            fk_poses = fk_batch(joint_names, values)
        poses = np.asarray(fk_poses, dtype=float)
        if poses.shape != (len(records), 4, 4):
            raise ValueError("批量FK返回的候选数量不一致")
        # 先全部验证，再提交，避免部分更正与旧的聚合指标混在一起。
        poses = [_pose(pose, "候选FK") for pose in poses]
        for row, pose in zip(records, poses):
            delta = pose[:3, 3] - goal[:3, 3]
            position = float(np.linalg.norm(delta))
            rotation = float(Rotation.from_matrix(goal[:3, :3].T @ pose[:3, :3]).magnitude())
            pt, rt = diagnostic.get("position_tolerance_m"), diagnostic.get("rotation_tolerance_rad")
            known = all(isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in (pt, rt))
            row.update(position_error_m=position, rotation_error_rad=rotation,
                       translation_delta_world_m=delta.tolist(), fk_tcp_pose_world=pose.tolist(),
                       pose_error_source="world_tcp_fk",
                       tcp_pose_converged=bool(position < pt and rotation < rt) if known else None)
        diagnostic["tcp_pose_error_reference"] = {
            "frame": "world_tcp", "requested_tcp_pose_world": goal.tolist(),
            "joint_names": list(joint_names), "position_metric": "euclidean_translation_m",
            "rotation_metric": "SO3_geodesic_angle_rad", "diagnostic_only": True}
    except Exception as error:
        diagnostic["tcp_pose_error_recalculation_error"] = f"{type(error).__name__}: {error}"
    if diagnostic.get("seeds"):
        chosen = int(diagnostic["selected_seed_index"])
        selected = records[chosen]
        for key in ("position_error_m", "rotation_error_rad", "pose_error_source", "tcp_pose_converged"):
            diagnostic[key] = selected[key]
        diagnostic["tcp_pose_converged_seed_count"] = sum(
            row.get("tcp_pose_converged") is True for row in records)
    # pose_converged及种子数保留原聚合指标诊断；实际TCP收敛另行声明，不冒充重新求解。
    diagnostic["pose_converged_source"] = "backend_aggregate_metrics"
    return diagnostic
