"""最小同步 cuRobo 基线；不含重试队列、碰撞放宽或隐式 fallback。"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import traceback
import time
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ..path_metrics import measure_linear_tcp_path, measure_pivot_tcp_path
from ..configuration import CollisionMode, resolve_collision_mode, collision_mode_metadata
from .approximate import ApproximateIKBudget, plan_with_approximate_ik

from ..model import (
    JointState as MotionJointState,
    JointTrajectory,
    MotionFailure,
    MotionFailureKind,
    MotionOperation,
    MotionRequest,
    MotionResult,
    PlannerMode,
)

ROOT = Path(__file__).resolve().parents[3]


def infer_failure_visualization_dir(scene_model: dict) -> Path | None:
    """把自动诊断限制在所属 run 的碰撞快照目录内。"""
    runs_root = (ROOT / "runs").resolve()
    for record in (scene_model.get("mesh") or {}).values():
        source = Path(record.get("file_path", ""))
        if not source.is_absolute():
            continue
        try:
            source.resolve().relative_to(runs_root)
        except ValueError:
            continue
        return source.resolve().parent / "curobo_no_solution"
    return None


def _record_postcheck_failure(result: dict, *, category: str, status: str,
                              message: str) -> None:
    """保存首要失败，同时列出后来发现的独立复核失败。"""
    if not result.get("success", False):
        additional = result.setdefault("additional_failure_categories", [])
        if category != result.get("failure_category") and category not in additional:
            additional.append(category)
        return
    result.update(success=False, failure_category=category,
                  status=status, message=message)


def add_tcp_tool_frame(robot_config: dict, tool_frame: str, link_from_tcp: np.ndarray,
                       *, tcp_frame: str = "gene_tcp") -> dict:
    """在原末端下注册固定TCP；不改变碰撞球、URDF或原末端的位置定义。"""
    config = copy.deepcopy(robot_config)
    kin = config["kinematics"]
    transform = np.asarray(link_from_tcp, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("link_from_tcp 必须为有限 4x4 矩阵")
    extra = kin.setdefault("extra_links", {})
    if tcp_frame in extra:
        raise ValueError(f"TCP frame 已存在: {tcp_frame}")
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    extra[tcp_frame] = {
        "fixed_transform": [*transform[:3, 3].tolist(), *xyzw[[3, 0, 1, 2]].tolist()],
        "joint_name": f"{tcp_frame}_joint", "joint_type": "FIXED",
        "link_name": tcp_frame, "parent_link_name": tool_frame,
    }
    frames = list(kin.get("tool_frames") or ())
    for name in (tool_frame, tcp_frame):
        if name not in frames:
            frames.append(name)
    kin["tool_frames"] = frames
    return config


def add_attached_collision_spheres(robot_config: dict, tool_frame: str,
                                   spheres_tool: list[list[float]]) -> dict:
    """将显式附着代理绑定到末端link，保留原机器人球和自碰撞忽略表。"""
    values = np.asarray(spheres_tool, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0 \
            or not np.all(np.isfinite(values)) or np.any(values[:, 3] <= 0):
        raise ValueError("附着碰撞代理必须为[N,4]有限正半径球")
    config = copy.deepcopy(robot_config)
    collision = config["kinematics"]["collision_spheres"]
    if tool_frame not in collision:
        raise ValueError(f"末端link没有碰撞球：{tool_frame}")
    collision[tool_frame].extend({"center": row[:3].tolist(), "radius": float(row[3])}
                                 for row in values)
    return config


def line_frame_rotation(direction: np.ndarray) -> np.ndarray:
    """把辅助TCP frame的局部Z轴对齐真实直线方向。"""
    vector = np.asarray(direction, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(magnitude) or magnitude <= 1e-9:
        raise ValueError("直线方向必须为非零有限三维向量")
    z = vector / magnitude
    reference = np.array([0., 0., 1.]) if abs(z[2]) < .9 else np.array([1., 0., 0.])
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def solver_runtime_options(mode: PlannerMode | str, random_seed: int) -> dict:
    """返回影响复现性的 cuRobo 运行参数。

    正式规划默认开启 CUDA Graph；显式 development 关闭图用于诊断对照。
    冷启动仍包含首次图捕获，记录时与热态求解耗时分开。
    """
    resolved = PlannerMode(mode)
    return {
        "use_cuda_graph": resolved is PlannerMode.CUDA_GRAPH_BENCHMARK,
        "random_seed": int(random_seed),
    }


def load_realman_motion_config(group: str, robot_metadata: dict, *, gripper_snapshot: dict | None = None) -> dict:
    """把已验证模型元数据映射到独立规划组；保留原碰撞球和忽略表。

    当前单臂规划不会自动加入另一臂；调用方必须通过 scene_model 显式提供
    其他臂和环境的几何，并在 scene_scope 记录覆盖范围，不能隐式宣称全场景。
    """
    if group not in robot_metadata["joint_groups"]:
        raise ValueError(f"未知关节组: {group}")
    if robot_metadata.get("profile") == "realman_single_pika":
        if gripper_snapshot is None:
            raise ValueError("Pika 规划必须提供当前开口的碰撞快照")
        from .profiles.pika import apply_pika_gripper_snapshot
        config = apply_pika_gripper_snapshot({}, gripper_snapshot, robot_metadata, group)
        kin = config["kinematics"]
        return dict(robot_config=config, joint_names=robot_metadata["joint_groups"][group],
                    planner_joint_names=kin["cspace"]["joint_names"],
                    gripper_snapshot=copy.deepcopy(gripper_snapshot),
                    world_from_base=robot_metadata["transforms"][group]["world_from_urdf_base"],
                    link_from_tcp=robot_metadata["transforms"][group]["link_from_tcp"],
                    tool_frame="link_7")
    asset = ROOT / "assets/robots/realman"
    config = yaml.safe_load((asset / "config" / f"realman_{group}.yml").read_text())
    kin = config["kinematics"]
    kin["urdf_path"] = str(asset / "urdf" / f"realman_{group}.urdf")
    kin["asset_root_path"] = str(asset / "urdf")
    names = kin["cspace"]["joint_names"]
    public_names = robot_metadata["joint_groups"][group]
    if len(names) != len(public_names):
        raise ValueError("规划关节和仿真关节数量不一致")
    if gripper_snapshot is not None:
        from .gripper import apply_gripper_snapshot
        config = apply_gripper_snapshot(config, gripper_snapshot, robot_metadata['grippers'][group], group)
    return dict(robot_config=config, joint_names=public_names, planner_joint_names=names, gripper_snapshot=copy.deepcopy(gripper_snapshot),
                world_from_base=robot_metadata["transforms"][group]["world_from_urdf_base"],
                link_from_tcp=robot_metadata["transforms"][group]["link_from_tcp"],
                tool_frame="link_7")


def ordered_positions(names, positions, expected):
    """强制具名且完整的关节输入，避免静默按七轴数组错配。"""
    values = np.asarray(positions, dtype=float)
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("关节名称重复、缺失或多余")
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise ValueError("关节位置必须为与名称等长的一维有限数组")
    lookup = dict(zip(names, values))
    return np.array([lookup[name] for name in expected])


def scene_collision_cache(scene_model: dict) -> dict[str, int]:
    """Size cuRobo's fixed obstacle buffers from the complete frozen scene."""
    if not isinstance(scene_model, dict):
        raise ValueError("scene_model must be a dictionary")
    result = {}
    for obstacle_type in ("cuboid", "mesh"):
        obstacles = scene_model.get(obstacle_type, {})
        if not isinstance(obstacles, dict):
            raise ValueError(f"scene_model[{obstacle_type!r}] must be a dictionary")
        # cuRobo allocates fixed GPU buffers. Keep an empty type allocatable so
        # a valid empty scene does not fall back to its unrelated default of 50.
        result[obstacle_type] = max(1, len(obstacles))
    return result


def empty_solver_result(operation: str) -> dict:
    """Normalize cuRobo's documented None return when goal IK has no seed."""
    if operation not in {"ik", "plan"}:
        raise ValueError(f"unknown solver operation: {operation}")
    return {
        "success": False,
        "status": "goal_ik_no_solution",
        "raw_result": "None",
        "raw_error": None,
        "message": "cuRobo did not find a successful IK seed",
        "failure_category": "goal_ik_no_solution",
    }


def _finite_solver_scalar(value):
    """只取 cuRobo 已筛出的最优种子数值，避免把整块 Tensor 写进报告。"""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value)
    if values.size == 0:
        return None
    number = float(values.reshape(-1)[0])
    return number if np.isfinite(number) else None


def solver_failure_diagnostic(raw, operation: str, solver=None) -> dict:
    """提取求解器真实可见的失败证据，不把约束失败直接叫作碰撞。

    cuRobo 的 ``feasible`` 同时包含关节限位、自碰撞和场景碰撞；它本身
    不能指出哪个物体。只有额外的碰撞查询才能确认具体碰撞对象。
    """
    if operation not in {"ik", "plan"}:
        raise ValueError(f"未知求解操作: {operation}")
    config = getattr(solver, "config", None)
    position_tolerance = _finite_solver_scalar(getattr(config, "position_tolerance", None))
    rotation_tolerance = _finite_solver_scalar(getattr(config, "orientation_tolerance", None))
    position_error = _finite_solver_scalar(getattr(raw, "position_error", None))
    rotation_error = _finite_solver_scalar(getattr(raw, "rotation_error", None))
    feasible_raw = getattr(raw, "feasible", None)
    if hasattr(feasible_raw, "detach"):
        feasible_raw = feasible_raw.detach().cpu().numpy()
    feasible_values = None if feasible_raw is None else np.asarray(feasible_raw)
    feasible = None if feasible_values is None or feasible_values.size == 0 else bool(feasible_values.reshape(-1)[0])
    position_converged = None if position_error is None or position_tolerance is None else position_error < position_tolerance
    rotation_converged = None if rotation_error is None or rotation_tolerance is None else rotation_error < rotation_tolerance
    pose_converged = None if position_converged is None or rotation_converged is None else position_converged and rotation_converged
    if operation == "ik":
        if pose_converged is False and feasible is False:
            reason = "pose_and_constraint"
        elif pose_converged is False:
            reason = "pose_not_converged"
        elif feasible is False:
            reason = "constraint_rejected"
        else:
            reason = "ik_solver_failed"
    else:
        reason = "trajectory_optimization_failed"
    return {
        "reason": reason,
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "position_tolerance_m": position_tolerance,
        "rotation_tolerance_rad": rotation_tolerance,
        "position_converged": position_converged,
        "rotation_converged": rotation_converged,
        "pose_converged": pose_converged,
        "constraint_feasible": feasible,
        "collision_confirmed": False,
    }


def describe_start_failure(check: dict) -> tuple[str, str, str]:
    """从已复核的事实生成短错误；允许的安装接触不算失败原因。"""
    facts = check.get("collision_diagnostic") or {}
    reasons = []
    if facts.get("self_collision"):
        reasons.append("机器人自身碰撞")
    if facts.get("scene_collision"):
        names = facts.get("colliding_obstacles") or []
        reasons.append("场景碰撞" + (f"（{', '.join(names)}）" if names else ""))
    if facts.get("joint_limit_violation"):
        reasons.append("关节限位违反")
    if reasons:
        category = "start_collision" if facts.get("self_collision") or facts.get("scene_collision") else "solver_rejected"
        status = "start_collision" if category == "start_collision" else "start_joint_limit"
        return status, category, "起始姿态：" + "；".join(reasons)
    return "start_invalid", "solver_rejected", "起始姿态未通过严格状态复核；请检查详细诊断"


def summarize_ik_failure(raw, solver) -> dict:
    """概括全部返回种子；可行性与位姿收敛分别报告，不猜测碰撞原因。"""
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    success = array(raw.success).reshape(-1).astype(bool)
    feasible = array(raw.feasible).reshape(-1).astype(bool)
    position = array(raw.position_error).reshape(-1).astype(float)
    rotation = array(raw.rotation_error).reshape(-1).astype(float)
    solutions = array(raw.solution).reshape(-1, array(raw.solution).shape[-1])
    count = len(success)
    if not count or any(len(values) != count for values in (feasible, position, rotation, solutions)):
        raise ValueError("IK 种子诊断维度不一致")
    position_tolerance = _finite_solver_scalar(solver.config.position_tolerance)
    rotation_tolerance = _finite_solver_scalar(solver.config.orientation_tolerance)
    if not position_tolerance or not rotation_tolerance:
        raise ValueError("IK solver 缺少正的位姿容差")
    converged = (position < position_tolerance) & (rotation < rotation_tolerance)
    # 优先解释满足其他约束的候选；否则查看位姿误差最小的失败候选。
    scores = position / position_tolerance + rotation / rotation_tolerance
    candidates = np.flatnonzero(feasible)
    if not len(candidates):
        candidates = np.arange(count)
    chosen = int(candidates[np.argmin(scores[candidates])])
    if not converged[chosen] and not feasible[chosen]:
        reason = "pose_and_constraint"
    elif not converged[chosen]:
        reason = "pose_not_converged"
    elif not feasible[chosen]:
        reason = "constraint_rejected"
    else:
        reason = "ik_solver_failed"
    return {
        "source": "followup_ik_solve_after_empty_plan",
        "seed_count": count,
        "successful_seed_count": int(success.sum()),
        "feasible_seed_count": int(feasible.sum()),
        "pose_converged_seed_count": int(converged.sum()),
        "position_tolerance_m": position_tolerance,
        "rotation_tolerance_rad": rotation_tolerance,
        "selected_seed_index": chosen,
        "selected_seed_reason": reason,
        "selected_joint_positions_rad": solutions[chosen].astype(float).tolist(),
        "collision_confirmed": False,
        "collision_checked": False,
        "seeds": [
            {
                "index": index,
                "success": bool(success[index]),
                "constraint_feasible": bool(feasible[index]),
                "pose_converged": bool(converged[index]),
                "position_error_m": float(position[index]),
                "rotation_error_rad": float(rotation[index]),
                "joint_positions_rad": solutions[index].astype(float).tolist(),
            }
            for index in range(count)
        ],
    }


def classify_world_obstruction(relaxed_ik_success: bool, strict_collision_check: dict | None) -> dict:
    """空环境解只作反事实诊断，不产生可执行的规划成功。"""
    if not relaxed_ik_success:
        reason = "unresolved_without_world"
    elif strict_collision_check is None:
        reason = "strict_collision_check_unavailable"
    elif strict_collision_check.get("scene_collision"):
        reason = "scene_collision_at_relaxed_solution"
    elif strict_collision_check.get("self_collision"):
        reason = "self_collision_at_relaxed_solution"
    elif strict_collision_check.get("joint_limit_violation"):
        reason = "joint_limit_at_relaxed_solution"
    else:
        reason = "failure_not_reproduced_at_relaxed_solution"
    return {
        "reason": reason,
        "diagnostic_only": True,
        "world_collision_disabled_for_probe": True,
        "relaxed_ik_success": bool(relaxed_ik_success),
        "blocking_obstacles": list((strict_collision_check or {}).get("colliding_obstacles", [])),
        "strict_collision_check": strict_collision_check,
    }


def classify_failure_detail(
    stage: str,
    diagnostic: dict | None = None,
    *,
    world_obstruction_probe: dict | None = None,
) -> dict:
    """把底层证据整理成稳定细分码，同时避免把有限搜索说成全局不可达。"""
    if stage not in {"goal_ik", "trajectory", "trajectory_postcheck"}:
        raise ValueError(f"未知失败阶段: {stage}")
    diagnostic = diagnostic or {}
    probe = world_obstruction_probe or {}
    collision = diagnostic.get("collision_check") or {}
    evidence_source = "failed_candidate"
    if (stage == "goal_ik" and probe.get("world_collision_disabled_for_probe")
            and probe.get("relaxed_ik_success")
            and probe.get("strict_collision_check")):
        collision = probe["strict_collision_check"]
        evidence_source = "world_relaxed_ik_solution"

    observed_constraints = []
    if collision.get("self_collision"):
        observed_constraints.append("self_collision")
    if collision.get("scene_collision"):
        observed_constraints.append("scene_collision")
    if collision.get("joint_limit_violation"):
        observed_constraints.append("joint_limit_violation")

    goal = stage == "goal_ik"
    candidate_reaches_goal = (
        diagnostic.get("pose_converged") is True
        or diagnostic.get("selected_seed_reason") in {"constraint_rejected", "ik_solver_failed"}
    )
    # 一个离目标还很远的失败 seed 即使碰撞，也不能证明目标本身因碰撞不可达。
    constraints = observed_constraints
    if goal and evidence_source == "failed_candidate" and not candidate_reaches_goal:
        constraints = []
    prefix = "goal" if goal else "path"
    waypoint = diagnostic.get("first_invalid_waypoint")
    waypoint_text = "" if waypoint is None else f"（首个失败 waypoint={waypoint}）"
    obstacle_names = list(collision.get("colliding_obstacles") or ())
    obstacle_text = f"：{', '.join(obstacle_names)}" if obstacle_names else ""

    if len(constraints) > 1:
        reason_code = f"{prefix}_multiple_constraints"
        human_summary = (
            ("目标候选" if goal else "候选路径")
            + waypoint_text + "同时违反多项约束：" + "、".join(constraints)
        )
        strength = "confirmed"
    elif constraints == ["self_collision"]:
        reason_code = f"{prefix}_self_collision"
        human_summary = ("目标关节姿态发生机器人自碰撞" if goal
                         else f"候选路径发生机器人自碰撞{waypoint_text}")
        strength = "confirmed"
    elif constraints == ["scene_collision"]:
        reason_code = f"{prefix}_scene_collision"
        human_summary = (("目标关节姿态与场景碰撞" if goal
                          else f"候选路径与场景碰撞{waypoint_text}") + obstacle_text)
        strength = "confirmed"
    elif constraints == ["joint_limit_violation"]:
        reason_code = f"{prefix}_joint_limit_violation"
        human_summary = ("目标候选违反关节限位（当前机械臂可操作空间约束）" if goal
                         else f"候选路径违反关节限位{waypoint_text}（当前机械臂可操作空间约束）")
        strength = "confirmed"
    elif (goal and probe.get("world_collision_disabled_for_probe")
          and not probe.get("relaxed_ik_success")):
        reason_code = "goal_pose_unreachable_within_search"
        human_summary = (
            "关闭场景碰撞并保留自碰撞检查后，目标 IK 在当前关节限位与有限种子搜索中仍未收敛；"
            "这支持工作空间/运动学不可达判断，但不构成全局不可达证明"
        )
        strength = "finite_search"
        evidence_source = "world_relaxed_ik_probe"
    elif goal and (
        diagnostic.get("pose_converged") is False
        or diagnostic.get("selected_seed_reason") in {"pose_not_converged", "pose_and_constraint"}
    ):
        reason_code = "goal_pose_not_converged"
        human_summary = "目标 IK 位姿误差未收敛；当前证据不能确认是碰撞还是绝对不可达"
        strength = "finite_search"
    elif not goal and stage == "trajectory_postcheck":
        reason_code = "path_postcheck_rejected"
        human_summary = "cuRobo 返回轨迹未通过独立碰撞/约束复核；详细碰撞对未能进一步确认"
        strength = "confirmed_invalid_state"
    elif not goal:
        reason_code = "path_optimization_failed_no_confirmed_collision"
        human_summary = "目标 IK 已有候选，但轨迹优化未返回完整可行路径；已检查证据未确认碰撞或关节限位"
        strength = "solver_result"
    else:
        reason_code = "goal_ik_failure_unresolved"
        human_summary = "目标 IK 求解失败，但当前记录不足以区分位姿不可达、碰撞或关节限位"
        strength = "insufficient_evidence"

    return {
        "stage": stage,
        "reason_code": reason_code,
        "human_summary": human_summary,
        "evidence_strength": strength,
        "evidence_source": evidence_source,
        "finite_search_not_global_proof": strength == "finite_search",
        "constraint_flags": constraints,
        "observed_candidate_constraint_flags": observed_constraints,
        "first_invalid_waypoint": waypoint,
        "collision_check": collision or None,
    }


class SynchronousPlanner:
    """一个具名关节组的 FK、单目标 IK 与完整轨迹规划。

    world_from_base 是 T_world_URDFbase，link_from_tcp 是 T_link7_TCP。
    scene_model 已在 URDF base 表达；scene_scope 是人类可读的覆盖范围声明。
    本类始终启用 self collision，保留配置中的 ignore 表；每个操作只调用一次。
    """
    def __init__(self, *, robot_config, joint_names, planner_joint_names,
                 world_from_base, link_from_tcp, tool_frame,
                 scene_model: dict, scene_scope: str, scene_snapshot_sha256: str | None = None,
                 mounting_contact_policy: dict | None = None, gripper_snapshot: dict | None = None,
                 attachment_spheres_tool: list[list[float]] | None = None,
                 held_object: dict | None = None,
                 planner_mode: PlannerMode | str = PlannerMode.CUDA_GRAPH_BENCHMARK,
                 random_seed: int = 123,
                 optimizer_collision_activation_distance: float = .01,
                 collision_mode: CollisionMode | str | None = None,
                 robot_protection_scene: dict | None = None,
                 diagnostic_output_dir: str | Path | None = None):
        if (not np.isfinite(optimizer_collision_activation_distance)
                or optimizer_collision_activation_distance < 0):
            raise ValueError('优化器软避障距离必须是非负有限米数')
        self.optimizer_collision_activation_distance = float(optimizer_collision_activation_distance)
        self.collision_mode = resolve_collision_mode(collision_mode)
        if self.collision_mode is CollisionMode.ROBOT_ONLY:
            scene_model = copy.deepcopy(robot_protection_scene or {})
            attachment_spheres_tool = None
            held_object = None
            if not scene_model:
                mounting_contact_policy = {}
            if robot_protection_scene is None:
                # 显式机器人快照的 scope 属于请求身份，不得加前缀后破坏 adapter 校验。
                scene_scope = "robot_only；无场景障碍或规划 attachment；" + scene_scope
        if not scene_scope or scene_model is None:
            raise ValueError("必须显式提供场景和覆盖范围，空诊断场景也需声明")
        from .gripper import gripper_result_metadata, _digest
        self.gripper_metadata = {**gripper_result_metadata(gripper_snapshot),
                                 **collision_mode_metadata(self.collision_mode)}
        self.gripper_metadata['optimizer_collision_activation_distance_m'] = self.optimizer_collision_activation_distance
        if gripper_snapshot is not None and _digest(robot_config['kinematics']['collision_spheres']) != gripper_snapshot['merged_sphere_sha256']:
            raise ValueError('Planner spheres do not match synchronized gripper snapshot')
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        self.torch = torch
        self._source_robot_config = copy.deepcopy(robot_config)
        self._gripper_snapshot = copy.deepcopy(gripper_snapshot)
        self.tcp_frame = "gene_tcp"
        self.robot_config = add_tcp_tool_frame(
            robot_config, tool_frame, np.asarray(link_from_tcp, float), tcp_frame=self.tcp_frame
        )
        if attachment_spheres_tool is not None:
            self.robot_config = add_attached_collision_spheres(
                self.robot_config, tool_frame, attachment_spheres_tool,
            )
        self.attachment_spheres_tool = copy.deepcopy(attachment_spheres_tool)
        self.held_object = copy.deepcopy(held_object)
        if held_object is not None:
            from .attachment import with_held_object
            self.robot_config=with_held_object(self.robot_config,tool_link=tool_frame,
                finger_links=held_object['finger_links'],spheres_link=held_object['spheres_link'])
        self.joint_names = list(joint_names)
        self.planner_joint_names = list(planner_joint_names)
        self.world_from_base = np.asarray(world_from_base, float)
        self.link_from_tcp = np.asarray(link_from_tcp, float)
        self.tool_frame = tool_frame
        self.scene_model = copy.deepcopy(scene_model)
        self.diagnostic_output_dir = (
            Path(diagnostic_output_dir).resolve()
            if diagnostic_output_dir is not None else
            infer_failure_visualization_dir(self.scene_model)
        )
        self._diagnostic_export_count = 0
        self.collision_cache = scene_collision_cache(self.scene_model)
        self.scene_scope = scene_scope
        self.planner_mode = PlannerMode(planner_mode)
        self.random_seed = int(random_seed)
        self.runtime_options = solver_runtime_options(self.planner_mode, self.random_seed)
        self.scene_snapshot_sha256 = scene_snapshot_sha256 or hashlib.sha256(json.dumps(self.scene_model, sort_keys=True).encode()).hexdigest()
        # 当前 cuRobo 配置加载器会原地包装 robot_cfg；各后端使用独立副本。
        self.kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(copy.deepcopy(self.robot_config)))
        if list(self.kinematics.joint_names) != self.planner_joint_names:
            raise ValueError("cuRobo 内部关节顺序与配置不一致")
        self.mounting_contact_policy = copy.deepcopy(mounting_contact_policy or {})
        self._validator = None
        if self.mounting_contact_policy.get("obstacle_names"):
            from ..collision import MountingCollisionValidator
            self._validator = MountingCollisionValidator(self.kinematics, self.scene_model, self.mounting_contact_policy)
        self._ik = self._planner = None
        self._ik_batches = {}
        self._collision_checker = None
        # 一个规划实例绑定固定场景；40 个抓取候选共用同一起始关节状态。
        self._start_validation_cache: dict[tuple[float, ...], dict] = {}

    def update_gripper_geometry(self, *, robot_config: dict, gripper_snapshot: dict) -> None:
        """同拓扑夹爪几何原位同步到所有 solver；保留 CUDA Graph 的 tensor 地址。

        不舍入开口，不复用旧球。源配置、原 arm 球、非动态锁关节和球数必须相同；
        调用方在任何异常后丢弃整个 planner，避免部分更新状态继续执行。
        """
        self._update_runtime_geometry(robot_config,gripper_snapshot,self.held_object)

    def update_held_geometry(self, *, robot_config, gripper_snapshot, held_object):
        """同拓扑持物球与实际夹爪一起更新；不能保留上一帧的attachment坐标。"""
        self._update_runtime_geometry(robot_config,gripper_snapshot,held_object)

    def _update_runtime_geometry(self, robot_config, gripper_snapshot, held_object):
        from .session import _geometry_reuse_settings
        old = _geometry_reuse_settings({"robot_config": self._source_robot_config,
                                      "gripper_snapshot": self._gripper_snapshot,"held_object":self.held_object})
        new = _geometry_reuse_settings({"robot_config": robot_config,
                                      "gripper_snapshot": gripper_snapshot,"held_object":held_object})
        if old != new:
            raise ValueError("动态更新改变机器人或持物结构，必须重建")
        config = add_tcp_tool_frame(robot_config, self.tool_frame, self.link_from_tcp,
                                    tcp_frame=self.tcp_frame)
        if self.attachment_spheres_tool is not None:
            config = add_attached_collision_spheres(config, self.tool_frame, self.attachment_spheres_tool)
        if held_object is not None:
            from .attachment import with_held_object
            config=with_held_object(config,tool_link=self.tool_frame,
                finger_links=held_object['finger_links'],spheres_link=held_object['spheres_link'])
        from curobo.kinematics import KinematicsCfg
        fresh = KinematicsCfg.from_robot_yaml_file(copy.deepcopy(config))
        solvers = [self._planner, self._ik, *self._ik_batches.values(), self._collision_checker]
        models = [self.kinematics] + [solver.kinematics for solver in solvers if solver is not None]
        # MotionPlanner 内的 IK/trajopt/graph 共享 KinematicsParams；这里同时
        # 包含独立 batch IK 和碰撞复核器，先逐一核对，再做原位 copy。
        seen = set()
        for model in models:
            params = model.config.kinematics_config
            if id(params) in seen:
                continue
            seen.add(id(params))
            if params.link_spheres.shape != fresh.kinematics_config.link_spheres.shape:
                raise ValueError("动态碰撞球 shape 改变")
            previous = model.get_self_collision_config()
            for field in ("collision_pairs", "sphere_padding"):
                if not self.torch.equal(getattr(previous, field), getattr(fresh.self_collision_config, field)):
                    raise ValueError("动态更新改变自碰撞对或 padding")
        seen.clear()
        for model in models:
            params = model.config.kinematics_config
            if id(params) not in seen:
                model.update_kinematics_config(fresh.kinematics_config)
                seen.add(id(params))
        self.held_object = copy.deepcopy(held_object)
        self.robot_config = config
        self._source_robot_config = copy.deepcopy(robot_config)
        self._gripper_snapshot = copy.deepcopy(gripper_snapshot)
        from .gripper import gripper_result_metadata
        self.gripper_metadata = {**gripper_result_metadata(gripper_snapshot),
                                 **collision_mode_metadata(self.collision_mode)}
        self._start_validation_cache.clear()

    def reset_seed(self) -> None:
        """重置求解采样而不销毁已捕获的 CUDA Graph。"""
        for solver in (self._planner, self._ik, *self._ik_batches.values()):
            if solver is not None:
                solver.reset_seed()

    # 仅本实现保证策略替换、validator重建与缓存失效；第三方factory默认不启用。
    supports_contact_policy_update = True

    def update_contact_policy(self, policy: dict) -> None:
        """保留GPU图/几何，完整替换阶段允许接触对；禁止跨阶段累积例外。"""
        from ..collision import MountingCollisionValidator
        policy = copy.deepcopy(policy or {})
        validator = None
        if policy.get("obstacle_names"):
            validator = MountingCollisionValidator(self.kinematics, self.scene_model, policy)
        self.mounting_contact_policy = policy
        self._validator = validator
        self._start_validation_cache.clear()

    def update_scene(self, *, scene_model: dict, scene_scope: str,
                     scene_snapshot_sha256: str,
                     mounting_contact_policy: dict | None = None,
                     gripper_snapshot: dict | None = None) -> None:
        """在碰撞拓扑兼容时更新所有已创建 solver 的 world。

        调用方负责用结构 key 判断对象集合、几何、机器人和 gripper 是否
        兼容。任一 backend 更新异常时本实例可能处于混合状态，调用方必须
        丢弃整个实例，不能继续使用。
        """
        if not scene_scope or scene_model is None:
            raise ValueError("scene update 必须提供明确场景和 scope")
        if scene_collision_cache(scene_model) != self.collision_cache:
            raise ValueError("新场景超过或改变当前 collision cache 拓扑")
        policy = copy.deepcopy(mounting_contact_policy or {})
        if policy != self.mounting_contact_policy:
            raise ValueError("mounting contact policy 改变必须先调用 update_contact_policy 或重建 planner")
        from .world_update import pose_updates, apply_pose_updates
        updates = pose_updates(self.scene_model, scene_model)
        if updates is not None:
            from curobo.types import Pose
            names = {name for objects in scene_model.values() for name in objects}
            worlds = []
            for solver in (self._planner, self._ik, *self._ik_batches.values()):
                if solver is not None:
                    worlds.append((solver.scene_collision_checker, names))
            if self._collision_checker is not None:
                worlds.append((self._collision_checker.scene_model, names))
            if self._validator is not None:
                body = set(policy.get("obstacle_names", []))
                for label, world in self._validator.checkers.items():
                    worlds.append((world, body if label == "body" else names - body))
            apply_pose_updates(worlds, updates, Pose.from_list)
        else:
            from curobo._src.geom.types import SceneCfg
            scene_cfg = SceneCfg.create(copy.deepcopy(scene_model))
            for solver in (self._planner, self._ik, *self._ik_batches.values(), self._collision_checker):
                if solver is not None:
                    solver.update_world(scene_cfg)
        self.scene_model = copy.deepcopy(scene_model)
        self.scene_scope = str(scene_scope)
        self.scene_snapshot_sha256 = str(scene_snapshot_sha256)
        from .gripper import gripper_result_metadata
        self.gripper_metadata = {**gripper_result_metadata(gripper_snapshot),
                                 **collision_mode_metadata(self.collision_mode)}
        if updates is None:
            if self.mounting_contact_policy.get("obstacle_names"):
                from ..collision import MountingCollisionValidator
                self._validator = MountingCollisionValidator(
                    self.kinematics, self.scene_model, self.mounting_contact_policy,
                )
            else:
                self._validator = None
        self._start_validation_cache.clear()

    def refresh_scene_metadata(self, *, scene_scope: str,
                               scene_snapshot_sha256: str,
                               gripper_snapshot: dict | None = None) -> None:
        """场景几何未变时只刷新追踪信息，不触发昂贵的 update_world。"""
        if not scene_scope or not scene_snapshot_sha256:
            raise ValueError("scene metadata 必须包含 scope 和 snapshot hash")
        self.scene_scope = str(scene_scope)
        self.scene_snapshot_sha256 = str(scene_snapshot_sha256)
        from .gripper import gripper_result_metadata
        self.gripper_metadata = {**gripper_result_metadata(gripper_snapshot),
                                 **collision_mode_metadata(self.collision_mode)}

    def close(self) -> None:
        """释放本实例创建的 cuRobo solver，供交互式调试长期反复规划。

        MotionGen 与各个 IK solver 可能共享底层碰撞对象，因此只调用各自
        公开的 destroy；同一 Python 对象最多释放一次。清理失败不应遮盖
        原来的规划或场景更新异常。
        """
        released: set[int] = set()
        solvers = [self._planner, self._ik, *self._ik_batches.values()]
        for solver in solvers:
            if solver is None or id(solver) in released:
                continue
            released.add(id(solver))
            destroy = getattr(solver, "destroy", None)
            if destroy is not None:
                try:
                    destroy()
                except Exception:
                    pass
        self._planner = None
        self._ik = None
        self._ik_batches.clear()
        self._collision_checker = None
        self._validator = None
        self._start_validation_cache.clear()

    def _state(self, names, positions):
        from curobo.types import JointState
        ordered = ordered_positions(names, positions, self.joint_names)
        tensor = self.torch.as_tensor(ordered, device="cuda", dtype=self.torch.float32).unsqueeze(0)
        return JointState.from_position(tensor, joint_names=self.planner_joint_names)

    def fk(self, joint_names, positions) -> np.ndarray:
        """返回配置指定 world 中的 TCP；不把 link_7 误当 SDK flange。"""
        state = self.kinematics.compute_kinematics(self._state(joint_names, positions))
        pose = state.tool_poses.get_link_pose(self.tool_frame)
        transform = np.eye(4)
        transform[:3, 3] = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
        quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        transform[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return self.world_from_base @ transform @ self.link_from_tcp

    def fk_batch(self, joint_names, positions) -> np.ndarray:
        """返回 (N,4,4) 的 world TCP；全部采样一次上传、一次 FK，保持输入次序。"""
        from curobo.types import JointState
        names = tuple(joint_names)
        if len(names) != len(set(names)) or set(names) != set(self.joint_names):
            raise ValueError("关节名称重复、缺失或多余")
        values = np.asarray(positions, dtype=float)
        if values.size == 0 and (values.shape == (0,) or values.shape == (0, len(names))):
            return np.empty((0, 4, 4), dtype=float)
        if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError("批量关节位置必须是 N×关节数的有限数组")
        # NumPy 列重排可能产生列优先 strides，cuRobo CUDA FK 要求连续关节矩阵。
        ordered = np.ascontiguousarray(values[:, [names.index(name) for name in self.joint_names]])
        tensor = self.torch.as_tensor(ordered, device="cuda", dtype=self.torch.float32)
        state = self.kinematics.compute_kinematics(
            JointState.from_position(tensor, joint_names=self.planner_joint_names))
        pose = state.tool_poses.get_link_pose(self.tool_frame)
        xyz = pose.position.detach().cpu().numpy().reshape(-1, 3)
        quat = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)
        transforms = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
        transforms[:, :3, 3] = xyz
        transforms[:, :3, :3] = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
        return self.world_from_base @ transforms @ self.link_from_tcp

    def _solution_positions(self, solution) -> np.ndarray:
        """按活动关节名提取，保留 batch/seed 维；Pika 解可能包含锁定指关节。"""
        return solution.reorder(self.planner_joint_names).position.detach().cpu().numpy()

    def _trajectory_geometry(self, positions) -> tuple[np.ndarray, np.ndarray]:
        """Batch FK for TCP paths and robot collision-sphere centers in world."""
        from curobo.types import JointState

        values = np.asarray(positions, dtype=float)
        if (values.ndim != 2 or values.shape[1] != len(self.joint_names)
                or len(values) == 0 or not np.all(np.isfinite(values))):
            raise ValueError("诊断轨迹必须是有限 [T,dof] 数组")
        tensor = self.torch.as_tensor(
            values, device="cuda", dtype=self.torch.float32,
        )
        state = self.kinematics.compute_kinematics(
            JointState.from_position(tensor, joint_names=self.planner_joint_names)
        )
        link_pose = state.tool_poses.get_link_pose(self.tool_frame)
        link_positions = link_pose.position.detach().cpu().numpy().reshape(-1, 3)
        link_quaternions = link_pose.quaternion.detach().cpu().numpy().reshape(-1, 4)
        link_rotations = Rotation.from_quat(
            link_quaternions[:, [1, 2, 3, 0]],
        ).as_matrix()
        tcp_base = link_positions + np.einsum(
            "nij,j->ni", link_rotations, self.link_from_tcp[:3, 3],
        )
        tcp_world = (
            self.world_from_base[:3, :3] @ tcp_base.T
        ).T + self.world_from_base[:3, 3]
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(
            len(values), -1, 4,
        )
        centers = spheres[..., :3]
        centers_world = np.einsum(
            "ij,ntj->nti", self.world_from_base[:3, :3], centers,
        ) + self.world_from_base[:3, 3]
        active = spheres[..., 3] > 0
        # Keep the cloud compact while preserving the whole arm at several times.
        step = max(1, len(values) // 8)
        sampled = centers_world[::step]
        sampled_active = active[::step]
        return tcp_world, sampled[sampled_active]

    @staticmethod
    def _optimizer_history(raw) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Return exact optimizer iterates and final all-seed joint trajectories."""
        debug = ((getattr(raw, "debug_info", None) or {}).get("solver") or {}).get(
            "debug", []
        )
        history = []
        for value in debug:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=float)
            if array.size and np.all(np.isfinite(array)):
                history.append(array)
        return history, (history[-1] if history else None)

    def _failure_output_path(self, diagnostic_label: str | None) -> Path:
        self._diagnostic_export_count += 1
        label = diagnostic_label or f"failure_{self._diagnostic_export_count:04d}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._") or "failure"
        root = Path(self.diagnostic_output_dir)
        candidate = root / safe
        suffix = 2
        while candidate.exists():
            candidate = root / f"{safe}_{suffix:02d}"
            suffix += 1
        return candidate

    def _export_failure_visualization(
        self, result: dict, *, start_positions, target, raw=None,
        diagnostic_label: str | None = None,
    ) -> None:
        """Export every available failed trajectory without changing solver outcome."""
        if self.diagnostic_output_dir is None:
            return
        traces = []
        history, final_all_seeds = self._optimizer_history(raw) if raw is not None else ([], None)
        if final_all_seeds is not None:
            values = np.asarray(final_all_seeds, dtype=float)
            if values.ndim >= 3 and values.shape[-1] == len(self.joint_names):
                paths = values.reshape(-1, values.shape[-2], values.shape[-1])
                for index, path in enumerate(paths):
                    tcp, spheres = self._trajectory_geometry(path)
                    validity = self._configuration_validity(path)
                    first_invalid = validity.index(False) if not all(validity) else None
                    collision = None
                    if first_invalid is not None:
                        collision = self._collision_diagnostic(
                            path[first_invalid],
                            context_label=f"trajopt_seed_{index}_waypoint_{first_invalid}",
                        )
                    traces.append({
                        "trace_id": f"trajopt_seed_{index:03d}",
                        "source": "curobo_optimizer_final_iteration_seed",
                        "synthetic": False,
                        "joint_positions_rad": path.tolist(),
                        "tcp_path_world_m": tcp.tolist(),
                        "robot_sphere_centers_world_m": spheres.tolist(),
                        "first_invalid_waypoint": first_invalid,
                        "collision_check": collision,
                    })
        if not traces:
            start_check = result.get("start_state_check") or {}
            collision = start_check.get("collision_diagnostic")
            if collision is not None and not start_check.get("valid", True):
                path = np.asarray(start_positions, dtype=float).reshape(1, -1)
                tcp, spheres = self._trajectory_geometry(path)
                traces.append({
                    "trace_id": "preflight_start_configuration",
                    "source": "exact_preflight_start_configuration",
                    "synthetic": False,
                    "note": (
                        "Planning stopped before solver execution because the exact "
                        "start configuration failed preflight validation."
                    ),
                    "joint_positions_rad": path.tolist(),
                    "tcp_path_world_m": tcp.tolist(),
                    "robot_sphere_centers_world_m": spheres.tolist(),
                    "first_invalid_waypoint": 0,
                    "collision_check": collision,
                })
        if not traces and raw is not None:
            solution = getattr(getattr(raw, "js_solution", None), "position", None)
            if solution is not None:
                path = solution.detach().cpu().numpy().reshape(-1, len(self.joint_names))
                if len(path):
                    tcp, spheres = self._trajectory_geometry(path)
                    validity = self._configuration_validity(path)
                    first_invalid = validity.index(False) if not all(validity) else None
                    collision = (self._collision_diagnostic(
                        path[first_invalid], context_label="best_failed_trajectory_waypoint",
                    ) if first_invalid is not None else None)
                    traces.append({
                        "trace_id": "best_failed_trajectory",
                        "source": "curobo_returned_best_failed_trajectory",
                        "synthetic": False,
                        "joint_positions_rad": path.tolist(),
                        "tcp_path_world_m": tcp.tolist(),
                        "robot_sphere_centers_world_m": spheres.tolist(),
                        "first_invalid_waypoint": first_invalid,
                        "collision_check": collision,
                    })
        diagnostic = result.get("ik_diagnostic") or {}
        if not traces and diagnostic.get("seeds"):
            start = np.asarray(start_positions, dtype=float)
            for seed in diagnostic["seeds"]:
                endpoint = np.asarray(seed["joint_positions_rad"], dtype=float)
                alpha = np.linspace(0.0, 1.0, 25)[:, None]
                # This path is diagnostic only: cuRobo stopped before TrajOpt.
                path = start[None] + alpha * (endpoint - start)[None]
                tcp, spheres = self._trajectory_geometry(path)
                validity = self._configuration_validity(path)
                first_invalid = validity.index(False) if not all(validity) else None
                collision = (self._collision_diagnostic(
                    path[first_invalid],
                    context_label=f"failed_ik_seed_{seed['index']}_interpolation",
                ) if first_invalid is not None else None)
                traces.append({
                    "trace_id": f"failed_ik_seed_{int(seed['index']):03d}",
                    "source": "diagnostic_joint_interpolation_to_failed_ik_seed",
                    "synthetic": True,
                    "note": "cuRobo returned no trajectory; this dashed path only exposes the failed IK endpoint in 3D",
                    "ik_seed": seed,
                    "joint_positions_rad": path.tolist(),
                    "tcp_path_world_m": tcp.tolist(),
                    "robot_sphere_centers_world_m": spheres.tolist(),
                    "first_invalid_waypoint": first_invalid,
                    "collision_check": collision,
                })
        if not traces:
            result["failure_visualization_error"] = (
                "cuRobo failure contained neither optimizer trajectories nor IK seed states"
            )
            return
        from gene_agent.visualization.curobo_failure import export_failure_bundle

        exact = sum(not item["synthetic"] for item in traces)
        output = self._failure_output_path(diagnostic_label)
        history_payload = {
            f"iteration_{index:04d}": np.asarray(value, dtype=np.float32)
            for index, value in enumerate(history)
        } or None
        result["failure_visualization"] = export_failure_bundle(
            output,
            scene_model=self.scene_model,
            target_pose_world=np.asarray(target, dtype=float),
            traces=traces,
            optimizer_history=history_payload,
            world_from_scene=self.world_from_base,
            metadata={
                "title": f"cuRobo {result.get('failure_category')} - {diagnostic_label or 'unlabeled'}",
                "failure_category": result.get("failure_category"),
                "failure_detail": result.get("failure_detail"),
                "backend_status": result.get("status"),
                "scene_scope": self.scene_scope,
                "scene_snapshot_sha256": self.scene_snapshot_sha256,
                "trace_semantics": (
                    "solid paths are exact final cuRobo optimizer seeds; dashed paths are "
                    "diagnostic interpolation only when cuRobo stopped before trajectory optimization"
                ),
                "exact_solver_trajectory_count": exact,
                "synthetic_trace_count": len(traces) - exact,
            },
        )

    def export_pre_execution_failure(
        self, positions, target, *, first_invalid_waypoint: int,
        diagnostic_label: str, precheck: dict,
    ) -> dict:
        """Export the complete accepted solver path rejected by a later precheck."""
        path = np.asarray(positions, dtype=float)
        if (path.ndim != 2 or path.shape[1] != len(self.joint_names)
                or len(path) == 0 or not np.all(np.isfinite(path))):
            raise ValueError("pre-execution diagnostic path must be finite [T,dof]")
        invalid = int(first_invalid_waypoint)
        if not 0 <= invalid < len(path):
            raise ValueError("first_invalid_waypoint is outside the diagnostic path")
        tcp, spheres = self._trajectory_geometry(path)
        collision = self._collision_diagnostic(
            path[invalid], context_label=f"pre_execution_waypoint_{invalid}",
        )
        collision["external_precheck"] = precheck
        from gene_agent.visualization.curobo_failure import export_failure_bundle

        output = self._failure_output_path(diagnostic_label)
        return export_failure_bundle(
            output,
            scene_model=self.scene_model,
            target_pose_world=np.asarray(target, dtype=float),
            traces=({
                "trace_id": "accepted_solver_path_rejected_by_precheck",
                "source": "exact_curobo_solver_path_then_external_precheck",
                "synthetic": False,
                "joint_positions_rad": path.tolist(),
                "tcp_path_world_m": tcp.tolist(),
                "robot_sphere_centers_world_m": spheres.tolist(),
                "first_invalid_waypoint": invalid,
                "collision_check": collision,
            },),
            world_from_scene=self.world_from_base,
            metadata={
                "title": f"pre-execution check failed - {diagnostic_label}",
                "failure_category": "collision_postcheck_rejected",
                "backend_status": "pre_execution_contact_rejected",
                "scene_scope": self.scene_scope,
                "scene_snapshot_sha256": self.scene_snapshot_sha256,
                "trace_semantics": (
                    "complete exact cuRobo solver path; the marked waypoint is the "
                    "first failure reported by the external pre-execution check"
                ),
                "exact_solver_trajectory_count": 1,
                "synthetic_trace_count": 0,
                "external_precheck": precheck,
            },
        )

    def export_execution_collision(
        self, positions, target, *, diagnostic_label: str,
        execution_failure: dict,
    ) -> dict:
        """Render the exact actual configuration where MuJoCo stopped execution.

        The cuRobo collision result and the external MuJoCo contact evidence stay
        separate in the artifact, because their proxy geometries need not agree.
        """
        configuration = np.asarray(positions, dtype=float)
        if (configuration.shape != (len(self.joint_names),)
                or not np.all(np.isfinite(configuration))):
            raise ValueError("execution collision configuration must be finite [dof]")
        path = configuration.reshape(1, -1)
        tcp, spheres = self._trajectory_geometry(path)
        collision = self._collision_diagnostic(
            configuration, context_label="actual_execution_stop_configuration",
        )
        collision["external_mujoco_execution_failure"] = execution_failure
        from gene_agent.visualization.curobo_failure import export_failure_bundle

        output = self._failure_output_path(diagnostic_label)
        return export_failure_bundle(
            output,
            scene_model=self.scene_model,
            target_pose_world=np.asarray(target, dtype=float),
            traces=({
                "trace_id": "actual_execution_collision_stop",
                "source": "actual_mujoco_stop_configuration_checked_by_curobo",
                "synthetic": False,
                "joint_positions_rad": path.tolist(),
                "tcp_path_world_m": tcp.tolist(),
                "robot_sphere_centers_world_m": spheres.tolist(),
                "first_invalid_waypoint": 0,
                "collision_check": collision,
            },),
            world_from_scene=self.world_from_base,
            metadata={
                "title": f"execution collision stop - {diagnostic_label}",
                "failure_category": "execution_collision_stop",
                "backend_status": execution_failure.get("failure_type"),
                "scene_scope": self.scene_scope,
                "scene_snapshot_sha256": self.scene_snapshot_sha256,
                "trace_semantics": (
                    "single exact actual MuJoCo stop configuration; cuRobo collision "
                    "proxy result is recorded separately from MuJoCo contact evidence"
                ),
                "exact_solver_trajectory_count": 1,
                "synthetic_trace_count": 0,
                "external_mujoco_execution_failure": execution_failure,
            },
        )

    def _goal(self, target, target_frame):
        from curobo.types import GoalToolPose, Pose
        if target_frame != "world_tcp":
            raise ValueError("当前仅接受明确的 world_tcp 目标")
        target = np.asarray(target, float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("目标必须为有限 4x4 齐次矩阵")
        if not np.allclose(target[3], [0, 0, 0, 1]) or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(target[:3, :3]) < 0:
            raise ValueError("目标必须是合法刚体变换")
        base_tcp = np.linalg.inv(self.world_from_base) @ target
        base_link = base_tcp @ np.linalg.inv(self.link_from_tcp)
        quat = Rotation.from_matrix(base_link[:3, :3]).as_quat()[[3, 0, 1, 2]]
        pose = Pose(position=self.torch.tensor(base_link[None, :3, 3], device="cuda", dtype=self.torch.float32),
                    quaternion=self.torch.tensor(quat[None], device="cuda", dtype=self.torch.float32))
        tcp_quat = Rotation.from_matrix(base_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]]
        tcp_pose = Pose(position=self.torch.tensor(base_tcp[None, :3, 3], device="cuda", dtype=self.torch.float32),
                        quaternion=self.torch.tensor(tcp_quat[None], device="cuda", dtype=self.torch.float32))
        # extra_links 注册了两个 tool frame；普通 IK/规划目标也必须为两者
        # 提供同一刚体目标，否则 cuRobo 的 reorder_links 在求解前就会拒绝。
        return GoalToolPose.from_poses({self.tool_frame: pose, self.tcp_frame: tcp_pose},
                                        ordered_tool_frames=[self.tool_frame, self.tcp_frame], num_goalset=1)

    def _linear_goal(self, target, target_frame, start_tcp_world: np.ndarray, *,
                     stationary_rotation: bool = False):
        """真实末端目标保持原姿态，辅助TCP目标只编码直线方向。"""
        from curobo.types import GoalToolPose, Pose

        self._goal(target, target_frame)  # 复用目标frame与刚体变换校验。
        base_target_tcp = np.linalg.inv(self.world_from_base) @ np.asarray(target, float)
        base_target_link = base_target_tcp @ np.linalg.inv(self.link_from_tcp)
        base_start_tcp = np.linalg.inv(self.world_from_base) @ np.asarray(start_tcp_world, float)
        direction = base_target_tcp[:3, 3] - base_start_tcp[:3, 3]
        if stationary_rotation and np.linalg.norm(direction) <= 1e-9:
            # pivot可原位旋转，没有平移方向时使用起点TCP轴建立辅助frame。
            # 真正末端目标、pivot路程/端点及碰撞后检全部保留；普通直线不改。
            direction = base_start_tcp[:3, 2]
        virtual_target = base_target_tcp.copy()
        virtual_target[:3, :3] = line_frame_rotation(direction)

        def as_pose(matrix):
            quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
            return Pose(
                position=self.torch.tensor(matrix[None, :3, 3], device="cuda", dtype=self.torch.float32),
                quaternion=self.torch.tensor(quaternion[None], device="cuda", dtype=self.torch.float32),
            )

        return GoalToolPose.from_poses(
            {self.tool_frame: as_pose(base_target_link), self.tcp_frame: as_pose(virtual_target)},
            ordered_tool_frames=[self.tool_frame, self.tcp_frame], num_goalset=1,
        )

    def _followup_ik_goal(self, operation, target, target_frame, solver_goal):
        """为规划失败诊断选择物理目标，避免把直线辅助 frame 当成真实 TCP。"""
        if operation in {"linear_tcp", "pivot_tcp"}:
            return self._goal(target, target_frame)
        return solver_goal

    @contextmanager
    def _linear_criteria(self, non_terminal_scale: float = 1.0):
        """约束仅作用于本次规划；退出后恢复两个tool frame的默认准则。"""
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        scale = float(non_terminal_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("non_terminal_scale 必须为有限非负数")

        if self._planner is None or not hasattr(self._planner, "update_tool_pose_criteria"):
            raise RuntimeError("cuRobo planner 不支持TCP ToolPoseCriteria")
        self._planner.update_tool_pose_criteria({
            self.tool_frame: ToolPoseCriteria(),
            self.tcp_frame: ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1., 1., 1., 0., 0., 0.],
                non_terminal_pose_axes_weight_factor=[scale, scale, 0., 0., 0., 0.],
                project_distance_to_goal=True,
            ),
        })
        try:
            yield
        finally:
            defaults = ToolPoseCriteria()
            self._planner.update_tool_pose_criteria({
                self.tool_frame: defaults, self.tcp_frame: ToolPoseCriteria(),
            })

    def _configuration_validity(self, positions) -> list[bool]:
        """独立验证离散关节姿态，给失败分类提供可复核的碰撞事实。"""
        if self._collision_checker is None:
            from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
            config = RobotCollisionCheckerCfg.load_from_config(
                robot_config=copy.deepcopy(self.robot_config),
                scene_model=self.scene_model,
                n_cuboids=self.collision_cache["cuboid"],
                n_meshes=self.collision_cache["mesh"],
                collision_activation_distance=0.0,
                self_collision_activation_distance=0.0,
            )
            self._collision_checker = RobotCollisionChecker(config)
        from contextlib import nullcontext
        from ..collision import optimizer_mounting_mask
        values = self.torch.as_tensor(
            np.asarray(positions, dtype=float), device="cuda", dtype=self.torch.float32
        ).reshape(1, -1, len(self.joint_names))
        mask_context = (
            optimizer_mounting_mask(
                self._collision_checker.kinematics, self.mounting_contact_policy
            )
            if self._validator else nullcontext()
        )
        with mask_context:
            valid = self._collision_checker.validate(values)
        return [bool(value) for value in valid.detach().cpu().reshape(-1).tolist()]

    def validate_joint_trajectory(self, joint_names, positions) -> dict:
        """在当前冻结场景中复核一条外部给定轨迹的离散与连续碰撞。

        候选筛选阶段产生的轨迹只有在实际阶段起点和当前碰撞世界中重新
        通过该检查后才可复用；不能因为它曾经规划成功就直接下发。
        """
        names = tuple(joint_names)
        values = np.asarray(positions, dtype=float)
        if (len(names) != len(set(names)) or set(names) != set(self.joint_names)
                or values.ndim != 2 or values.shape[1] != len(names)
                or len(values) < 2 or not np.isfinite(values).all()):
            raise ValueError("待复核轨迹必须包含完整唯一关节和至少两个有限 waypoint")
        ordered = np.ascontiguousarray(
            values[:, [names.index(name) for name in self.joint_names]], dtype=float,
        )
        validity = self._configuration_validity(ordered)
        result = {
            "valid": bool(all(validity)),
            "checked_waypoints": len(validity),
            "first_invalid_index": None,
            "source": "current_scene_discrete_collision_validation",
        }
        if not all(validity):
            result["first_invalid_index"] = validity.index(False)
            return result
        if self._validator is not None:
            result = self._validator.check(ordered, self.planner_joint_names)
            result["source"] = "current_scene_continuous_collision_validation"
        return result

    def _start_configuration_check(self, positions) -> dict:
        """缓存严格起点复核，避免 optimizer 的宽泛 mounting mask 漏掉外物。

        optimizer mask 只是求解技巧，会把若干近基座 link 对*所有*外物的
        球一起关掉。真正允许的接触仅限 policy 中的指定 link×本体障碍物，
        因此必须再用 MountingCollisionValidator 的分组检查放行起点。
        """
        key = tuple(float(value) for value in positions)
        if key in self._start_validation_cache:
            return copy.deepcopy(self._start_validation_cache[key])
        optimizer_valid = self._configuration_validity([key])[0]
        strict = self._validator.check([key], self.planner_joint_names) if self._validator else None
        valid = optimizer_valid and (strict is None or strict["valid"])
        result = {
            "valid": valid,
            "optimizer_valid": optimizer_valid,
            "strict_mounting_check": strict,
        }
        if not valid:
            try:
                result["collision_diagnostic"] = self._collision_diagnostic(key, context_label="start_state")
            except Exception as error:
                result["collision_diagnostic_error"] = f"{type(error).__name__}: {error}"
        self._start_validation_cache[key] = copy.deepcopy(result)
        return result

    def _collision_sphere_records(self, state, *, scene_hits: dict[int, set[str]],
                                  self_hits: set[int]) -> list[dict]:
        """Serialize the exact active cuRobo spheres at one checked configuration."""
        spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
        params = self._collision_checker.kinematics.config.kinematics_config
        link_ids = params.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
        link_names = {int(value): str(key)
                      for key, value in params.link_name_to_idx_map.items()}
        rotation = self.world_from_base[:3, :3]
        translation = self.world_from_base[:3, 3]
        records = []
        for index, sphere in enumerate(spheres):
            radius = float(sphere[3])
            if radius <= 0.0:
                continue
            center = rotation @ sphere[:3] + translation
            obstacles = sorted(scene_hits.get(index, set()))
            types = []
            if obstacles:
                types.append("scene_collision")
            if index in self_hits:
                types.append("self_collision")
            records.append({
                "sphere_index": index,
                "link": link_names.get(int(link_ids[index]), f"link_index_{int(link_ids[index])}"),
                "center_world_m": center.tolist(),
                "radius_m": radius,
                "collision_types": types,
                "colliding_obstacles": obstacles,
            })
        return records

    def _collision_diagnostic(self, positions, *, context_label: str = "failed_solver_best_candidate") -> dict:
        """按严格接触策略定位失败姿态的限位、自碰撞和场景物体。

        外部物体始终使用完整碰撞球；只有指定的本体障碍物才临时屏蔽
        policy 内的安装 link。逐物体查询后恢复原 enable 状态。
        """
        from ..collision import optimizer_mounting_mask

        if self._collision_checker is None:
            self._configuration_validity([positions])
        checker = self._collision_checker
        query = self.torch.as_tensor(positions, device="cuda", dtype=self.torch.float32).reshape(1, 1, -1)
        checker.setup_batch_tensors(1, 1)
        full_state = checker.get_kinematics(query)
        checked_state = full_state
        bounds = bool((checker.get_bound(query) > 0).any().item())
        self_indices: set[int] = set()
        self_pairs = []
        if checker.self_collision_cost is not None:
            self_value = checker.get_self_collision(full_state.robot_spheres)
            self_hit = bool((self_value > 0).any().item())
            if self_hit:
                try:
                    distances = checker.self_collision_cost._pair_distance.detach().reshape(-1)
                    pairs = checker.kinematics.get_self_collision_config().collision_pairs
                    pairs = pairs.detach().reshape(-1, 2)
                    active = distances[:len(pairs)] > 0
                    for first, second in pairs[active].detach().cpu().numpy().tolist():
                        first, second = int(first), int(second)
                        self_indices.update((first, second))
                        self_pairs.append([first, second])
                except (AttributeError, TypeError, RuntimeError, ValueError):
                    # cuRobo versions without the pair-distance debug buffer still
                    # retain the aggregate self-collision result below.
                    pass
        else:
            self_hit = None
        obstacles = []
        allowed_mounting = []
        scene_hits: dict[int, set[str]] = {}
        world = checker.scene_model
        if checker.collision_constraint is not None and world is not None:
            states = []
            for key in ("cuboids", "meshes", "voxels"):
                collection = getattr(world.data, key, None)
                if collection is None:
                    continue
                names = tuple(str(name) for name in collection.get_names(0))
                enabled = [bool(collection.enable[0, index].item()) for index in range(len(names))]
                states.append((collection, names, enabled))
            body_names = set(self.mounting_contact_policy.get("obstacle_names", []))
            active_items = [
                (collection, name)
                for collection, names, enabled in states
                for name, active in zip(names, enabled) if active
            ]
            # 先检测所有外物；mounting mask 的 FK 可能重用内部 GPU buffer，
            # 随后对本体逐项重新计算完整 FK，避免残留的屏蔽球污染外物诊断。
            active_items.sort(key=lambda item: item[1] in body_names)
            try:
                for collection, name in active_items:
                    for other, other_names, other_enabled in states:
                        for obstacle, was_enabled in zip(other_names, other_enabled):
                            other.set_enabled(obstacle, was_enabled and obstacle == name, 0)
                    full_state = checker.get_kinematics(query)
                    checked_state = full_state
                    full_scores = checker.get_collision_constraint(full_state).detach().reshape(-1)
                    full_hit = bool((full_scores > 0).any().item())
                    if not full_hit:
                        continue
                    if name in body_names and self._validator:
                        with optimizer_mounting_mask(checker.kinematics, self.mounting_contact_policy):
                            masked_state = checker.get_kinematics(query)
                            restricted_scores = checker.get_collision_constraint(masked_state).detach().reshape(-1)
                            restricted_hit = bool((restricted_scores > 0).any().item())
                        (obstacles if restricted_hit else allowed_mounting).append(name)
                        hit_scores = restricted_scores if restricted_hit else None
                    else:
                        obstacles.append(name)
                        hit_scores = full_scores
                    if hit_scores is not None:
                        for sphere_index in self.torch.nonzero(
                            hit_scores > 0, as_tuple=False,
                        ).detach().cpu().reshape(-1).tolist():
                            scene_hits.setdefault(int(sphere_index), set()).add(name)
            finally:
                for collection, names, enabled in states:
                    for name, active in zip(names, enabled):
                        collection.set_enabled(name, active, 0)
        # Kinematics buffers are reused internally; recompute after every
        # temporary mounting mask has been restored before serializing spheres.
        checked_state = checker.get_kinematics(query)
        sphere_records = self._collision_sphere_records(
            checked_state, scene_hits=scene_hits, self_hits=self_indices,
        )
        return {
            "joint_limit_violation": bounds,
            "self_collision": self_hit,
            "scene_collision": bool(obstacles),
            "colliding_obstacles": obstacles,
            "allowed_mounting_contacts": allowed_mounting,
            "checked_configuration": context_label,
            "robot_collision_spheres_world": sphere_records,
            "self_collision_sphere_pairs": self_pairs,
            "collision_sphere_summary": {
                "active_count": len(sphere_records),
                "scene_colliding_count": sum(
                    "scene_collision" in record["collision_types"] for record in sphere_records
                ),
                "self_colliding_count": sum(
                    "self_collision" in record["collision_types"] for record in sphere_records
                ),
                "attribution": (
                    "per_sphere" if (scene_hits or self_indices)
                    else "aggregate_only" if (self_hit or obstacles)
                    else "no_collision"
                ),
            },
        }

    def _diagnose_world_obstruction(self, start, goal) -> dict:
        """只在 IK 零成功时比较无环境障碍的解，并用原场景复核。"""
        if self.collision_mode is CollisionMode.ROBOT_ONLY:
            return {"world_collision_disabled_for_probe": False, "relaxed_ik_success": None,
                    "reason": "not_applicable_robot_only", "blocking_obstacles": []}
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

        empty_scene = {"cuboid": {}, "mesh": {}}
        solver = InverseKinematics(InverseKinematicsCfg.create(
            robot=copy.deepcopy(self.robot_config), scene_model=empty_scene,
            collision_cache=scene_collision_cache(empty_scene),
            self_collision_check=True, num_seeds=16, **self.runtime_options,
        ))
        try:
            raw = solver.solve_pose(goal, current_state=start)
            success = bool(raw.success.any().item())
            collision = None
            error = None
            if success:
                positions = raw.solution.detach().cpu().numpy().reshape(-1, len(self.joint_names))[0]
                try:
                    collision = self._collision_diagnostic(
                        positions, context_label="world_relaxed_ik_solution"
                    )
                except Exception as diagnostic_error:
                    error = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            result = classify_world_obstruction(success, collision)
            if error:
                result["strict_collision_check_error"] = error
            return result
        finally:
            # 诊断也有 CUDA Graph/solver 资源；失败时同样释放，不缓存旧碰撞状态。
            try:
                solver.destroy()
            except Exception:
                pass


    def _solve(self, operation, joint_names, positions, target, target_frame, *,
               max_attempts: int = 1, non_terminal_scale: float = 1.0,
               pivot_limits: tuple[float, float] | None = None,
               diagnostic_label: str | None = None):
        if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts 必须是 1 到 5 的整数")
        if operation == "pivot_tcp":
            if (pivot_limits is None or len(pivot_limits) != 2
                    or not np.isfinite(pivot_limits).all()
                    or any(value <= 0 for value in pivot_limits)):
                raise ValueError("旋转阶段须给出正的 TCP 最大偏移与总路程上限")
        self._failure_diagnostic_budget = None
        started = time.perf_counter()
        phase_timing = {}
        try:
            from contextlib import nullcontext
            from ..collision import optimizer_mounting_mask
            linear = operation in {"linear_tcp", "pivot_tcp"}
            start_tcp_world = self.fk(joint_names, positions) if linear else None
            start = self._state(joint_names, positions)
            if operation == "pivot_tcp":
                goal = self._linear_goal(target, target_frame, start_tcp_world, stationary_rotation=True)
            else:
                goal = (self._linear_goal(target, target_frame, start_tcp_world)
                        if linear else self._goal(target, target_frame))
            start_check = self._start_configuration_check(
                ordered_positions(joint_names, positions, self.joint_names)
            )
            phase_timing["input_and_start_check_s"] = time.perf_counter()-started
            if not start_check["valid"]:
                status, category, message = describe_start_failure(start_check)
                result = dict(
                    success=False, **self.gripper_metadata, status=status,
                    operation=operation, raw_result=None, raw_error=None,
                    message=message,
                    failure_category=category, joint_names=self.joint_names,
                    scene_scope=self.scene_scope,
                    scene_snapshot_sha256=self.scene_snapshot_sha256,
                    planner_mode=self.planner_mode.value,
                    use_cuda_graph=self.runtime_options["use_cuda_graph"],
                    random_seed=self.random_seed,
                    collision_verification={"valid": False, "first_invalid_index": 0,
                                            "start_check": start_check},
                    start_state_check=start_check,
                    duration_s=time.perf_counter()-started, timing=phase_timing,
                )
                try:
                    self._export_failure_visualization(
                        result,
                        start_positions=ordered_positions(
                            joint_names, positions, self.joint_names,
                        ),
                        target=target,
                        diagnostic_label=diagnostic_label,
                    )
                except Exception as visualization_error:
                    result["failure_visualization_error"] = (
                        f"{type(visualization_error).__name__}: {visualization_error}"
                    )
                return result
            initialization_started = time.perf_counter()
            if operation == "ik":
                if self._ik is None:
                    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
                    self._ik = InverseKinematics(InverseKinematicsCfg.create(robot=copy.deepcopy(self.robot_config), scene_model=self.scene_model,
                        collision_cache=self.collision_cache,
                        optimizer_collision_activation_distance=self.optimizer_collision_activation_distance,
                        self_collision_check=True, num_seeds=16, **self.runtime_options))
                phase_timing["solver_initialization_s"] = time.perf_counter()-initialization_started
                solver_started = time.perf_counter()
                with optimizer_mounting_mask(self._ik.kinematics, self.mounting_contact_policy) if self._validator else nullcontext():
                    raw = self._ik.solve_pose(goal, current_state=start)
            else:
                if self._planner is None:
                    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
                    self._planner = MotionPlanner(MotionPlannerCfg.create(robot=copy.deepcopy(self.robot_config), scene_model=self.scene_model,
                        collision_cache=self.collision_cache,
                        optimizer_collision_activation_distance=self.optimizer_collision_activation_distance,
                        self_collision_check=True, num_ik_seeds=16, num_trajopt_seeds=4,
                        interpolation_dt=0.025,
                        store_debug=(self.diagnostic_output_dir is not None
                                     and self.planner_mode is PlannerMode.DEVELOPMENT),
                        **self.runtime_options))
                phase_timing["solver_initialization_s"] = time.perf_counter()-initialization_started
                solver_started = time.perf_counter()
                with self._linear_criteria(non_terminal_scale) if linear else nullcontext():
                    with optimizer_mounting_mask(self._planner.kinematics, self.mounting_contact_policy) if self._validator else nullcontext():
                        cspace_goal = getattr(self, "_cspace_goal_positions", None)
                        if cspace_goal is not None:
                            raw = self._planner.plan_cspace(
                                self._state(self.joint_names, cspace_goal),
                                start, max_attempts=max_attempts,
                            )
                        else:
                            raw = self._planner.plan_pose(goal, start, max_attempts=max_attempts)
            self.torch.cuda.synchronize()
            phase_timing["solver_call_including_graph_capture_s"] = time.perf_counter()-solver_started
            postprocess_started = time.perf_counter()
            if raw is None:
                empty = dict(
                    empty_solver_result("plan" if linear else operation),
                    **self.gripper_metadata,
                    operation=operation,
                    joint_names=self.joint_names,
                    scene_scope=self.scene_scope,
                    scene_snapshot_sha256=self.scene_snapshot_sha256,
                    planner_mode=self.planner_mode.value,
                    use_cuda_graph=self.runtime_options["use_cuda_graph"],
                    random_seed=self.random_seed,
                    duration_s=time.perf_counter()-started, timing=phase_timing,
                    start_state_check=start_check,
                )
                options = getattr(self, 'diagnostic_options', None)
                budget = None
                if options is not None and options.mode == 'on_failure':
                    from .diagnostics import DiagnosticBudget
                    budget = self._failure_diagnostic_budget = DiagnosticBudget(options)
                if operation != "ik" and (budget is None or budget.allow('followup_ik', solver=True)):
                    try:
                        # 上游 plan_pose 丢弃无成功种子时的 IKResult；独立再求一次
                        # 只用于诊断，并明确记录其来源，不伪称为原规划内部结果。
                        diagnostic_goal = self._followup_ik_goal(
                            operation, target, target_frame, goal,
                        )
                        diagnostic_raw = self._planner.ik_solver.solve_pose(
                            diagnostic_goal, current_state=start,
                            return_seeds=self._planner.ik_solver.config.num_seeds,
                        )
                        diagnostic = summarize_ik_failure(
                            diagnostic_raw, self._planner.ik_solver,
                        )
                        try:
                            if budget is not None and not budget.allow("selected_candidate_collision"):
                                raise TimeoutError("diagnostic budget exhausted")
                            diagnostic["collision_check"] = self._collision_diagnostic(
                                diagnostic["selected_joint_positions_rad"]
                            )
                            diagnostic["collision_checked"] = True
                            diagnostic["collision_confirmed"] = bool(
                                diagnostic["collision_check"]["self_collision"]
                                or diagnostic["collision_check"]["scene_collision"]
                            )
                        except Exception as diagnostic_error:
                            diagnostic["collision_check_error"] = (
                                f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                            )
                        empty["ik_diagnostic"] = diagnostic
                        if diagnostic["successful_seed_count"] == 0 and (budget is None or budget.allow("world_relaxed_ik", solver=True)):
                            try:
                                diagnostic["world_obstruction_probe"] = (
                                    self._diagnose_world_obstruction(start, goal)
                                )
                            except Exception as diagnostic_error:
                                diagnostic["world_obstruction_probe_error"] = (
                                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                                )
                    except Exception as diagnostic_error:
                        empty["ik_diagnostic_error"] = (
                            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                        )
                    empty["duration_s"] = time.perf_counter()-started
                    detail = classify_failure_detail(
                        "goal_ik",
                        empty.get("ik_diagnostic"),
                        world_obstruction_probe=(empty.get("ik_diagnostic") or {}).get(
                            "world_obstruction_probe"
                        ),
                    )
                    empty["failure_detail"] = detail
                    empty["message"] = detail["human_summary"]
                    try:
                        self._export_failure_visualization(
                            empty,
                            start_positions=ordered_positions(
                                joint_names, positions, self.joint_names,
                            ),
                            target=target,
                            diagnostic_label=diagnostic_label,
                        )
                    except Exception as visualization_error:
                        empty["failure_visualization_error"] = (
                            f"{type(visualization_error).__name__}: {visualization_error}"
                        )
                if "failure_detail" not in empty:
                    detail = classify_failure_detail("goal_ik")
                    empty["failure_detail"] = detail
                    empty["message"] = detail["human_summary"]
                if ("failure_visualization" not in empty
                        and "failure_visualization_error" not in empty):
                    empty["failure_visualization_error"] = (
                        "cuRobo returned no IK seed states or optimizer trajectory; "
                        "a trajectory visualization cannot be generated without inventing a path"
                    )
                return empty
            success = bool(raw.success.all().item())
            result = dict(success=success, status=str(getattr(raw, "status", "success" if success else "solver_failed")),
                          raw_result=repr(raw), raw_error=None, failure_category=None if success else "solver_rejected")
            if not success:
                options=getattr(self,'diagnostic_options',None)
                budget=None
                if options is not None and options.mode=='on_failure':
                    from .diagnostics import DiagnosticBudget
                    budget=self._failure_diagnostic_budget=DiagnosticBudget(options)
                solver = self._ik if operation == "ik" else self._planner.trajopt_solver
                diagnostic = solver_failure_diagnostic(raw, "plan" if linear else operation, solver)
                if operation == "ik":
                    # Knowledge 预览可显示失败种子的真实 FK；该姿态不是成功 IK。
                    candidate = getattr(raw, "solution", None)
                    if candidate is not None:
                        values = candidate.detach().cpu().numpy().reshape(-1, len(self.joint_names))[0]
                        if np.isfinite(values).all():
                            diagnostic["joint_positions_rad"] = values.astype(float).tolist()
                if operation == "ik" and diagnostic["constraint_feasible"] is False:
                    solution = getattr(raw, "solution", None)
                    if solution is not None:
                        candidate_q = solution.detach().cpu().numpy().reshape(-1, len(self.joint_names))[0]
                        try:
                            if budget is not None and not budget.allow('failed_ik_constraint_check'):
                                raise TimeoutError('diagnostic budget exhausted')
                            diagnostic["collision_check"] = self._collision_diagnostic(candidate_q)
                            diagnostic["collision_confirmed"] = bool(
                                diagnostic["collision_check"]["self_collision"]
                                or diagnostic["collision_check"]["scene_collision"]
                            )
                        except Exception as diagnostic_error:
                            diagnostic["collision_check_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                if operation != "ik":
                    solution = getattr(getattr(raw, "js_solution", None), "position", None)
                    if solution is not None:
                        try:
                            sampled = solution.detach().cpu().numpy().reshape(-1, len(self.joint_names))
                            if getattr(getattr(self, "diagnostic_options", None), "mode", "off") == "on_failure":
                                from .diagnostics import retain_failed_trajectories
                                result["diagnostic_trajectories"] = retain_failed_trajectories(solution.detach().cpu().numpy(), len(self.joint_names))
                            if budget is not None:
                                from .diagnostics import check_failed_trajectories
                                checked=check_failed_trajectories(self,result['diagnostic_trajectories'],budget)
                                diagnostic['sampled_waypoints']=checked['sampled_waypoints']
                                first=checked['first_invalid']
                                candidate=None if first is None else first[2]
                                if first is not None:
                                    diagnostic['trajectory_index']=first[0]
                                    diagnostic['first_invalid_waypoint']=first[1]
                            else:
                                validity=self._configuration_validity(sampled)
                                diagnostic['sampled_waypoints']=len(validity)
                                first=None if all(validity) else validity.index(False)
                                candidate=None if first is None else sampled[first]
                                if first is not None:diagnostic['first_invalid_waypoint']=first
                            if candidate is not None:
                                if budget is not None and not budget.allow('failed_trajectory_pair_query'):
                                    raise TimeoutError('diagnostic budget exhausted')
                                diagnostic['collision_check']=self._collision_diagnostic(candidate)
                                diagnostic['collision_confirmed']=bool(diagnostic['collision_check']['self_collision'] or diagnostic['collision_check']['scene_collision'])
                        except Exception as diagnostic_error:
                            diagnostic["collision_check_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                result["solver_diagnostic"] = diagnostic
                detail = classify_failure_detail(
                    "goal_ik" if operation == "ik" else "trajectory", diagnostic,
                )
                result["failure_detail"] = detail
                if operation == "ik" and detail["reason_code"] == "goal_pose_not_converged":
                    result["failure_category"] = "ik_pose_not_converged"
                elif operation == "ik" and detail["reason_code"] in {
                    "goal_self_collision", "goal_scene_collision", "goal_multiple_constraints",
                }:
                    result["failure_category"] = "goal_collision"
                elif operation == "ik":
                    result["failure_category"] = "ik_constraint_rejected"
                elif detail["reason_code"] in {
                    "path_self_collision", "path_scene_collision", "path_multiple_constraints",
                }:
                    result["failure_category"] = "path_collision"
                else:
                    result["failure_category"] = "path_not_found"
                result["message"] = detail["human_summary"]
            if success:
                solution = raw.js_solution if operation == "ik" else raw.get_interpolated_plan()
                if operation != "ik":
                    result["interpolation_dt_s"] = 0.025
                # cuRobo 的 IKResult 可以把已锁定的 Pika 指关节也放进完整
                # js_solution；执行边界只接受具名活动七轴，不能按长度硬 reshape。
                result["positions_rad"] = self._solution_positions(solution).reshape(-1, len(self.joint_names)).tolist()
                if linear:
                    tcp_poses = self.fk_batch(self.joint_names, result["positions_rad"])
                    tcp_points = tcp_poses[:, :3, 3]
                    final_tcp = tcp_poses[-1]
                    final_orientation_error = float(np.linalg.norm(
                        Rotation.from_matrix(
                            np.asarray(target)[:3, :3].T @ final_tcp[:3, :3]
                        ).as_rotvec()
                    ))
                    result["tcp_path_world_m"] = [point.tolist() for point in tcp_points]
                    if operation == "pivot_tcp":
                        metrics = measure_pivot_tcp_path(
                            tcp_points, start_tcp_world[:3, 3], np.asarray(target)[:3, 3])
                        result["pivot_path_metrics"] = {
                            **metrics.to_dict(),
                            "goal_orientation_error_rad": final_orientation_error,
                            "max_excursion_allowed_m": pivot_limits[0],
                            "max_path_length_allowed_m": pivot_limits[1],
                        }
                        if (not metrics.satisfies(
                                max_excursion_m=pivot_limits[0],
                                max_path_length_m=pivot_limits[1], endpoint_m=.005)
                                or final_orientation_error > .05):
                            result.update(
                                success=False, status="pivot_path_rejected",
                                message="旋转 TCP 轨迹超出活动范围、总路程或端点上限",
                                failure_category="constraint_violation",
                            )
                    else:
                        metrics = measure_linear_tcp_path(
                            tcp_points, start_tcp_world[:3, 3], np.asarray(target)[:3, 3])
                        result["linear_path_metrics"] = metrics.to_dict()
                        result["linear_path_metrics"]["goal_orientation_error_rad"] = final_orientation_error
                        # 平移阶段逐段检查偏离、后退和越界；终点也必须接近目标。
                        if (not metrics.satisfies(
                            lateral_m=.005, backward_step_m=.001,
                            cumulative_backward_m=.002, overshoot_m=.002,
                            endpoint_m=.005,
                        ) or final_orientation_error > .05):
                            result.update(
                                success=False, status="linear_path_rejected",
                                message="TCP轨迹未通过直线度/单向性/端点复核",
                                failure_category="constraint_violation",
                            )
                validity = self._configuration_validity(result["positions_rad"])
                result["collision_verification"] = {
                    "valid": all(validity),
                    "checked_waypoints": len(validity),
                    "first_invalid_index": None,
                }
                if not all(validity):
                    first_invalid = validity.index(False)
                    result["collision_verification"]["first_invalid_index"] = first_invalid
                    postcheck_diagnostic = {"first_invalid_waypoint": first_invalid}
                    try:
                        postcheck_diagnostic["collision_check"] = self._collision_diagnostic(
                            result["positions_rad"][first_invalid],
                            context_label=f"postcheck_waypoint_{first_invalid}",
                        )
                        result["collision_verification"]["collision_diagnostic"] = (
                            postcheck_diagnostic["collision_check"]
                        )
                    except Exception as diagnostic_error:
                        result["collision_verification"]["collision_diagnostic_error"] = (
                            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                        )
                    _record_postcheck_failure(
                        result, category="collision_postcheck_rejected",
                        status="collision_postcheck_rejected",
                        message="cuRobo 返回结果未通过独立离散碰撞复核",
                    )
                    if result.get("failure_category") == "collision_postcheck_rejected":
                        detail = classify_failure_detail(
                            "trajectory_postcheck", postcheck_diagnostic,
                        )
                        result["failure_detail"] = detail
                        result["message"] = detail["human_summary"]
                if self._validator and all(validity):
                    verification = self._validator.check(result["positions_rad"], self.planner_joint_names)
                    result["collision_verification"] = verification
                    if not verification["valid"]:
                        _record_postcheck_failure(
                            result, category="collision_postcheck_rejected",
                            status="collision_postcheck_rejected",
                            message="cuRobo 返回结果未通过连续碰撞复核",
                        )
                        if result.get("failure_category") == "collision_postcheck_rejected":
                            detail = classify_failure_detail("trajectory_postcheck")
                            result["failure_detail"] = detail
                            result["message"] = detail["human_summary"]
            if not result.get("success") and operation != "ik" and result.get(
                    "failure_category") in {
                        "path_collision", "path_not_found", "solver_rejected",
                        "collision_postcheck_rejected", "constraint_violation",
                    }:
                try:
                    self._export_failure_visualization(
                        result,
                        start_positions=ordered_positions(
                            joint_names, positions, self.joint_names,
                        ),
                        target=target,
                        raw=raw,
                        diagnostic_label=diagnostic_label,
                    )
                except Exception as visualization_error:
                    result["failure_visualization_error"] = (
                        f"{type(visualization_error).__name__}: {visualization_error}"
                    )
            return dict(result, **self.gripper_metadata, operation=operation, joint_names=self.joint_names, scene_scope=self.scene_scope, scene_snapshot_sha256=self.scene_snapshot_sha256,
                        planner_mode=self.planner_mode.value, use_cuda_graph=self.runtime_options["use_cuda_graph"],
                        random_seed=self.random_seed, start_state_check=start_check,
                        duration_s=time.perf_counter()-started, timing=phase_timing)
        except Exception as error:
            return dict(success=False, **self.gripper_metadata, status=type(error).__name__, operation=operation, raw_error=traceback.format_exc(),
                        message=str(error), failure_category="input_validation_error" if isinstance(error, ValueError) else "backend_exception",
                        joint_names=self.joint_names, scene_scope=self.scene_scope,
                        scene_snapshot_sha256=self.scene_snapshot_sha256,
                        planner_mode=self.planner_mode.value,
                        use_cuda_graph=self.runtime_options["use_cuda_graph"],
                        random_seed=self.random_seed, duration_s=time.perf_counter()-started, timing=phase_timing)

        finally:
            if 'postprocess_started' in locals():
                phase_timing['postprocess_and_builtin_diagnostics_s'] = time.perf_counter()-postprocess_started
            phase_timing['total_s'] = time.perf_counter()-started

    def solve_ik(self, joint_names, positions, target, *, target_frame="world_tcp",
                 diagnostic_label: str | None = None):
        return self._solve(
            "ik", joint_names, positions, target, target_frame,
            diagnostic_label=diagnostic_label,
        )

    def plan_linear_tcp(self, joint_names, positions, target, *, target_frame="world_tcp",
                        max_attempts: int = 1, non_terminal_scale: float = 5.0,
                        diagnostic_label: str | None = None):
        """规划 TCP 直线段，并在求解阶段强化非终点横向偏差代价。

        后验直线度/单向性复核仍是硬门；较高的默认权重用于避免求解器
        先跨到另一冗余关节分支，再由后验检查淘汰整条大绕行轨迹。
        """
        return self._solve("linear_tcp", joint_names, positions, target, target_frame,
                           max_attempts=max_attempts,
                           non_terminal_scale=non_terminal_scale,
                           diagnostic_label=diagnostic_label)

    def plan_joint_target(self, joint_names, positions, target_positions,
                          target_pose, *, target_frame="world_tcp",
                          max_attempts: int = 1,
                          diagnostic_label: str | None = None):
        """规划到显式关节目标，同时保留该阶段的位姿和碰撞审计记录.

        仅用于已注册的过渡目标。目标关节状态由调用方从场景配置读取；
        轨迹仍经过 cuRobo 全世界、自碰撞和 MuJoCo 执行后检查。
        """
        goal = np.asarray(target_positions, dtype=float)
        if goal.shape != (len(self.joint_names),) or not np.isfinite(goal).all():
            raise ValueError("关节目标必须是与规划关节同长度的有限向量")
        self._cspace_goal_positions = goal.copy()
        try:
            return self._solve("plan", joint_names, positions, target_pose,
                               target_frame, max_attempts=max_attempts,
                               diagnostic_label=diagnostic_label)
        finally:
            self._cspace_goal_positions = None

    def plan_pivot_tcp(self, joint_names, positions, target, *, target_frame="world_tcp",
                       max_attempts: int = 1, max_tcp_excursion_m: float,
                       max_tcp_path_length_m: float,
                       diagnostic_label: str | None = None):
        """用 cuRobo 规划旋转，限制末端活动半径及路程，不要求平移单向。"""
        return self._solve(
            "pivot_tcp", joint_names, positions, target, target_frame,
            max_attempts=max_attempts,
            pivot_limits=(max_tcp_excursion_m, max_tcp_path_length_m),
            diagnostic_label=diagnostic_label,
        )

    def solve_ik_batch(
        self,
        joint_names,
        positions,
        targets,
        *,
        target_frame="world_tcp",
        diagnose_failures: bool | str = True,
    ) -> tuple[dict, ...]:
        """一次 GPU 调用筛选多个候选，可选择是否逐项生成失败归因。

        ``duration_s`` 为一次 batch 墙钟耗时的均摊值，使调用方累加候选
        耗时不会重复计算；完整墙钟耗时另存在 ``batch_total_duration_s``。

        Knowledge/Skill 候选筛选使用 ``when_no_success``：一批中有可达
        候选时跳过其余失败姿态的逐物体碰撞查询；整批无解时保留原有的
        逐候选详细诊断。显式布尔值继续保持原有语义。
        """
        targets = tuple(targets)
        if not targets:
            return ()
        if diagnose_failures not in (True, False, "when_no_success"):
            raise ValueError("diagnose_failures 必须是布尔值或 when_no_success")
        started = time.perf_counter()
        count = len(targets)
        common = dict(
            **self.gripper_metadata, operation="ik", joint_names=self.joint_names,
            scene_scope=self.scene_scope,
            scene_snapshot_sha256=self.scene_snapshot_sha256,
            planner_mode=self.planner_mode.value,
            use_cuda_graph=self.runtime_options["use_cuda_graph"],
            random_seed=self.random_seed,
            batch_size=count,
        )
        try:
            from contextlib import nullcontext
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
            from curobo.types import GoalToolPose, JointState
            from ..collision import optimizer_mounting_mask

            ordered = ordered_positions(joint_names, positions, self.joint_names)
            start_check = self._start_configuration_check(ordered)
            if not start_check["valid"]:
                status, category, message = describe_start_failure(start_check)
                elapsed = time.perf_counter() - started
                return tuple(dict(
                    common, success=False, status=status, failure_category=category,
                    message=message, raw_error=None, duration_s=elapsed / count,
                    batch_total_duration_s=elapsed,
                    collision_verification={"valid": False, "first_invalid_index": 0,
                                            "start_check": start_check},
                ) for _ in targets)

            goals = [self._goal(target, target_frame) for target in targets]
            goal = GoalToolPose(
                tool_frames=goals[0].tool_frames,
                position=self.torch.cat([item.position for item in goals], dim=0),
                quaternion=self.torch.cat([item.quaternion for item in goals], dim=0),
            )
            start = self._state(joint_names, positions)
            batch_start = JointState.from_position(
                start.position.expand(count, -1).clone(),
                joint_names=self.planner_joint_names,
            )
            if count not in self._ik_batches:
                self._ik_batches[count] = InverseKinematics(InverseKinematicsCfg.create(
                    robot=copy.deepcopy(self.robot_config), scene_model=self.scene_model,
                    collision_cache=self.collision_cache,
                    optimizer_collision_activation_distance=self.optimizer_collision_activation_distance,
                    self_collision_check=True, num_seeds=16,
                    max_batch_size=count, **self.runtime_options,
                ))
            solver = self._ik_batches[count]
            with optimizer_mounting_mask(solver.kinematics, self.mounting_contact_policy) if self._validator else nullcontext():
                raw = solver.solve_pose(goal, current_state=batch_start)
            self.torch.cuda.synchronize()

            success = raw.success.detach().cpu().numpy().reshape(count, -1)[:, 0].astype(bool)
            feasible = raw.feasible.detach().cpu().numpy().reshape(count, -1)[:, 0].astype(bool)
            position_error = raw.position_error.detach().cpu().numpy().reshape(count, -1)[:, 0]
            rotation_error = raw.rotation_error.detach().cpu().numpy().reshape(count, -1)[:, 0]
            solutions = self._solution_positions(raw.js_solution).reshape(count, -1, len(self.joint_names))[:, 0]
            if not all(np.all(np.isfinite(values)) for values in (position_error, rotation_error, solutions)):
                raise ValueError("cuRobo batch IK 返回非有限解或误差")
            # 后处理在同一调用内完成，避免逐候选 Python/GPU 往返。
            valid = self._configuration_validity(solutions)
            diagnose_this_batch = (
                diagnose_failures is True or
                (diagnose_failures == "when_no_success" and not bool(np.any(success & valid)))
            )
            elapsed = time.perf_counter() - started
            rows = []
            for index in range(count):
                item = dict(common, duration_s=elapsed / count,
                            batch_total_duration_s=elapsed, raw_error=None,
                            status="success" if success[index] else "solver_failed")
                item["collision_verification"] = {
                    "valid": bool(valid[index]), "checked_waypoints": 1,
                    "first_invalid_index": None if valid[index] else 0,
                }
                if success[index] and valid[index]:
                    item.update(success=True, positions_rad=[solutions[index].tolist()],
                                failure_category=None)
                    if self._validator:
                        strict = self._validator.check([solutions[index]], self.planner_joint_names)
                        item["collision_verification"] = strict
                        if not strict["valid"]:
                            item.update(success=False, status="collision_postcheck_rejected",
                                        failure_category="collision_postcheck_rejected",
                                        message="目标解未通过严格安装接触复核")
                else:
                    row = SimpleNamespace(
                        position_error=np.array([position_error[index]]),
                        rotation_error=np.array([rotation_error[index]]),
                        feasible=np.array([feasible[index]]),
                    )
                    diagnostic = solver_failure_diagnostic(row, "ik", solver)
                    if diagnose_this_batch and (not valid[index] or not feasible[index]):
                        try:
                            diagnostic["collision_check"] = self._collision_diagnostic(
                                solutions[index]
                            )
                            diagnostic["collision_confirmed"] = bool(
                                diagnostic["collision_check"]["self_collision"] or
                                diagnostic["collision_check"]["scene_collision"]
                            )
                        except Exception as error:
                            diagnostic["collision_check_error"] = f"{type(error).__name__}: {error}"
                    item.update(success=False, solver_diagnostic=diagnostic)
                    detail = classify_failure_detail("goal_ik", diagnostic)
                    item["failure_detail"] = detail
                    if success[index] and not valid[index]:
                        item.update(status="collision_postcheck_rejected",
                                    failure_category="collision_postcheck_rejected",
                                    message=detail["human_summary"])
                    elif detail["reason_code"] == "goal_pose_not_converged":
                        item.update(failure_category="ik_pose_not_converged",
                                    message=detail["human_summary"])
                    elif detail["reason_code"] in {
                        "goal_self_collision", "goal_scene_collision", "goal_multiple_constraints",
                    }:
                        item.update(failure_category="goal_collision",
                                    message=detail["human_summary"])
                    else:
                        item.update(failure_category="ik_constraint_rejected",
                                    message=detail["human_summary"])
                rows.append(item)
            # 包含逐项复核及必要的失败诊断；不能在后处理之前结束计时。
            elapsed = time.perf_counter() - started
            for item in rows:
                item.update(duration_s=elapsed/count, batch_total_duration_s=elapsed)
            return tuple(rows)
        except Exception as error:
            elapsed = time.perf_counter() - started
            return tuple(dict(common, success=False, status=type(error).__name__,
                              message=str(error), raw_error=traceback.format_exc(),
                              failure_category="input_validation_error" if isinstance(error, ValueError) else "backend_exception",
                              duration_s=elapsed / count, batch_total_duration_s=elapsed)
                         for _ in targets)

    def plan(self, joint_names, positions, target, *, target_frame="world_tcp",
             max_attempts: int = 1, diagnostic_label: str | None = None):
        return self._solve("plan", joint_names, positions, target, target_frame,
                           max_attempts=max_attempts,
                           diagnostic_label=diagnostic_label)


# 保留历史 import；typed 适配的唯一实现按职责拆分。
from .adapter import CuroboPlannerAdapter
