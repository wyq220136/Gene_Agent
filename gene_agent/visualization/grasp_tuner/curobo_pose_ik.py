"""微波炉 Knowledge 的 cuRobo IK 适配：只检查位姿、关节和自身碰撞。"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion.curobo import SynchronousPlanner, load_realman_motion_config
from gene_agent.motion.curobo.gripper import export_gripper_snapshot
from gene_agent.motion.curobo.profiles.pika import export_pika_gripper_snapshot
from .pose_ik import MuJoCoPoseIK, PoseIKResult


def select_preview_candidate(planner, joint_names: tuple[str, ...], goal_pose: np.ndarray,
                             solutions: np.ndarray, feasible: list[bool]) -> dict | None:
    """在候选中先选通过约束的姿态，再按真实 world TCP 的 FK 误差排序。"""
    candidates = []
    for index, (q, valid) in enumerate(zip(np.asarray(solutions), feasible)):
        if not np.isfinite(q).all():
            continue
        actual = planner.fk(joint_names, q)
        position = float(np.linalg.norm(actual[:3, 3] - goal_pose[:3, 3]))
        rotation = math.degrees(float(np.linalg.norm(Rotation.from_matrix(
            goal_pose[:3, :3] @ actual[:3, :3].T).as_rotvec())))
        if not np.isfinite([position, rotation]).all():
            continue
        candidates.append({"index": index, "joint_positions_rad": q,
                           "constraint_feasible": bool(valid),
                           "position_error_m": position, "rotation_error_deg": rotation})
    if not candidates:
        return None
    # 与 MuJoCo 预览使用相同的平移/旋转权重；失败候选仅用于解释和画图。
    return min(candidates, key=lambda item: (
        not item["constraint_feasible"],
        item["position_error_m"] + .002 * item["rotation_error_deg"],
    ))


def diagnose_mujoco_candidate(goal_pose: np.ndarray, mujoco_ik: PoseIKResult,
                              planner: SynchronousPlanner, joint_names: tuple[str, ...],
                              *, position_tolerance_m: float | None,
                              rotation_tolerance_rad: float | None) -> dict:
    """用 cuRobo 的 FK/约束复核 MuJoCo 解，区分已知可行解与单次搜索失败。"""
    report = {
        "mujoco_reachable": bool(mujoco_ik.reachable),
        "mujoco_position_error_m": float(mujoco_ik.position_error_m),
        "mujoco_rotation_error_deg": float(mujoco_ik.rotation_error_deg),
        "candidate_joint_positions_rad": mujoco_ik.joint_positions,
    }
    if mujoco_ik.joint_positions is None:
        return {**report, "conclusion": "mujoco_candidate_missing"}
    q = np.asarray(mujoco_ik.joint_positions, dtype=float)
    actual = np.asarray(planner.fk(joint_names, q), dtype=float)
    position = float(np.linalg.norm(goal_pose[:3, 3] - actual[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        goal_pose[:3, :3] @ actual[:3, :3].T).as_rotvec()))
    check = planner._collision_diagnostic(q)
    tolerance_known = (position_tolerance_m is not None and rotation_tolerance_rad is not None
                       and position_tolerance_m > 0 and rotation_tolerance_rad > 0)
    within = (position < position_tolerance_m and rotation < rotation_tolerance_rad) if tolerance_known else None
    if not mujoco_ik.reachable:
        conclusion = "mujoco_also_not_strict"
    elif check.get("joint_limit_violation"):
        conclusion = "mujoco_candidate_joint_limit"
    elif check.get("self_collision"):
        conclusion = "mujoco_candidate_self_collision"
    elif within is False:
        conclusion = "different_pose_tolerance"
    elif within is True and check.get("joint_limit_violation") is False and check.get("self_collision") is False:
        conclusion = "solver_search_missed_known_valid_candidate"
    else:
        conclusion = "candidate_validity_unknown"
    return {**report, "curobo_fk_position_error_m": position,
            "curobo_fk_rotation_error_deg": math.degrees(rotation),
            "curobo_position_tolerance_m": position_tolerance_m,
            "curobo_rotation_tolerance_deg": None if rotation_tolerance_rad is None else math.degrees(rotation_tolerance_rad),
            "candidate_within_curobo_tolerance": within,
            "candidate_check": check, "conclusion": conclusion}


class CuroboPoseIK:
    """每个夹爪开口同步机器人球；场景障碍留给 Skill/规划阶段。"""

    def __init__(self, case):
        self.case = case
        self.group = case.snapshot["active_group"]
        self.robot = case.metadata["robot"]
        self.names = tuple(self.robot["joint_groups"][self.group])
        self._cache: dict[tuple, tuple[SynchronousPlanner, np.ndarray]] = {}
        self._diagnostic_data = None

    def _planner(self, *, joint_name: str | None, angle_rad: float | None,
                 opening_width_m: float | None):
        key = (joint_name, None if angle_rad is None else round(float(angle_rad), 7),
               None if opening_width_m is None else round(float(opening_width_m), 6))
        if key in self._cache:
            return self._cache[key]
        # “检查全部”会遍历许多门角；不同时持有几十份 GPU 求解器。
        self._cache.clear()
        model = self.case.model
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, self.case.data)
        if joint_name is not None:
            data.qpos[model.joint(joint_name).qposadr[0]] = float(angle_rad)
        gripper = self.robot["grippers"][self.group]
        if opening_width_m is not None and gripper.get("mode") == "parallel_slide_proxy":
            width = float(np.clip(opening_width_m, *gripper["width_range_m"]))
            for name in gripper["joint_names"]:
                data.qpos[model.joint(name).qposadr[0]] = width / 2
        mujoco.mj_forward(model, data)
        # 夹爪球仍随开口更新，以保留自身碰撞检查；不导出任何场景物体。
        if gripper.get("mode") == "parallel_slide_proxy":
            gripper_snapshot = export_gripper_snapshot(model, data, self.robot, self.group)
        elif gripper.get("mode") == "pika_urdf":
            gripper_snapshot = export_pika_gripper_snapshot(model, data, self.robot, self.group)
        else:
            gripper_snapshot = None
        settings = load_realman_motion_config(self.group, self.robot,
                                              gripper_snapshot=gripper_snapshot)
        planner = SynchronousPlanner(
            **settings, scene_model={"cuboid": {}, "mesh": {}},
            scene_scope="Knowledge 纯位姿 IK：无场景障碍；保留关节限位与机器人自身碰撞",
            mounting_contact_policy={},
        )
        start = np.asarray([data.qpos[model.joint(name).qposadr[0]] for name in self.names], dtype=float)
        self._cache[key] = (planner, start)
        self._diagnostic_data = data
        return planner, start

    def solve(self, goal_pose: np.ndarray, *, joint_name: str | None = None,
              angle_rad: float | None = None, opening_width_m: float | None = None) -> PoseIKResult:
        planner, start = self._planner(joint_name=joint_name, angle_rad=angle_rad,
                                       opening_width_m=opening_width_m)
        result = planner.solve_ik(self.names, start, goal_pose)
        diagnostic = result.get("solver_diagnostic") or {}
        values = result.get("positions_rad") if result.get("success") else diagnostic.get("joint_positions_rad")
        q = None if values is None else np.asarray(values, dtype=float).reshape(-1, len(self.names))[0]
        if q is not None and not np.isfinite(q).all():
            q = None
        feasible = (True if result.get("success") else diagnostic.get("constraint_feasible"))
        preview = None
        preview_error = None
        preview_check = diagnostic.get("collision_check")
        if not result.get("success") and planner._ik is not None:
            try:
                # 标准 solve_pose 在失败时仅返回一个代表姿态。再次搜索并显式
                # 返回全部种子，仅用于 Knowledge 诊断，不改变严格 IK 的结论。
                raw = planner._ik.solve_pose(
                    planner._goal(goal_pose, "world_tcp"),
                    current_state=planner._state(self.names, start),
                    return_seeds=planner._ik.config.num_seeds,
                )
                solutions = raw.solution.detach().cpu().numpy().reshape(-1, len(self.names))
                reported_feasible = raw.feasible.detach().cpu().numpy().reshape(-1).astype(bool)
                if len(solutions) != len(reported_feasible):
                    raise ValueError("cuRobo 返回的种子和可行性数量不一致")
                independently_valid = planner._configuration_validity(solutions)
                valid = [bool(a and b) for a, b in zip(reported_feasible, independently_valid)]
                preview = select_preview_candidate(planner, self.names, goal_pose, solutions, valid)
                if preview is not None:
                    q = preview["joint_positions_rad"]
                    feasible = preview["constraint_feasible"]
                    preview_check = planner._collision_diagnostic(
                        q, context_label="knowledge_followup_preview_seed")
            except Exception as error:
                preview_error = f"{type(error).__name__}: {error}"
        if q is not None:
            actual = planner.fk(self.names, q)
            position = float(np.linalg.norm(actual[:3, 3] - goal_pose[:3, 3]))
            rotation = math.degrees(float(np.linalg.norm(Rotation.from_matrix(
                goal_pose[:3, :3] @ actual[:3, :3].T).as_rotvec())))
        else:
            position = diagnostic.get("position_error_m")
            rotation_rad = diagnostic.get("rotation_error_rad")
            position = float(position) if position is not None else math.inf
            rotation = math.degrees(float(rotation_rad)) if rotation_rad is not None else math.inf
        comparison = None
        if not result.get("success"):
            source = self._diagnostic_data or self.case.data
            mujoco_solver = MuJoCoPoseIK(self.case.model, source, self.names,
                                         self.robot["tcp_sites"][self.group])
            mujoco_ik = mujoco_solver.solve(goal_pose)
            try:
                comparison = diagnose_mujoco_candidate(
                    goal_pose, mujoco_ik, planner, self.names,
                    position_tolerance_m=diagnostic.get("position_tolerance_m"),
                    rotation_tolerance_rad=diagnostic.get("rotation_tolerance_rad"),
                )
            except Exception as error:
                comparison = {"mujoco_reachable": mujoco_ik.reachable,
                              "conclusion": "comparison_failed",
                              "error": f"{type(error).__name__}: {error}"}
        evidence = {
            "solver_reason": diagnostic.get("reason"),
            "position_tolerance_m": diagnostic.get("position_tolerance_m"),
            "rotation_tolerance_deg": None if diagnostic.get("rotation_tolerance_rad") is None else
            math.degrees(float(diagnostic["rotation_tolerance_rad"])),
            "failed_seed_constraint_feasible": diagnostic.get("constraint_feasible"),
            "failed_seed_check": diagnostic.get("collision_check"),
            "preview_source": "followup_all_seeds" if preview is not None else "original_returned_seed",
            "preview_seed_index": None if preview is None else preview["index"],
            "preview_constraint_feasible": feasible,
            "preview_search_error": preview_error,
            "mujoco_candidate_comparison": comparison,
        } if not result.get("success") else None
        return PoseIKResult(bool(result.get("success")), position, rotation,
                            None if q is None else tuple(float(x) for x in q),
                            backend="curobo", constraint_feasible=feasible,
                            failure_category=None if result.get("success") else
                            result.get("failure_category") or result.get("status"),
                            collision_check=preview_check, diagnostic=evidence)
