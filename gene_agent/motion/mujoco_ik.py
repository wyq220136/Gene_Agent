"""MuJoCo CPU 姿态 IK 与短路径检查；用于无 GPU 的仿真开发。"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


class MujocoPosePlanner:
    """从当前真实状态规划，但所有 IK 与碰撞探针只修改 MjData 副本。"""

    def __init__(self, model, data, robot_metadata: dict, group: str):
        self.model, self.data = model, data
        self.group = group
        self.names = tuple(robot_metadata["joint_groups"][group])
        self.site_name = robot_metadata["tcp_sites"][group]
        self.ids = [model.joint(name).id for name in self.names]
        self.qpos_addresses = [int(model.jnt_qposadr[index]) for index in self.ids]
        self.low = np.asarray(model.jnt_range[self.ids, 0], dtype=float) + 1e-5
        self.high = np.asarray(model.jnt_range[self.ids, 1], dtype=float) - 1e-5
        self.root_id = model.body(robot_metadata["group_root_bodies"][group]).id

    def _copy(self):
        preview = mujoco.MjData(self.model)
        mujoco.mj_copyData(preview, self.model, self.data)
        return preview

    def _tcp(self, preview) -> np.ndarray:
        site = preview.site(self.site_name)
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return pose

    def _solve(self, preview, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float, float]:
        def residual(q):
            preview.qpos[self.qpos_addresses] = q
            mujoco.mj_forward(self.model, preview)
            actual = self._tcp(preview)
            rot_error = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            return np.r_[actual[:3, 3] - target[:3, 3], .1 * rot_error,
                         .0005 * (q - seed)]

        fit = least_squares(residual, np.clip(seed, self.low, self.high),
                            bounds=(self.low, self.high), max_nfev=160)
        residual(fit.x)
        actual = self._tcp(preview)
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        rotation_error = float(np.linalg.norm(Rotation.from_matrix(
            target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
        return fit.x, position_error, rotation_error

    def _is_robot_body(self, body_id: int) -> bool:
        while body_id:
            if body_id == self.root_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _contact_owner(self, geom_id: int) -> str:
        geom = self.model.geom(geom_id)
        if geom.name.startswith("brush_long_stand_"):
            return "brush_stand"
        if geom.name == "scene_table":
            return "table"
        if geom.name == "scene_floor":
            return "floor"
        return self.model.body(int(self.model.geom_bodyid[geom_id])).name

    def _set_attached(self, preview, T_tcp_object: np.ndarray, object_name: str) -> None:
        pose = self._tcp(preview) @ T_tcp_object
        joint_id = self.model.joint(f"{object_name}_free").id
        qpos = int(self.model.jnt_qposadr[joint_id])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, pose[:3, :3].reshape(9))
        preview.qpos[qpos:qpos + 7] = [*pose[:3, 3], *quat]

    def _forbidden_contacts(self, preview, allowed: set[frozenset[str]],
                            *, attached_object: str | None, sample_index: int) -> list[dict]:
        invalid = []
        for contact_index in range(preview.ncon):
            contact = preview.contact[contact_index]
            if float(contact.dist) >= -.001:
                continue
            one, two = int(contact.geom1), int(contact.geom2)
            owner_pair = frozenset((self._contact_owner(one), self._contact_owner(two)))
            robot_involved = (self._is_robot_body(int(self.model.geom_bodyid[one]))
                              or self._is_robot_body(int(self.model.geom_bodyid[two])))
            brush_involved = "brush" in owner_pair
            attached_involved = attached_object in owner_pair if attached_object else False
            if (robot_involved or brush_involved or attached_involved) and owner_pair not in allowed:
                invalid.append({"waypoint": sample_index, "geom_names": [
                    self.model.geom(one).name, self.model.geom(two).name],
                    "body_names": sorted(owner_pair), "penetration_m": -float(contact.dist)})
                if len(invalid) >= 12:
                    break
        return invalid

    def _check_path(self, positions: np.ndarray, *, allowed_contacts: Iterable[tuple[str, str]],
                    attached_object: str | None, T_tcp_object: np.ndarray | None,
                    moving_body_names: Iterable[str] = ()) -> list[dict]:
        allowed = {frozenset(pair) for pair in allowed_contacts}
        preview = self._copy()
        moving_names = tuple(moving_body_names)
        if moving_names and not attached_object:
            raise ValueError("随行物体检查需要附着物 frame")
        # 杯内球不能留在起点世界坐标作为静态障碍。先保存其相对杯子的
        # 位姿，在每个规划关节点随杯更新；实际球运动另由执行采样复核。
        moving = []
        if moving_names:
            mujoco.mj_forward(self.model, preview)
            anchor = preview.body(attached_object)
            anchor_rotation = anchor.xmat.reshape(3, 3)
            anchor_position = anchor.xpos.copy()
            for name in moving_names:
                body = self.model.body(name)
                joint_id = int(self.model.body_jntadr[body.id])
                if (int(self.model.body_jntnum[body.id]) != 1 or
                        int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)):
                    raise ValueError(f"随行物体必须只有一个 freejoint: {name}")
                frame = preview.body(name)
                moving.append((int(self.model.jnt_qposadr[joint_id]),
                               anchor_rotation.T @ (frame.xpos - anchor_position),
                               anchor_rotation.T @ frame.xmat.reshape(3, 3)))
        # 规划时按已测得的 TCP 相对位姿放置附着物；MuJoCo 在线状态从不改写。
        if attached_object:
            eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                      f"sim_attach_{self.group}_{attached_object}")
            if eq_id >= 0:
                preview.eq_active[eq_id] = 0
        invalid = []
        for index, q in enumerate(positions):
            preview.qpos[self.qpos_addresses] = q
            mujoco.mj_forward(self.model, preview)
            if attached_object:
                self._set_attached(preview, T_tcp_object, attached_object)
                mujoco.mj_forward(self.model, preview)
                if moving:
                    anchor = preview.body(attached_object)
                    rotation = anchor.xmat.reshape(3, 3)
                    for address, relative_position, relative_rotation in moving:
                        preview.qpos[address:address + 3] = (
                            anchor.xpos + rotation @ relative_position)
                        quat = np.empty(4)
                        mujoco.mju_mat2Quat(
                            quat, (rotation @ relative_rotation).reshape(9))
                        preview.qpos[address + 3:address + 7] = quat
                    mujoco.mj_forward(self.model, preview)
            invalid.extend(self._forbidden_contacts(
                preview, allowed, attached_object=attached_object, sample_index=index))
            if len(invalid) >= 12:
                return invalid
        return invalid

    def check_observed_contacts(self, samples: Iterable[dict], *,
                                allowed_contacts: Iterable[tuple[str, str]]) -> list[dict]:
        """复核真实执行回放里的每个 qpos 样本，不改在线 MuJoCo 状态。"""
        allowed = {frozenset(pair) for pair in allowed_contacts}
        preview = self._copy()
        invalid = []
        for index, sample in enumerate(samples):
            qpos = np.asarray(sample["qpos"], dtype=float)
            if qpos.shape != preview.qpos.shape or not np.isfinite(qpos).all():
                raise ValueError("执行回放 qpos 与当前模型不符")
            preview.qpos[:] = qpos
            mujoco.mj_forward(self.model, preview)
            invalid.extend(self._forbidden_contacts(
                preview, allowed, attached_object="brush", sample_index=index))
            if len(invalid) >= 12:
                break
        return invalid

    def plan_to_pose(self, target_pose, *, path_kind: str, duration_s: float,
                     allowed_contacts: Iterable[tuple[str, str]] = (),
                     attached_object: str | None = None,
                     T_tcp_object=None,
                     approximate_ik_budget: Mapping[str, float] | None = None) -> dict:
        """生成关节路径或逐点直线 TCP 路径；返回原始失败原因供 Skill 记录。"""
        target = np.asarray(target_pose, dtype=float)
        if (target.shape != (4, 4) or not np.isfinite(target).all()
                or not np.allclose(target[3], [0, 0, 0, 1])):
            raise ValueError("target_pose 必须是有限 4x4 刚体位姿")
        if path_kind not in {"joint", "linear_tcp"}:
            raise ValueError("path_kind 只能是 joint 或 linear_tcp")
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s 必须为正")
        budget = None
        if approximate_ik_budget is not None:
            budget = dict(approximate_ik_budget)
            if (set(budget) != {"max_position_error_m", "max_rotation_error_rad"}
                    or any(not np.isfinite(value) or value <= 0 for value in budget.values())):
                raise ValueError("近似 IK 预算须提供正的有限位置与角度误差上限")
        attached_pose = None if T_tcp_object is None else np.asarray(T_tcp_object, dtype=float)
        if (attached_object is None) != (attached_pose is None):
            raise ValueError("附着物名称和 T_tcp_object 必须同时提供")
        if attached_pose is not None and attached_pose.shape != (4, 4):
            raise ValueError("T_tcp_object 必须是 4x4")
        start = self._copy()
        mujoco.mj_forward(self.model, start)
        start_q = np.asarray(start.qpos[self.qpos_addresses], dtype=float).copy()
        start_pose = self._tcp(start)
        probe = self._copy()
        rows = [start_q]
        failures = []
        approximate_waypoints = []
        max_position_error = max_rotation_error = 0.
        if path_kind == "joint":
            end, position_error, rotation_error = self._solve(probe, target, start_q)
            rows = list(np.linspace(start_q, end, 25))
            max_position_error, max_rotation_error = position_error, rotation_error
        else:
            distance = float(np.linalg.norm(target[:3, 3] - start_pose[:3, 3]))
            count = max(2, int(np.ceil(distance / .012)) + 1)
            orientations = Slerp([0, 1], Rotation.from_matrix(
                [start_pose[:3, :3], target[:3, :3]]))
            position_error = rotation_error = 0.0
            for fraction in np.linspace(0, 1, count)[1:]:
                waypoint = np.eye(4)
                waypoint[:3, :3] = orientations([fraction]).as_matrix()[0]
                waypoint[:3, 3] = (1 - fraction) * start_pose[:3, 3] + fraction * target[:3, 3]
                q, position_error, rotation_error = self._solve(probe, waypoint, rows[-1])
                rows.append(q)
                max_position_error = max(max_position_error, position_error)
                max_rotation_error = max(max_rotation_error, rotation_error)
                strict_waypoint = position_error <= .005 and rotation_error <= .04
                budget_waypoint = (budget is not None and
                                   position_error <= budget["max_position_error_m"] and
                                   rotation_error <= budget["max_rotation_error_rad"])
                if not strict_waypoint and budget_waypoint:
                    approximate_waypoints.append({"fraction": float(fraction),
                                                  "position_error_m": position_error,
                                                  "rotation_error_rad": rotation_error})
                if not strict_waypoint and not budget_waypoint:
                    failures.append({"fraction": float(fraction),
                                     "position_error_m": position_error,
                                     "rotation_error_rad": rotation_error})
                    break
        positions = np.asarray(rows)
        if (not np.isfinite(positions).all()
                or not np.isfinite([position_error, rotation_error,
                                    max_position_error, max_rotation_error]).all()):
            return {"success": False, "failure_type": "ik_non_finite_candidate",
                    "path_collision_checked": False,
                    "target_tcp_world": target.tolist(), "approximate_ik_budget": budget}
        actual_low = np.asarray(self.model.jnt_range[self.ids, 0], dtype=float)
        actual_high = np.asarray(self.model.jnt_range[self.ids, 1], dtype=float)
        violation = np.maximum.reduce((actual_low - positions,
                                       positions - actual_high,
                                       np.zeros_like(positions)))
        start_violation = float(np.max(violation[0]))
        # position actuator 偶尔把已执行起点带出标称关节限约 1 mrad。
        # 仅允许规划轨迹单调回到限内；新目标和中间点不能进一步越界。
        recovering_start = (start_violation <= .002
                            and np.all(violation[1:] <= violation[:-1] + 1e-8)
                            and np.max(violation[-1]) <= 1e-8)
        if not recovering_start:
            return {"success": False, "failure_type": "ik_joint_constraint_violation",
                    "path_collision_checked": False,
                    "target_tcp_world": target.tolist(), "approximate_ik_budget": budget,
                    "start_joint_limit_violation_rad": start_violation,
                    "max_path_joint_limit_violation_rad": float(np.max(violation))}
        strict_converged = (max_position_error <= .005 and max_rotation_error <= .04)
        accepted_by_budget = (budget is not None and not failures and
                              max_position_error <= budget["max_position_error_m"] and
                              max_rotation_error <= budget["max_rotation_error_rad"])
        endpoint_probe = self._copy()
        endpoint_probe.qpos[self.qpos_addresses] = rows[-1]
        mujoco.mj_forward(self.model, endpoint_probe)
        actual_fk = self._tcp(endpoint_probe)
        approximation = {
            "approximate": bool(not strict_converged and accepted_by_budget),
            "strict_converged": bool(strict_converged),
            "budget": budget,
            "original_target_tcp_world": target.tolist(),
            "actual_fk_tcp_world": actual_fk.tolist(),
            "joint_solution_rad": rows[-1].tolist(),
            "raw_position_error_m": float(position_error),
            "raw_rotation_error_rad": float(rotation_error),
            "max_waypoint_position_error_m": float(max_position_error),
            "max_waypoint_rotation_error_rad": float(max_rotation_error),
            "accepted_waypoints": approximate_waypoints,
            "constraint_feasible": None,
        }
        if not strict_converged and not accepted_by_budget:
            return {"success": False, "failure_type": "ik_pose_not_converged",
                    "position_error_m": position_error, "rotation_error_rad": rotation_error,
                    "ik_failures": failures, "path_collision_checked": False,
                    "approximate_ik": approximation}
        contacts = self._check_path(positions, allowed_contacts=allowed_contacts,
                                    attached_object=attached_object,
                                    T_tcp_object=attached_pose)
        approximation["constraint_feasible"] = not contacts
        return {"success": not contacts, "failure_type": "path_collision" if contacts else None,
                "path_kind": path_kind, "joint_names": list(self.names),
                "times_s": np.linspace(0, duration_s, len(positions)).tolist(),
                "positions_rad": positions.tolist(),
                "position_error_m": position_error, "rotation_error_rad": rotation_error,
                "path_collision_checked": True, "forbidden_contacts": contacts,
                "start_joint_limit_recovery_rad": start_violation,
                "planned_start_tcp_world": start_pose.tolist(), "target_tcp_world": target.tolist(),
                "attached_object": attached_object, "approximate_ik": approximation}
