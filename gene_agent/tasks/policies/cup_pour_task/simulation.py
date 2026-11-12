"""杯把抓取、分段倾倒与放回的连续 MuJoCo Task。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback
from gene_agent.tasks.results import finish_task_report, record_task_exception

import mujoco
import numpy as np
from benchmarks.catalog import identity
from gene_agent.tasks.policies import identity as policy_identity
from scipy.spatial.transform import Rotation

from gene_agent.tasks.tabletop.pick import _body_pose, _write
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.motion.curobo.scene import filter_collision_snapshot
from gene_agent.simulation.diagnostics import query_diagnostics

from .pick_backend import CupPickRun


LIP_IN_CUP_M = np.array([0., -.035, .095])
from benchmarks.cup_pour_task.evaluation import (
    MIN_BALLS_IN_TARGET, MAX_BALLS_OUTSIDE, pour_postcondition as _pour_postcondition, ball_membership, placement_postcondition,
)
CARRY_CUP_BASE_ABOVE_BOWL_BODY_M = .135




def target_tcp_for_cup_pose(current_tcp: np.ndarray, current_cup: np.ndarray,
                            desired_cup: np.ndarray) -> np.ndarray:
    """按执行后实测 TCP→杯子关系生成目标，不重用计划中的理想抓取变换。"""
    return (np.asarray(desired_cup, dtype=float) @
            np.linalg.inv(np.linalg.inv(np.asarray(current_tcp, dtype=float)) @
                          np.asarray(current_cup, dtype=float)))


def cup_carry_pose(current_cup: np.ndarray, nominal_cup: np.ndarray,
                   bowl_position: np.ndarray) -> np.ndarray:
    """搬运时抬高杯底，并将任意初始 yaw 转为名义倾倒方向。"""
    current = np.asarray(current_cup, dtype=float)
    nominal = np.asarray(nominal_cup, dtype=float)
    bowl = np.asarray(bowl_position, dtype=float)
    if (current.shape != (4, 4) or nominal.shape != (4, 4) or
            bowl.shape != (3,) or not all(np.isfinite(item).all()
                                         for item in (current, nominal, bowl))):
        raise ValueError("杯子与碗的搬运位姿必须是有限的 4×4/3D 数据")
    goal = nominal.copy()
    goal[:3, 3] = [bowl[0] + .01, bowl[1] + .04,
                    max(current[2, 3], bowl[2] + CARRY_CUP_BASE_ABOVE_BOWL_BODY_M)]
    return goal


def tilt_cup_about_lip(cup_pose: np.ndarray, lip_in_cup: np.ndarray,
                       angle_deg: float) -> np.ndarray:
    """世界 X 分段转杯，保持杯口朝目标碗的一侧位置。"""
    cup = np.asarray(cup_pose, dtype=float)
    lip = np.asarray(lip_in_cup, dtype=float)
    if cup.shape != (4, 4) or lip.shape != (3,) or not np.isfinite(cup).all():
        raise ValueError("杯子位姿和局部杯口点必须是有限 4×4/3D 数据")
    pivot = (cup @ np.r_[lip, 1.])[:3]
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler("x", angle_deg, degrees=True).as_matrix() @ cup[:3, :3]
    goal[:3, 3] = pivot - goal[:3, :3] @ lip
    return goal


def low_angle_upright_goal(cup_pose: np.ndarray, angle_deg: float) -> np.ndarray:
    """扶正时向远离目标碗的世界 -X 微移 5 mm，避开实测 IK 边界。"""
    goal = tilt_cup_about_lip(cup_pose, LIP_IN_CUP_M, angle_deg)
    goal[0, 3] -= .005
    return goal


def vertical_retreat_waypoints(tcp_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """松杯后分两段竖直撤离，总高度仍为 7 cm，保持实际 TCP 朝向。"""
    tcp = np.asarray(tcp_pose, dtype=float)
    if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
        raise ValueError("撤离起点必须是有限的 4×4 TCP 位姿")
    lower, upper = tcp.copy(), tcp.copy()
    lower[2, 3] += .035
    upper[2, 3] += .07
    return lower, upper


class CupPourTaskRun(CupPickRun):
    """三个 Skill 消费同一物理状态；只在已验证双指接触后可使用辅助 weld。"""

    def __init__(self, scene_dir: str | Path, output_dir: str | Path,
                 *, simulation_weld: bool = False):
        super().__init__(scene_dir, output_dir, simulation_weld=simulation_weld)
        self.object_spheres_fine = self._collision_spheres(max_cell_m=.012)
        self.target_name = self.scene["roles"]["target_container"]
        self.ball_names = tuple(self.model.body(i).name for i in range(self.model.nbody)
                                if self.model.body(i).name.startswith("pour_ball_"))
        self.initial_cup_pose: np.ndarray | None = None
        self.initial_cargo: set[str] = set()
        self.attached_stages: set[str] = set()
        self.report_task = {
            "schema_version": 1, "task": "cup_pour_balls_into_bowl_and_return",
            "status": "running", "success": False,
            "planner_backend": "curobo", "execution_backend": "mujoco_physical",
            "simulation_assistance": self.report["simulation_assistance"],
            "scene_xml_path": str(self.effective_xml),
            "scene_xml_sha256": self.report["scene_xml_sha256"],
            "source_scene_xml_sha256": self.report["source_scene_xml_sha256"],
            "code_sources_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[4] / name)
                                       .read_bytes()).hexdigest()
                for name in ("gene_agent/tasks/policies/cup_pour_task/pick_backend.py",
                             "gene_agent/knowledge/grasp/cup_handle.py",
                             "gene_agent/tasks/policies/cup_pour_task/simulation.py",
                             "benchmarks/tabletop.py",
                             "gene_agent/tasks/policies/cup_pour_task/policy.py", "benchmarks/cup_pour_task/definition.json", "benchmarks/cup_pour_task/evaluation.py", "benchmarks/cup_pour_task/scene.py", "gene_agent/tasks/tabletop/pick.py", "gene_agent/tasks/results.py", "gene_agent/motion/curobo/runtime_bridge.py")},
            "cup_name": self.object_name, "target_bowl": self.target_name,
            "ball_count": len(self.ball_names),
            "initial_distribution": self.scene.get("distribution_sample"),
            "skills": {}, "stages": {},
        }
        self.report_task["benchmark"] = identity("cup_pour_task")
        self.report_task["policy_id"] = "cup_pour_task/default"
        self.report_task["policy"] = policy_identity("cup_pour_task")

        _write(self.output_dir / "task_result.json", self.report_task)

    def _ball_membership(self) -> dict[str, set[str]]:
        return ball_membership(self.data, self.object_name, self.target_name, self.ball_names)

    def _observe_task(self, stage: str) -> dict:
        membership = self._ball_membership()
        row = {"stage": stage,
               "observation": self.runtime.query(
                   object_names=(self.object_name, self.target_name),
                   robot_groups=(self.group,)),
               "ball_membership": {key: sorted(value) for key, value in membership.items()},
               "simulation_attachments": self.runtime.simulation_attachments()}
        _write(self.output_dir / "task_observations" / f"{stage}.json", row)
        return row

    def _current_target_tcp(self, desired_cup: np.ndarray) -> np.ndarray:
        observation = self.observe()
        tcp = np.asarray(observation["robots"][self.group]["T_world_tcp"])
        cup = np.asarray(observation["objects"][self.object_name]["T_world_object"])
        return target_tcp_for_cup_pose(tcp, cup, desired_cup)

    def _allowed_contacts(self, stage: str) -> list[tuple[str, str]]:
        allowed = super()._allowed_contacts(stage)
        if stage in self.attached_stages:
            allowed.extend((self.object_name, name) for name in self.ball_names)
        if stage == "place_descend":
            allowed.append((self.object_name, "table"))
        return allowed

    def _attachment(self, stage: str, observation: dict) -> dict | None:
        if stage == "lift":
            return super()._attachment(stage, observation)
        if stage in self.attached_stages:
            if not self.simulation_weld:
                contacts = query_diagnostics(self.runtime, body_names=(self.object_name,))["contacts"]
                if not self._contact_classifier(contacts, None,
                                                object_body=self.object_name)["success"]:
                    raise RuntimeError("持杯阶段失去双指杯把接触")
            result = super()._attachment("lift", observation)
            if stage in {"pour_lower_50", "pour_raise", "return_above",
                         "place_descend"}:
                # 杯口靠近碗沿以及回桌时，粗球会把真实净空误报为
                # 起点碰撞。精细附着代理仍接受完整 MuJoCo 路径复核。
                pose = np.asarray(self.observe()["objects"][self.object_name]
                                  ["T_world_object"])
                local = self.object_spheres_fine
                centers = (pose[:3, :3] @ local[:, :3].T).T + pose[:3, 3]
                result["spheres_world_m"] = np.column_stack(
                    (centers, local[:, 3])).tolist()
                result["coverage"] = (
                    "cup_bottom_wall_handle_box_cylinder_aabb_cells_12mm")
            return result
        return None

    def _snapshot_filter_task(self, stage: str, snapshot: dict) -> dict:
        if stage not in self.attached_stages:
            return snapshot
        return filter_collision_snapshot(
            snapshot, self.output_dir / "motion_curobo" / "collision" /
            f"{stage}_cup_contents_scope", keep_body_names=None,
            rejected_body_names=set(self.ball_names),
            rejected_geom_names={"scene_table"} if stage == "place_descend" else set(),
            attached_object_id=self.object_name,
            scope="持杯球只在附着物规划快照中排除；MuJoCo 完整几何逐点复核",
        )

    def _postcheck_task(self, stage: str, plan: dict, _snapshot: dict) -> dict:
        positions = np.asarray(plan.get("positions_rad"), dtype=float)
        if positions.ndim != 2 or len(positions) < 2:
            return {"success": False, "reason": "missing_joint_trajectory"}
        sampled = [positions[0]]
        for start, end in zip(positions[:-1], positions[1:]):
            steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / .025)))
            sampled.extend(np.linspace(start, end, steps + 1)[1:])
        observed = self.observe()
        tcp = np.asarray(observed["robots"][self.group]["T_world_tcp"])
        cup = np.asarray(observed["objects"][self.object_name]["T_world_object"])
        attached = stage in self.attached_stages
        tcp_to_cup = np.linalg.inv(tcp) @ cup if attached else None
        moving_contents = tuple(sorted(
            self._ball_membership()["in_cup"] & self.initial_cargo)) if attached else ()
        forbidden = self.collision_checker._check_path(
            np.asarray(sampled), allowed_contacts=self._allowed_contacts(stage),
            attached_object=self.object_name if attached else None,
            T_tcp_object=tcp_to_cup, moving_body_names=moving_contents)
        lip_excursion = None
        if stage.startswith(("tilt_", "upright_")):
            probe = mujoco.MjData(self.model)
            probe.qpos[:] = self.data.qpos
            addresses = [self.model.joint(name).qposadr[0]
                         for name in self.robot["joint_groups"][self.group]]
            pivot = (cup @ np.r_[LIP_IN_CUP_M, 1.])[:3]
            deviations = []
            for joints in sampled:
                probe.qpos[addresses] = joints
                mujoco.mj_forward(self.model, probe)
                site = probe.site("pika_tcp")
                current_tcp = np.eye(4)
                current_tcp[:3, :3] = site.xmat.reshape(3, 3)
                current_tcp[:3, 3] = site.xpos
                lip = (current_tcp @ tcp_to_cup @ np.r_[LIP_IN_CUP_M, 1.])[:3]
                deviations.append(float(np.linalg.norm(lip - pivot)))
            lip_excursion = max(deviations)
        return {"success": not forbidden and
                (lip_excursion is None or lip_excursion <= .017),
                "forbidden_contacts": forbidden,
                "moving_contents_postcheck": {
                    "mode": "rigid_with_measured_cup_transform",
                    "body_count": len(moving_contents),
                    "actual_execution_contacts_checked_separately": True},
                "max_lip_excursion_m": lip_excursion,
                "max_lip_excursion_allowed_m": .017 if lip_excursion is not None else None,
                "sample_count": len(sampled), "source": "full_mujoco_geometry_before_curobo_execution"}

    def _task_motion(self, stage: str, desired_cup: np.ndarray | None = None,
                     *, target_tcp: np.ndarray | None = None,
                     linear: bool = False, attached: bool = True) -> dict:
        if (desired_cup is None) == (target_tcp is None):
            raise ValueError("必须且只能提供杯子目标或 TCP 目标")
        before = self._observe_task(stage + "_start")
        target = (self._current_target_tcp(desired_cup) if desired_cup is not None else
                  np.asarray(target_tcp, dtype=float))
        if attached:
            self.attached_stages.add(stage)
        pivot = stage.startswith(("tilt_", "upright_"))
        bridge = CuroboRuntimeBridge(
            self.runtime, self.output_dir / "motion_curobo", group=self.group,
            linear_tcp_stages=(stage,) if linear else (),
            pivot_tcp_stages=({stage: {"max_tcp_excursion_m": .035,
                                       "max_tcp_path_length_m": .06}}
                              if pivot else {}),
            endpoint_tolerance_rad=.05, settle_timeout_s=1.,
            replay_object_names=(self.object_name, self.target_name),
            max_attempts_by_stage={stage: 2},
            execution_time_scale_by_stage={
                stage: (2.5 if pivot else 1.0 if stage.startswith("pour_lower") else 3.0)},
            attachment_for_stage=self._attachment,
            contact_policy_for_stage=self._contact_policy,
            snapshot_filter_for_stage=self._snapshot_filter_task,
            plan_postcheck_for_stage=self._postcheck_task,
            disabled_body_names_for_stage=(
                (lambda name, _obs: (self.object_name,)
                 if name.startswith("retreat_") else ())
                if stage.startswith("retreat_") else None),
        )
        result = bridge(stage, target, before["observation"]["state_id"])
        execution = result.get("execution")
        if execution is not None:
            forbidden = self.collision_checker.check_observed_contacts(
                execution.get("replay_samples", ()),
                allowed_contacts=self._allowed_contacts(stage))
            result["observed_forbidden_contacts"] = forbidden
            result["success"] = bool(result["success"] and not forbidden)
            if pivot:
                start_cup = np.asarray(before["observation"]["objects"][self.object_name]
                                       ["T_world_object"])
                point = (start_cup @ np.r_[LIP_IN_CUP_M, 1.])[:3]
                deviation = max(float(np.linalg.norm(
                    (np.asarray(sample["objects"][self.object_name]
                                ["T_world_object"]) @ np.r_[LIP_IN_CUP_M, 1.])[:3]
                    - point)) for sample in execution.get("replay_samples", ()))
                result["physical_lip_motion_check"] = {
                    "max_lip_excursion_m": deviation,
                    "allowed_m": .02, "success": deviation <= .02}
                result["success"] = bool(result["success"] and deviation <= .02)
        result["target_tcp_world"] = target.tolist()
        result["end"] = self._observe_task(stage + "_end")
        self.report_task["stages"][stage] = result
        _write(self.output_dir / "task_result.json", self.report_task)
        return result

    def _hold(self, stage: str, seconds: float) -> dict:
        start = self._observe_task(stage + "_start")
        execution = self.runtime.execute_hold(
            duration_s=seconds, planned_state_id=start["observation"]["state_id"],
            replay_object_names=(self.object_name, self.target_name), replay_group=self.group)
        result = {"success": bool(execution["success"]), "execution": execution,
                  "end": self._observe_task(stage + "_end")}
        self.report_task["stages"][stage] = result
        _write(self.output_dir / "task_result.json", self.report_task)
        return result

    def _run_pour(self) -> bool:
        bowl = self.data.body(self.target_name).xpos.copy()
        nominal = np.asarray(self.scene["nominal_object_poses"][self.object_name]
                             ["T_world_object"], dtype=float)
        desired = cup_carry_pose(_body_pose(self.data.body(self.object_name)), nominal, bowl)
        # 先在杯子原位抬高，再水平搬运；避免经过碗沿时杯底过低。
        raised = _body_pose(self.data.body(self.object_name))
        # 此抓取姿态在杯子原位再抬约 4 cm 后接近严格 IK 边界。
        # 剩余高度随搬运段取得，避免在起点要求不可达的高位姿。
        raised[2, 3] = min(desired[2, 3], raised[2, 3] + .035)
        if not self._task_motion("carry_raise", raised, linear=True)["success"]:
            self.report_task["failure_stage"] = "carry_raise"
            return False
        carry = self._task_motion("carry_above", desired)
        if not carry["success"]:
            self.report_task["failure_stage"] = "carry_above"
            return False
        self._hold("carry_hold", .3)
        # 到达碗正上方后才沿世界 Z 降到倾倒高度；搬运段保持大净空。
        approach = _body_pose(self.data.body(self.object_name))
        approach[2, 3] = bowl[2] + .115
        if not self._task_motion("pour_approach", approach, linear=True)["success"]:
            self.report_task["failure_stage"] = "pour_approach"
            return False
        retained = self._ball_membership()["in_cup"] & self.initial_cargo
        if len(retained) < .95 * len(self.initial_cargo):
            self.report_task["failure_stage"] = "carry_postcondition"
            return False
        final_angle = 0
        for angle in (15, 30, 45, 50):
            stage = f"tilt_{angle}"
            goal = tilt_cup_about_lip(_body_pose(self.data.body(self.object_name)),
                                      LIP_IN_CUP_M, float(angle - final_angle))
            result = self._task_motion(stage, goal)
            if not result["success"]:
                self.report_task["failure_stage"] = stage
                return False
            final_angle = angle
            self._hold(f"tilt_settle_{angle}", .25)
            membership = self._ball_membership()
            if len(membership["in_target"] & self.initial_cargo) >= MIN_BALLS_IN_TARGET:
                break
        else:
            # 最大只倾斜 50°；杯口再降低少许并静置，让近杯沿的球流出。
            lowered = _body_pose(self.data.body(self.object_name))
            lowered[2, 3] -= .03
            if not self._task_motion("pour_lower_50", lowered, linear=True)["success"]:
                self.report_task["failure_stage"] = "pour_lower_50"
                return False
            if not self._hold("hold_50_after_lower", 4.)["success"]:
                self.report_task["failure_stage"] = "hold_50_after_lower"
                return False
        self._hold("pour_settle", 1.)
        membership = self._ball_membership()
        transferred = membership["in_target"] & self.initial_cargo
        spilled = membership["outside"] & self.initial_cargo
        success = _pour_postcondition(transferred, spilled)
        self.report_task["skills"]["pour"] = {
            "success": success, "final_tilt_deg": final_angle,
            "initial_cargo_count": len(self.initial_cargo),
            "min_balls_in_target": MIN_BALLS_IN_TARGET,
            "max_balls_outside": MAX_BALLS_OUTSIDE,
            "transferred_count": len(transferred), "spilled_count": len(spilled),
            "transferred_ball_ids": sorted(transferred),
            "spilled_ball_ids": sorted(spilled),
            "remaining_in_cup_ids": sorted(membership["in_cup"] & self.initial_cargo)}
        _write(self.output_dir / "task_result.json", self.report_task)
        if not success:
            self.report_task["failure_stage"] = "pour_postcondition"
            return False
        return True

    def _run_place(self) -> bool:
        poured = self.report_task["skills"]["pour"]["final_tilt_deg"]
        # 低角度倾倒的杯口本就远离碗沿；小幅竖直抬离即可。
        # 原 90° 倾倒需要更大的抬升净空。
        raised = _body_pose(self.data.body(self.object_name))
        raised[2, 3] += .025 if poured <= 50 else .075
        if not self._task_motion("pour_raise", raised, linear=True)["success"]:
            self.report_task["failure_stage"] = "pour_raise"
            return False
        current_angle = poured
        for angle in range((poured - 1) // 15 * 15, -1, -15):
            current_cup = _body_pose(self.data.body(self.object_name))
            goal = (low_angle_upright_goal(current_cup, float(angle - current_angle))
                    if angle == 0 and poured <= 50 else
                    tilt_cup_about_lip(current_cup, LIP_IN_CUP_M,
                                       float(angle - current_angle)))
            if angle == 0:
                # 大倾角回位需要侧移避开 IK 死角；50° 以内保持杯沿
                # 的原位置，避免横移把剩余小球甩向夹爪。
                if poured > 50:
                    goal[1, 3] -= .03
                stage = "return_turn_upright"
            else:
                stage = f"upright_{angle}"
            if not self._task_motion(stage, goal)["success"]:
                self.report_task["failure_stage"] = stage
                return False
            current_angle = angle
            if angle == 15 and poured <= 50:
                # 杯内仍有大部分球，扶正前让它们落稳。
                if not self._hold("upright_settle_15", .8)["success"]:
                    self.report_task["failure_stage"] = "upright_settle_15"
                    return False
            if angle == 0 and poured <= 50:
                # 杯子水平移动回原位前，让杯口附近的球沉入杯内。
                if not self._hold("upright_settle_0", 1.)["success"]:
                    self.report_task["failure_stage"] = "upright_settle_0"
                    return False
            if angle == 45:
                # 先尝试竖直抬离碗沿。低角度时它只是可选净空动作：
                # 若尚未执行就被直线规划拒绝，后续扶正仍会完整复核碰撞。
                clear = _body_pose(self.data.body(self.object_name))
                clear[2, 3] += .04
                clear_result = self._task_motion("clear_bowl", clear, linear=True)
                if not clear_result["success"]:
                    if poured > 50 or clear_result.get("execution") is not None:
                        self.report_task["failure_stage"] = "clear_bowl"
                        return False
                    self.report_task["optional_clear_bowl"] = {
                        "skipped_after_plan_failure": True,
                        "planner_status": clear_result.get("plan", {}).get("status")}
                    _write(self.output_dir / "task_result.json", self.report_task)
        above = self.initial_cup_pose.copy()
        above[2, 3] = .085
        if not self._task_motion("return_above", above)["success"]:
            self.report_task["failure_stage"] = "return_above"
            return False
        placed = self.initial_cup_pose.copy()
        placed[2, 3] = .004
        if not self._task_motion("place_descend", placed, linear=True)["success"]:
            self.report_task["failure_stage"] = "place_descend"
            return False
        before_release = self._observe_task("release_start")
        if self.simulation_weld:
            self.runtime.release_simulation_object(self.group, self.object_name)
        after_release = self._observe_task("released")
        opened = self.runtime.execute_gripper_command(
            group=self.group, width_m=.08, duration_s=.4,
            planned_state_id=after_release["observation"]["state_id"], timeout_s=.7,
            replay_object_names=(self.object_name, self.target_name))
        self.report_task["stages"]["release_open"] = {
            "success": bool(opened["success"]), "execution": opened,
            "start": before_release, "end": self._observe_task("release_open_end")}
        _write(self.output_dir / "task_result.json", self.report_task)
        if not opened["success"]:
            self.report_task["failure_stage"] = "release_open"
            return False
        tcp = np.asarray(self.observe()["robots"][self.group]["T_world_tcp"])
        for stage, waypoint in zip(("retreat_low", "retreat_high"),
                                   vertical_retreat_waypoints(tcp)):
            if not self._task_motion(stage, target_tcp=waypoint,
                                     linear=True, attached=False)["success"]:
                self.report_task["failure_stage"] = stage
                return False
        self._hold("place_settle", .6)
        cup = _body_pose(self.data.body(self.object_name))
        translation = float(np.linalg.norm(
            cup[:2, 3] - self.initial_cup_pose[:2, 3]))
        angle = float(np.linalg.norm(Rotation.from_matrix(
            cup[:3, :3] @ self.initial_cup_pose[:3, :3].T).as_rotvec()))
        success = placement_postcondition(translation, angle, cup[2, 3],
                                          self.runtime.simulation_attachments())
        self.report_task["skills"]["place"] = {
            "success": success, "xy_error_m": translation,
            "orientation_error_rad": angle,
            "final_cup_pose_world": cup.tolist(),
            "initial_cup_pose_world": self.initial_cup_pose.tolist()}
        if not success:
            self.report_task["failure_stage"] = "place_postcondition"
        return success


    def run_pick(self):
        return super().run()

    def cargo_goal_satisfied(self, transferred, spilled):
        return _pour_postcondition(transferred, spilled)

    def save_task(self):
        _write(self.output_dir / "task_result.json", self.report_task)

    def pick(self) -> bool:
        self.report_task["initial"] = self._observe_task("initial")
        pick = self.run_pick()
        self.report_task["skills"]["pick"] = {
            "success": bool(pick["success"]),
            "result_path": str(self.output_dir / "pick/result.json")
                if (self.output_dir / "pick/result.json").is_file() else None,
            "bilateral_handle_contact": bool((pick.get("contact") or {}).get("bilateral_pad_contact")),
            "simulation_assisted": bool(pick.get("simulation_assisted_pick_success"))}
        if not pick["success"]:
            self.report_task.update(failure_stage=pick.get("failure_stage") or "pick", error=pick.get("error"))
            if pick.get("traceback"):
                self.report_task["traceback"] = pick["traceback"]
            return False
        self.initial_cup_pose = np.asarray(self.report_task["initial"]["observation"]["objects"][self.object_name]["T_world_object"], dtype=float)
        self.initial_cargo = self._ball_membership()["in_cup"]
        self.report_task["initial_cargo_count"] = len(self.initial_cargo)
        self.report_task["initial_cargo_ball_ids"] = sorted(self.initial_cargo)
        if len(self.initial_cargo) < .95 * len(self.ball_names):
            self.report_task["failure_stage"] = "pick_ball_retention"
            return False
        return True

    def pour(self) -> bool:
        return self._run_pour()

    def place(self) -> bool:
        if not self._run_place():
            return False
        membership = self._ball_membership()
        transferred = membership["in_target"] & self.initial_cargo
        spilled = membership["outside"] & self.initial_cargo
        final_ok = self.cargo_goal_satisfied(transferred, spilled)
        self.report_task["final_cargo_postcondition"] = {
            "success": final_ok, "transferred_count": len(transferred),
            "spilled_count": len(spilled),
            "remaining_in_cup_count": len(membership["in_cup"] & self.initial_cargo)}
        if not final_ok:
            self.report_task["failure_stage"] = "final_cargo_postcondition"
        return final_ok

    def record_exception(self, error):
        record_task_exception(self.report_task, error)

    def finish(self, success, scope):
        return finish_task_report(self.report_task, success=success, scope=scope,
            full_scope="place", observe_final=lambda: self._observe_task("final"), save=self.save_task)

    def run_task(self, *, until_skill: str = "place") -> dict:
        from .policy import run
        return run(self, until_skill=until_skill)
