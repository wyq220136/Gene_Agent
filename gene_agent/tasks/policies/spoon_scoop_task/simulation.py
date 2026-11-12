"""勺子舀球、移碗、倾倒的连续 MuJoCo Task。"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import traceback
from gene_agent.tasks.results import finish_task_report, record_task_exception

import mujoco
import numpy as np
from benchmarks.catalog import identity
from gene_agent.tasks.policies import identity as policy_identity
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import load_concept
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.motion.curobo.scene import filter_collision_snapshot
from gene_agent.simulation.diagnostics import query_diagnostics

from benchmarks.spoon_scoop_task.evaluation import balls_in_spoon, balls_in_target, pour_postcondition

from .pick_backend import SpoonPickRun
from gene_agent.tasks.tabletop.pick import _body_pose, _write


class SpoonScoopTaskRun(SpoonPickRun):
    """同一 Runtime 的三个 Skill；每段只消费上一步的实际 MuJoCo 状态。"""

    SCOOP_STAGES = ("source_above", "source_pitch", "source_entry",
                    "source_sweep", "source_lift_low",
                    "source_level")
    POUR_STAGES = ("target_above", "target_tilt_30", "target_tilt_45",
                   "target_tilt_60", "target_tilt_75", "target_settle")

    def __init__(self, scene_dir: str | Path, output_dir: str | Path,
                 *, simulation_weld: bool = False,
                 grasp_handle_x_offset_m: float = .006,
                 source_above_z_m: float = .15):
        super().__init__(scene_dir, output_dir, simulation_weld=simulation_weld,
                         grasp_handle_x_offset_m=grasp_handle_x_offset_m)
        self.source_name = self.scene["roles"]["source_container"]
        self.source_above_z_m = float(source_above_z_m)
        self.target_name = self.scene["roles"]["target_container"]
        self.ball_names = tuple(self.model.body(index).name for index in range(self.model.nbody)
                                if self.model.body(index).name.startswith("scoop_ball_"))
        self.scoop_part = load_concept(self.concept_path, part_name="scoop_bowl").concept
        self.cargo_ids: set[str] = set()
        self.report_task = {
            "schema_version": 1, "task": "scoop_balls_into_other_bowl",
            "status": "running", "success": False, "planner_backend": "curobo",
            "execution_backend": "mujoco_physical", "simulation_assistance":
            self.report["simulation_assistance"],
            "scene_xml_path": str(self.effective_xml),
            "scene_xml_sha256": self.report["scene_xml_sha256"],
            "source_scene_xml_sha256": self.report["source_scene_xml_sha256"],
            "code_sources_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[4] / name)
                                       .read_bytes()).hexdigest()
                for name in (
                    "gene_agent/tasks/policies/spoon_scoop_task/simulation.py",
                    "gene_agent/tasks/policies/spoon_scoop_task/pick_backend.py",
                    "benchmarks/tabletop.py",
                    "gene_agent/tasks/policies/spoon_scoop_task/policy.py", "benchmarks/spoon_scoop_task/definition.json", "benchmarks/spoon_scoop_task/evaluation.py", "benchmarks/spoon_scoop_task/scene.py", "gene_agent/motion/curobo/runtime_bridge.py",
                )
            },
            "source_bowl": self.source_name, "target_bowl": self.target_name,
            "tool_name": self.object_name,
            "scene_task": self.scene["task"],
            "initial_distribution": self.scene.get("distribution_sample"),
            "ball_count": len(self.ball_names), "skills": {}, "stages": {},
            "source_above_z_m": self.source_above_z_m,
        }
        self.report_task["benchmark"] = identity("spoon_scoop_task")
        self.report_task["policy_id"] = "spoon_scoop_task/default"
        self.report_task["policy"] = policy_identity("spoon_scoop_task")

        _write(self.output_dir / "task_result.json", self.report_task)

    def _scoop_center(self) -> np.ndarray:
        spoon = _body_pose(self.data.body(self.object_name))
        return (spoon @ self.scoop_part.pose_in_object)[:3, 3]

    def _scoop_local_ball_positions(self) -> dict[str, list[float]]:
        inverse = np.linalg.inv(_body_pose(self.data.body(self.object_name)))
        return {name: (inverse[:3, :3] @ self.data.body(name).xpos +
                       inverse[:3, 3]).tolist() for name in self.ball_names}

    def _balls_in_spoon(self) -> set[str]:
        return balls_in_spoon(self._scoop_local_ball_positions())

    def _balls_in_target(self) -> set[str]:
        return balls_in_target(self.data, self.target_name, self.ball_names)

    def _observe_task(self, stage: str) -> dict:
        row = {
            "stage": stage,
            "observation": self.runtime.query(
                object_names=(self.object_name, self.source_name, self.target_name),
                robot_groups=(self.group,)),
            "scoop_center_world_m": self._scoop_center().tolist(),
            "balls_in_spoon": sorted(self._balls_in_spoon()),
            "balls_in_target": sorted(self._balls_in_target()),
        }
        _write(self.output_dir / "task_observations" / f"{stage}.json", row)
        return row

    def _target_for_scoop_center(self, center_world) -> np.ndarray:
        tcp = np.asarray(self.observe()["robots"][self.group]["T_world_tcp"], dtype=float)
        target = tcp.copy()
        target[:3, 3] += np.asarray(center_world, dtype=float) - self._scoop_center()
        return target

    def _target_for_tilt(self, center_world, *, angle_deg: float,
                         axis: str = "y") -> np.ndarray:
        current_tcp = np.asarray(self.observe()["robots"][self.group]["T_world_tcp"])
        current_spoon = _body_pose(self.data.body(self.object_name))
        tcp_to_spoon = np.linalg.inv(current_tcp) @ current_spoon
        rotation = Rotation.from_euler(axis, angle_deg, degrees=True).as_matrix()
        desired_spoon = np.eye(4)
        desired_spoon[:3, :3] = rotation @ current_spoon[:3, :3]
        desired_spoon[:3, 3] = (np.asarray(center_world, dtype=float) -
                                desired_spoon[:3, :3] @
                                self.scoop_part.pose_in_object[:3, 3])
        return desired_spoon @ np.linalg.inv(tcp_to_spoon)

    def _current_grip(self) -> dict:
        contacts = query_diagnostics(self.runtime, body_names=(self.object_name,))["contacts"]
        result = self._contact_classifier(contacts, None, object_body=self.object_name)
        if not result["success"]:
            raise RuntimeError("持勺阶段失去有效双指接触")
        return result

    @staticmethod
    def _is_tilt_stage(stage: str) -> bool:
        return stage.startswith("target_tilt_")

    def _allowed_contacts(self, stage: str) -> list[tuple[str, str]]:
        allowed = [(self.object_name, "pika_gripper_left_link"),
                   (self.object_name, "pika_gripper_right_link")]
        if stage in {"source_pitch", "source_entry", "source_sweep",
                     "source_lift_low", "source_lift", "source_level"}:
            allowed.append((self.object_name, self.source_name))
            allowed.extend((self.object_name, name) for name in self.ball_names)
        if stage in {"target_above", "target_settle"} or self._is_tilt_stage(stage):
            allowed.append((self.object_name, self.target_name))
            allowed.extend((self.object_name, name) for name in self.cargo_ids)
        return allowed

    def _attachment(self, stage: str, observation: dict) -> dict | None:
        if stage in (*self.SCOOP_STAGES, "target_above") or self._is_tilt_stage(stage):
            # weld 模式依赖启用时保存的双指接触证据；逐段再次要求持续接触
            # 会把伺服微小漂移误判为掉勺。无辅助模式仍要求实时双指接触。
            if not self.simulation_weld:
                self._current_grip()
            return super()._attachment("lift", observation)
        return super()._attachment(stage, observation)

    def _snapshot_filter_task(self, stage: str, snapshot: dict) -> dict:
        rejected = set()
        if stage in {"source_pitch", "source_entry", "source_sweep",
                     "source_lift_low", "source_lift", "source_level"}:
            rejected.add(self.source_name)
            rejected.update(self.ball_names)
        if stage == "target_above" or self._is_tilt_stage(stage):
            rejected.update(self.cargo_ids)
        if self._is_tilt_stage(stage):
            rejected.add(self.target_name)
        if not rejected:
            return snapshot
        return filter_collision_snapshot(
            snapshot, self.output_dir / "motion_curobo" / "collision" /
            f"{stage}_contact_scope",
            keep_body_names=None, rejected_body_names=rejected,
            attached_object_id=self.object_name,
            scope="仅本阶段目标物理接触和已盛球从规划 world 排除；完整 MuJoCo 逐点复核",
        )

    def _postcheck_task(self, stage: str, plan: dict, _snapshot: dict) -> dict:
        positions = np.asarray(plan.get("positions_rad"), dtype=float)
        if positions.ndim != 2 or len(positions) < 2:
            return {"success": False, "reason": "missing_joint_trajectory"}
        sampled = [positions[0]]
        for start, end in zip(positions[:-1], positions[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(end - start)) / .025)))
            sampled.extend(np.linspace(start, end, count + 1)[1:])
        observed = self.observe()
        tcp = np.asarray(observed["robots"][self.group]["T_world_tcp"])
        spoon = np.asarray(observed["objects"][self.object_name]["T_world_object"])
        forbidden = self.collision_checker._check_path(
            np.asarray(sampled), allowed_contacts=self._allowed_contacts(stage),
            attached_object=self.object_name, T_tcp_object=np.linalg.inv(tcp) @ spoon)
        pivot_excursion = None
        if self._is_tilt_stage(stage):
            # 用 FK 复核整条规划轨迹的勺头中心，防止关节空间绕行把球甩出。
            probe = mujoco.MjData(self.model)
            probe.qpos[:] = self.data.qpos
            joint_adr = [self.model.joint(name).qposadr[0]
                         for name in self.robot["joint_groups"][self.group]]
            tcp_to_spoon = np.linalg.inv(tcp) @ spoon
            local_center = self.scoop_part.pose_in_object[:3, 3]
            pivot = self._scoop_center()
            deviations = []
            for joints in sampled:
                probe.qpos[joint_adr] = joints
                mujoco.mj_forward(self.model, probe)
                site = probe.site("pika_tcp")
                actual_tcp = np.eye(4)
                actual_tcp[:3, :3] = site.xmat.reshape(3, 3)
                actual_tcp[:3, 3] = site.xpos
                actual_spoon = actual_tcp @ tcp_to_spoon
                center = (actual_spoon @ np.r_[local_center, 1.])[:3]
                deviations.append(float(np.linalg.norm(center - pivot)))
            pivot_excursion = max(deviations)
        return {"success": not forbidden and
                (pivot_excursion is None or pivot_excursion <= .015),
                "forbidden_contacts": forbidden,
                "sample_count": len(sampled), "max_joint_step_rad": .025,
                "max_scoop_center_excursion_m": pivot_excursion,
                "max_scoop_center_excursion_allowed_m": .015 if pivot_excursion is not None else None,
                "source": "full_mujoco_geometry_before_curobo_trajectory_execution"}

    def _ball_contact_ids(self, replay_samples) -> set[str]:
        probe = mujoco.MjData(self.model)
        contacted = set()
        for sample in replay_samples:
            probe.qpos[:] = sample["qpos"]
            mujoco.mj_forward(self.model, probe)
            for contact in probe.contact:
                bodies = {self.model.body(int(self.model.geom_bodyid[geom])).name
                          for geom in (contact.geom1, contact.geom2)}
                if self.object_name in bodies:
                    contacted.update(name for name in bodies if name in self.ball_names)
        return contacted

    def _task_motion(self, stage: str, target: np.ndarray,
                     *, linear_tcp: bool = False) -> dict:
        before = self._observe_task(stage + "_start")
        state_id = before["observation"]["state_id"]
        bridge = CuroboRuntimeBridge(
            self.runtime, self.output_dir / "motion_curobo", group=self.group,
            linear_tcp_stages=(stage,) if linear_tcp else (),
            pivot_tcp_stages=({stage: {"max_tcp_excursion_m": .025,
                                       "max_tcp_path_length_m": .05}}
                              if stage == "target_tilt_30" else {}),
            endpoint_tolerance_rad=.05, settle_timeout_s=1.,
            replay_object_names=(self.object_name,),
            max_attempts_by_stage={stage: 2},
            execution_time_scale_by_stage={stage: (
                3.0 if stage in {"source_above", "target_above"} else 2.0)},
            attachment_for_stage=self._attachment,
            contact_policy_for_stage=self._contact_policy,
            snapshot_filter_for_stage=self._snapshot_filter_task,
            plan_postcheck_for_stage=self._postcheck_task,
        )
        result = bridge(stage, target, state_id)
        execution = result.get("execution")
        if execution is not None:
            forbidden = self.collision_checker.check_observed_contacts(
                execution.get("replay_samples", ()),
                allowed_contacts=self._allowed_contacts(stage))
            result["observed_forbidden_contacts"] = forbidden
            result["success"] = bool(result["success"] and not forbidden)
            result["spoon_ball_contact_ids"] = sorted(
                self._ball_contact_ids(execution.get("replay_samples", ())))
            if self._is_tilt_stage(stage):
                pivot = np.asarray(before["scoop_center_world_m"], dtype=float)
                local = self.scoop_part.pose_in_object[:3, 3]
                samples = execution.get("replay_samples", ())
                centers = []
                tcp_positions = []
                for sample in samples:
                    pose = np.asarray(sample["objects"][self.object_name]
                                      ["T_world_object"], dtype=float)
                    centers.append(pose[:3, 3] + pose[:3, :3] @ local)
                    tcp_positions.append(np.asarray(sample["robots"][self.group]
                                                    ["T_world_tcp"], dtype=float)[:3, 3])
                if centers:
                    actual_excursion = float(np.max(np.linalg.norm(
                        np.asarray(centers) - pivot, axis=1)))
                    tcp_positions = np.asarray(tcp_positions)
                    result["physical_pour_motion_check"] = {
                        "max_scoop_center_excursion_m": actual_excursion,
                        "max_scoop_center_excursion_allowed_m": .02,
                        "tcp_path_length_m": float(np.sum(np.linalg.norm(
                            np.diff(tcp_positions, axis=0), axis=1))),
                        "tcp_max_excursion_m": float(np.max(np.linalg.norm(
                            tcp_positions - tcp_positions[0], axis=1))),
                        "success": actual_excursion <= .02,
                    }
                    result["success"] = bool(result["success"] and
                                             actual_excursion <= .02)
        result["target_tcp_world"] = np.asarray(target).tolist()
        result["start_observation_ref"] = str(
            self.output_dir / "task_observations" / f"{stage}_start.json")
        result["end"] = self._observe_task(stage + "_end")
        self.report_task["stages"][stage] = result
        _write(self.output_dir / "task_result.json", self.report_task)
        return result

    def _hold(self, stage: str, duration_s: float) -> dict:
        start = self._observe_task(stage + "_start")
        execution = self.runtime.execute_hold(
            duration_s=duration_s, planned_state_id=start["observation"]["state_id"],
            replay_object_names=(self.object_name,), replay_group=self.group)
        result = {"success": bool(execution["success"]), "execution": execution,
                  "end": self._observe_task(stage + "_end")}
        self.report_task["stages"][stage] = result
        _write(self.output_dir / "task_result.json", self.report_task)
        return result

    def _run_scoop(self) -> bool:
        source = self.data.body(self.source_name).xpos.copy()
        entry_x = source[0] - .025
        sweep_x = source[0] + .025
        entry_y = source[1] - .03
        contacts = set()
        stages = (
            ("source_above", lambda: self._target_for_scoop_center(
                [entry_x, entry_y, self.source_above_z_m]), False),
            # TCP 保持较高，勺头下倾后进入球堆。倾角在提升后恢复。
            ("source_pitch", lambda: self._target_for_tilt(
                [entry_x, entry_y, .11], angle_deg=35.), False),
            ("source_entry", lambda: self._target_for_scoop_center(
                [entry_x, entry_y, .07]), True),
            ("source_sweep", lambda: self._target_for_scoop_center(
                [sweep_x, entry_y, .06]), True),
            ("source_lift_low", lambda: self._target_for_scoop_center(
                [sweep_x, entry_y, .12]), True),
            ("source_level", lambda: self._target_for_tilt(
                [sweep_x, entry_y, .12], angle_deg=-35.), False),
        )
        for stage, target_fn, linear in stages:
            result = self._task_motion(
                stage, target_fn(), linear_tcp=linear)
            contacts.update(result.get("spoon_ball_contact_ids", ()))
            if not result["success"]:
                self.report_task["failure_stage"] = stage
                return False
        held = self._hold("source_hold", .3)
        self.cargo_ids = self._balls_in_spoon() & contacts
        self.report_task["skills"]["scoop"] = {
            "success": bool(held["success"] and self.cargo_ids),
            "spoon_ball_contact_ids": sorted(contacts),
            "cargo_ball_ids": sorted(self.cargo_ids),
            "source_bowl_center_world_m": source.tolist(),
        }
        _write(self.output_dir / "task_result.json", self.report_task)
        if not self.report_task["skills"]["scoop"]["success"]:
            self.report_task["failure_stage"] = "scoop_postcondition"
            return False
        return True

    def _run_pour(self) -> bool:
        target_bowl = self.data.body(self.target_name).xpos.copy()
        over = [target_bowl[0], target_bowl[1], .16]
        moved = self._task_motion("target_above", self._target_for_scoop_center(over))
        if not moved["success"]:
            self.report_task["failure_stage"] = "target_above"
            return False
        self._hold("target_pretilt_hold", .25)
        retained = self._balls_in_spoon() & self.cargo_ids
        self.report_task["skills"]["carry"] = {
            "success": bool(retained), "retained_cargo_ball_ids": sorted(retained)}
        if not retained:
            self.report_task["failure_stage"] = "carry_postcondition"
            return False
        # 每段围绕实际勺头中心旋转；起点来自上一段 MuJoCo 执行终态。
        # 30° 初段使用 cuRobo 原位旋转规划；其后小角度关节规划
        # 同样由完整勺头中心轨迹约束复核。
        for angle in (30, 45, 60, 75):
            stage = f"target_tilt_{angle}"
            pivot = self._scoop_center().copy()
            tilted = self._task_motion(
                stage, self._target_for_tilt(
                    pivot, angle_deg=30. if angle == 30 else 15., axis="x"))
            if not tilted["success"]:
                self.report_task["failure_stage"] = stage
                return False
            if self.cargo_ids <= self._balls_in_target():
                break
        settled = self._hold("target_settle", 1.0)
        transferred = self._balls_in_target() & self.cargo_ids
        self.report_task["skills"]["pour"] = {
            "success": pour_postcondition(settled["success"], self.cargo_ids, transferred),
            "transferred_ball_ids": sorted(transferred),
            "cargo_ball_ids": sorted(self.cargo_ids),
            "final_tilt_deg": angle,
            "target_bowl_center_world_m": target_bowl.tolist(),
        }
        if not self.report_task["skills"]["pour"]["success"]:
            self.report_task["failure_stage"] = "pour_postcondition"
            return False
        return True


    def run_pick(self):
        return super().run()

    def save_task(self):
        _write(self.output_dir / "task_result.json", self.report_task)

    def pick(self) -> bool:
        self.report_task["initial"] = self._observe_task("initial")
        self.save_task()
        pick = self.run_pick()
        path = self.output_dir / "pick/result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        self.report_task["skills"]["pick"] = {
            "success": bool(pick["success"]),
            "physical_pick_success": bool(result.get("physical_pick_success")),
            "result_path": str(path) if result else None}
        if not pick["success"]:
            self.report_task["failure_stage"] = pick.get("failure_stage") or "pick"
            if pick.get("error"):
                self.report_task.update(error=pick["error"], traceback=pick.get("traceback"))
            return False
        return True

    def scoop(self) -> bool:
        return self._run_scoop()

    def pour(self) -> bool:
        return self._run_pour()

    def record_exception(self, error):
        record_task_exception(self.report_task, error)

    def finish(self, success, scope):
        return finish_task_report(self.report_task, success=success, scope=scope,
            full_scope="pour", observe_final=lambda: self._observe_task("final"), save=self.save_task)

    def run_task(self, *, until_skill: str = "pour") -> dict:
        from .policy import run
        return run(self, until_skill=until_skill)
