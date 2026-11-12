"""刷子清扫方块的同步 MuJoCo Task；只从 fresh reset 开始。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from benchmarks.catalog import identity
from gene_agent.tasks.results import annotate_task_result
from gene_agent.tasks.policies import identity as policy_identity
from PIL import Image
from scipy.spatial.transform import Rotation

from gene_agent.robots.contact.pika import classify_pika_brush_grasp
from benchmarks.sweep_cube_task.evaluation import evaluate_cube_in_dustpan
from gene_agent.knowledge.geometry import load_concept
from gene_agent.knowledge.sweep import (make_brush_grasp_pose, make_sweep_layout,
                                        make_sweep_contact_pose_candidates,
                                        align_sweep_contact_candidates)
from gene_agent.motion.mujoco_ik import MujocoPosePlanner
from gene_agent.scene.tabletop_randomization import sample_tabletop_layout
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.simulation.execution import scale_position_actuator_kp
from gene_agent.simulation.runtime import MujocoRuntime
from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
from gene_agent.skills.library.pick.brush import brush_pick_targets, brush_lift_target
from gene_agent.robots.pika_geometry import pika_grasp_center_tcp
from gene_agent.skills.library.pick.brush import load_brush_pick_profile
from gene_agent.skills.library.pick.model import GraspSettings
from gene_agent.skills.library.sweep.skill import run_sweep_skill


OBJECTS = ("brush", "cube", "cube_red", "cube_blue", "dustpan")
TARGET_BLOCKS = ("cube", "cube_red", "cube_blue")


def approximate_ik_budget_for_stage(stage_name: str) -> dict[str, float] | None:
    """只对非接触目标开放受限近似 IK；扫动与抓紧保持严格。"""
    if stage_name.endswith("__retreat"):
        return {"max_position_error_m": .003, "max_rotation_error_rad": .08}
    if stage_name.endswith("__move_behind_cube"):
        return {"max_position_error_m": .010, "max_rotation_error_rad": .065}
    if stage_name == "pregrasp":
        # 离抓取点原有 80 mm 预位；目标最多移动 25 mm，后续抓取仍严格。
        return {"max_position_error_m": .025, "max_rotation_error_rad": .065}
    return None


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    if path.parent.name == 'stages' and not path.stem.endswith('_plan'):
        print(f"[{time.strftime('%H:%M:%S')}] 阶段记录 {path.stem}："
              f"success={value.get('success', '见记录')}；"
              f"failure={value.get('failure_type') or '无'}", flush=True)


def _pose(body) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = body.xmat.reshape(3, 3)
    transform[:3, 3] = body.xpos
    return transform



def _merge_contact_settling(motion, hold):
    """保留两段原始执行，合成真实阶段终态和时长，避免回放与摘要矛盾。"""
    success = bool(motion["success"] and hold["success"])
    return {"success": success, "status": "completed" if success else "failed",
            "failure_type": motion.get("failure_type") or hold.get("failure_type"),
            "start_state_id": motion["start_state_id"], "end_state_id": hold["end_state_id"],
            "sim_duration_s": motion["sim_duration_s"] + hold["sim_duration_s"],
            "group": motion.get("group"), "joint_names": motion.get("joint_names"),
            "samples": [*motion.get("samples", ()), *hold.get("samples", ())],
            "replay_samples": [*motion.get("replay_samples", ()), *hold.get("replay_samples", ())],
            "motion_execution": motion, "contact_settling": hold}

def scene_with_robot_gravity_compensation(source, destination, robot_metadata):
    """复用已有机器人重力补偿，在run内落盘有效XML，不改采集场景。"""
    import xml.etree.ElementTree as ET
    from gene_agent.simulation import configure_robot_gravity_compensation
    source, destination = Path(source).resolve(), Path(destination).resolve()
    tree = ET.parse(source)
    for element in tree.getroot().iter():
        if element.get("file"):
            element.set("file", str((source.parent / element.get("file")).resolve()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    configure_robot_gravity_compensation(destination, robot_metadata, ("left",))
    return destination


from benchmarks.sweep_cube_task.scene import place_initial_objects, record_settled_placement

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[4] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs


class SweepCubeRun:
    def __init__(self, scene_dir: str | Path, output_dir: str | Path,
                 *, randomize: bool = False, seed: int = 0,
                 diagnostic_cpu_ik: bool = False, grasp_tilt_deg: float = -20.,
                 robot_gravity_compensation: bool = False,
                 prepared_output_dir: bool = False):
        if not np.isfinite(grasp_tilt_deg) or abs(grasp_tilt_deg) > 45.:
            raise ValueError("抓刷倾角应在±45度内")
        self.grasp_tilt_deg = float(grasp_tilt_deg)
        self.diagnostic_cpu_ik = bool(diagnostic_cpu_ik)
        self.motion_backend = "diagnostic_cpu_ik" if diagnostic_cpu_ik else "curobo"
        self._pending_contact_candidates: dict[str, list[dict]] = {}
        self.scene_dir = Path(scene_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        if prepared_output_dir:
            if not self.output_dir.is_dir() or (self.output_dir / "result.json").exists():
                raise ValueError("预先创建的运行目录无效或已经有结果")
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        self.concept_dir = self.scene_dir / "concepts"
        self.scene = json.loads((self.scene_dir / "scene.json").read_text(encoding="utf-8"))
        self.robot = json.loads((self.scene_dir / "robot/robot_metadata.json").read_text(encoding="utf-8"))
        source_scene = self.scene_dir / "scene.xml"
        effective_scene = (scene_with_robot_gravity_compensation(
            source_scene, self.output_dir / "simulation_scene.xml", self.robot)
            if robot_gravity_compensation else source_scene)
        self.model = mujoco.MjModel.from_xml_path(str(effective_scene))
        # 刷子任务专用仿真辅助刚性；仅在双侧接触核验后激活。
        # 侧壁仍做规划及实际执行检查，不能依靠硬约束穿过障碍。
        brush_weld = self.model.equality("sim_attach_left_brush").id
        self.model.eq_solimp[brush_weld, :2] = .9999
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.placement = self._place_initial_objects(randomize=randomize, seed=seed)
        if 'placement_rules' in self.scene:
            # 规则可能移动固定簸箕；把实际模型写到新 run，回放不能再加载旧 XML。
            effective_scene=self.output_dir/'placement_scene.xml'
            mujoco.mj_saveLastXML(str(effective_scene),self.model)
        self.group = "left"
        self.robot_group_names = self.robot["joint_groups"][self.group]
        # 机器人 position actuator 原 kp=150 难以抗重力接近桌面；仅此仿真 run 放大 5 倍。
        self.controller_kp = scale_position_actuator_kp(
            self.model, self.robot_group_names, 5.)
        self.runtime = MujocoRuntime(self.model, self.data, {
            "robot": self.robot, "world_frame": self.scene["world_frame"],
            "simulation_attachment_pairs": self.scene["simulation_attachment_pairs"]})
        self.planner = MujocoPosePlanner(self.model, self.data, self.robot, self.group)
        self.report = {"schema_version": 1, "task": "sweep_cube_into_dustpan",
                       "status": "running", "success": False,
                       "planner_backend": "mujoco_cpu_pose_ik" if diagnostic_cpu_ik else "curobo",
                       "execution_backend": (
                           "mujoco_cpu_pose_ik_physical_execution" if diagnostic_cpu_ik else
                           "curobo_planning_mujoco_physical_execution"),
                       "evaluation_scope": (
                           "diagnostic_cpu_ik_only" if diagnostic_cpu_ik else
                           "curobo_stage_collision_planning_with_full_mujoco_postcheck"),
                       "knowledge_candidate_screen": (
                           "curobo_pure_pose_ik_then_mujoco_cpu_path_diagnostic"
                           if diagnostic_cpu_ik else "curobo_full_trajectory"),
                       "scene_xml_path": str(effective_scene),
                       "scene_xml_sha256": hashlib.sha256(
                           effective_scene.read_bytes()).hexdigest(),
                       "source_scene_xml_path": str(source_scene),
                       "source_scene_xml_sha256": hashlib.sha256(source_scene.read_bytes()).hexdigest(),
                       "robot_gravity_compensation": bool(np.any(self.model.body_gravcomp)),
                       "gravity_compensated_bodies": [self.model.body(i).name
                           for i in np.flatnonzero(self.model.body_gravcomp)],
                       "concept_sha256": {name: hashlib.sha256(
                           (self.scene_dir / self.scene["objects"][name].get(
                               "concept_path", f"concepts/{name}.json")).read_bytes()).hexdigest()
                           for name in OBJECTS if name != "dustpan" or
                           (self.concept_dir / "dustpan.json").is_file()},
                       "controller_kp_scale": 5., "grasp_tilt_deg": self.grasp_tilt_deg,
                       "brush_weld_solver": {"solref": self.model.eq_solref[brush_weld].tolist(),
                                             "solimp": self.model.eq_solimp[brush_weld].tolist()},
                       "simulation_assistance": "weld_after_verified_bilateral_inner_finger_contact",
                       "stages": {}, "failure_stage": None,
                       "placement": self.placement}
        self.report["benchmark"] = identity("sweep_cube_task")
        self.report["policy_id"] = "sweep_cube_task/default"
        self.report["policy"] = policy_identity("sweep_cube_task")

        source_root = Path(__file__).resolve().parents[4]
        sources = ("gene_agent/knowledge/sweep.py", "gene_agent/skills/library/sweep/skill.py",
                   "gene_agent/tasks/policies/sweep_cube_task/simulation.py",
                   "gene_agent/tasks/policies/sweep_long_task/policy.py",
                   "gene_agent/tasks/policies/sweep_long_task/simulation.py",
                   "gene_agent/tasks/results.py",
                   "gene_agent/tasks/policies/sweep_cube_task/execution/selection.py",
                   "gene_agent/tasks/policies/sweep_cube_task/execution/curobo.py",
                   "gene_agent/tasks/policies/sweep_cube_task/execution/grip.py",
                   "benchmarks/sweep_cube_task/evaluation.py",
                   "gene_agent/tasks/policies/sweep_cube_task/policy.py", "benchmarks/sweep_cube_task/definition.json", "benchmarks/sweep_cube_task/evaluation.py", "benchmarks/sweep_cube_task/scene.py", "benchmarks/sweep_long_task/scene.py", "gene_agent/motion/curobo/profiles/pika.py")
        self.report["code_sources_sha256"] = {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in sources}
        from gene_agent.skills.library.pick.brush import brush_pick_source_hashes
        self.report['code_sources_sha256'].update(brush_pick_source_hashes())
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.initial_brush_z = float(self.data.body("brush").xpos[2])
        self._record_settled_placement()
        self.report["initial_state"] = self.observe()
        self.runtime.replay_sample(object_names=OBJECTS,robot_groups=(self.group,))
        if not diagnostic_cpu_ik:
            from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import brush_collision_spheres_object
            self._brush_spheres_object = brush_collision_spheres_object(self.model, self.data, max_cell_m=.012)
        self._render("initial")



    def observe(self) -> dict:
        return self.runtime.query(object_names=OBJECTS, robot_groups=(self.group,))

    def _render(self, name: str) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = 145, -30, .95
        camera.lookat[:] = [.44, 0, .10]
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        with mujoco.Renderer(self.model, height=720, width=960) as renderer:
            renderer.update_scene(self.data, camera=camera, scene_option=option)
            path = self.output_dir / "renders" / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(path)

    def _attached_pose(self) -> np.ndarray | None:
        if not self.runtime.simulation_attachments():
            return None
        obs = self.observe()
        tcp = np.asarray(obs["robots"][self.group]["T_world_tcp"])
        brush = np.asarray(obs["objects"]["brush"]["T_world_object"])
        return np.linalg.inv(tcp) @ brush

    def _move(self, name: str, target: np.ndarray, path_kind: str, duration_s: float,
              *, allowed_contacts=()) -> dict:
        if not self.diagnostic_cpu_ik:
            return self._move_curobo(name, target, path_kind, duration_s,
                                     allowed_contacts=allowed_contacts)
        before = self.observe()
        attached = self._attached_pose()
        plan = self.planner.plan_to_pose(
            target, path_kind=path_kind, duration_s=duration_s,
            allowed_contacts=allowed_contacts,
            attached_object="brush" if attached is not None else None,
            T_tcp_object=attached,
            approximate_ik_budget=approximate_ik_budget_for_stage(name))
        row = {"start_state_id": before["state_id"], "target_tcp_world": target.tolist(),
               "path_kind": path_kind,
               "allowed_contact_pairs": [list(pair) for pair in allowed_contacts],
               "plan": plan, "execution": None}
        self.report["stages"][name] = row
        _write(self.output_dir / "stages" / f"{name}_plan.json", row)
        if not plan["success"]:
            return {"success": False, "executed": False,
                    "failure_type": plan["failure_type"], "plan": plan}
        execution = self.runtime.execute_joint_trajectory(
            group=self.group, joint_names=plan["joint_names"],
            times_s=plan["times_s"], positions_rad=plan["positions_rad"],
            planned_state_id=before["state_id"], timeout_s=duration_s + .8,
            endpoint_tolerance_rad=.03, settle_timeout_s=.6,
            replay_object_names=OBJECTS)
        after = self.observe()
        actual = np.asarray(after["robots"][self.group]["T_world_tcp"])
        position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        rotation_error = float(np.linalg.norm(Rotation.from_matrix(
            target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
        forbidden_observed = self.planner.check_observed_contacts(
            execution.get("replay_samples", ()), allowed_contacts=allowed_contacts)
        # 物理执行跟踪误差独立于 IK 误差；下一阶段从实际终点重规划。
        success = bool(execution["success"] and position_error <= .025
                       and rotation_error <= .12 and not forbidden_observed)
        approximation = plan.get("approximate_ik") or {}
        if approximation.get("approximate"):
            budget = approximation["budget"]
            retreat_clearance = None
            if name.endswith("__retreat"):
                start_brush_z = float(np.asarray(
                    before["objects"]["brush"]["T_world_object"])[2, 3])
                end_brush_z = float(np.asarray(
                    after["objects"]["brush"]["T_world_object"])[2, 3])
                retreat_clearance = end_brush_z - start_brush_z
            row["approximate_ik_postcheck"] = {
                "actual_tcp_world": actual.tolist(),
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
                "budget": budget,
                "retreat_brush_lift_m": retreat_clearance,
                "success": (position_error <= .025 and rotation_error <= .12
                            and not forbidden_observed and
                            (retreat_clearance is None or retreat_clearance >= .05)),
            }
            success = success and row["approximate_ik_postcheck"]["success"]
        row.update(execution=execution, end_state=after,
                   actual_tcp_position_error_m=position_error,
                   actual_tcp_rotation_error_rad=rotation_error,
                   forbidden_observed_contacts=forbidden_observed,
                   success=success,
                   failure_type=(None if success else
                                 "execution_forbidden_contact" if forbidden_observed else
                                 execution.get("failure_type")
                                 or "physical_tcp_endpoint_error"))
        _write(self.output_dir / "stages" / f"{name}.json", row)
        return {"success": success, "failure_type": row["failure_type"],
                "plan": plan, "execution": execution}

    def _move_curobo(self, name: str, target: np.ndarray, path_kind: str,
                     duration_s: float, *, allowed_contacts=(), execute=True) -> dict:
        """每段从真实状态冻结碰撞场景，cuRobo 规划后才交给 MuJoCo 执行。"""
        print(f"[{time.strftime('%H:%M:%S')}] cuRobo {'规划并执行' if execute else '候选预检（不执行）'}："
              f"{name} ({path_kind})", flush=True)
        from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
        from gene_agent.motion.curobo.scene import filter_collision_snapshot

        before = self.observe()
        attached = self._attached_pose()
        active = self.runtime.simulation_attachments()
        allowed = tuple(tuple(pair) for pair in allowed_contacts)
        candidate_rows = self._pending_contact_candidates.pop(name, None)
        targets = ([np.asarray(row["target_tcp_world"], dtype=float)
                    for row in candidate_rows] if candidate_rows else [np.asarray(target, dtype=float)])
        trials = []

        def attachment_for_stage(_stage: str, observation: dict) -> dict | None:
            if attached is None:
                return None
            if len(active) != 1 or active[0]["object_instance_id"] != "brush":
                raise ValueError("持刷阶段缺少接触核验后的仿真附着证据")
            brush = np.asarray(self.observe()["objects"]["brush"]["T_world_object"], dtype=float)
            local = self._brush_spheres_object
            centers = (brush[:3, :3] @ local[:, :3].T).T + brush[:3, 3]
            return {"object_instance_id": "brush",
                    "spheres_world_m": np.column_stack((centers, local[:, 3])).tolist(),
                    "coverage": "compiled_collision_mesh_per_geom_aabb_grid_12mm",
                    "physical_hold_evidence": active[0]}

        def snapshot_filter(stage: str, snapshot: dict) -> dict:
            rejected_bodies = set()
            rejected_geoms = set()
            if name == "approach":
                rejected_bodies.add("brush")
            if name == "lift" and self.scene.get("grasp_mode") == "world_minus_z":
                # 刷头从支架内拔出，唯独起始接触的支架需要阶段白名单。
                rejected_geoms.update(("brush_long_stand_left", "brush_long_stand_right"))
            if attached is not None:
                for first, second in allowed:
                    if first == "brush" and second in TARGET_BLOCKS + ("dustpan",):
                        rejected_bodies.add(second)
                    if first == "brush" and second == "table":
                        rejected_geoms.add("scene_table")
            if not rejected_bodies and not rejected_geoms:
                return snapshot
            return filter_collision_snapshot(
                snapshot, self.output_dir / "motion_curobo" / "collision" /
                f"{stage}_contact_scope",
                keep_body_names=None, rejected_body_names=rejected_bodies,
                rejected_geom_names=rejected_geoms,
                attached_object_id="brush" if attached is not None else None,
                scope=("保留完整来源快照；仅阶段允许接触的障碍从 cuRobo world 移除；"
                       "执行前以完整 MuJoCo 几何逐点复核所有非白名单接触"),
            )

        def mounting_policy(_stage: str, snapshot: dict) -> dict:
            table_names = [row["obstacle_name"] for row in snapshot["records"]
                           if row["geom_name"] == "scene_table"]
            return {"robot_links": ["base_link"] if table_names else [],
                    "obstacle_names": table_names,
                    "source": "sweep_scene_base_link_table_mounting_spheres",
                    "semantics": "仅固定底座球与桌面的已知安装交叠"}

        def postcheck(_stage: str, plan: dict, _snapshot: dict) -> dict:
            positions = np.asarray(plan.get("positions_rad"), dtype=float)
            if positions.ndim != 2 or len(positions) < 2:
                return {"success": False, "reason": "missing_joint_trajectory"}
            sampled = [positions[0]]
            for start, end in zip(positions[:-1], positions[1:]):
                count = max(1, int(np.ceil(np.max(np.abs(end - start)) / .025)))
                sampled.extend(np.linspace(start, end, count + 1)[1:])
            forbidden = self.planner._check_path(
                np.asarray(sampled), allowed_contacts=allowed,
                attached_object="brush" if attached is not None else None,
                T_tcp_object=attached)
            from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import check_brush_pan_walls
            forbidden.extend(check_brush_pan_walls(self.planner, sampled, attached))
            return {"success": not forbidden, "sample_count": len(sampled),
                    "max_joint_step_rad": .025,
                    "forbidden_contacts": forbidden,
                    "source": "full_mujoco_geometry_before_curobo_trajectory_execution"}

        chosen = None
        for index, trial_target in enumerate(targets):
            trial_name = f"{name}_candidate_{index}" if candidate_rows else name
            bridge = CuroboRuntimeBridge(
                self.runtime, self.output_dir / "motion_curobo", group=self.group,
                attachment_for_stage=attachment_for_stage,
                snapshot_filter_for_stage=snapshot_filter,
                contact_policy_for_stage=mounting_policy,
                plan_postcheck_for_stage=postcheck,
                linear_tcp_stages=(trial_name,) if path_kind == "linear_tcp" else (),
                endpoint_tolerance_rad=.03, settle_timeout_s=.6,
                replay_object_names=OBJECTS,
                max_attempts_by_stage={trial_name: 2},
                approximate_ik_by_stage=(
                    {trial_name: approximate_ik_budget_for_stage(name)}
                    if approximate_ik_budget_for_stage(name) is not None else {}),
                execution_time_scale_by_stage=(
                    {trial_name: 2.0} if name.endswith("__sweep") else {}),
            )
            method = bridge if execute else bridge.plan_only
            result = method(trial_name, trial_target, before["state_id"])
            trials.append({"candidate_index": index, "target_tcp_world": trial_target.tolist(),
                           "plan_success": bool(result["plan"].get("success")),
                           "failure_category": result["plan"].get("failure_category"),
                           "execution_started": result.get("execution") is not None})
            chosen = (result, trial_target, index)
            if result["success"] or result.get("execution") is not None:
                break

        result, target, index = chosen
        if not execute:
            return result
        plan, execution = result["plan"], result.get("execution")
        if result["success"] and execution is not None and name.endswith(("__lower_to_contact", "__sweep")):
            # 关节门槛只表示轨迹结束；接触阶段再稳态保持，并检查实际TCP与接触。
            hold = self.runtime.execute_hold(duration_s=.5, planned_state_id=self.observe()["state_id"],
                replay_object_names=OBJECTS, replay_group=self.group)
            execution = _merge_contact_settling(execution, hold)
            result["success"] = bool(result["success"] and execution["success"])
        after = self.observe() if execution is not None else None
        actual = (np.asarray(after["robots"][self.group]["T_world_tcp"])
                  if after is not None else None)
        position_error = (float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                          if actual is not None else None)
        rotation_error = (float(np.linalg.norm(Rotation.from_matrix(
            target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
                          if actual is not None else None)
        forbidden_observed = (self.planner.check_observed_contacts(
            execution.get("replay_samples", ()), allowed_contacts=allowed)
            if execution is not None else [])
        from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import check_observed_brush_pan_walls
        if execution is not None:
            forbidden_observed.extend(check_observed_brush_pan_walls(
                self.planner, execution.get("replay_samples", ())))
        from gene_agent.tasks.policies.sweep_cube_task.execution.grip import measure_weld_drift
        grip_drift = (measure_weld_drift(self.model, self.data, execution.get("replay_samples", ()))
                      if execution is not None else None)
        contact_stage = name.endswith(("__lower_to_contact", "__sweep"))
        position_tolerance = .005 if contact_stage else .025
        rotation_tolerance = .05 if contact_stage else .12
        success = bool(result["success"] and execution is not None
                       and position_error <= position_tolerance and rotation_error <= rotation_tolerance
                       and not forbidden_observed
                       and (grip_drift is None or grip_drift["success"]))
        row = {"start_state_id": before["state_id"],
               "target_tcp_world": target.tolist(), "path_kind": path_kind,
               "requested_duration_s": duration_s,
               "allowed_contact_pairs": [list(pair) for pair in allowed],
               "planning_backend": "curobo", "candidate_trials": trials,
               "selected_candidate_index": index if candidate_rows else None,
               "plan": plan, "execution": execution, "end_state": after,
               "actual_tcp_position_error_m": position_error,
               "actual_tcp_rotation_error_rad": rotation_error,
               "actual_tcp_tolerances": {"position_m": position_tolerance, "rotation_rad": rotation_tolerance},
               "forbidden_observed_contacts": forbidden_observed,
               "grip_drift_diagnostic": grip_drift,
               "success": success,
               "failure_type": (None if success else
                                "execution_forbidden_contact" if forbidden_observed else
                                "simulation_grip_drift" if grip_drift is not None and not grip_drift["success"] else
                                (execution or {}).get("failure_type")
                                or plan.get("failure_category")
                                or "physical_tcp_endpoint_error")}
        self.report["stages"][name] = row
        _write(self.output_dir / "stages" / f"{name}_plan.json", row)
        if after is not None:
            _write(self.output_dir / "stages" / f"{name}.json", row)
        return {"success": success, "failure_type": row["failure_type"],
                "plan": plan, "execution": execution}

    def _target_for_brush(self, desired_xyz: np.ndarray) -> np.ndarray:
        observation = self.observe()
        tcp = np.asarray(observation["robots"][self.group]["T_world_tcp"])
        brush = np.asarray(observation["objects"]["brush"]["T_world_object"])
        target = tcp.copy()
        target[:3, 3] += desired_xyz - brush[:3, 3]
        return target

    def _contact_target_for_brush(self, desired_xyz: np.ndarray,
                                  *, stage_label: str, direction_xy: np.ndarray,
                                  target_name: str = "cube",
                                  recovery_inside_pan: bool = False, cube_goal_world_xyz=None) -> np.ndarray:
        """从功能刷面候选中选首条纯 cuRobo IK 和诊断路径都可行的接触 pose。"""
        from gene_agent.motion.curobo.core import SynchronousPlanner, load_realman_motion_config
        from gene_agent.motion.curobo.profiles.pika import export_pika_gripper_snapshot

        observation = self.observe()
        robot_state = observation["robots"][self.group]
        tcp = np.asarray(robot_state["T_world_tcp"], dtype=float)
        brush = np.asarray(observation["objects"]["brush"]["T_world_object"], dtype=float)
        base_target = self._target_for_brush(desired_xyz)
        table_geom = self.model.geom("scene_table")
        table_z = float(self.data.geom_xpos[table_geom.id][2] + table_geom.size[2])
        candidates = make_sweep_contact_pose_candidates(
            base_target, tcp, brush, self.concept_dir / "brush.json",
            sweep_direction_world_xy=direction_xy, table_top_z_m=table_z)
        candidates = align_sweep_contact_candidates(
            candidates, tcp, {n: np.asarray(r["T_world_object"])
                             for n, r in observation["objects"].items()},
            self.concept_dir, target_name=target_name,
            recovery_inside_pan=recovery_inside_pan, cube_goal_world_xyz=cube_goal_world_xyz,
            goal_inset_extra_m=float(self.scene.get("goal_inset_extra_m", 0.)))
        if not self.diagnostic_cpu_ik:
            stage_name = f"{stage_label}__lower_to_contact"
            self._pending_contact_candidates[stage_name] = candidates
            self.report["knowledge"][stage_label]["contact_candidate_screen"] = {
                "scene_scope": "逐个候选由完整场景 cuRobo 直线规划决定；无 CPU 筛选",
                "table_top_z_m": table_z, "candidates": candidates,
            }
            _write(self.output_dir / "knowledge" / f"{stage_label}_contact_candidates.json",
                   self.report["knowledge"][stage_label]["contact_candidate_screen"])
            return np.asarray(candidates[0]["target_tcp_world"], dtype=float)
        gripper = export_pika_gripper_snapshot(self.model, self.data, self.robot, self.group)
        settings = load_realman_motion_config(self.group, self.robot,
                                              gripper_snapshot=gripper)
        planner = SynchronousPlanner(
            **settings, scene_model={"cuboid": {}, "mesh": {}},
            scene_scope="清扫 Knowledge 候选纯 pose IK：关节限位和自身碰撞；场景路径另查")
        T_tcp_brush = np.linalg.inv(tcp) @ brush
        allowed = (*TABLE_CONTACTS, *(("brush", name) for name in TARGET_BLOCKS))
        results = []
        selected = None
        for index, candidate in enumerate(candidates):
            target = np.asarray(candidate["target_tcp_world"], dtype=float)
            ik = planner.solve_ik(robot_state["joint_names"],
                                  robot_state["joint_positions_rad"], target)
            row = {"index": index, **candidate,
                   "curobo_pure_ik": ik, "mujoco_diagnostic_path": None}
            if ik.get("failure_category") == "start_collision":
                # Knowledge 的纯 pose 可达性独立于此刻起点。Pika 的保守
                # 球近似偶尔判实际运行姿态为自碰撞；保留原始诊断，用已知
                # 初始状态仅复查目标 pose。此结果不可当作正式路径成功。
                initial = self.report["initial_state"]["robots"][self.group]
                ik = planner.solve_ik(initial["joint_names"],
                                      initial["joint_positions_rad"], target)
                row["curobo_initial_state_ik"] = ik
            if ik.get("success") or index == 0:
                path = self.planner.plan_to_pose(
                    target, path_kind="linear_tcp", duration_s=2.,
                    allowed_contacts=allowed, attached_object="brush",
                    T_tcp_object=T_tcp_brush)
                row["mujoco_diagnostic_path"] = path
                if path.get("success"):
                    if ik.get("success") or index == 0:
                        row["selected_basis"] = (
                            "curobo_pure_ik_and_mujoco_diagnostic_path"
                            if ik.get("success") else
                            "cpu_diagnostic_only_curobo_rejected_original")
                        selected = target
                        results.append(row)
                        break
            results.append(row)
        self.report["knowledge"][stage_label]["contact_candidate_screen"] = {
            "scene_scope": "cuRobo 纯 IK + MuJoCo CPU 诊断路径；不计正式 cuRobo 任务评测",
            "table_top_z_m": table_z,
            "selected_index": None if selected is None else len(results) - 1,
            "selected_basis": None if selected is None else results[-1]["selected_basis"],
            "candidates": results,
        }
        _write(self.output_dir / "knowledge" / f"{stage_label}_contact_candidates.json",
               self.report["knowledge"][stage_label]["contact_candidate_screen"])
        if selected is None:
            raise ValueError(f"{stage_label} 没有同时通过纯 cuRobo IK 与诊断路径的接触候选")
        return selected

    def _pika_contact_classifier(self, contacts, descriptor, *, object_body: str) -> dict:
        if object_body != "brush" or descriptor["mode"] != "pika_urdf":
            raise ValueError("此任务接触分类仅对应 Pika 抓刷柄")
        frozen = self.runtime.copy_data()
        sites = {side: frozen.site(f"pika_{side}_inner").xpos.copy()
                 for side in ("left", "right")}
        handle = load_concept(self.concept_dir / "brush.json", part_name="grip_handle").concept
        handle_world = (_pose(frozen.body("brush")) @ handle.pose_in_object)[:3, 3]
        return classify_pika_brush_grasp(contacts, sites, handle_world)

    def _pick(self) -> dict:
        initial = self.observe()
        brush = np.asarray(initial["objects"]["brush"]["T_world_object"])
        profile = load_brush_pick_profile('simulation', overrides={**self.scene.get('pick_overrides', {}),
            'grasp_tilt_deg': self.grasp_tilt_deg})
        self.report['pick_profile'] = profile.to_dict()
        if self.scene.get("grasp_mode") == "world_minus_z":
            handle = load_concept(self.concept_dir / "brush.json", part_name="grip_handle").concept
            grasp = np.eye(4)
            grasp[:3, :3] = (Rotation.from_euler("z", float(self.scene.get("grasp_yaw_deg", 0.)),
                                                  degrees=True).as_matrix()
                            @ np.column_stack(([0., 0., -1.], [0., 1., 0.], [1., 0., 0.])))
            grasp[:3, 3] = (brush @ handle.pose_in_object)[:3, 3] + [0., 0., .002]
        else:
            grasp = brush_pick_targets(brush, self.concept_dir / 'brush.json', profile,
                grasp_center_tcp_m=pika_grasp_center_tcp(self.runtime.robot_metadata,self.group))['approach']
        pregrasp = grasp.copy()
        approach = .064 if self.scene.get("grasp_mode") == "world_minus_z" else profile.values["approach_distance_m"]
        pregrasp[2, 3] += approach
        self.report['pick_geometry'] = {'grasp_mode': self.scene.get('grasp_mode', 'object_relative'),
            'approach_world_z_m': approach,
            'grasp_target': grasp.tolist(), 'pregrasp_target': pregrasp.tolist(),
            'lift_m': .08 if self.scene.get('grasp_mode') == 'world_minus_z' else profile.values['lift_distance_m'],
            'scene_grasp_yaw_deg': self.scene.get('grasp_yaw_deg')}
        if self.scene.get('grasp_mode') != 'world_minus_z':
            self.report['pick_geometry'].update(
                grasp_center_tcp_m=pika_grasp_center_tcp(self.runtime.robot_metadata,self.group).tolist(),
                position_rule='align_robot_grasp_region_with_handle_concept',
                tcp_above_handle_m_applied=False)

        def motion(stage, target, state_id):
            if self.observe()["state_id"] != state_id:
                raise ValueError("Pick 上一步执行状态已变化")
            contacts = ((*GRIP_CONTACTS, ("brush", "table")) if stage == "lift"
                        else GRIP_CONTACTS if stage != "pregrasp" else ())
            if stage == "lift" and self.scene.get("grasp_mode") == "world_minus_z":
                contacts = (*contacts, ("brush", "brush_stand"))
            return self._move(stage, target, "joint" if stage == "pregrasp" else "linear_tcp",
                              3.0 if stage == "pregrasp" else 2.5,
                              allowed_contacts=contacts)

        def evidence(stage, observation, target):
            _write(self.output_dir / "pick" / "observations" / f"{stage}.json", {
                "observation": observation,
                "target_tcp_world": None if target is None else np.asarray(target).tolist()})

        return run_pick_skill(
            self.runtime, self.output_dir / "pick", candidate={
                "candidate_id": "brush_grip_handle_top", "T_world_tcp": grasp},
            motion_stage=motion, object_id="brush", group=self.group,
            diagnostics=lambda: query_diagnostics(self.runtime, body_names=("brush",)),
            evidence_writer=evidence,
            settings=GraspSettings(approach_distance_m=approach,
                                   lift_distance_m=.08 if self.scene.get("grasp_mode") == "world_minus_z" else profile.values["lift_distance_m"],
                                   open_width_m=profile.values["open_width_m"], close_width_m=0.,
                                   hold_duration_s=profile.values["hold_duration_s"]),
            selected_pregrasp_pose=pregrasp,
            lift_target_builder=(None if self.scene.get('grasp_mode') == 'world_minus_z'
                                 else lambda actual: brush_lift_target(actual, pregrasp, profile)),
            contact_classifier=self._pika_contact_classifier,
            after_contact_verified=lambda contact: self.runtime.attach_simulation_object(
                self.group, "brush", contact_evidence=contact))

    def _sweep_target(self, target_name: str, stage_label: str,
                      *, recovery_inside_pan: bool = False, cube_goal_world_xyz=None, stop_after=None) -> bool:
        poses = {name: np.asarray(row["T_world_object"])
                 for name, row in self.observe()["objects"].items()}
        try:
            layout = make_sweep_layout(poses, self.concept_dir,
                                       target_name=target_name,
                                       recovery_inside_pan=recovery_inside_pan,
                                       cube_goal_world_xyz=cube_goal_world_xyz,
                                       goal_inset_extra_m=float(getattr(self, "scene", {}).get("goal_inset_extra_m", 0.)))
        except ValueError as exc:
            self.report["knowledge"][stage_label] = {
                "target_name": target_name, "error": str(exc),
                "object_poses_world": {name: pose.tolist() for name, pose in poses.items()}}
            failure = f"{stage_label}__knowledge"
            self.report["sweeps"][stage_label] = {
                "target_name": target_name, "success": False,
                "failure_stage": failure}
            self.report["failure_stage"] = failure
            return False
        self.report["knowledge"][stage_label] = {
            "target_name": target_name,
            "direction_world_xy": layout.direction_world_xy.tolist(),
            "brush_start_world_xyz": layout.brush_start_world_xyz.tolist(),
            "brush_end_world_xyz": layout.brush_end_world_xyz.tolist(),
            "cube_goal_world_xyz": layout.cube_goal_world_xyz.tolist()}
        prepared = None
        retreat_lift_m = .05
        retreat_delta_world_m = None
        if not self.diagnostic_cpu_ik:
            from gene_agent.tasks.policies.sweep_cube_task.execution.selection import select_sweep_sequence
            selected, screen = select_sweep_sequence(self, target_name, stage_label,
                                                    recovery_inside_pan=recovery_inside_pan,
                                                    cube_goal_world_xyz=cube_goal_world_xyz)
            self.report["knowledge"][stage_label]["sequence_screen"] = screen
            if selected is None:
                failure = f"{stage_label}__sequence_screen"
                self.report["failure_stage"] = failure
                self.report["sweeps"][stage_label] = {"target_name": target_name, "success": False,
                                                     "failure_stage": failure}
                return False
            prepared = selected["targets"]
            retreat_lift_m = selected["retreat_lift_m"]
            retreat_delta_world_m = selected["retreat_delta_world_m"]
        try:
            sweep = run_sweep_skill(
                layout, contacts=SWEEP_CONTACTS_PROFILE, initial_brush_z=self.initial_brush_z,
                move_stage=self._move, target_for_brush=self._target_for_brush,
                contact_target_for_brush=lambda xyz: self._contact_target_for_brush(
                    xyz, stage_label=stage_label, direction_xy=layout.direction_world_xy,
                    target_name=target_name, recovery_inside_pan=recovery_inside_pan,
                    cube_goal_world_xyz=cube_goal_world_xyz),
                sweep_end_for_current_pose=lambda: make_sweep_layout(
                    {n: np.asarray(r["T_world_object"])
                     for n, r in self.observe()["objects"].items()},
                    self.concept_dir, target_name=target_name,
                    recovery_inside_pan=recovery_inside_pan,
                    cube_goal_world_xyz=cube_goal_world_xyz,
                    goal_inset_extra_m=float(self.scene.get("goal_inset_extra_m", 0.))).brush_end_world_xyz,
                current_brush_xyz=lambda: np.asarray(
                    self.observe()["objects"]["brush"]["T_world_object"])[:3, 3],
                stage_prefix=f"{stage_label}__", sweep_targets=TARGET_BLOCKS,
                prepared_targets=prepared, retreat_lift_m=retreat_lift_m,
                retreat_delta_world_m=retreat_delta_world_m,
                contact_target_name=target_name, stop_after=stop_after)
        except ValueError as exc:
            failure = f"{stage_label}__contact_candidates"
            self.report["sweeps"][stage_label] = {
                "target_name": target_name, "success": False,
                "failure_stage": failure, "error": str(exc)}
            self.report["failure_stage"] = failure
            return False
        self.report["sweeps"][stage_label] = {
            "target_name": target_name, "success": sweep["success"],
            "failure_stage": sweep["failure_stage"],
            "requested_scope": sweep["requested_scope"], "failure": sweep.get("failure")}
        if not sweep["success"]:
            self.report["failure_stage"] = sweep["failure_stage"]
            return False
        self._render("after_sweep_" + stage_label)
        self.report["sweeps"][stage_label]["interim_goal"] = evaluate_cube_in_dustpan(
            self.model, self.data, self.concept_dir, object_name=target_name)
        return True

    def _all_goals(self) -> dict:
        goals = {name: evaluate_cube_in_dustpan(
            self.model, self.data, self.concept_dir, object_name=name)
            for name in TARGET_BLOCKS}
        return {"success": all(row["success"] for row in goals.values()),
                "targets": goals}





    def pick(self) -> bool:
        if not self.placement["valid_initial_state"]:
            self.fail("invalid_initial_state")
            return False
        pick = self._pick()
        self.report["pick"] = {key: value for key, value in pick.items() if key != "stages"}
        if not pick["success"]:
            self.fail(f"pick/{pick['failure_stage']}")
            return False
        if not self.verify_pick_clearance():
            return False
        self._render("after_pick")
        return True

    def begin_sweeping(self, rotation, order):
        self.sweep_reference_brush_rotation = rotation.copy()
        self.report.update(sweep_reference_brush_rotation=rotation.tolist(),
                           target_order=order, knowledge={}, sweeps={})

    def sweep(self, target_name, stage_label, **kwargs):
        return self._sweep_target(target_name, stage_label, **kwargs)

    def goals(self):
        return self._all_goals()

    def failure(self):
        stage = self.report["failure_stage"] or ""
        label = stage.removesuffix("__sequence_screen")
        return {"stage": stage, "screen": self.report.get("knowledge", {}).get(
            label, {}).get("sequence_screen", {})}

    def clear_failure(self):
        # 仅由 policy 在核对纯预检证据后调用；原阶段记录仍保留。
        self.report["failure_stage"] = None

    def fail(self, stage):
        self.report["failure_stage"] = stage

    def record_progress(self, **values):
        self.report.update(values)

    def record_alignment(self, label, local, success):
        self.report["sweeps"][label]["alignment_postcondition"] = {
            "actual_pan_local_xyz": local.tolist(), "lateral_limit_m": .01,
            "success": success}
        if not success:
            self.fail(f"{label}__postcondition")

    def finish_sweeping(self):
        self._render("after_sweep")
        return self.release_and_settle()

    def run(self, *, policy=None, selection='task', target_name='cube', stop_after=None) -> dict:
        if selection not in {'task','pick','sweep'} or target_name not in TARGET_BLOCKS:
            raise ValueError('未知清扫调试范围或目标')
        if stop_after not in {None,'move_behind_cube','lower_to_contact','sweep','retreat'}:
            raise ValueError('未知清扫停止阶段')
        if stop_after is not None and selection != 'sweep':
            raise ValueError('阶段停止只适用于单次 sweep')
        if selection != 'task' and policy is not None:
            raise ValueError('独立 Skill 不能同时覆盖完整 Task policy')
        scope = 'full' if selection == 'task' else selection + ('/'+stop_after if stop_after else '')
        print(f"[{time.strftime('%H:%M:%S')}] 场景初始化完成，开始抓刷与清扫", flush=True)
        try:
            if selection == 'task':
                if policy is None:
                    from .policy import run as policy
                policy(self, alignment_lateral_trigger_m=float(
                    self.scene.get("alignment_lateral_trigger_m", .015)),
                    target_order_policy=self.scene.get("target_order_policy", "closest_first"))
                self.report['success'] = bool(self.report.get('goal',{}).get('success')
                                               and self.report['failure_stage'] is None)
            else:
                ready = self.pick()
                if ready and selection == 'sweep':
                    actual = self.observe()['objects']['brush']['T_world_object']
                    self.begin_sweeping(np.asarray(actual)[:3,:3], [target_name])
                    ready = self.sweep(target_name, 'debug_'+target_name, stop_after=stop_after)
                self.report['success'] = bool(ready and self.report['failure_stage'] is None)
            self.report["status"] = "completed" if self.report["success"] else "failed"
        except Exception as exc:
            self.report["status"] = "failed"
            self.report["failure_stage"] = self.report["failure_stage"] or "exception"
            self.report["error"] = f"{type(exc).__name__}: {exc}"
            self.report["traceback"] = traceback.format_exc()
        finally:
            self.report["final_state"] = self.observe()
            annotate_task_result(self.report, scope=scope, full_scope="full",
                                 actual_final=self.report["final_state"], report_ref="result.json")
            self._render("final")
            _write(self.output_dir / "result.json", self.report)
        return self.report

    def _place_initial_objects(self, *, randomize, seed):
        return place_initial_objects(self, randomize=randomize, seed=seed)

    def _record_settled_placement(self):
        return record_settled_placement(self)

    def verify_pick_clearance(self):
        if self.scene.get("grasp_mode") != "world_minus_z":
            return True
        # 抓取必须实际沿世界 -Z 下行；拔出后刷头须彻底离开支架。
        samples = self.report["stages"]["approach"]["execution"]["replay_samples"]
        target_xy = np.asarray(self.report["stages"]["approach"]["target_tcp_world"])[:2, 3]
        preview = mujoco.MjData(self.model)
        angles, deviations = [], []
        for sample in samples:
            preview.qpos[:] = sample["qpos"]
            mujoco.mj_forward(self.model, preview)
            site = preview.site("pika_tcp")
            axis = site.xmat.reshape(3, 3)[:, 0]
            angles.append(float(np.rad2deg(np.arccos(np.clip(-axis[2], -1., 1.)))))
            deviations.append(float(np.linalg.norm(site.xpos[:2] - target_xy)))
        stand_contacts = []
        for contact in self.data.contact:
            names = [self.model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
            if (any(n.startswith("brush_collision") for n in names)
                    and any(n.startswith("brush_long_stand_") for n in names)
                    and contact.dist <= .001):
                stand_contacts.append({"geom_names": names, "distance_m": float(contact.dist)})
        verified = bool(angles and max(angles) <= 3. and max(deviations) <= .005
                        and not stand_contacts)
        self.report["pick"]["top_down_and_stand_postcheck"] = {
            "success": verified, "sample_count": len(angles),
            "max_downward_axis_error_deg": max(angles, default=None),
            "max_xy_deviation_m": max(deviations, default=None),
            "stand_contacts_after_lift": stand_contacts,
            "angle_limit_deg": 3., "xy_limit_m": .005}
        if not verified:
            self.report["failure_stage"] = "pick/top_down_and_stand_postcheck"
            return False

        return True

    def release_and_settle(self):
        self.runtime.release_simulation_object(self.group, "brush")
        before_release = self.observe()
        release = self.runtime.execute_gripper_command(
            group=self.group, width_m=.04, duration_s=.7,
            planned_state_id=before_release["state_id"], replay_object_names=OBJECTS)
        self.report["stages"]["release"] = release
        _write(self.output_dir / "stages/release.json", release)
        if not release["success"]:
            self.report["failure_stage"] = "release"
            return False
        before_hold = self.observe()
        hold = self.runtime.execute_hold(
            duration_s=.5, planned_state_id=before_hold["state_id"],
            replay_object_names=OBJECTS, replay_group=self.group)
        self.report["stages"]["settle"] = hold
        _write(self.output_dir / "stages/settle.json", hold)
        if not hold["success"]:
            self.report["failure_stage"] = "settle"
            return False
        return True
