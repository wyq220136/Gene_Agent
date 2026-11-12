"""杯/勺共用的 MuJoCo Pick 接线；当前用于左臂 Pika。

子类提供抓取几何、碰撞代理与接触判据；阶段执行复用现有 Skill，
不把勺子特例作为所有任务的默认行为。
"""

from __future__ import annotations
from benchmarks.catalog import identity
from gene_agent.tasks.policies import identity as policy_identity

import hashlib
import json
import os
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from gene_agent.knowledge.geometry import load_concept
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.motion.mujoco_ik import MujocoPosePlanner
from benchmarks.tabletop import TEMPLATE
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.simulation.execution import scale_position_actuator_kp
from gene_agent.simulation.runtime import MujocoRuntime
from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
from gene_agent.skills.library.pick.model import GraspSettings


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _body_pose(body) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose


def _make_run_scene(source: Path, destination: Path, object_name: str,
                    *, simulation_weld: bool) -> None:
    """生成本次场景；辅助模式才加入非活动 weld，源资产不改。"""
    tree = ET.parse(source)
    root = tree.getroot()
    for element in root.iter():
        if element.get("file"):
            element.set("file", str((source.parent / element.get("file")).resolve()))
    if simulation_weld:
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(equality, "weld", name=f"sim_attach_left_{object_name}",
                      body1="pika_gripper_base_link", body2=object_name,
                      active="false")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


class TabletopPickRun:
    """一次性同步 Pick；正式模式只使用 cuRobo 生成机器人运动轨迹。"""

    def __init__(self, scene_dir: str | Path, output_dir: str | Path,
                 *, simulation_weld: bool = True,
                 grasp_handle_x_offset_m: float = 0.):
        self.scene_dir = Path(scene_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.scene = json.loads((self.scene_dir / "scene.json").read_text(encoding="utf-8"))
        self.object_name = self.scene["roles"][self.OBJECT_ROLE]
        if self.object_name not in self.SUPPORTED_OBJECTS:
            raise ValueError(f"当前抓取 Skill 不接受场景物体 {self.object_name}")
        self.group = "left"
        self.simulation_weld = bool(simulation_weld)
        if not np.isfinite(grasp_handle_x_offset_m) or abs(grasp_handle_x_offset_m) > .015:
            raise ValueError("勺柄抓取点局部 X 微调必须在 ±15 mm 内")
        self.grasp_handle_x_offset_m = float(grasp_handle_x_offset_m)
        self._verified_contact = None
        # scene.json 沿用模板的 robot.metadata_path，基准目录是模板场景。
        self.robot = json.loads((TEMPLATE.parent / self.scene["robot"]["metadata_path"])
                                .read_text(encoding="utf-8"))
        self.robot_metadata_path = (TEMPLATE.parent /
                                    self.scene["robot"]["metadata_path"]).resolve()
        self.concept_path = Path(self.scene["concept_paths"][self.object_name])
        self.grip = load_concept(self.concept_path, part_name="grasp_handle").concept
        self.source_xml = Path(self.scene["scene_xml_path"]).resolve()
        self.effective_xml = self.output_dir / "scene.xml"
        _make_run_scene(self.source_xml, self.effective_xml, self.object_name,
                        simulation_weld=self.simulation_weld)
        self.model = mujoco.MjModel.from_xml_path(str(self.effective_xml))
        if self.simulation_weld:
            weld = self.model.equality(f"sim_attach_left_{self.object_name}").id
            self.model.eq_solimp[weld, :2] = .9999
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.robot_kp_scale = scale_position_actuator_kp(
            self.model, self.robot["joint_groups"][self.group], 5.)
        # Pika 原始滑块控制在 1 s 内只有单侧碰到细勺柄；增加仿真控制增益，
        # 保留原碰撞 mesh 和实际接触检查。这个数值只属于本次仿真 run。
        self.gripper_actuators = {}
        for name in self.robot["grippers"][self.group]["actuator_names"]:
            actuator = self.model.actuator(name).id
            self.gripper_actuators[name] = {
                "original_kp": float(self.model.actuator_gainprm[actuator, 0]),
                "original_force_range_n": self.model.actuator_forcerange[actuator].tolist(),
                "effective_kp": 500., "effective_force_range_n": [-20., 20.],
            }
            self.model.actuator_gainprm[actuator, 0] = 500.
            self.model.actuator_biasprm[actuator, 1] = -500.
            self.model.actuator_forcerange[actuator] = [-20., 20.]
        self.runtime = MujocoRuntime(self.model, self.data, {
            "robot": self.robot, "world_frame": self.scene["world_frame"],
            "simulation_attachment_pairs": (
                {self.group: self.object_name} if self.simulation_weld else {}),
        })
        # CPU 模块只逐关节复核碰撞；不调用它的 IK 或运动规划入口。
        self.collision_checker = MujocoPosePlanner(self.model, self.data,
                                                   self.robot, self.group)
        self.object_spheres = self._collision_spheres()
        self.report = {
            "schema_version": 1, "skill": self.SKILL_NAME, "task": self.scene["task"],
            "status": "running", "success": False,
            "planner_backend": "curobo", "execution_backend": "mujoco_physical",
            "scene_xml_path": str(self.effective_xml),
            "scene_xml_sha256": hashlib.sha256(self.effective_xml.read_bytes()).hexdigest(),
            "source_scene_xml_path": str(self.source_xml),
            "source_scene_xml_sha256": hashlib.sha256(self.source_xml.read_bytes()).hexdigest(),
            "concept_path": str(self.concept_path),
            "concept_sha256": hashlib.sha256(self.concept_path.read_bytes()).hexdigest(),
            "robot_metadata_sha256": hashlib.sha256(
                self.robot_metadata_path.read_bytes()).hexdigest(),
            "asset_manifest_sha256": hashlib.sha256(
                Path(self.scene["asset_manifest_path"]).read_bytes()).hexdigest(),
            "code_sources_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[3] / name)
                                     .read_bytes()).hexdigest()
                for name in (
                    "gene_agent/tasks/tabletop/pick.py",
                    "gene_agent/skills/library/pick/planning.py",
                    "gene_agent/skills/library/pick/skill.py",
                    "gene_agent/skills/library/pick/model.py",
                    "gene_agent/skills/contracts.py",
                    "gene_agent/skills/runtime/operation.py",
                    "gene_agent/skills/runtime/runner.py",
                    "gene_agent/skills/runtime/planning.py",
                    "gene_agent/skills/runtime/poses.py",
                    "gene_agent/simulation/pick_backend.py",
                    "gene_agent/motion/curobo/runtime_bridge.py",
                    "gene_agent/simulation/runtime.py",
                ) + self.EXTRA_CODE_SOURCES
            },
            "robot_controller_kp_scale": 5.,
            "gripper_actuators": self.gripper_actuators,
            "simulation_assistance": (
                "weld_after_verified_bilateral_inner_finger_contact"
                if self.simulation_weld else "none"),
            "grasp_handle_x_offset_m": self.grasp_handle_x_offset_m,
            "stages": {},
        }
        self.report["benchmark"] = identity(self.BENCHMARK_ID)
        self.report["policy_id"] = f"{self.BENCHMARK_ID}/pick"
        self.report["policy"] = policy_identity(self.BENCHMARK_ID, variant="pick")
        _write(self.output_dir / "result.json", self.report)


    def observe(self) -> dict:
        return self.runtime.query(object_names=(self.object_name,),
                                  robot_groups=(self.group,))


    def _attachment(self, stage: str, observation: dict) -> dict | None:
        if stage != "lift":
            return None
        if self.simulation_weld:
            active = self.runtime.simulation_attachments()
            if len(active) != 1 or active[0]["object_instance_id"] != self.object_name:
                raise ValueError("抬升规划缺少双指接触后激活的仿真夹持证据")
            physical_hold_evidence = active[0]
        else:
            if not self._verified_contact or not self._verified_contact.get(
                    "bilateral_pad_contact"):
                raise ValueError("无辅助抬升缺少实际双指接触证据")
            physical_hold_evidence = self._verified_contact
        pose = np.asarray(self.observe()["objects"][self.object_name]["T_world_object"])
        local = self.object_spheres
        centers = (pose[:3, :3] @ local[:, :3].T).T + pose[:3, 3]
        return {
            "object_instance_id": self.object_name,
            "spheres_world_m": np.column_stack((centers, local[:, 3])).tolist(),
            "coverage": "compiled_spoon_handle_and_scoop_box_aabb_cells_22mm",
            "physical_hold_evidence": physical_hold_evidence,
        }

    def _after_contact_verified(self, contact: dict) -> dict | None:
        if self.simulation_weld:
            return self.runtime.attach_simulation_object(
                self.group, self.object_name, contact_evidence=contact)
        # 只留给 cuRobo 的持物碰撞代理作规划输入；物体仍由 MuJoCo 接触物理自由运动。
        self._verified_contact = {
            "mode": "verified_bilateral_contact_no_weld",
            "object_instance_id": self.object_name,
            "bilateral_pad_contact": bool(contact["bilateral_pad_contact"]),
            "contact_count": int(contact["contact_count"]),
            "sim_time_s": float(self.data.time),
        }
        return None

    def _contact_policy(self, _stage: str, snapshot: dict) -> dict:
        table = [row["obstacle_name"] for row in snapshot["records"]
                 if row["geom_name"] == "scene_table"]
        return {"robot_links": ["base_link"] if table else [],
                "obstacle_names": table,
                "source": "spoon_scene_base_link_table_mounting_spheres",
                "semantics": "仅机械臂固定底座与桌面的安装交叠"}


    def _allowed_contacts(self, stage: str) -> list[tuple[str, str]]:
        allowed = ([(self.object_name, "pika_gripper_left_link"),
                    (self.object_name, "pika_gripper_right_link")]
                   if stage != "pregrasp" else [])
        if stage == "lift":
            allowed.append((self.object_name, "table"))
        return allowed

    def _postcheck(self, stage: str, plan: dict, _snapshot: dict) -> dict:
        positions = np.asarray(plan.get("positions_rad"), dtype=float)
        if positions.ndim != 2 or len(positions) < 2:
            return {"success": False, "reason": "missing_joint_trajectory"}
        samples = [positions[0]]
        for start, end in zip(positions[:-1], positions[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(end - start)) / .025)))
            samples.extend(np.linspace(start, end, count + 1)[1:])
        allowed = self._allowed_contacts(stage)
        attached = None
        if stage == "lift":
            observed = self.observe()
            attached = (np.linalg.inv(np.asarray(observed["robots"][self.group]["T_world_tcp"]))
                        @ np.asarray(observed["objects"][self.object_name]["T_world_object"]))
        forbidden = self.collision_checker._check_path(
            np.asarray(samples), allowed_contacts=allowed,
            attached_object=self.object_name if attached is not None else None,
            T_tcp_object=attached)
        return {"success": not forbidden, "sample_count": len(samples),
                "max_joint_step_rad": .025, "forbidden_contacts": forbidden,
                "source": "full_mujoco_geometry_before_curobo_trajectory_execution"}

    def _motion(self, stage: str, target: np.ndarray, planned_state_id: str) -> dict:
        if self.observe()["state_id"] != planned_state_id:
            raise ValueError("Pick 的计划起点必须等于上一阶段实际状态")
        bridge = CuroboRuntimeBridge(
            self.runtime, self.output_dir / "motion_curobo", group=self.group,
            linear_tcp_stages=("approach", "lift"),
            endpoint_tolerance_rad=.05, settle_timeout_s=1.,
            replay_object_names=(self.object_name,),
            max_attempts_by_stage={stage: 2},
            approximate_ik_by_stage=({"pregrasp": {
                "max_position_error_m": .012, "max_rotation_error_rad": .08}}
                if stage == "pregrasp" else {}),
            attachment_for_stage=self._attachment,
            disabled_body_names_for_stage=lambda name, _obs: (
                (self.object_name,) if name == "approach" else ()),
            contact_policy_for_stage=self._contact_policy,
            snapshot_filter_for_stage=self._snapshot_filter,
            plan_postcheck_for_stage=self._postcheck,
        )
        result = bridge(stage, target, planned_state_id)
        execution = result.get("execution")
        if execution is not None:
            allowed = self._allowed_contacts(stage)
            forbidden = self.collision_checker.check_observed_contacts(
                execution.get("replay_samples", ()), allowed_contacts=allowed)
            result["observed_forbidden_contacts"] = forbidden
            result["success"] = bool(result["success"] and not forbidden)
        return result

    def run(self, *, stop_after="pick") -> dict:
        body_pose = _body_pose(self.data.body(self.object_name))
        grasp = self._grasp_target(body_pose)
        pregrasp = grasp.copy()
        pregrasp[2, 3] += .07
        self.report["grasp_target_tcp_world"] = grasp.tolist()
        self.report["pregrasp_target_tcp_world"] = pregrasp.tolist()
        self.report["initial_observation"] = self.observe()
        _write(self.output_dir / "result.json", self.report)

        def evidence(stage, observation, target):
            _write(self.output_dir / "pick" / "observations" / f"{stage}.json", {
                "observation": observation,
                "target_tcp_world": None if target is None else np.asarray(target).tolist()})

        try:
            pick = run_pick_skill(
                self.runtime, self.output_dir / "pick", stop_after=stop_after,
                candidate={"candidate_id": f"{self.object_name}_{self.GRASP_CANDIDATE_SUFFIX}",
                           "T_world_tcp": grasp},
                motion_stage=self._motion, object_id=self.object_name,
                group=self.group,
                diagnostics=lambda: query_diagnostics(self.runtime,
                                                       body_names=(self.object_name,)),
                evidence_writer=evidence,
                settings=GraspSettings(approach_distance_m=.07,
                                       lift_distance_m=.09, open_width_m=.08,
                                       close_width_m=0., hold_duration_s=.5),
                close_duration_s=1., selected_pregrasp_pose=pregrasp,
                contact_classifier=self._contact_classifier,
                after_contact_verified=self._after_contact_verified,
            )
            self.report["requested_scope"] = stop_after
            self.report["operation_success"] = pick["operation_success"]
            self.report["actual_final"] = pick["actual_final"]
            self.report["failure"] = pick.get("failure")
            self.report["pick_result_path"] = str(self.output_dir / "pick" / "result.json")
            self.report["status"] = pick["status"]
            self.report["success"] = bool(pick["success"])
            self.report["failure_stage"] = pick.get("failure_stage")
            self.report["contact"] = pick.get("contact")
            self.report["lift_height_m"] = pick.get("lift_height_m")
            self.report["hold_height_m"] = pick.get("hold_height_m")
            self.report["simulation_assisted_pick_success"] = pick.get(
                "simulation_assisted_pick_success", False)
        except Exception as error:
            self.report["status"] = "failed"
            self.report["failure_stage"] = "exception"
            self.report["error"] = f"{type(error).__name__}: {error}"
            self.report["traceback"] = traceback.format_exc()
        self.report["final_observation"] = self.observe()
        _write(self.output_dir / "result.json", self.report)
        return self.report

