"""把已在 Viser 验证的门阶段接到同步 MuJoCo 执行。"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import traceback

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession, _json_value
from gene_agent.tasks.policies.microwave_task.simulation_door.probe import probe_planned_door_contacts
from gene_agent.tasks.policies.microwave_task.simulation_door.execution import _DoorContactTrace
from gene_agent.feedback.state import model_fingerprint, restore_state, save_state
from gene_agent.simulation.checks import evaluate_checks
from gene_agent.simulation.execution import scale_position_actuator_kp
from gene_agent.simulation.runtime import MujocoRuntime
from gene_agent.skills.library.door.sequence import door_skill_angle_range, door_skill_sequence
from gene_agent.skills.library import door as door_skill
from gene_agent.skills.contracts import OperationState, OperationStage, CheckResult
from gene_agent.skills.runtime.operation import CallbackOperationBackend


_PHYSICAL_DOOR_SELECTIONS = frozenset({
    "handle_pull", "push_open", "push_close", "microwave_door_open",
    "microwave_door_close", "microwave_door_cycle",
})


def _door_contact_policy_failure(
    phase, stage_contact_pairs: dict, *,
    stage_start_time_s: float | None = None,
    door_motion_rad: float | None = None,
) -> str | None:
    """依据原阶段接触语义判定，不能把门角变化当作无碰撞成功。"""
    if not stage_contact_pairs:
        return None
    if phase.skill_id == "task_transition":
        # 旧 Task1 从仍贴着门的最后推门 pose 直接直线撤回默认 TCP。
        # 出发瞬间的轻微指垫残留接触是退离动作的一部分；只接受
        # 前 0.15 s、总法向冲量不超过 0.2 Ns、门角变化不超过 1°，
        # 且接触部位全部为指垫。中途重新碰门或非指垫碰门仍失败。
        if (phase.name == "return_default_tcp"
                and stage_start_time_s is not None
                and door_motion_rad is not None
                and abs(door_motion_rad) <= np.deg2rad(1.)
                and all(value["classification"] == "pad_target"
                        and float(value["last_contact_time_s"])
                        - stage_start_time_s <= .15
                        for value in stage_contact_pairs.values())
                and sum(float(value["normal_impulse_ns"])
                        for value in stage_contact_pairs.values()) <= .2):
            return None
        return "forbidden_task_transition_contact"
    if phase.skill_id == "handle_pull" and (
            phase.contact_policy == "handle_contact"
            or phase.name in {"close", "release"}):
        # close 必须碰到把手；release 的第一帧也可能保留此前接触，
        # 而旧阶段表对这两段仍写着 avoid_door。这里只修正物理验收
        # 的解释，不改轨迹或 Knowledge；真实把手接触由终态检查确认。
        return None
    if phase.contact_policy == "avoid_door":
        return ("unexpected_precontact_door_contact" if phase.name == "pre_contact"
                else "forbidden_robot_door_contact")
    if any(value["classification"] == "forbidden"
           for value in stage_contact_pairs.values()):
        return "forbidden_robot_door_contact"
    return None


class ExperimentSession:
    """一个实验只拥有一份物理状态；规划器每阶段读取这份状态的副本。"""

    def __init__(self, selection: str, mode: str, config: Path, seed: int,
                 output_dir: Path | None = None, *, speed_scale: float = 1.0,
                 speed_scale_by_phase: dict[str, float] | None = None):
        if mode != "physics":
            raise ValueError(f"尚未接入实验会话模式：{mode}")
        if selection not in _PHYSICAL_DOOR_SELECTIONS:
            raise ValueError(f"物理执行尚未接入：{selection}")
        if int(seed) != 0:
            raise ValueError("当前门实验仅支持已验证的 seed=0 初始场景")
        if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("物理轨迹速度比例必须在 (0, 1] 内")
        self.selection = selection
        self.mode = mode
        self.seed = int(seed)
        self.speed_scale = float(speed_scale)
        self.output_dir = Path(output_dir or Path("runs") /
                               f"microwave-experiment-{datetime.now():%Y%m%d-%H%M%S-%f}")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "stages").mkdir(exist_ok=True)
        self.planning = DoorDebugSession(config_path=Path(config))
        self.planning.output_dir = self.output_dir / "planning"
        self.planning.start_skill(selection)
        try:
            self.arm_position_kp_scale = float(self.planning.config.get(
                "physics_arm_position_kp_scale", 1.0
            ))
        except (TypeError, ValueError) as error:
            raise ValueError("position actuator kp 比例必须为正有限数") from error
        model = self.planning.case.model
        robot = self.planning.case.metadata["robot"]
        arm_joint_names = robot["joint_groups"][self.planning.group]
        self.arm_position_kp_by_joint = scale_position_actuator_kp(
            model, arm_joint_names, self.arm_position_kp_scale
        )
        # 仅调仿真 actuator 的跟踪刚度；关节力矩上限、Skill 轨迹和
        # cuRobo 碰撞场景均保持原值。变更后的模型参与检查点指纹。
        mujoco.mj_forward(model, self.planning.case.data)
        (self.output_dir / "config_initial.json").write_text(
            json.dumps(self.planning.config, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8"
        )
        configured_scales = (speed_scale_by_phase if speed_scale_by_phase is not None
                             else self.planning.config.get("physics_speed_scale_by_phase", {}))
        known_phases = {phase.phase_id for phase in self.planning.phases}
        motion_phases = {phase.phase_id for phase in self.planning.phases
                         if phase.motion_kind != "gripper"}
        for phase_id, value in configured_scales.items():
            if phase_id not in known_phases or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"无效的物理阶段速度比例：{phase_id}={value}")
        self.speed_scale_by_phase = dict(configured_scales)
        configured_kp = self.planning.config.get(
            "physics_arm_position_kp_scale_by_phase", {}
        )
        if not isinstance(configured_kp, dict):
            raise ValueError("阶段 position actuator kp 必须是阶段到比例的映射")
        self.arm_position_kp_scale_by_phase = {}
        for phase_id, value in configured_kp.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"无效的阶段 kp 比例：{phase_id}={value}") from error
            if phase_id not in motion_phases or not np.isfinite(parsed) or parsed <= 0:
                raise ValueError(f"无效的阶段 kp 比例：{phase_id}={value}")
            self.arm_position_kp_scale_by_phase[phase_id] = parsed
        physical_data = mujoco.MjData(model)
        mujoco.mj_copyData(physical_data, model, self.planning.case.data)
        mujoco.mj_forward(model, physical_data)
        self.runtime = MujocoRuntime(model, physical_data, self.planning.case.metadata)
        self.rows: list[dict] = []
        self.prefix_stages: list[str] = []
        self.prefix_run_ref: str | None = None
        self.prefix_planning_success = True
        self.prefix_execution_success = True
        self.stopped = False
        self._door_joint = model.joint("microwave_door_joint")
        self._traces: dict[str, _DoorContactTrace] = {}
        self.checks: dict[str, dict] = {}
        self.initialization = {
            "kind": ("explicit_reset_scenario" if selection in {"push_open", "push_close"}
                     else "fresh_closed_door"),
            "door_angle_rad": float(physical_data.qpos[self._door_joint.qposadr[0]]),
        }

    def save_checkpoint(self, path: Path) -> Path:
        """在阶段边界保存真实物理状态和规划进度，供独立实验局部重试。"""
        if self.stopped:
            raise ValueError("失败后的状态不能作为下一阶段检查点")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=False)
        model = self.runtime.model
        current = self._observation()
        metadata = {
            "schema_version": 1,
            "selection": self.selection,
            "seed": self.seed,
            "model_fingerprint": model_fingerprint(model),
            "state_id": current["state_id"],
            "scene_revision": self.runtime.scene_revision,
            "phase_index": self.planning.phase_index,
            "stop_index": self.planning.stop_index,
            "next_phase_id": (self.planning.phases[self.planning.phase_index].phase_id
                              if self.planning.phase_index < self.planning.stop_index else None),
            "config_effective": self.planning.config,
            "knowledge_state": self.planning.knowledge_state,
            "checks": self.checks,
            "contact_traces": {name: trace.summary() for name, trace in self._traces.items()},
            "attachment_evidence": self.runtime._simulation_attachment_evidence,
            "speed_scale": self.speed_scale,
            "speed_scale_by_phase": self.speed_scale_by_phase,
            "physical_controller": {
                "arm_position_kp_scale": self.arm_position_kp_scale,
                "arm_position_kp_by_joint": self.arm_position_kp_by_joint,
                "arm_position_kp_scale_by_phase": self.arm_position_kp_scale_by_phase,
            },
            "source_run": str(self.output_dir),
            "prefix_stages": self.prefix_stages + [row["stage_id"] for row in self.rows],
            "prefix_planning_success": (self.prefix_planning_success and
                                        all(row["planning_success"] for row in self.rows)),
            "prefix_execution_success": (self.prefix_execution_success and
                                         all(row["execution_success"] for row in self.rows)),
        }
        save_state(model, self.runtime.data, destination / "state.npz")
        (destination / "checkpoint.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False,
                       default=_json_value) + "\n", encoding="utf-8"
        )
        return destination

    def restore_checkpoint(self, path: Path) -> None:
        """先验证模型和 Skill，再恢复状态；旧规划轨迹与 solver 缓存均丢弃。"""
        if self.rows:
            raise ValueError("只能在新实验开始前恢复检查点")
        source = Path(path)
        metadata = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("未知检查点格式")
        if metadata.get("selection") != self.selection or metadata.get("seed") != self.seed:
            raise ValueError("检查点 selection/seed 与当前实验不一致")
        # 夹碗辅助 weld 会在运行中更新 eq_data。验证时临时装入保存值，
        # 同时检查其余编译模型参数；无论成功失败都恢复当前模型，不提前
        # 改变在线状态。真正的积分状态仍由 restore_state() 原子核对。
        model = self.runtime.model
        previous_eq_data = model.eq_data.copy()
        try:
            with np.load(source / "state.npz", allow_pickle=False) as archive:
                saved_eq_data = archive["eq_data"]
                if saved_eq_data.shape != model.eq_data.shape:
                    raise ValueError("检查点模型指纹与当前物理模型不一致")
                model.eq_data[:] = saved_eq_data
                if (metadata.get("model_fingerprint") != model_fingerprint(model)
                        or str(archive["model_fingerprint"]) != metadata["model_fingerprint"]):
                    raise ValueError("检查点模型指纹与当前物理模型不一致")
        finally:
            model.eq_data[:] = previous_eq_data
        if metadata.get("scene_revision") != self.runtime.scene_revision:
            raise ValueError("检查点场景版本与当前实验不一致")
        for skill_id, saved in metadata["contact_traces"].items():
            trace = self._trace_for(skill_id)
            if sorted(trace.allowed_door_geom_names) != saved["allowed_door_geom_names"]:
                raise ValueError(f"检查点接触策略已变化：{skill_id}")
        # 先重建 Knowledge 和阶段列表，再接回真实状态。rebuild() 只重置
        # 规划副本；物理 runtime 始终由 state.npz 恢复，不写名义门角。
        self.planning.config = metadata["config_effective"]
        self.planning.knowledge_state = metadata["knowledge_state"]
        self.planning.rebuild()
        phase_index = int(metadata["phase_index"])
        stop_index = int(metadata["stop_index"])
        if not 0 <= phase_index <= stop_index <= len(self.planning.phases):
            raise ValueError("检查点阶段索引无效")
        next_id = (self.planning.phases[phase_index].phase_id
                   if phase_index < stop_index else None)
        if next_id != metadata["next_phase_id"]:
            raise ValueError("检查点阶段定义与当前代码不一致")
        restore_state(self.runtime.model, self.runtime.data, source / "state.npz")
        if self._observation()["state_id"] != metadata["state_id"]:
            raise ValueError("检查点实际状态校验失败")
        self.runtime._simulation_attachment_evidence = metadata["attachment_evidence"]
        self.planning.phase_index = phase_index
        self.planning.stop_index = stop_index
        self.planning.planner_pool.clear()
        self.planning.planned_q = None
        self.planning.plan_result = None
        self.planning.plan_revision = None
        self._sync_planning_from_actual()
        self.checks = metadata["checks"]
        self._traces.clear()
        for skill_id, saved in metadata["contact_traces"].items():
            trace = self._trace_for(skill_id)
            trace.samples = saved["door_angle_samples"]
            trace.contact_pairs = saved["contact_pairs"]
            trace.max_forbidden_depth_m = saved["max_forbidden_depth_m"]
            trace.pad_handle_normal_impulse_ns = saved["pad_handle_normal_impulse_ns"]
            trace.pad_target_normal_impulse_ns = saved["pad_target_normal_impulse_ns"]
        self.prefix_stages = metadata["prefix_stages"]
        self.prefix_run_ref = metadata["source_run"]
        self.prefix_planning_success = bool(metadata["prefix_planning_success"])
        self.prefix_execution_success = bool(metadata["prefix_execution_success"])
        self.speed_scale = float(metadata.get("speed_scale", self.speed_scale))
        self.speed_scale_by_phase = dict(metadata.get(
            "speed_scale_by_phase", self.speed_scale_by_phase
        ))
        self.arm_position_kp_scale_by_phase = dict(metadata.get(
            "physical_controller", {}
        ).get("arm_position_kp_scale_by_phase", {}))
        self.initialization = {
            "kind": "checkpoint_restore",
            "checkpoint_ref": str(source),
            "state_id": metadata["state_id"],
            "door_angle_rad": self._observation()["door_angle_rad"],
        }

    def apply_parameters(self, patch: dict) -> dict:
        """在阶段边界编辑尚未开始的推门 Knowledge 或物理执行速度。

        接触目标仍由原 Knowledge 生成；编辑后废弃旧轨迹并重建阶段。
        已执行 Skill 的目标不能从后续检查点回写，否则前缀事实与新目标
        不再一致。换物理模型或场景摆位须重新 reset，不能走此入口。
        """
        if self.stopped:
            raise ValueError("实验已停止；须从合法检查点或 fresh reset 重试")
        if not isinstance(patch, dict) or not patch or set(patch) - {
            "knowledge_state", "physics_speed_scale_by_phase",
            "physics_arm_position_kp_scale_by_phase",
        }:
            raise ValueError("参数补丁只支持 Knowledge、阶段速度和阶段 arm kp")
        if self.planning.phase_index >= self.planning.stop_index:
            raise ValueError("当前 Skill 已完成，没有可修改的后续阶段")
        completed = set(self.prefix_stages + [row["stage_id"] for row in self.rows])
        future_phases = self.planning.phases[
            self.planning.phase_index:self.planning.stop_index
        ]
        future_phase_ids = {phase.phase_id for phase in future_phases}
        future_motion_ids = {phase.phase_id for phase in future_phases
                             if phase.motion_kind != "gripper"}
        active_skills = {phase.skill_id for phase in future_phases}
        proposed_knowledge = copy.deepcopy(self.planning.knowledge_state)
        proposed_speed = dict(self.speed_scale_by_phase)
        proposed_kp = dict(self.arm_position_kp_scale_by_phase)
        changed_skills: list[str] = []
        changed_phases: list[str] = []
        edits = patch.get("knowledge_state", {})
        if not isinstance(edits, dict):
            raise ValueError("knowledge_state 必须按 Skill 给出参数对象")
        for skill_id, values in edits.items():
            if skill_id not in {"push_open", "push_close"} or skill_id not in active_skills:
                raise ValueError(f"当前实验不支持编辑这个 Skill 的 Knowledge：{skill_id}")
            if any(stage.startswith(f"{skill_id}/") for stage in completed):
                raise ValueError(f"Skill {skill_id} 已执行；须恢复到该 Skill 之前的检查点")
            if not isinstance(values, dict) or set(values) != {"door_push"} or \
                    not isinstance(values["door_push"], dict) or not values["door_push"]:
                raise ValueError("当前只支持门推压 Skill 的 door_push 非空参数补丁")
            current = copy.deepcopy(proposed_knowledge.get(skill_id, {}))
            current.setdefault("door_push", {}).update(values["door_push"])
            proposed_knowledge[skill_id] = current
            changed_skills.append(skill_id)
        speed_edits = patch.get("physics_speed_scale_by_phase", {})
        if not isinstance(speed_edits, dict):
            raise ValueError("physics_speed_scale_by_phase 必须为阶段到速度的映射")
        for phase_id, value in speed_edits.items():
            if phase_id in completed:
                raise ValueError(f"阶段 {phase_id} 已执行；不能改写前序速度")
            try:
                parsed_speed = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"无效的物理阶段速度比例：{phase_id}={value}") from error
            if phase_id not in future_phase_ids or not np.isfinite(parsed_speed) or \
                    not 0 < parsed_speed <= 1:
                raise ValueError(f"无效的物理阶段速度比例：{phase_id}={value}")
            proposed_speed[phase_id] = parsed_speed
            changed_phases.append(phase_id)
        kp_edits = patch.get("physics_arm_position_kp_scale_by_phase", {})
        if not isinstance(kp_edits, dict):
            raise ValueError("阶段 arm kp 必须为阶段到比例的映射")
        for phase_id, value in kp_edits.items():
            if phase_id in completed:
                raise ValueError(f"阶段 {phase_id} 已执行；不能改写前序 kp")
            try:
                parsed_kp = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"无效的阶段 kp 比例：{phase_id}={value}") from error
            if phase_id not in future_motion_ids or not np.isfinite(parsed_kp) or parsed_kp <= 0:
                raise ValueError(f"无效的阶段 kp 比例：{phase_id}={value}")
            proposed_kp[phase_id] = parsed_kp
            if phase_id not in changed_phases:
                changed_phases.append(phase_id)
        if not changed_skills and not changed_phases:
            raise ValueError("参数补丁没有任何有效改动")

        old_knowledge = self.planning.knowledge_state
        phase_index, stop_index = self.planning.phase_index, self.planning.stop_index
        old_phase_ids = tuple(phase.phase_id for phase in self.planning.phases)
        if changed_skills:
            self.planning.knowledge_state = proposed_knowledge
            try:
                self.planning.rebuild()
                if tuple(phase.phase_id for phase in self.planning.phases) != old_phase_ids:
                    raise ValueError("参数修改改变了 Skill 阶段结构；请从 fresh reset 开始")
            except Exception:
                self.planning.knowledge_state = old_knowledge
                self.planning.rebuild()
                self.planning.phase_index = phase_index
                self.planning.stop_index = stop_index
                self._sync_planning_from_actual()
                raise
            self.planning.phase_index = phase_index
            self.planning.stop_index = stop_index
            self.planning.planner_pool.clear()
            self._sync_planning_from_actual()
        self.speed_scale_by_phase = proposed_speed
        self.arm_position_kp_scale_by_phase = proposed_kp
        result = {
            "changed_skills": changed_skills,
            "changed_phases": changed_phases,
            "patch": patch,
            "knowledge_state_effective": self.planning.knowledge_state,
            "speed_scale_by_phase_effective": self.speed_scale_by_phase,
            "arm_position_kp_scale_by_phase_effective": self.arm_position_kp_scale_by_phase,
            "next_stage_id": (self.planning.phases[self.planning.phase_index].phase_id
                              if self.planning.phase_index < self.planning.stop_index else None),
            "actual_state_id": self._observation()["state_id"],
        }
        (self.output_dir / "parameters_applied.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                       default=_json_value) + "\n", encoding="utf-8"
        )
        return result

    def _observation(self) -> dict:
        observation = self.runtime.query(
            object_names=("microwave_door",), robot_groups=(self.planning.group,)
        )
        observation["door_angle_rad"] = float(
            self.runtime.data.qpos[self._door_joint.qposadr[0]]
        )
        return observation

    def _sync_planning_from_actual(self) -> None:
        """同步全部积分状态，避免视觉会话的名义终点成为下阶段起点。"""
        mujoco.mj_copyData(
            self.planning.case.data, self.runtime.model, self.runtime.data
        )
        mujoco.mj_forward(self.runtime.model, self.planning.case.data)

    def _refresh_profile_reposition_from_actual(self) -> None:
        """旧推门换点在阶段边界读取实测 TCP 和门角，不使用名义 40°。"""
        phase = self.planning.phases[self.planning.phase_index]
        if phase.skill_id != "push_open" or phase.name not in {
            "retract_up", "new_precontact", "new_contact",
        }:
            return
        profile = self.planning._load_push_open_profile()
        if profile is None or "reposition_after_40" not in profile:
            return
        actual_angle = float(self.runtime.data.qpos[self._door_joint.qposadr[0]])
        if phase.name == "retract_up":
            site_name = self.runtime.robot_metadata["tcp_sites"][self.planning.group]
            site = self.runtime.data.site(site_name)
            target = np.eye(4)
            target[:3, :3] = site.xmat.reshape(3, 3)
            target[:3, 3] = site.xpos.copy()
            target[2, 3] += float(profile["reposition_after_40"]["retreat_world_z_m"])
        else:
            pre, contact = self.planning.push_open_recontact_poses(actual_angle)
            target = pre if phase.name == "new_precontact" else contact
        phases = list(self.planning.phases)
        phases[self.planning.phase_index] = replace(
            phase, tcp_pose_world=target, door_angle_rad=actual_angle,
        )
        self.planning.phases = tuple(phases)

    def _save_row(self, row: dict) -> dict:
        stage_name = row["stage_id"].replace("/", "_")
        destination = self.output_dir / "stages" / f"{stage_name}.json"
        destination.write_text(
            json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False,
                       default=_json_value) + "\n",
            encoding="utf-8",
        )
        self.rows.append(row)
        return row

    def _trace_for(self, skill_id: str) -> _DoorContactTrace:
        if skill_id not in self._traces:
            allowed = ("microwave_handle_grip_collision",) if skill_id == "handle_pull" \
                else ("microwave_door_panel_collision",)
            extra = (self.planning.config.get(
                "physics_push_allowed_gripper_contact_geoms", ())
                if skill_id in {"push_open", "push_close"} else ())
            if not isinstance(extra, (list, tuple)) or any(
                not isinstance(name, str) or not name or
                mujoco.mj_name2id(self.runtime.model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
                for name in extra
            ):
                raise ValueError("推门允许接触的夹爪 geom 必须是场景中存在的名称")
            self._traces[skill_id] = _DoorContactTrace(
                self.runtime.model, self.runtime.robot_metadata,
                self.planning.group, allowed_door_geom_names=allowed,
                allowed_gripper_geom_names=tuple(extra),
            )
        return self._traces[skill_id]

    def _finish_door_skill(self, skill_id: str) -> dict:
        """仅用实际门角、该阶段允许的接触和稳定时间判断 Skill 结果。"""
        trace = self._trace_for(skill_id)
        target_deg = (self.planning.config["pull_angle_deg"] if skill_id == "handle_pull"
                      else 0. if skill_id == "push_close"
                      else self.planning.config["push_angle_deg"])
        result = evaluate_checks([{
            "kind": "door_angle",
            "target_rad": float(np.deg2rad(target_deg)),
            "tolerance_rad": float(np.deg2rad(5.0)),
            "stable_window_s": .3,
            "required_contact_key": ("pad_handle_contact_count" if skill_id == "handle_pull"
                                     else "allowed_target_contact_count"),
        }], trace.samples)[0]
        self.checks[skill_id] = result
        self._save_trace(skill_id)
        return result

    def _save_trace(self, skill_id: str) -> None:
        contact_path = self.output_dir / f"{skill_id}_contact.json"
        contact_path.write_text(
            json.dumps(self._trace_for(skill_id).summary(), ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def _refresh_push_start_from_actual(self) -> None:
        """复用旧 Task1 的阶段二门角重估规则，输入改为 MuJoCo 实测角。"""
        actual = float(self.runtime.data.qpos[self._door_joint.qposadr[0]])
        start, _ = door_skill_angle_range("push_open", self.planning.config, actual)
        self.planning.config["push_start_angle_deg"] = start
        self.planning.rebuild()
        indices = [index for index, phase in enumerate(self.planning.phases)
                   if phase.skill_id == "push_open"]
        self.planning.phase_index = indices[0]
        self.planning.stop_index = (
            len(self.planning.phases) if self.selection in {
                "microwave_door_close", "microwave_door_cycle"
            } else indices[-1] + 1
        )
        self._sync_planning_from_actual()
        (self.output_dir / "config_effective.json").write_text(
            json.dumps(self.planning.config, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8"
        )

    def _refresh_close_start_from_actual(self) -> None:
        """任务过渡后用真实门角重建尚未执行的关门 Knowledge。"""
        actual = float(self.runtime.data.qpos[self._door_joint.qposadr[0]])
        start, _ = door_skill_angle_range("push_close", self.planning.config, actual)
        self.planning.config["close_start_angle_deg"] = start
        self.planning.rebuild()
        indices = [index for index, phase in enumerate(self.planning.phases)
                   if phase.skill_id == "push_close"]
        self.planning.phase_index = indices[0]
        self.planning.stop_index = indices[-1] + 1
        # rebuild() 只重建目标阶段，绝不写真实 MuJoCo 积分状态；关门
        # 规划的起点仍取刚完成任务过渡时的实际机械臂与门姿态。
        self._sync_planning_from_actual()
        (self.output_dir / "config_effective.json").write_text(
            json.dumps(self.planning.config, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8"
        )

    def _record_exception(self, phase, category: str, actual_start: dict,
                          error: Exception) -> dict:
        """把后端 traceback 留在本次实验，不让一段异常抹掉已完成前缀。"""
        name = phase.phase_id.replace("/", "_")
        relative = f"stages/{name}.traceback.txt"
        (self.output_dir / relative).write_text(traceback.format_exc(), encoding="utf-8")
        self.stopped = True
        return self._save_row({
            "stage_id": phase.phase_id,
            "source_format": "mujoco_physical",
            "selection": self.selection,
            "planning_success": False if category == "planning_exception" else True,
            "execution_success": False,
            "goal_success": False,
            "failure_category": category,
            "message": f"{type(error).__name__}: {error}",
            "traceback_ref": relative,
            "start_state_id": actual_start["state_id"],
            "end_state_id": self._observation()["state_id"],
            "actual_start": actual_start,
            "actual_end": self._observation(),
        })

    def _door_path_probes(self, phase, actual_start: dict,
                          names: tuple[str, ...], points: np.ndarray) -> dict:
        """同一条真实规划轨迹分别按名义门角、固定起始门角做只读几何诊断。"""
        allowed_names = (
            ("microwave_door_panel_collision",) if phase.contact_policy == "door_contact"
            else ("microwave_handle_grip_collision",) if phase.contact_policy == "handle_contact"
            else ()
        )
        allowed_robot = (tuple(self.planning.config.get(
            "physics_push_allowed_gripper_contact_geoms", ()))
            if phase.skill_id in {"push_open", "push_close"} else ())
        probes = [("nominal_door_path_probe", phase.door_angle_rad)]
        if phase.contact_policy in {"door_contact", "handle_contact"}:
            # 门在指垫接触前不能自行跟随名义轨迹，因此另查保持起始角
            # 时最先碰门的部位；它只用于诊断，不阻断物理执行。
            probes.append(("frozen_start_door_path_probe",
                           actual_start["door_angle_rad"]))
        result = {}
        for key, target_angle in probes:
            try:
                result[key] = probe_planned_door_contacts(
                    self.runtime.model, self.runtime.data, self.runtime.robot_metadata,
                    group=self.planning.group, joint_names=tuple(names),
                    positions_rad=points,
                    door_start_rad=actual_start["door_angle_rad"],
                    door_target_rad=target_angle,
                    allowed_door_geom_names=allowed_names,
                    allowed_robot_geom_names=allowed_robot,
                )
            except Exception as error:
                result[key] = {
                    "status": "diagnostic_error",
                    "message": f"{type(error).__name__}: {error}",
                }
        return result

    def probe_next(self) -> dict:
        """只规划并检查下一段，不驱动物理状态、不推进 Skill 阶段。"""
        if self.stopped or self.planning.phase_index >= self.planning.stop_index:
            raise ValueError("当前没有可检查的下一阶段")
        self._sync_planning_from_actual()
        self._refresh_profile_reposition_from_actual()
        phase = self.planning.phases[self.planning.phase_index]
        if phase.motion_kind == "gripper":
            raise ValueError("夹爪命令没有 TCP 规划轨迹；请执行到下一运动阶段")
        actual_start = self._observation()
        try:
            planned, selection = self._plan_current_phase()
            phase = self.planning.phases[self.planning.phase_index]
            result = {
                "status": "plan_inspected",
                "stage_id": phase.phase_id,
                "start_state_id": actual_start["state_id"],
                "planning_success": bool(planned.get("success")),
                "planning": planned,
                "candidate_selection": selection,
                "execution_success": None,
                "goal_success": None,
            }
            if result["planning_success"]:
                points = np.asarray(self.planning.planned_q, dtype=float)
                names = tuple(self.planning.planned_joint_names)
                start_q = np.asarray([
                    self.runtime.data.qpos[self.runtime.model.joint(name).qposadr[0]]
                    for name in names
                ])
                if len(points) < 2 or points.shape[1:] != start_q.shape or \
                        float(np.max(np.abs(points[0] - start_q))) > .01:
                    result["planning_success"] = False
                    result["failure_category"] = "trajectory_start_mismatch"
                else:
                    result.update(self._door_path_probes(phase, actual_start,
                                                         names, points))
            else:
                result["failure_category"] = planned.get("failure_category")
            (self.output_dir / "probe_next.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                           default=_json_value) + "\n", encoding="utf-8"
            )
            return result
        except Exception as error:
            traceback_ref = "probe_next.traceback.txt"
            (self.output_dir / traceback_ref).write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            result = {
                "status": "planning_exception",
                "stage_id": phase.phase_id,
                "start_state_id": actual_start["state_id"],
                "planning_success": False,
                "execution_success": None,
                "goal_success": None,
                "failure_category": "planning_exception",
                "message": f"{type(error).__name__}: {error}",
                "traceback_ref": traceback_ref,
            }
            (self.output_dir / "probe_next.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2,
                           allow_nan=False) + "\n", encoding="utf-8"
            )
            return result
        finally:
            # 探针只保留报告和可复用的热 planner；下一次 step 必须从
            # 未改变的实际状态重新规划，不能误用探针轨迹。
            self.planning.planned_q = None
            self.planning.plan_result = None
            self.planning.plan_revision = None

    def _plan_current_phase(self) -> tuple[dict, dict | None]:
        """复用 Viser 的逐角两批候选 IK 筛选，再规划当前真实阶段。"""
        import gene_agent.visualization.microwave.catalog as visual_catalog

        # Viser 可在规划副本写名义门角；物理在线状态绝不能受规划影响。
        actual_before = float(self.runtime.data.qpos[self._door_joint.qposadr[0]])
        selection = visual_catalog._select_task1_push_phase(self.planning)
        if selection is not None and not selection["success"]:
            return ({
                "success": False,
                "operation": "knowledge_candidate_selection",
                "failure_category": "goal_ik_no_solution",
                "message": "旧 Knowledge 当前门角所有候选 IK 不可达",
                "knowledge_candidate_selection": selection,
            }, selection)
        planned = self.planning.plan_next()
        actual_after = float(self.runtime.data.qpos[self._door_joint.qposadr[0]])
        if abs(actual_after - actual_before) > 1e-9:
            raise RuntimeError("规划过程改变了 MuJoCo 实际门角")
        if selection is not None:
            planned["knowledge_candidate_selection"] = selection
        return planned, selection

    def step(self) -> dict:
        """规划一个原有门阶段，随后用 actuator 执行；失败立即停止。"""
        if self.stopped or self.planning.phase_index >= self.planning.stop_index:
            return {"complete": True, "stopped": self.stopped}
        self._sync_planning_from_actual()
        self._refresh_profile_reposition_from_actual()
        phase = self.planning.phases[self.planning.phase_index]
        actual_start = self._observation()
        start_state_id = actual_start["state_id"]
        if (phase.skill_id == 'handle_pull' and phase.name == 'release'
                and self.planning.config.get('check_pull_before_release', False)):
            # 松爪前检查原角度/稳定窗口，避免松开后才发现末段已经脱把手。
            # 原25±5度和0.3秒判据均保持；失败时保留当前抓持状态与接触记录。
            check = self._finish_door_skill('handle_pull')
            if check['status'] != 'pass':
                self.stopped = True
                return self._save_row({
                    'stage_id': phase.phase_id,
                    'source_format': 'mujoco_physical',
                    'planning_success': True, 'execution_success': False,
                    'goal_success': False, 'failure_category': check['code'],
                    'goal_check': check, 'start_state_id': start_state_id,
                    'end_state_id': start_state_id,
                    'actual_start': actual_start, 'actual_end': actual_start,
                })
        if self.rows:
            previous_end = self.rows[-1]["actual_end"]["door_angle_rad"]
            if abs(actual_start["door_angle_rad"] - previous_end) > 1e-6:
                self.stopped = True
                return self._save_row({
                    "stage_id": phase.phase_id,
                    "source_format": "mujoco_physical",
                    "planning_success": False,
                    "execution_success": False,
                    "goal_success": False,
                    "failure_category": "door_angle_discontinuity",
                    "message": "上一阶段终点与本阶段起点的实际门角不连续",
                    "previous_end_door_angle_rad": previous_end,
                    "actual_start": actual_start,
                    "actual_end": actual_start,
                })
        try:
            planned, selection = self._plan_current_phase()
            # gripper 命令会在 plan_next() 内立即推进索引；它的阶段对象
            # 仍是调用前的 phase。运动阶段不推进，可读取候选筛选改写后的目标。
            if phase.motion_kind != "gripper":
                phase = self.planning.phases[self.planning.phase_index]
        except Exception as error:
            return self._record_exception(phase, "planning_exception", actual_start,
                                          error)
        row = {
            "stage_id": phase.phase_id,
            "source_format": "mujoco_physical",
            "selection": self.selection,
            "seed": self.seed,
            "physical_speed_scale": self.speed_scale_by_phase.get(
                phase.phase_id, self.speed_scale
            ),
            "physical_arm_position_kp_scale": self.arm_position_kp_scale_by_phase.get(
                phase.phase_id, self.arm_position_kp_scale
            ),
            "operation": planned.get("operation"),
            "planning_success": bool(planned.get("success")),
            "execution_success": False,
            "goal_success": None,
            "start_state_id": start_state_id,
            "end_state_id": start_state_id,
            "actual_start": actual_start,
            "actual_end": actual_start,
            "planning": planned,
            "candidate_selection": selection,
            "execution": None,
        }
        if not row["planning_success"]:
            row["failure_category"] = planned.get("failure_category")
            self.stopped = True
            return self._save_row(row)
        trace = self._trace_for(phase.skill_id)
        contact_start = len(trace.samples)
        pair_start = {name: dict(value) for name, value in trace.contact_pairs.items()}
        if phase.motion_kind == "gripper":
            try:
                execution = self.runtime.execute_gripper_command(
                    group=self.planning.group,
                    width_m=float(phase.gripper_width_m),
                    duration_s=.4,
                    planned_state_id=start_state_id,
                    step_observer=trace.observe,
                    replay_object_names=("microwave_door",),
                )
            except Exception as error:
                self._save_trace(phase.skill_id)
                return self._record_exception(phase, "execution_exception",
                                              actual_start, error)
        else:
            points = np.asarray(self.planning.planned_q, dtype=float)
            names = self.planning.planned_joint_names
            start_q = np.asarray([
                self.runtime.data.qpos[self.runtime.model.joint(name).qposadr[0]]
                for name in names
            ])
            if len(points) < 2 or points.shape[1:] != start_q.shape or \
                    float(np.max(np.abs(points[0] - start_q))) > .01:
                row["planning_success"] = False
                row["failure_category"] = "trajectory_start_mismatch"
                self.stopped = True
                return self._save_row(row)
            row.update(self._door_path_probes(phase, actual_start, names, points))
            dt = float(planned.get("interpolation_dt_s") or .025)
            # 仅改变物理执行速度，不改 Viser 已验收的轨迹形状与关键目标。
            times = np.arange(len(points), dtype=float) * dt / row["physical_speed_scale"]
            def stop_on_forbidden_contact(_model, _data, sample):
                current_pairs = {
                    name: {
                        "classification": value["classification"],
                        "step_count": (
                            value["step_count"]
                            - pair_start.get(name, {}).get("step_count", 0)
                        ),
                        "normal_impulse_ns": (
                            value["normal_impulse_ns"]
                            - pair_start.get(name, {}).get("normal_impulse_ns", 0.0)
                        ),
                        "max_penetration_depth_m": value["max_penetration_depth_m"],
                        "first_contact_time_s": value["first_contact_time_s"],
                        "last_contact_time_s": value["last_contact_time_s"],
                        "first_contact_world_m": value["first_contact_world_m"],
                        "last_contact_world_m": value["last_contact_world_m"],
                    }
                    for name, value in trace.contact_pairs.items()
                    if value["step_count"] > pair_start.get(name, {}).get("step_count", 0)
                }
                failure = _door_contact_policy_failure(
                    phase, current_pairs,
                    stage_start_time_s=actual_start["sim_time_s"],
                    door_motion_rad=(
                        float(_data.qpos[self._door_joint.qposadr[0]])
                        - actual_start["door_angle_rad"]
                    ),
                )
                if failure is None:
                    return None
                return {
                    "failure_type": failure,
                    "stage": phase.phase_id,
                    "sample_time_s": sample.get("time_s"),
                    "contact_pairs": current_pairs,
                    "source": "mujoco_actual_contact_during_execution",
                }
            try:
                execution = self.runtime.execute_joint_trajectory(
                    group=self.planning.group,
                    joint_names=names,
                    times_s=times,
                    positions_rad=points,
                    planned_state_id=start_state_id,
                    step_observer=trace.observe,
                    stop_condition=stop_on_forbidden_contact,
                    replay_object_names=("microwave_door",),
                    # 物理控制指令发完后，实际机械臂仍可能落后数厘米。
                    # 等待短时间并检查真实关节终点，避免下一阶段从尚未
                    # 到位的位置重新规划，随后出现“规划成功但没碰到门”。
                    endpoint_tolerance_rad=.05,
                    settle_timeout_s=1.5,
                    position_kp_scale=(
                        row["physical_arm_position_kp_scale"]
                        / self.arm_position_kp_scale
                    ),
                )
            except Exception as error:
                self._save_trace(phase.skill_id)
                return self._record_exception(phase, "execution_exception",
                                              actual_start, error)
            failure_type = str(execution.get("failure_type") or "")
            if (not execution.get("success")
                    and ("collision" in failure_type or "contact" in failure_type)):
                try:
                    stopped = self._observation()
                    robot = stopped["robots"][self.planning.group]
                    execution["failure_visualization"] = (
                        self.planning.export_execution_collision(
                            phase.phase_id,
                            robot["joint_names"], robot["joint_positions_rad"],
                            {
                                "failure_type": execution.get("failure_type"),
                                "failure_step_index": execution.get("failure_step_index"),
                                "failure_evidence": execution.get("failure_evidence"),
                                "end_state_id": execution.get("end_state_id"),
                            },
                        )
                    )
                except Exception as visualization_error:
                    execution["failure_visualization_error"] = (
                        f"{type(visualization_error).__name__}: "
                        f"{visualization_error}"
                    )
            # 视觉播放器的 advance() 会写名义门角。物理执行只使用实际
            # MuJoCo 终态，因此这里仅推进阶段索引，不调用 advance()。
            if execution["success"]:
                self.planning.phase_index += 1
                self.planning.planned_q = None
        next_index = self.planning.phase_index
        next_skill = (self.planning.phases[next_index].skill_id
                      if next_index < self.planning.stop_index else None)
        if (execution["success"] and phase.skill_id in {"push_open", "push_close"}
                and next_skill != phase.skill_id):
            # 最后一段轨迹刚到目标时，最近 0.3 s 的窗口仍包含推门运动。
            # 保持当前 actuator 目标并继续推进真实物理，随后再检查门角
            # 是否稳定；不能把检查阈值放宽，也不能写门关节 qpos。
            try:
                hold = self.runtime.execute_hold(
                    duration_s=.35,
                    planned_state_id=self._observation()["state_id"],
                    step_observer=trace.observe,
                    replay_object_names=("microwave_door",),
                    replay_group=self.planning.group,
                )
            except Exception as error:
                row.update(
                    execution=execution,
                    actual_end=self._observation(),
                    execution_success=False,
                    goal_success=False,
                    failure_category="post_motion_hold_exception",
                    message=f"{type(error).__name__}: {error}",
                )
                self.stopped = True
                self._save_trace(phase.skill_id)
                return self._save_row(row)
            execution["post_motion_hold"] = {
                "success": hold["success"],
                "failure_type": hold["failure_type"],
                "sim_duration_s": hold["sim_duration_s"],
                "end_state_id": hold["end_state_id"],
            }
            execution["motion_end_state_id"] = execution["end_state_id"]
            execution["end_state_id"] = hold["end_state_id"]
            execution["motion_sim_duration_s"] = execution["sim_duration_s"]
            execution["sim_duration_s"] += hold["sim_duration_s"]
            execution["replay_samples"].extend(hold["replay_samples"][1:])
            if not hold["success"]:
                execution["success"] = False
                execution["failure_type"] = hold["failure_type"]
        row["execution"] = execution
        row["contact_sample_range"] = [contact_start, len(trace.samples)]
        row["stage_contact_pairs"] = {
            name: {
                "classification": value["classification"],
                "step_count": value["step_count"] - pair_start.get(name, {}).get("step_count", 0),
                "normal_impulse_ns": (value["normal_impulse_ns"]
                                      - pair_start.get(name, {}).get("normal_impulse_ns", 0.0)),
                "max_penetration_depth_m": value["max_penetration_depth_m"],
                "first_contact_in_skill_time_s": value["first_contact_time_s"],
                "first_contact_in_skill_world_m": value["first_contact_world_m"],
                "last_contact_time_s": value["last_contact_time_s"],
                "last_contact_world_m": value["last_contact_world_m"],
            }
            for name, value in trace.contact_pairs.items()
            if value["step_count"] > pair_start.get(name, {}).get("step_count", 0)
        }
        row["execution_success"] = bool(execution["success"])
        row["actual_end"] = self._observation()
        row["end_state_id"] = row["actual_end"]["state_id"]
        if phase.name == "pre_contact" and phase.skill_id in {"push_open", "push_close"}:
            contacts = sum((item["pad_target_contact_count"]
                            + item["forbidden_door_contact_count"]) > 0
                           for item in trace.samples[contact_start:])
            door_delta = abs(row["actual_end"]["door_angle_rad"]
                             - actual_start["door_angle_rad"])
            row["precontact_contact_steps"] = contacts
            row["precontact_door_motion_rad"] = door_delta
        contact_failure = _door_contact_policy_failure(
            phase, row["stage_contact_pairs"],
            stage_start_time_s=actual_start["sim_time_s"],
            door_motion_rad=(row["actual_end"]["door_angle_rad"]
                             - actual_start["door_angle_rad"]),
        )
        if (phase.skill_id == "task_transition" and row["stage_contact_pairs"]
                and contact_failure is None):
            row["departure_pad_contact_accepted"] = True
        if (contact_failure is None and phase.name == "pre_contact"
                and row.get("precontact_door_motion_rad", 0.) > np.deg2rad(2)):
            contact_failure = "unexpected_precontact_door_motion"
        if contact_failure is not None:
            forbidden = [name for name, value in row["stage_contact_pairs"].items()
                         if value["classification"] == "forbidden"]
            row["goal_success"] = False
            row["failure_category"] = contact_failure
            row["rejected_contact_pairs"] = (
                list(row["stage_contact_pairs"]) if phase.contact_policy == "avoid_door"
                else forbidden
            )
            row["forbidden_contact_pairs"] = forbidden
            self.stopped = True
            self._save_trace(phase.skill_id)
            return self._save_row(row)
        if not row["execution_success"]:
            row["failure_category"] = execution.get("failure_type")
            self.stopped = True
            self._save_trace(phase.skill_id)
        else:
            next_index = self.planning.phase_index
            next_skill = (self.planning.phases[next_index].skill_id
                          if next_index < self.planning.stop_index else None)
            if next_skill != phase.skill_id:
                if phase.skill_id == "task_transition":
                    # 上方已逐步检查机器人与门接触；过渡本身没有门角目标，
                    # 不能错误地套用 push 的指垫接触验收。
                    row["goal_check"] = {
                        "kind": "transition_clearance", "status": "pass",
                        "source": "mujoco_actual_contacts",
                    }
                    row["goal_success"] = True
                    if next_skill == "push_close":
                        self._refresh_close_start_from_actual()
                else:
                    check = self._finish_door_skill(phase.skill_id)
                    row["goal_check"] = check
                    row["goal_success"] = check["status"] == "pass"
                    if not row["goal_success"]:
                        row["failure_category"] = check["code"]
                        self.stopped = True
                    elif (phase.skill_id, next_skill) == door_skill_sequence("open") and self.selection in {
                            "microwave_door_open", "microwave_door_close",
                            "microwave_door_cycle"}:
                        self._refresh_push_start_from_actual()
        return self._save_row(row)

    def run(self, until_stage: str | None = None) -> dict:
        """同步运行；提前停在指定阶段只记 partial，不算整项通过。"""
        if until_stage is not None and until_stage not in {
                p.phase_id for p in self.planning.phases[self.planning.phase_index:self.planning.stop_index]}:
            raise ValueError('未知或已执行的门停止阶段')

        def stages():
            # 推门点可在拉门后按实际门角重建，不提前冻结全部轨迹或阶段列表。
            while not self.stopped and self.planning.phase_index < self.planning.stop_index:
                i = self.planning.phase_index
                phase = self.planning.phases[i]
                next_skill = (self.planning.phases[i+1].skill_id
                              if i+1 < self.planning.stop_index else None)
                yield OperationStage(phase.phase_id, next_skill != phase.skill_id)
                if phase.phase_id == until_stage:
                    break

        def observe():
            value = self._observation()
            return OperationState(value['state_id'], 'mujoco', value)

        def perform(name):
            row = self.step()
            check = row.get('goal_check')
            failure = None
            success = (row.get('planning_success') and row.get('execution_success')
                       and row.get('goal_success') is not False)
            if not success:
                failure = {'code': row.get('failure_category') or 'door_stage_failed',
                           'source': 'mujoco_door', 'message': row.get('message') or name,
                           'report_ref': 'stages/' + row['stage_id'].replace('/','_') + '.json'}
            return {'success': bool(success), 'failure': failure, 'execution': row.get('execution'),
                    'check': CheckResult(check.get('status','unknown'),
                        check.get('source') or 'mujoco_door', check) if check else None}

        flow = door_skill.run(CallbackOperationBackend(observe=observe, perform=perform,
                             evidence_mode='mujoco'), stages=stages())
        if not flow.success:
            self.stopped = True
        complete = not self.stopped and self.planning.phase_index >= self.planning.stop_index
        report = {
            "schema_version": 1,
            "source_format": "mujoco_physical",
            "shared_flow": {**flow.to_dict(), "requested_scope": until_stage or "complete"},
            "selection": self.selection,
            "seed": self.seed,
            "speed_scale": self.speed_scale,
            "speed_scale_by_phase": self.speed_scale_by_phase,
            "physical_controller": {
                "arm_position_kp_scale": self.arm_position_kp_scale,
                "arm_position_kp_by_joint": self.arm_position_kp_by_joint,
                "arm_position_kp_scale_by_phase": self.arm_position_kp_scale_by_phase,
            },
            "knowledge_state_effective": self.planning.knowledge_state,
            "scene_ref": self.planning.case.metadata["xml_path"],
            "robot_group": self.planning.group,
            "initialization": self.initialization,
            "status": "completed" if complete else "failed" if self.stopped else "partial",
            "planning_success": self.prefix_planning_success and all(
                row["planning_success"] for row in self.rows
            ),
            "execution_success": self.prefix_execution_success and all(
                row["execution_success"] for row in self.rows
            ),
            "goal_success": complete and all(
                check["status"] == "pass" for check in self.checks.values()
            ) and bool(self.checks),
            "checks": self.checks,
            "stages": [row["stage_id"] for row in self.rows],
            "prefix_stages": self.prefix_stages,
            "prefix_run_ref": self.prefix_run_ref,
            "failure_stage": (self.rows[-1]["stage_id"] if self.stopped and self.rows
                              else None),
            "failure_category": (self.rows[-1].get("failure_category")
                                 if self.stopped and self.rows else None),
            "actual_final": self._observation(),
        }
        (self.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report


__all__ = ["ExperimentSession"]
