"""按旧 Task1 顺序，把现有 Skill 从同一 MuJoCo 实际状态串起来。"""

from __future__ import annotations
from gene_agent.motion.configuration import configured_motion, current_motion_config, motion_configuration

import gc
import json
import traceback
from time import perf_counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from benchmarks.catalog import identity
from gene_agent.tasks.policies import identity as policy_identity
from benchmarks.microwave_task.evaluation import evaluate_final

from gene_agent.tasks.policies.microwave_task.simulation_door.session import ExperimentSession
from gene_agent.tasks.policies.microwave_task.simulation_bowl.extract import run_physical_extract_grasp
from gene_agent.tasks.policies.microwave_task.simulation_bowl.retreat import run_assisted_extract_retreat
from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import run_physical_insert_bowl
from gene_agent.tasks.policies.microwave_task.simulation_knob import run_planned_knob_execution
from gene_agent.tasks.policies.microwave_task.simulation_bowl.place import run_assisted_table_place
from gene_agent.tasks.policies.microwave_task.planning.knob import run_rotate_knob_0_planning
from gene_agent.feedback.state import save_state, state_id, model_fingerprint
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.tasks.policies.microwave_task.simulation_bowl.pick import run_experiment
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult
from gene_agent.tasks.policies.microwave_task.policy import (
    MicrowaveTaskPolicy, default_tcp_transition, run_microwave_task1_flow,
    operation_parameters,
)
from gene_agent.skills.library.door.sequence import DoorPhase
from .subtasks import resolve_subtask, validate_stage_stop
from .operations.bowl import check_bowl_access
from benchmarks.microwave_task.scene import resolve_settings, build_initial_scene, environment_settings
from gene_agent.simulation.execution import scale_position_actuator_kp


ROOT = Path(__file__).resolve().parents[4]


def audit_push_paths(run: Path) -> dict:
    """分别报告规划直线约束与实际 TCP 偏离，不能混为一项成功。"""
    rows = []
    for partition in ("open_first", "close_first", "open_second", "close_second"):
        report = json.loads((Path(run) / partition / "report.json").read_text(
            encoding="utf-8"))
        for stage_id in report["stages"]:
            if not (stage_id.startswith(("push_open/", "push_close/"))
                    and "_push_" in stage_id.split("/")[-1]):
                continue
            path = Path(run) / partition / "stages" / f"{stage_id.replace('/', '_')}.json"
            stage = json.loads(path.read_text(encoding="utf-8"))
            planned = stage.get("planning") or {}
            metrics = planned.get("linear_path_metrics") or {}
            replay = (stage.get("execution") or {}).get("replay_samples") or []
            points = np.asarray([sample["robots"]["right"]["T_world_tcp"]
                                 for sample in replay], dtype=float)[:, :3, 3]
            axis = points[-1] - points[0]
            length = float(np.linalg.norm(axis))
            if length <= 1e-8:
                raise ValueError(f"推门阶段实际 TCP 没有有效位移：{partition}/{stage_id}")
            axis /= length
            projection = (points - points[0]) @ axis
            lateral = np.linalg.norm(points - points[0]
                                     - np.outer(projection, axis), axis=1)
            rows.append({
                "stage_id": f"{partition}/{stage_id}",
                "operation": stage.get("operation"),
                "requested_operation": planned.get("requested_operation"),
                "fallback": planned.get("fallback"),
                "planned_max_lateral_deviation_m": metrics.get("max_lateral_deviation_m"),
                "actual_max_lateral_deviation_m": float(lateral.max()),
                "actual_max_backward_step_m": float(
                    np.maximum(0., -np.diff(projection)).max()),
            })
    if not rows:
        raise ValueError("完整任务没有可审核的推门运动段")
    return {
        "stage_count": len(rows),
        "all_planned_linear_tcp": all(
            row["operation"] == row["requested_operation"] == "linear_tcp"
            and row["fallback"] is None
            and row["planned_max_lateral_deviation_m"] is not None
            for row in rows
        ),
        "max_planned_lateral_deviation_m": max(
            row["planned_max_lateral_deviation_m"] for row in rows),
        "max_actual_lateral_deviation_m": max(
            row["actual_max_lateral_deviation_m"] for row in rows),
        "max_actual_backward_step_m": max(
            row["actual_max_backward_step_m"] for row in rows),
        "stages": rows,
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _prepare_scene(output: Path, settings: dict) -> tuple[Path, Path]:
    """只建立一份 GT 场景；支撑高度与仿真 weld 都在编译前确定。"""
    scene_dir = output / "scene"
    object_poses = {
        "microwave": np.asarray(settings["microwave_pose_world"], dtype=float),
        "bowl": np.asarray(settings["bowl_pose_world"], dtype=float),
    }
    model, data, metadata, placement = build_initial_scene(environment_settings(settings),scene_dir / 'source_scene')
    _write(output / 'placement.json',placement)
    if not placement['valid']:
        reasons = [*placement.get('invalid_reasons', []),
                   *placement.get('initial_check', {}).get('invalid_reasons', [])]
        labels = {'table_boundary': '超出声明的摆位边界',
                  'outside_table_support': '超出桌面支撑范围'}
        details = '; '.join(
            f"{row.get('object') or row.get('bodies') or '场景'}: "
            f"{labels.get(row['reason'], row['reason'])} ({row['reason']})" for row in reasons)
        raise ValueError(f'微波炉初态无效：{details or "原因未记录"}；'
                         '见 placement.json；尚未进入 IK 或运动规划')
    export_collision_scene(model, data, metadata["robot"], "right",
                           scene_dir / "collision")
    _write(scene_dir / "case_meta.json", {
        "revision": 0, "seed": int(settings["seed"]), "joint_overrides": {},
    })
    scene_config = {
        "schema_version": 1, "seed": int(settings["seed"]),
        "table": settings.get('table'),
        "object_poses_world": {key: value.tolist() for key, value in object_poses.items()},
        "turntable_height_offset_m": settings["turntable_height_offset_m"],
        "microwave_floor_height_offset_m": settings["microwave_floor_height_offset_m"],
        "extract_arm_kp_scale": settings["extract_arm_kp_scale"],
    }
    scene_config_path = output / "scene_config.json"
    _write(scene_config_path, scene_config)
    door_config = json.loads((ROOT / settings["door_config_template"]).read_text(
        encoding="utf-8"))
    door_config["scene_run"] = str(scene_dir)
    door_config["microwave_pose_world"] = object_poses["microwave"].tolist()
    door_config["physics_arm_position_kp_scale"] = float(settings["door_arm_kp_scale"])
    door_config_path = output / "door_config.json"
    _write(door_config_path, door_config)
    return door_config_path, scene_config_path


def _transition_after_open(session: ExperimentSession) -> dict:
    """旧 Task1 的默认 TCP 撤离是 Task 过渡，不塞进单独开门 Skill。"""
    index = next(i for i, phase in enumerate(session.planning.phases)
                 if phase.skill_id == "task_transition")
    session.planning.phase_index = index
    session.planning.stop_index = index + 1
    session._sync_planning_from_actual()
    return session.step()


def _continue_close(runtime, config: Path, output: Path, *, prepare_only=False) -> dict:
    """从当前实际姿态回默认 TCP，再用同源 Knowledge 物理推关门。"""
    session = ExperimentSession("microwave_door_cycle", "physics", config, 0,
                                output_dir=output)
    session.runtime = runtime
    session.initialization = {
        "kind": "handoff_from_live_task", "source_state_id": state_id(runtime.model, runtime.data),
        "door_angle_rad": float(runtime.data.qpos[
            runtime.model.joint("microwave_door_joint").qposadr[0]]),
    }
    session.planning.phase_index = next(i for i, phase in enumerate(session.planning.phases)
                                        if phase.skill_id == "task_transition")
    session.planning.stop_index = len(session.planning.phases)
    session._sync_planning_from_actual()
    return {"report": None if prepare_only else session.run(), "session": session}


def _continue_before_reopen(runtime, config: Path, output: Path) -> dict:
    """第二次开门前先退离把手，再缓慢回到默认 TCP。

    旋钮操作结束时的实际姿态可能把前臂带到把手附近。直接把该姿态
    送入 handle_pull 的 pregrasp 会让规划器在一条短路径上同时扭转多
    个关节，物理执行时容易擦到把手。这里显式插入一个避障退离段，
    然后再用完整场景碰撞规划回默认 TCP；两段都从同一实际状态续接。
    """
    output = Path(output)
    session = ExperimentSession("microwave_door_cycle", "physics", config,
                                0, output_dir=output)
    session.runtime = runtime
    session.initialization = {
        "kind": "handoff_from_live_task",
        "source_state_id": state_id(runtime.model, runtime.data),
        "door_angle_rad": float(runtime.data.qpos[
            runtime.model.joint("microwave_door_joint").qposadr[0]]),
    }
    planning = session.planning
    transition_index = next(
        i for i, phase in enumerate(planning.phases)
        if phase.skill_id == "task_transition"
    )
    actual = session._observation()
    group = planning.group
    current_tcp = np.asarray(actual["robots"][group]["T_world_tcp"], dtype=float)
    retreat = current_tcp.copy()
    # 旋钮操作结束后的姿态不是稳定的自由空间姿态。先保持当前腕部
    # 朝向抬高并外移，避免在贴近门把手时同时扭转多个关节；第二段
    # 使用配置的 home 关节分支回位，避免末端位姿 IK 选到自碰撞冗余解。
    # 旋钮终态位于工作空间边缘，横向外移会迫使腕部切换冗余分支。
    # 先保持当前姿态只上抬一小段，获得把手净空，再回配置的 home C-space。
    retreat[2, 3] += 0.02
    transition = default_tcp_transition(
        planning.default_tcp_pose_world,
        door_angle_rad=float(actual["door_angle_rad"]),
        gripper_width_m=float(actual["robots"][group]["gripper"]["width_m"]),
        phase_id="task/before_reopen/return_default_tcp",
    )
    retreat_phase = DoorPhase(
        "task/before_reopen/retreat_from_door", "task_transition",
        "retreat_from_door", retreat, float(actual["door_angle_rad"]),
        float(actual["robots"][group]["gripper"]["width_m"]), "plan",
        "avoid_door",
    )
    planning.phases = (
        planning.phases[:transition_index]
        + (retreat_phase, transition)
        + planning.phases[transition_index + 1:]
    )
    planning.phase_index = transition_index
    planning.stop_index = transition_index + 2
    home = planning.case.metadata["robot"]["home_joint_positions"]
    names = tuple(planning.case.metadata["robot"]["joint_groups"][group])
    planning.joint_targets_by_phase[transition.phase_id] = [
        float(home[name]) for name in names
    ]
    session._sync_planning_from_actual()
    # 仅降低物理执行速度，规划目标和碰撞边界保持不变。
    session.speed_scale_by_phase[retreat_phase.phase_id] = 0.45
    session.speed_scale_by_phase[transition.phase_id] = 0.35
    report = session.run()
    report["safe_reposition"] = {
        "retreat_distance_m": 0.0,
        "retreat_lift_m": 0.02,
        "return_target": "configured_home_joint_positions_cspace",
        "retreat_target_tcp_pose_world": retreat.tolist(),
        "retreat_source": "actual_tcp_world_z_clearance",
        "speed_scale_by_phase": {
            retreat_phase.phase_id: 0.45,
            transition.phase_id: 0.35,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"report": report, "session": session}


def _continue_before_knob(runtime, config: Path, output: Path) -> dict:
    """关门后把右臂归一到配置的安全关节分支，再开始旋钮候选筛选。

    七自由度机械臂可能以不同肘/腕构型到达同一默认 TCP。旋钮规划若只
    消费这个 TCP，会把上游抓碗产生的冗余分支带入头部附近。这里从实际
    状态规划到命名 home 关节目标；目标来自机器人配置，不绑定具体 seed。
    """
    output = Path(output)
    session = ExperimentSession(
        "microwave_door_cycle", "physics", config, 0, output_dir=output,
    )
    session.runtime = runtime
    session.initialization = {
        "kind": "handoff_from_live_task",
        "source_state_id": state_id(runtime.model, runtime.data),
        "door_angle_rad": float(runtime.data.qpos[
            runtime.model.joint("microwave_door_joint").qposadr[0]
        ]),
    }
    planning = session.planning
    actual = session._observation()
    group = planning.group
    current_tcp = np.asarray(actual["robots"][group]["T_world_tcp"], dtype=float)
    handle = np.asarray(
        runtime.data.geom("microwave_handle_grip_collision").xpos, dtype=float,
    )
    away = current_tcp[:2, 3] - handle[:2]
    norm = float(np.linalg.norm(away))
    if norm < 1e-6:
        away = np.array([-1.0, 0.0], dtype=float)
        norm = 1.0
    retreat = current_tcp.copy()
    # 该姿态位于当前工作空间边缘；固定的大幅抬升会让 IK 失去解。
    # 小幅外移先解除门板附近的指垫净空，再由 C-space 规划完成归位。
    retreat[:2, 3] += away / norm * 0.015
    retreat[2, 3] += 0.01
    retreat_phase = DoorPhase(
        "task/before_knob/retreat_from_closed_door", "task_transition",
        "retreat_from_door", retreat, float(actual["door_angle_rad"]),
        float(actual["robots"][group]["gripper"]["width_m"]),
        "plan", "avoid_door",
    )
    transition = default_tcp_transition(
        planning.default_tcp_pose_world,
        door_angle_rad=float(actual["door_angle_rad"]),
        gripper_width_m=float(actual["robots"][group]["gripper"]["width_m"]),
        phase_id="task/before_knob/normalize_right_arm",
    )
    planning.phases = (retreat_phase, transition)
    planning.phase_index = 0
    planning.stop_index = 2
    home = planning.case.metadata["robot"]["home_joint_positions"]
    names = tuple(planning.case.metadata["robot"]["joint_groups"][group])
    target = [float(home[name]) for name in names]
    planning.joint_targets_by_phase[transition.phase_id] = target
    session._sync_planning_from_actual()
    session.speed_scale_by_phase[retreat_phase.phase_id] = 0.45
    session.speed_scale_by_phase[transition.phase_id] = 0.35
    report = session.run()
    report["safe_branch_normalization"] = {
        "source_state_id": session.initialization["source_state_id"],
        "target_source": "robot_metadata.home_joint_positions",
        "joint_names": list(names),
        "target_positions_rad": target,
        "retreat_target_tcp_pose_world": retreat.tolist(),
        "retreat_source": "actual_tcp_away_from_microwave_handle_xy",
        "retreat_distance_m": 0.015,
        "retreat_lift_m": 0.01,
        "collision_scope": "full_scene_with_self_collision",
        "speed_scale_by_phase": {
            retreat_phase.phase_id: 0.45,
            transition.phase_id: 0.35,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"report": report, "session": session}


def _continue_open(runtime, config: Path, output: Path) -> dict:
    # 二次开门的真实开口存在约3.3mm跟踪偏差；原15mm命令抓持裕量不足。
    # 原001/018/006同状态单因素验证10mm可持续拉到23°以上。首轮设置不变。
    reopen_config = json.loads(Path(config).read_text(encoding='utf-8'))
    reopen_config['handle_close_width_m'] = float(reopen_config.get(
        'reopen_handle_close_width_m', .010))
    reopen_config['check_pull_before_release'] = True
    reopen_path = Path(output).parent / f'{Path(output).name}_config.json'
    _write(reopen_path, reopen_config)
    session = ExperimentSession("microwave_door_open", "physics", reopen_path, 0,
                                output_dir=output)
    session.runtime = runtime
    session.initialization = {
        "kind": "handoff_from_live_task", "source_state_id": state_id(runtime.model, runtime.data),
        "door_angle_rad": float(runtime.data.qpos[
            runtime.model.joint("microwave_door_joint").qposadr[0]]),
    }
    session._sync_planning_from_actual()
    return {"report": session.run(), "session": session}


def _knob_plan(runtime, closed_session: ExperimentSession, output: Path,
               settings: dict) -> dict:
    """仅换已调通的抓取姿态参数；旋钮阶段顺序、直线段均来自旧 Skill。"""
    task = json.loads((ROOT / "data/task_1/skills/knob_0_task.json").read_text(encoding="utf-8"))
    task["microwave_pose_world"] = settings["microwave_pose_world"]
    profile_path = ROOT / task["grasp_profile_path"]
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    params = profiles["profiles"][task["grasp_profile_name"]]["parameters"]
    params["azimuth_deg"]["values"] = [settings["knob_azimuth_deg"]]
    params["tip_x_deg"]["values"] = [settings["knob_tip_x_deg"]]
    new_profile = output / "knob_profile.json"
    _write(new_profile, profiles)
    task["grasp_profile_path"] = str(new_profile)
    task_path = output / "knob_task.json"
    _write(task_path, task)
    closed_session._sync_planning_from_actual()
    return run_rotate_knob_0_planning(
        output / "knob_plan", task_path=task_path,
        case=closed_session.planning.case,
    )


def _release_finished_planner(session: ExperimentSession) -> None:
    """子任务结束后释放 cuRobo 图和缓存，保留 MuJoCo 实际状态与运行记录。

    完整任务会依次建立四个门规划会话；已完成的会话仍保存报告和场景，
    但不再需要其 CUDA solver。显式回收可避免末次关门与此前的图争显存。
    """
    session.planning.planner_pool.clear()
    session.planning.plan_result = None
    session.planning.planned_q = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MicrowaveOperationFailure(RuntimeError):
    def __init__(self, message, report_ref):
        super().__init__(message)
        self.report_ref = report_ref


class MicrowavePreconditionError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SimulationBowlHandoff:
    """仿真专用实际操作证据；模型绑定不传给真机 policy。"""
    report: dict
    start_state_id: str
    end_state_id: str
    model_fingerprint: str


class MicrowaveSimulationSession:
    """持有同一 MuJoCo Runtime 和操作证据；policy 不接管这些后端资源。

    prefix 从本会话 fresh reset 跑必要前缀；current 只执行所选操作，
    必须由当前实际状态及本会话证据满足前置条件，不读取任意旧报告冒充持物。
    """
    @configured_motion("config", override_argument="collision_mode")
    def __init__(self, output_dir: Path, *, config: Path, record_video=False,
                 seed=None, randomize=None, collision_mode=None):
        settings = resolve_settings(json.loads(Path(config).read_text(encoding='utf-8')),
                                    seed=seed,randomize=randomize)
        settings["motion"] = current_motion_config().to_dict()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'resolved_settings.json',settings)
        _write(output / 'resolved_inputs.json', {
            'benchmark_environment': environment_settings(settings),
            'policy_parameters': operation_parameters(settings),
            'adapter': {'door_config_template': settings['door_config_template'],
                        'motion': settings['motion']},
            'source': str(Path(config).resolve()),
            'note': '旧 JSON 保留兼容；场景只接收 benchmark_environment，数值不复制为第二套默认值',
        })
        door_config, scene_config = _prepare_scene(output, settings)
        summary: dict = {
            "schema_version": 1, "task": "microwave_task1", "status": "running",
            "source_format": "mujoco_physical_with_declared_assistance",
            "evidence_mode": "assisted_pick_extract_and_kinematic_knob",
            "config_ref": str(Path(config).resolve()), "scene_ref": str(output / "scene/source_scene/scene.xml"),
            "subtasks": {}, "goal_success": False, "fully_physical_success": False,
            "planner_backend": "curobo", "seed": settings["seed"],
            "motion": current_motion_config().to_dict(),
            "placement_ref": str(output / "placement.json"),
            "placement": settings.get("placement_record"),
        }

        summary["benchmark"] = identity("microwave_task")
        summary["policy_id"] = "microwave_task/default"
        summary["policy"] = policy_identity("microwave_task")

        opened = ExperimentSession("microwave_door_open", "physics", door_config, 0,
                                   output_dir=output / "open_first")
        runtime = opened.runtime
        runtime.replay_sample(object_names=('bowl',),robot_groups=('left','right'))
        runtime.metadata["simulation_attachment_pairs"] = dict(
            settings["simulation_attachment_pairs"])
        runtime.metadata["gravity_compensated_groups"] = []
        # 共享 Task 决定子任务次序；这些回调只装配当前环境的设备与参数。
        # 第二次开门前的安全过渡由下方适配器从实际状态动态生成。
        policy = MicrowaveTaskPolicy(before_reopen_transition=True)
        summary["environment_policy"] = {
            "include_knob": True, "before_reopen_transition": True,
            "reason": "第二次开门前先退离把手，再缓慢回默认 TCP；完整场景避障",
        }
        summary["steps"] = []
        context = {"opened": opened}

        self.settings, self.output = settings, output
        self.door_config, self.scene_config = door_config, scene_config
        self.summary, self.runtime, self.opened = summary, runtime, opened
        self.policy, self.context = policy, context
        self.record_video = record_video
        self.bowl_handoffs: dict[str, SimulationBowlHandoff] = {}
        self._run_index = 0

    def record(self, name: str, success: bool, report_ref: str, *, state_before: str,
               detail: dict | None = None, failure_report_ref: str | None = None) -> None:
        summary, runtime, output = self.summary, self.runtime, self.output
        summary["subtasks"][name] = {
            "success": bool(success), "report_ref": report_ref,
            "start_state_id": state_before,
            "end_state_id": state_id(runtime.model, runtime.data),
            **(detail or {}),
        }
        _write(output / "summary.json", summary)
        if not success:
            summary["status"] = "failed"
            summary["failure_subtask"] = name
            _write(output / "summary.json", summary)
            raise MicrowaveOperationFailure(f"操作停止于 {name}；详见 {report_ref}",
                                            failure_report_ref or report_ref)

    def observe(self):
        runtime = self.runtime
        return OperationState(state_id(runtime.model, runtime.data), "mujoco", {
            "sim_time_s": float(runtime.data.time),
            "observation": runtime.query(object_names=("bowl",), robot_groups=("right",)),
        })

    def transition_close(self, key, partition):
        runtime, door_config, output = self.runtime, self.door_config, self.output
        context = self.context
        session = _continue_close(runtime, door_config, output / partition,
                                  prepare_only=True)["session"]
        context[key] = {"session": session}
        # stop_index 保持完整关闭范围，step() 沿用原关门起点重估逻辑。
        transition = session.step()
        return transition

    def perform(self, name):
        settings, output = self.settings, self.output
        summary, runtime = self.summary, self.runtime
        context, opened = self.context, self.opened
        door_config, scene_config = self.door_config, self.scene_config
        record_video = self.record_video
        before = state_id(runtime.model, runtime.data)
        row = {"step_id": name, "start_state_id": before, "status": "started"}
        summary["steps"].append(row)
        _write(output / "summary.json", summary)
        started, sim_started = perf_counter(), float(runtime.data.time)
        ref, result = None, {}
        try:
            self.require_preconditions(name)
            if name == "open_door_first":
                result = opened.run()
                success = result["goal_success"]
                ref = "open_first/report.json"
                self.record("open_first", success, ref, state_before=before)
            elif name in {"after_open_transition", "after_reopen_transition"}:
                session = opened if name == "after_open_transition" else context["open_second"]["session"]
                result = _transition_after_open(session)
                success = result["execution_success"] and result.get("failure_category") is None
                partition = "open_first" if name == "after_open_transition" else "open_second"
                ref = f"{partition}/stages/task_open_to_close_return_default_tcp.json"
                self.record(name, success, ref, state_before=before)
                if name == "after_open_transition":
                    save_state(runtime.model, runtime.data, output / "state_after_open.npz")
                _release_finished_planner(session)
            elif name == "pick_bowl":
                result, success, ref = self._pick_bowl(before)
            elif name == "insert_bowl":
                # 放入单独由 Task 调用，不能藏在 Pick 的 continuation 决策内。
                result, success, ref = self._insert_bowl(before)
            elif name in {"before_close_first_transition", "before_close_second_transition"}:
                if name == "before_close_first_transition":
                    self.restore_control("before_close_first_transition")
                key = "close_first" if name == "before_close_first_transition" else "close_second"
                result = self.transition_close(key, key)
                success = result["execution_success"] and result.get("failure_category") is None
                ref = f"{key}/stages/task_open_to_close_return_default_tcp.json"
            elif name == "before_reopen_transition":
                transition_output = output / "before_reopen_transition"
                result = _continue_before_reopen(
                    runtime, door_config, transition_output
                )
                context["before_reopen_transition"] = result
                success = result["report"]["shared_flow"]["success"]
                ref = "before_reopen_transition/report.json"
            elif name in {"close_door_first", "close_door_second"}:
                key = "close_first" if name == "close_door_first" else "close_second"
                session = context[key]["session"]
                result = session.run()
                context[key]["report"] = result
                success = result["goal_success"]
                ref = f"{key}/report.json"
                self.record(key, success, ref, state_before=before)
                if key == "close_first":
                    save_state(runtime.model, runtime.data, output / "state_after_close.npz")
                _release_finished_planner(session)
            elif name == "rotate_knob":
                closed = context.get("close_first", {}).get("session")
                if closed is None:
                    closed = ExperimentSession("microwave_door_cycle", "physics", door_config, 0,
                                               output_dir=output / "knob_context")
                    closed.runtime = runtime
                    context["knob_context"] = {"session": closed}
                normalization = _continue_before_knob(
                    runtime, door_config, output / "before_knob_transition",
                )
                context["before_knob_transition"] = normalization
                if not normalization["report"]["shared_flow"]["success"]:
                    self.record(
                        "rotate_knob", False,
                        "before_knob_transition/report.json", state_before=before,
                    )
                _release_finished_planner(normalization["session"])
                plan = _knob_plan(runtime, closed, output, settings)
                context["knob_plan"] = plan
                if not plan["success"]:
                    self.record("rotate_knob", False, "knob_plan/skill_rotate_knob_0_result.json", state_before=before)
                result = run_planned_knob_execution(runtime, plan, output / "knob_execution",
                    **self.stage_stop_kwargs(name))
                success = result["operation_success"] if self.stage_stop_kwargs(name) else result["goal_success"]
                ref = "knob_execution/result.json"
                self.record("rotate_knob", success, ref, state_before=before,
                       detail={"evidence_mode": result["evidence_mode"]})
                save_state(runtime.model, runtime.data, output / "state_after_knob.npz")
            elif name == "open_door_second":
                result = _continue_open(runtime, door_config, output / "open_second")
                context["open_second"] = result
                success = result["report"]["goal_success"]
                ref = "open_second/report.json"
                self.record("open_second", success, ref, state_before=before)
            elif name == "extract_bowl":
                result, success, ref = self._extract_bowl(before)
            elif name == "place_bowl_on_table":
                result, success, ref = self._place_bowl_on_table(before)
            else:
                raise ValueError(f"未实现的仿真 Task 适配：{name}")
            row.update(success=bool(success), status="executed" if success else "failed", report_ref=ref)
            if not success:
                row['failure'] = self.failure_details(result, ref)
        except Exception as error:
            success = False
            row.update(success=False, status="failed", exception=f"{type(error).__name__}: {error}")
            diagnostic = output / "diagnostics" / f"{name}.traceback.txt"
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_text(traceback.format_exc(), encoding="utf-8")
            reference = getattr(error, 'report_ref', None) or ref
            if reference is None:
                # 操作方法可能已写完原报告，却在录像/落盘收尾时抛错，
                # 尚未来得及返回 result。继续保留原 planner/执行失败证据。
                reports = {
                    'pick_bowl': ('pick_insert/result.json',),
                    'insert_bowl': ('pick_insert/physical_insert/insert_result.json',),
                    'extract_bowl': ('extract_retreat/result.json', 'extract_grasp/result.json'),
                    'place_bowl_on_table': ('table_place/result.json',),
                }
                reference = next((path for path in reports.get(name, ())
                                  if (output / path).is_file()), None)
            row['failure'] = {**self.failure_details(result, reference, error),
                              'traceback_ref': str(diagnostic.relative_to(output))}
            summary.update(status="failed", failure_subtask=name)
        row["end_state_id"] = state_id(runtime.model, runtime.data)
        row.update(wall_time_s=perf_counter()-started, sim_start_s=sim_started,
                   sim_end_s=float(runtime.data.time))
        _write(output / "summary.json", summary)
        facts = result.get('report', result)
        execution_facts = ({'execution_status': facts['shared_flow'].get('execution_status', 'unknown')}
                           if facts.get('shared_flow') else
                           {'execution': facts['execution']} if 'execution' in facts else
                           {'execution_status': 'unknown'})
        return {"success": bool(success), **execution_facts, "check": CheckResult(
            "pass" if success else "fail", "mujoco_subtask_checks",
            {"report_ref": row.get("report_ref"),
             "evidence_mode": "assisted_pick_extract_and_kinematic_knob"})}

    def _pick_bowl(self, before):
        settings, output, runtime = self.settings, self.output, self.runtime
        scene_config, record_video = self.scene_config, self.record_video
        self.activate_load_control("pick_bowl")
        result = run_experiment(
            output / "pick_insert", runtime=runtime, candidate_id="auto",
            depth_m=float(settings["pick_depth_m"]), scene_config_path=scene_config,
            skill_profile="pick_bowl_legacy_visual", simulation_weld_after_contact=True,
            arm_kp_scale=1., record_video=record_video)
        self.capture_bowl_handoff("pick", result, before)
        success = result["success"]
        ref = "pick_insert/result.json"
        return result, bool(success), ref

    def _insert_bowl(self, before):
        settings, output, runtime = self.settings, self.output, self.runtime
        context, record_video = self.context, self.record_video
        recorder = None
        if record_video:
            from gene_agent.visualization.trajectory import TrajectoryRecorder
            recorder = TrajectoryRecorder(
                output / "pick_insert/insert_visual", target_body="bowl",
                width=640, height=480, fps=10, sample_interval_s=.1,
                combine_mp4=True, detail_view="gripper_side", extra_view="contact",
                overview_camera={"lookat": [.42, .26, .13], "distance": .85,
                                 "azimuth": -10., "elevation": -25.},
                robot_metadata=runtime.robot_metadata, active_group="right")
        try:
            result = run_physical_insert_bowl(
                runtime, output / "pick_insert", recorder, self.bowl_handoffs["pick"].report,
                insert_depth_offset_m=float(settings["insert_depth_offset_m"]),
                insert_tcp_table_angle_max_deg=float(
                    settings.get("insert_tcp_table_angle_max_deg", 70.)),
                insert_max_bowl_tilt_deg=float(
                    settings.get("insert_max_bowl_tilt_deg", 14.)),
                retreat_clearance_margin_m=float(
                    settings.get("insert_retreat_clearance_margin_m", .005)),
                retreat_minimum_lift_m=float(
                    settings.get("insert_retreat_minimum_lift_m", 0.)),
                retreat_rotation_local_deg=tuple(settings["insert_retreat_rotation_local_deg"]),
                **self.stage_stop_kwargs("insert_bowl"))
        finally:
            if recorder is not None:
                evidence = recorder.finalize()
                context["insert_video"] = evidence
        pick = self.bowl_handoffs["pick"].report
        pick.update(continuation=result,
                    compound_success=bool(pick.get("success") and result.get("physical_place_success")),
                    fully_physical_compound_success=bool(pick.get("physical_pick_success") and result.get("physical_place_success")),
                    compound_mode="simulation_assisted")
        if record_video:
            pick["insert_visual_evidence"] = context.get("insert_video")
        _write(output / "pick_insert/result.json", pick)
        success = result["operation_success"] if self.stage_stop_kwargs("insert_bowl") else pick["compound_success"]
        ref = "pick_insert/physical_insert/insert_result.json"
        self.record("pick_insert", success, "pick_insert/result.json",
               state_before=self.bowl_handoffs["pick"].start_state_id,
               detail={"evidence_mode": pick["compound_mode"]}, failure_report_ref=ref)
        save_state(runtime.model, runtime.data, output / "state_after_insert.npz")
        return result, bool(success), ref

    def _extract_bowl(self, before):
        settings, output, runtime = self.settings, self.output, self.runtime
        record_video = self.record_video
        self.activate_load_control("extract_bowl")
        extract = run_physical_extract_grasp(runtime, output / "extract_grasp", record_video=record_video)
        self.record("extract_grasp", extract["physical_grasp_prefix_success"],
               "extract_grasp/result.json", state_before=before)
        if self.stage_stop_kwargs("extract_bowl"):
            extract.update(requested_scope="grasp", operation_success=True,
                           actual_final=asdict(self.observe()))
            _write(output / 'extract_grasp/result.json', extract)
            return extract, True, "extract_grasp/result.json"
        retreat_start = state_id(runtime.model, runtime.data)
        result = run_assisted_extract_retreat(runtime, output / "extract_retreat", extract, record_video=record_video)
        self.capture_bowl_handoff("retreat", result, retreat_start)
        success = result["assisted_extract_success"]
        ref = "extract_retreat/result.json"
        self.record("extract_retreat", success, ref, state_before=retreat_start,
               detail={"evidence_mode": "simulation_assisted"})
        return result, bool(success), ref

    def _place_bowl_on_table(self, before):
        output, runtime = self.output, self.runtime
        context, record_video = self.context, self.record_video
        result = run_assisted_table_place(runtime, output / "table_place", self.bowl_handoffs["retreat"].report, record_video=record_video,
            **self.stage_stop_kwargs("place_bowl_on_table"))
        context["placed"] = result
        success = result["operation_success"] if self.stage_stop_kwargs("place_bowl_on_table") else result["physical_table_place_success"]
        ref = "table_place/result.json"
        self.record("table_place", success, ref, state_before=before)
        save_state(runtime.model, runtime.data, output / "state_after_table_place.npz")
        self.restore_control("place_bowl_on_table")
        return result, bool(success), ref

    def failure_details(self, result, reference, error=None):
        """将原操作的失败阶段与分类提到摘要，报告仍是事实来源。"""
        raw = result if isinstance(result, dict) else {}
        if not raw and reference and (self.output / reference).is_file():
            raw = json.loads((self.output / reference).read_text(encoding='utf-8'))
        raw = raw.get('continuation') or raw
        first = raw.get('first_failure') or {}
        if not isinstance(first, dict):
            first = {'message': str(first)}
        phase = raw.get('failure_stage') or first.get('stage') or first.get('stage_id')
        stages = raw.get('stages') or {}
        if isinstance(stages, dict):
            failed = stages.get(phase) or {}
        else:
            failed = next((r for r in stages if isinstance(r, dict)
                           and r.get('name', r.get('stage_id')) == phase), {})
        planning, execution = failed.get('planning') or {}, failed.get('execution') or {}
        code = (first.get('code') or raw.get('failure_category') or planning.get('failure_category')
                or execution.get('failure_type') or getattr(error, 'code', None)
                or ('operation_exception' if error else 'operation_failed'))
        message = (first.get('message') or raw.get('error') or planning.get('raw_error')
                   or planning.get('status') or execution.get('message') or raw.get('failure_reasons')
                   or str(error or '操作后置条件未通过'))
        return {'code': code, 'message': str(message), 'stage': phase, 'report_ref': reference}

    def require_preconditions(self, name):
        """入口检查当前事实；接触/夹持的细检查继续由原操作实现负责。"""
        runtime, context = self.runtime, self.context
        if not np.isfinite(runtime.data.qpos).all():
            raise MicrowavePreconditionError('invalid_state', '实际关节/物体状态包含非有限值')
        angle = float(runtime.data.qpos[runtime.model.joint('microwave_door_joint').qposadr[0]])
        if name in {'pick_bowl', 'insert_bowl', 'extract_bowl'}:
            # 沿用门操作的 5° 终态容差，目标来自该场景实际配置。
            target = json.loads((ROOT / self.settings['door_config_template']).read_text(encoding='utf-8'))['push_angle_deg']
            if not check_bowl_access(angle, target_angle_rad=np.deg2rad(target)).passed:
                raise MicrowavePreconditionError('precondition_failed', '门未处于已打开的操作范围；请先开门并完成原 Task 过渡')
        if name == 'rotate_knob' and abs(angle) > np.deg2rad(5):
            raise MicrowavePreconditionError('precondition_failed', '旋钮操作需要当前门已关闭')
        key = {'insert_bowl': 'pick', 'place_bowl_on_table': 'retreat'}.get(name)
        if key:
            if key not in self.bowl_handoffs:
                raise MicrowavePreconditionError('missing_handoff', f'{name} 缺少本会话的 {key} 实际操作证据')
            if self.bowl_handoffs[key].model_fingerprint != model_fingerprint(runtime.model):
                raise MicrowavePreconditionError('incompatible_handoff', f'{key} 证据模型与当前几何/控制参数不兼容')
            if self.bowl_handoffs[key].end_state_id != state_id(runtime.model, runtime.data):
                raise MicrowavePreconditionError('stale_handoff', f'{key} 证据终态与当前状态不一致；禁止使用旧持物报告')

    def capture_bowl_handoff(self, key: str, report: dict, start_state_id: str):
        self.bowl_handoffs[key] = SimulationBowlHandoff(
            report, start_state_id, state_id(self.runtime.model, self.runtime.data),
            model_fingerprint(self.runtime.model))

    def activate_load_control(self, stage: str) -> None:
        settings, context, runtime = self.settings, self.context, self.runtime
        ratio = float(settings["extract_arm_kp_scale"]) / float(settings["door_arm_kp_scale"])
        if "arm_kp_ratio" in context:
            if not np.isclose(float(context["arm_kp_ratio"]), ratio):
                raise MicrowavePreconditionError(
                    "control_state_conflict", "负载控制增益已激活且比例不一致")
            return
        scale_position_actuator_kp(
            runtime.model, runtime.robot_metadata["joint_groups"]["right"], ratio)
        context["arm_kp_ratio"] = ratio
        self.summary.setdefault("control_transitions", []).append({
            "stage": stage,
            "from_arm_position_kp_scale": float(settings["door_arm_kp_scale"]),
            "to_arm_position_kp_scale": float(settings["extract_arm_kp_scale"]),
            "ratio": ratio,
        })

    def restore_control(self, stage: str = "session_close"):
        ratio = self.context.pop('arm_kp_ratio', None)
        if ratio is not None:
            scale_position_actuator_kp(self.runtime.model,
                self.runtime.robot_metadata['joint_groups']['right'], 1. / ratio)
            self.summary.setdefault("control_transitions", []).append({
                "stage": stage,
                "from_arm_position_kp_scale": float(self.settings["extract_arm_kp_scale"]),
                "to_arm_position_kp_scale": float(self.settings["door_arm_kp_scale"]),
                "ratio": 1. / float(ratio),
            })

    def close(self):
        """释放 solver 和临时控制设置；不改机器人或物体的实际位姿。"""
        self.restore_control()
        sessions = [self.opened] + [row['session'] for row in self.context.values()
                                   if isinstance(row, dict) and 'session' in row]
        for session in {id(s): s for s in sessions}.values():
            _release_finished_planner(session)

    def stage_stop_kwargs(self, name):
        return {'stop_after': self._stop_after} if getattr(self, '_stop_selection', None) == name else {}

    def run(self, *, selection='all', preparation='prefix', stop_after=None):
        validate_stage_stop(selection, stop_after)
        self._stop_selection = selection if stop_after else None
        self._stop_after = stop_after
        plan = resolve_subtask(selection, policy=self.policy)
        steps = plan.steps(preparation)
        if self._run_index and preparation == 'prefix':
            raise ValueError('prefix 只能从新会话开始；同会话继续操作请用 current')
        # 当前记录格式每个既有阶段有固定路径，不能覆盖已执行的阶段证据。
        previous = {r['step_id'] for r in self.summary['steps']}
        if previous.intersection(steps):
            raise ValueError('所选步骤已有运行记录；请新建会话，避免覆盖原证据')
        self._run_index += 1
        summary, runtime, output = self.summary, self.runtime, self.output
        begin = len(summary['steps'])
        summary.update(status='running', requested_scope=f'{selection}/{stop_after}' if stop_after else selection,
                       stop_after=stop_after,
                       preparation=preparation, task_success=None,
                       operation_success=False, goal_success=False if selection == 'all' else None,
                       prefix_steps=list(plan.prefix_steps) if preparation == 'prefix' else [],
                       target_steps=list(plan.target_steps))
        summary.pop('failure', None)
        summary.pop('failure_subtask', None)
        started = perf_counter()
        with motion_configuration(self.settings['motion']):
            flow = run_microwave_task1_flow(CallbackOperationBackend(
                observe=self.observe, perform=self.perform, evidence_mode='mujoco'),
                policy=self.policy, selection=selection, preparation=preparation)
        summary['shared_flow'] = flow.to_dict()
        rows = summary['steps'][begin:]
        for row in rows:
            row['role'] = 'target' if row['step_id'] in plan.target_steps else 'prefix'
        summary['timing'] = {
            'wall_s': perf_counter() - started,
            'prefix_wall_s': sum(row['wall_time_s'] for row in rows if row['role'] == 'prefix'),
            'target_wall_s': sum(row['wall_time_s'] for row in rows if row['role'] == 'target'),
            'prefix_sim_s': sum(row['sim_end_s']-row['sim_start_s'] for row in rows if row['role'] == 'prefix'),
            'target_sim_s': sum(row['sim_end_s']-row['sim_start_s'] for row in rows if row['role'] == 'target'),
            'source': 'perf_counter adapter wall; MuJoCo actual sim time (not execution wall)',
            'planning_wall_s': None,
            'planning_note': '原门阶段 planning.timing 有独立耗时；其他操作未全量记录，不以总墙钟替代',
        }
        if not flow.success:
            self.restore_control()
        summary['actual_final'] = asdict(self.observe())
        summary['final_state_id'] = summary['actual_final']['state_id']
        summary['operation_success'] = bool(flow.success)
        summary['status'] = ('partial' if stop_after else 'completed') if flow.success else 'failed'
        if stop_after:
            save_state(runtime.model, runtime.data, output / 'debug_stop_state.npz')
        if not flow.success:
            failed = rows[-1]
            summary.update(failure_subtask=flow.failure_stage, failure={
                'step_id': flow.failure_stage, 'role': failed['role'],
                **failed.get('failure', {'code': flow.failure_code, 'message': '操作检查未通过'})})
            save_state(runtime.model, runtime.data, output / 'failure_state.npz')
        elif selection == 'all':
            final = runtime.query(object_names=('bowl',), robot_groups=('right',))
            door_angle = float(runtime.data.qpos[runtime.model.joint('microwave_door_joint').qposadr[0]])
            knob_angle = float(runtime.data.qpos[runtime.model.joint('knob_0_joint').qposadr[0]])
            summary['final_checks'] = evaluate_final(door_angle, knob_angle,
                self.context['knob_plan']['joint_target_angle_rad'],
                self.context['placed']['physical_table_place_success'], final['simulation_attachments'])
            summary['goal_success'] = all(summary['final_checks'][key] for key in (
                'door_closed', 'knob_at_target', 'bowl_on_table', 'no_simulation_attachment'))
            summary['task_success'] = summary['goal_success']
            summary['push_motion_audit'] = audit_push_paths(output)
            summary['status'] = 'completed' if summary['goal_success'] else 'failed'
            save_state(runtime.model, runtime.data, output / 'final_state.npz')
        if selection == 'all' and not flow.success:
            summary['task_success'] = False
        # 每次调用保留独立摘要；内部报告仍使用原路径。
        reference = f'calls/{self._run_index:02d}_{selection}/subtask.json'
        summary['call_report_ref'] = reference
        _write(output / reference, summary)
        _write(output / 'summary.json', summary)
        return json.loads(json.dumps(summary))


@configured_motion('config', override_argument='collision_mode')
def run_microwave_task1_physical(output_dir: Path, *, config: Path,
                                 record_video=False, seed=None, randomize=None,
                                 collision_mode=None, selection='all', preparation='prefix', stop_after=None) -> dict:
    # 非法选择在任何场景构建之前拒绝。
    resolve_subtask(selection).steps(preparation)
    validate_stage_stop(selection, stop_after)
    session = MicrowaveSimulationSession(output_dir, config=config, record_video=record_video,
        seed=seed, randomize=randomize, collision_mode=collision_mode)
    try:
        return session.run(selection=selection, preparation=preparation, stop_after=stop_after)
    finally:
        session.close()


__all__ = ['run_microwave_task1_physical', 'MicrowaveSimulationSession', 'audit_push_paths']
