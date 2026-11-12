"""把门 Skill 的已有规划轨迹送入 MuJoCo，检查门是否由真实接触带动。

此模块只消费规划记录和该记录保存的场景。执行过程中绝不写门关节
qpos/ctrl；门角变化只能来自机器人与门的 MuJoCo 接触。该检查是
独立 Skill 的物理验收入口，不把旧版运动学回放当作成功证据。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.feedback.state import state_id
from gene_agent.simulation import MujocoRuntime, set_joint_positions
from gene_agent.skills.library.door.sequence import door_angle_check


def _is_descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def classify_door_contact(
    robot_geom_name: str, door_geom_name: str,
    pad_geom_names: set[str], allowed_door_geom_names: tuple[str, ...] | frozenset[str],
    allowed_gripper_geom_names: set[str] | frozenset[str],
) -> str:
    """按阶段明确允许的几何接触分类；指根只在调用方授权时放行。

    门板接触可以由指垫或指定指根带动。其他机械臂部位、错误的门几何
    仍需作为诊断事实保留，不能因门角达标而一并视为正常接触。
    """
    if door_geom_name not in allowed_door_geom_names:
        return "forbidden"
    if robot_geom_name in pad_geom_names:
        return ("pad_handle" if door_geom_name == "microwave_handle_grip_collision"
                else "pad_target")
    if robot_geom_name in allowed_gripper_geom_names:
        return "gripper_target"
    return "forbidden"


class _DoorContactTrace:
    """在真实仿真步上统计接触，而不是从 keypose 反推接触。"""

    def __init__(self, model: mujoco.MjModel, robot: dict, group: str,
                 allowed_door_geom_names: tuple[str, ...] = (
                     "microwave_handle_grip_collision",),
                 allowed_gripper_geom_names: tuple[str, ...] = ()):
        self.arm_root = model.body(robot["group_root_bodies"][group]).id
        self.door_root = model.body("microwave_door").id
        self.pad_names = set(robot["grippers"][group]["pad_geom_names"])
        self.allowed_door_geom_names = frozenset(allowed_door_geom_names)
        self.allowed_gripper_geom_names = frozenset(allowed_gripper_geom_names)
        self.door_joint = model.joint("microwave_door_joint").qposadr[0]
        self.samples: list[dict] = []
        self.contact_pairs: dict[str, dict] = {}
        self.max_forbidden_depth_m = 0.0
        self.pad_handle_normal_impulse_ns = 0.0
        self.pad_target_normal_impulse_ns = 0.0

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, _sample: dict) -> None:
        step_pad = 0
        step_handle = 0
        step_other = 0
        step_allowed_target = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            first_body = int(model.geom_bodyid[first])
            second_body = int(model.geom_bodyid[second])
            if (_is_descendant(model, first_body, self.arm_root)
                    and _is_descendant(model, second_body, self.door_root)):
                arm_geom, door_geom = first, second
            elif (_is_descendant(model, second_body, self.arm_root)
                  and _is_descendant(model, first_body, self.door_root)):
                arm_geom, door_geom = second, first
            else:
                continue
            arm_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, arm_geom)
            door_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, door_geom)
            classification = classify_door_contact(
                arm_name, door_name, self.pad_names,
                self.allowed_door_geom_names, self.allowed_gripper_geom_names,
            )
            allowed = classification != "forbidden"
            key = f"{arm_name} ↔ {door_name}"
            row = self.contact_pairs.setdefault(key, {
                "classification": classification,
                "step_count": 0, "max_penetration_depth_m": 0.0,
                "normal_impulse_ns": 0.0,
                "first_contact_time_s": float(data.time),
                "first_contact_world_m": contact.pos.tolist(),
            })
            wrench = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, index, wrench)
            depth = max(0.0, -float(contact.dist))
            impulse = max(0.0, float(wrench[0])) * float(model.opt.timestep)
            row["step_count"] += 1
            row["max_penetration_depth_m"] = max(row["max_penetration_depth_m"], depth)
            row["normal_impulse_ns"] += impulse
            row["last_contact_time_s"] = float(data.time)
            row["last_contact_world_m"] = contact.pos.tolist()
            if allowed:
                step_allowed_target += 1
                if classification in {"pad_target", "pad_handle"}:
                    step_pad += 1
                    self.pad_target_normal_impulse_ns += impulse
                    if classification == "pad_handle":
                        step_handle += 1
                        self.pad_handle_normal_impulse_ns += impulse
            else:
                step_other += 1
                self.max_forbidden_depth_m = max(self.max_forbidden_depth_m, depth)
        # 保留真实角度轨迹，方便诊断「指垫接触了但门没动」与「门被别处撞开」。
        self.samples.append({
            "sim_time_s": float(data.time),
            "door_angle_rad": float(data.qpos[self.door_joint]),
            "pad_target_contact_count": step_pad,
            "allowed_target_contact_count": step_allowed_target,
            "pad_handle_contact_count": step_handle,
            "forbidden_door_contact_count": step_other,
        })

    def summary(self) -> dict:
        return {
            "pad_handle_normal_impulse_ns": self.pad_handle_normal_impulse_ns,
            "pad_target_normal_impulse_ns": self.pad_target_normal_impulse_ns,
            "allowed_door_geom_names": sorted(self.allowed_door_geom_names),
            "allowed_gripper_geom_names": sorted(self.allowed_gripper_geom_names),
            "max_forbidden_depth_m": self.max_forbidden_depth_m,
            "contact_pairs": self.contact_pairs,
            "door_angle_samples": self.samples,
        }


def load_door_runtime(planned_run: str | Path, *,
                      gravity_compensation: bool = True) -> MujocoRuntime:
    """仅在独立 Skill 的起点 reset；完整任务只创建一次 Runtime。"""
    source = Path(planned_run).resolve()
    reports = list(source.glob("skill_*_door_result.json"))
    if len(reports) != 1:
        raise ValueError("规划 run 必须恰好包含一个门 Skill 结果")
    plan = json.loads(reports[0].read_text(encoding="utf-8"))
    task = json.loads((source / "task_config.json").read_text(encoding="utf-8"))
    robot = json.loads((source / "source_scene/robot/robot_metadata.json").read_text(
        encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(source / plan["model_ref"]))
    data = mujoco.MjData(model)
    group = task["robot_group"]
    if gravity_compensation:
        root_body = model.body(robot["group_root_bodies"][group]).id
        for body in range(model.nbody):
            if _is_descendant(model, body, root_body):
                model.body_gravcomp[body] = 1.0
    set_joint_positions(model, data, robot["home_joint_positions"])
    initial = {task["joint_name"]: float(plan["joint_start_angle_rad"])}
    start_arm = task.get("start_robot_joint_positions_rad")
    if start_arm is not None:
        names = robot["joint_groups"][group]
        if len(start_arm) != len(names):
            raise ValueError("规划记录中的初始机械臂关节数量不匹配")
        initial.update(zip(names, map(float, start_arm)))
    set_joint_positions(model, data, initial)
    return MujocoRuntime(model, data, {
        "robot": robot, "world_frame": robot["world_frame"],
        "xml_path": str(source / plan["model_ref"]),
        "source_door_run_ref": str(source.parent),
        "seed": int(task["seed"]),
        "gravity_compensated_groups": [group] if gravity_compensation else [],
        "simulation_attachment_pairs": dict(
            task.get("simulation_attachment_pairs") or {}),
    })


def execute_saved_door_plan(
    planned_run: str | Path,
    output_dir: str | Path,
    *,
    gravity_compensation: bool = True,
    angle_tolerance_rad: float = 0.05,
    forbidden_depth_limit_m: float = 0.003,
    close_width_override_m: float | None = None,
    runtime: MujocoRuntime | None = None,
    visual_recorder=None,
) -> dict:
    """从保存的场景 reset 一次，实际驱动机器人并测量门的接触转角。

    `planned_run` 必须是 `run_door_planning` 的成功记录。这里允许
    单个 Skill 的显式初始门角；完整 Task 将另行从上个 Skill 的真实
    结束状态续接，不能调用这个 reset 入口来冒充连续执行。
    """
    source = Path(planned_run).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = list(source.glob("skill_*_door_result.json"))
    if len(reports) != 1:
        raise ValueError("规划 run 必须恰好包含一个门 Skill 结果")
    plan_path = reports[0]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("success") or plan.get("source_format") != "kinematic_plan":
        raise ValueError("只能物理执行完整成功的门规划记录")
    task = json.loads((source / "task_config.json").read_text(encoding="utf-8"))
    robot = json.loads((source / "source_scene/robot/robot_metadata.json").read_text(
        encoding="utf-8"))
    scene_path = source / plan["model_ref"]
    group = task["robot_group"]
    reused_runtime = runtime is not None
    if runtime is None:
        runtime = load_door_runtime(source, gravity_compensation=gravity_compensation)
        model, data = runtime.model, runtime.data
    else:
        # 连续 Skill 必须沿用同一个真实积分状态，不能从规划记录 reset。
        model, data = runtime.model, runtime.data
        if tuple(runtime.robot_metadata["joint_groups"][group]) != tuple(robot["joint_groups"][group]):
            raise ValueError("复用的 Runtime 与门规划机器人关节组不一致")
        actual_start = float(data.qpos[model.joint(task["joint_name"]).qposadr[0]])
        if abs(actual_start - float(plan["joint_start_angle_rad"])) > angle_tolerance_rad:
            raise ValueError("复用 Runtime 的实际门角与规划起点不一致，必须重新规划")
    result = {
        "schema_version": 1, "skill": plan["skill"], "source_format": "mujoco_physical",
        "planned_run": str(source), "planned_report_sha256": hashlib.sha256(
            plan_path.read_bytes()).hexdigest(),
        "scene_ref": str(scene_path), "robot_group": group,
        "gravity_compensation": gravity_compensation,
        "reused_runtime_without_reset": reused_runtime,
        "actual_run_start_angle_rad": float(data.qpos[model.joint(task["joint_name"]).qposadr[0]]),
        "close_width_override_m": close_width_override_m,
        "joint_name": task["joint_name"],
        "joint_start_angle_rad": float(plan["joint_start_angle_rad"]),
        "joint_target_angle_rad": float(plan["joint_target_angle_rad"]),
        "stages": [], "success": False,
    }
    joint_adr = model.joint(task["joint_name"]).qposadr[0]
    trace = _DoorContactTrace(model, robot, group)
    for stage in plan["stages"]:
        if stage["status"] != "planned":
            break
        start_angle = float(data.qpos[joint_adr])
        before_count = len(trace.samples)
        record_stage = f"handle_pull_{stage['name']}"
        if visual_recorder is not None:
            visual_recorder.begin_stage(record_stage, model, data)
        def observe_step(observed_model, observed_data, sample):
            trace.observe(observed_model, observed_data, sample)
            if visual_recorder is not None:
                visual_recorder.observe(record_stage, observed_model, observed_data, sample)
        if stage["operation"] == "gripper_command":
            width_m = (close_width_override_m if stage["name"] == "close"
                       and close_width_override_m is not None
                       else float(stage["gripper_width_m"]))
            execution = runtime.execute_gripper_command(
                group=group, width_m=width_m,
                duration_s=0.4, planned_state_id=state_id(model, data),
                step_observer=observe_step,
                replay_object_names=("microwave_door", "bowl"),
            )
        else:
            trajectory = (stage.get("result") or {}).get("trajectory")
            if trajectory is None:
                raise ValueError(f"阶段 {stage['name']} 缺少规划轨迹")
            execution = runtime.execute_joint_trajectory(
                group=group, joint_names=trajectory["joint_names"],
                times_s=trajectory["times_s"], positions_rad=trajectory["positions_rad"],
                planned_state_id=state_id(model, data),
                step_observer=observe_step,
                replay_object_names=("microwave_door", "bowl"),
            )
        if visual_recorder is not None:
            visual_recorder.end_stage(record_stage, model, data)
        row = {
            "name": stage["name"], "operation": stage["operation"],
            "planned_target_angle_rad": stage["end_angle_rad"],
            "actual_start_angle_rad": start_angle,
            "actual_end_angle_rad": float(data.qpos[joint_adr]),
            "execution_success": bool(execution["success"]),
            "execution_failure_type": execution["failure_type"],
            "execution_ref": f"stages/{len(result['stages']):02d}_{stage['name']}.json",
            "contact_sample_range": [before_count, len(trace.samples)],
        }
        stage_path = output / row["execution_ref"]
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(json.dumps(execution, ensure_ascii=False, allow_nan=False)
                              + "\n", encoding="utf-8")
        result["stages"].append(row)
        if not execution["success"]:
            break
    result["actual_final_angle_rad"] = float(data.qpos[joint_adr])
    result["contact"] = trace.summary()
    angle_check = door_angle_check(
        result["actual_final_angle_rad"], target_rad=result["joint_target_angle_rad"],
        tolerance_rad=angle_tolerance_rad, source="mujoco_gt",
    )
    result["angle_postcondition"] = angle_check.to_dict()
    target_error = angle_check.evidence["error_rad"]
    result["angle_error_rad"] = target_error
    result["physical_door_motion_rad"] = abs(
        result["actual_final_angle_rad"] - result["joint_start_angle_rad"])
    result["success"] = bool(
        len(result["stages"]) == len(plan["stages"])
        and all(row["execution_success"] for row in result["stages"])
        and angle_check.passed
        and trace.pad_handle_normal_impulse_ns > 0.01
        and trace.max_forbidden_depth_m <= forbidden_depth_limit_m
    )
    result["failure_reasons"] = []
    if len(result["stages"]) != len(plan["stages"]):
        result["failure_reasons"].append("execution_stopped_before_last_stage")
    if target_error > angle_tolerance_rad:
        result["failure_reasons"].append("physical_door_angle_not_reached")
    if trace.pad_handle_normal_impulse_ns <= 0.01:
        result["failure_reasons"].append("no_effective_pad_handle_contact")
    if trace.max_forbidden_depth_m > forbidden_depth_limit_m:
        result["failure_reasons"].append("forbidden_robot_door_penetration")
    (output / "physical_door_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["load_door_runtime", "execute_saved_door_plan"]
