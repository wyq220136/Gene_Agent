"""微波炉推门的同步规划、执行与接触验收；消费同一个 MuJoCo Runtime。"""
from __future__ import annotations
from gene_agent.motion.curobo.scene import planner_scene_options

import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.planning.bowl_search import _contacts
from gene_agent.feedback.state import save_state, state_id
from gene_agent.motion.curobo.core import SynchronousPlanner, load_realman_motion_config
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.robots.gripper import measure_gripper_width


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _plan_with_retries(plan_once: Callable[[], dict], *, max_attempts: int) -> tuple[dict, list[dict]]:
    """只重复搜索同一个目标，不改变碰撞策略或 TCP；返回真实选用的规划。"""
    if max_attempts < 1:
        raise ValueError("max_attempts 必须至少为 1")
    attempts = []
    for _ in range(max_attempts):
        plan = plan_once()
        attempts.append({"success": bool(plan["success"]),
                         "failure_category": plan.get("failure_category"),
                         "duration_s": plan.get("duration_s")})
        # 起点已碰撞时，相同状态和相同碰撞场景重复搜索不会改变前提。
        if plan["success"] or plan.get("failure_category") == "start_collision":
            break
    return plan, attempts


def run_push_door(runtime, stages, *, group: str, output_dir, visual_recorder=None,
                  angle_tolerance_deg: float = 5., max_plan_attempts: int = 4) -> dict:
    """预接触保留门碰撞；推门段仅在规划里排除活动门，MuJoCo 实体接触照常启用。

    每段只从上段实际结束关节状态重新规划。规划失败或实测接触不符即停止，
    不把后续目标位姿回放为已执行结果。
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model, data, robot = runtime.model, runtime.data, runtime.robot_metadata
    names = robot["joint_groups"][group]
    pads = set(robot["grippers"][group]["pad_geom_names"])
    door_adr = model.joint("microwave_door_joint").qposadr[0]
    report = {"schema_version": 1, "success": False, "failure_stage": None,
              "group": group, "stages": []}
    for stage in stages:
        name = stage.name
        moving = stage.target_angle_deg is not None
        retracting = name.startswith("retract_")
        door_contact_allowed = moving or retracting
        before = float(np.degrees(data.qpos[door_adr]))
        snapshot = export_collision_scene(
            model, data, robot, group, output / "collision" / name,
            moving_body_names=("microwave_door",) if door_contact_allowed else (),
        )
        config = load_realman_motion_config(
            group, robot, gripper_snapshot=snapshot.get("gripper_snapshot"),
        )
        planner = SynchronousPlanner(
            **config, **planner_scene_options(snapshot),
        )
        q = [float(data.qpos[model.joint(joint).qposadr[0]]) for joint in names]
        linear = stage.operation.value == "linear_tcp"
        method = planner.plan_linear_tcp if linear else planner.plan
        attempt_number = 0

        def plan_once() -> dict:
            nonlocal attempt_number
            attempt_number += 1
            attempt = method(names, q, stage.tcp_pose_world,
                             target_frame="world_tcp")
            _write(output / "plans" / f"{name}_attempt_{attempt_number:02d}.json",
                   attempt)
            return attempt

        plan, planning_attempts = _plan_with_retries(
            plan_once,
            max_attempts=max_plan_attempts,
        )
        for index, attempt in enumerate(planning_attempts, start=1):
            attempt["plan_ref"] = f"plans/{name}_attempt_{index:02d}.json"
        _write(output / "plans" / f"{name}.json", plan)
        tcp_site = data.site(robot["tcp_sites"][group])
        start_tcp = tcp_site.xpos.copy()
        row = {"name": name, "operation": stage.operation.value,
               "start_tcp_world_m": start_tcp.tolist(),
               "target_tcp_world_m": stage.tcp_pose_world[:3, 3].tolist(),
               "tcp_target_displacement_m": float(np.linalg.norm(
                   stage.tcp_pose_world[:3, 3] - start_tcp)),
               "start_angle_deg": before, "target_angle_deg": stage.target_angle_deg,
               "plan_success": bool(plan["success"]),
               "planning_attempts": planning_attempts,
               "planning_failure": plan.get("failure_category"),
               "linear_path_metrics": plan.get("linear_path_metrics"),
               "collision_snapshot_ref": f"collision/{name}/collision_snapshot.json",
               "allowed_contact_geoms": sorted(pads) if door_contact_allowed else []}
        report["stages"].append(row)
        if not plan["success"]:
            row["failure_category"] = plan.get("failure_category") or "planning_failed"
            report["failure_stage"] = name
            break
        contacts: dict[str, float] = {}
        impulses: dict[str, float] = {}
        peak_forces: dict[str, float] = {}
        first_forbidden_panel_contact: dict = {}
        peak_forbidden_panel_contact: dict = {}
        peak_actuator_load: dict[str, float] = {}
        fixed_collisions: dict[str, float] = {}
        fixed_contact_pairs: dict[str, float] = {}
        arm_body = model.body(robot["group_root_bodies"][group]).id
        microwave_body = model.body("microwave").id

        def observe(mm, dd, sample):
            if visual_recorder is not None:
                visual_recorder.observe(name, mm, dd, sample)
            for actuator_id in range(mm.nu):
                actuator_name = mm.actuator(actuator_id).name
                if actuator_name.startswith(f"{group}_"):
                    peak_actuator_load[actuator_name] = max(
                        peak_actuator_load.get(actuator_name, 0.),
                        abs(float(dd.actuator_force[actuator_id])),
                    )
            for contact_index, contact in enumerate(dd.contact):
                first = mm.geom(contact.geom1).name
                second = mm.geom(contact.geom2).name
                if "microwave_door_panel_collision" in (first, second):
                    arm_geom = second if first == "microwave_door_panel_collision" else first
                    if arm_geom.startswith(f"{group}_"):
                        contacts[arm_geom] = max(contacts.get(arm_geom, 0.),
                                                 max(0., -float(contact.dist)))
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(mm, dd, contact_index, wrench)
                        force = max(0., float(wrench[0]))
                        impulses[arm_geom] = impulses.get(arm_geom, 0.) + force * mm.opt.timestep
                        peak_forces[arm_geom] = max(peak_forces.get(arm_geom, 0.), force)
                        if arm_geom not in pads:
                            # 仅靠每段最大受力无法判断是进入接触点时就碰到
                            # 指根，还是门转动后才扫到指根。记录首次事件与
                            # 最大力事件，便于按真实时间和位置截取失败画面。
                            event = {
                                "time_s": float(sample["time_s"]),
                                "robot_geom": arm_geom,
                                "door_geom": "microwave_door_panel_collision",
                                "contact_position_world_m": contact.pos.tolist(),
                                "signed_distance_m": float(contact.dist),
                                "normal_force_n": force,
                                "actual_door_angle_deg": float(np.degrees(dd.qpos[door_adr])),
                                "actual_tcp_position_world_m": dd.site(
                                    robot["tcp_sites"][group]).xpos.tolist(),
                            }
                            if not first_forbidden_panel_contact:
                                first_forbidden_panel_contact.update(event)
                            if (not peak_forbidden_panel_contact
                                    or force > peak_forbidden_panel_contact["normal_force_n"]):
                                peak_forbidden_panel_contact.update(event)
            fixed = _contacts(mm, dd, first_root=arm_body,
                              second_root=microwave_body,
                              allowed_second_geoms={"microwave_door_panel_collision"})
            if fixed["count"]:
                fixed_collisions["max_depth_m"] = max(
                    fixed_collisions.get("max_depth_m", 0.),
                    fixed["max_penetration_depth_m"],
                )
                for pair in fixed["pairs"]:
                    label = f"{pair['first_geom']} ↔ {pair['second_geom']}"
                    fixed_contact_pairs[label] = max(
                        fixed_contact_pairs.get(label, 0.),
                        pair["penetration_depth_m"],
                    )

        positions = plan["positions_rad"]
        times = (np.arange(len(positions)) * plan["interpolation_dt_s"]).tolist()
        if visual_recorder is not None:
            visual_recorder.begin_stage(name, model, data)
        execution = runtime.execute_joint_trajectory(
            group=group, joint_names=names, times_s=times,
            positions_rad=positions, planned_state_id=state_id(model, data),
            timeout_s=max(times[-1] + 2., 1.), endpoint_tolerance_rad=.06,
            settle_timeout_s=1.5, step_observer=observe,
        )
        if visual_recorder is not None:
            visual_recorder.end_stage(name, model, data)
        _write(output / "executions" / f"{name}.json", execution)
        save_state(model, data, output / "states" / f"{name}.npz")
        after = float(np.degrees(data.qpos[door_adr]))
        # MuJoCo 执行后的实际关节状态可能偏离规划终点。立即用本段的
        # cuRobo 碰撞场景复核，避免把下一段一开始才报出的起点碰撞
        # 错误归因到下一段目标；活动门仍按本段接触策略处理。
        q_after = [float(data.qpos[model.joint(joint).qposadr[0]])
                   for joint in planner.joint_names]
        post_execution_collision = planner._start_configuration_check(q_after)
        if "collision_diagnostic" in post_execution_collision:
            post_execution_collision["collision_diagnostic"][
                "checked_configuration"] = "post_execution_state"
        forbidden = sorted(set(contacts) - pads)
        forbidden_handle_contact = any(
            "microwave_handle_grip_collision" in pair
            for pair in fixed_contact_pairs
        )
        row.update({"execution_success": bool(execution["success"]),
                    "execution_failure": execution.get("failure_type"),
                    "actual_angle_deg": after,
                    "robot_door_contact_depth_m": contacts,
                    "robot_door_normal_impulse_ns": impulses,
                    "robot_door_peak_normal_force_n": peak_forces,
                    "peak_actuator_force": peak_actuator_load,
                    "final_gripper_width_m": measure_gripper_width(
                        model, data, robot, group)["width_m"],
                    "forbidden_robot_door_contacts": forbidden,
                    "first_forbidden_panel_contact": (
                        first_forbidden_panel_contact or None),
                    "peak_forbidden_panel_contact": (
                        peak_forbidden_panel_contact or None),
                    "fixed_microwave_contact_depth_m": fixed_collisions,
                    "fixed_microwave_contact_pairs_m": fixed_contact_pairs,
                    "forbidden_handle_contact": forbidden_handle_contact,
                    "post_execution_planner_collision_check": post_execution_collision})
        if moving:
            row["success"] = bool(execution["success"]
                                  and abs(after - stage.target_angle_deg) <= angle_tolerance_deg
                                  and (any(key in pads for key in contacts)
                                       or (abs(after - before) <= 2.
                                           and abs(after - stage.target_angle_deg) <= angle_tolerance_deg))
                                  and not forbidden
                                  and not forbidden_handle_contact
                                  and max(fixed_collisions.values(), default=0.) <= .003
                                  and post_execution_collision["valid"])
        elif retracting:
            row["success"] = bool(execution["success"] and abs(after - before) <= 2.
                                  and not forbidden
                                  and not forbidden_handle_contact
                                  and max(fixed_collisions.values(), default=0.) <= .003
                                  and post_execution_collision["valid"])
        else:
            row["success"] = bool(execution["success"] and abs(after - before) <= 2.
                                  and not contacts
                                  and not forbidden_handle_contact
                                  and max(fixed_collisions.values(), default=0.) <= .003
                                  and post_execution_collision["valid"])
        if not row["success"]:
            if not execution["success"]:
                row["failure_category"] = execution.get("failure_type") or "execution_failed"
            elif not post_execution_collision["valid"]:
                row["failure_category"] = "post_execution_collision"
            elif forbidden:
                row["failure_category"] = "forbidden_robot_door_contact"
            elif forbidden_handle_contact:
                row["failure_category"] = "forbidden_handle_contact"
            elif max(fixed_collisions.values(), default=0.) > .003:
                row["failure_category"] = "fixed_microwave_collision"
            elif moving:
                row["failure_category"] = "door_angle_or_contact_mismatch"
            else:
                row["failure_category"] = "unexpected_door_motion_or_contact"
            report["failure_stage"] = name
            break
    else:
        report["success"] = True
    report["final_angle_deg"] = float(np.degrees(data.qpos[door_adr]))
    _write(output / "result.json", report)
    return report


__all__ = ["run_push_door"]
