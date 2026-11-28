"""仿真策略的现场规划适配；预测只用于筛选，不作为已执行或夹持证据。"""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import mujoco
import numpy as np
from gene_agent.feedback.runtime_timing import timed_runtime_call
from gene_agent.motion.model import (CollisionPolicy, JointState, MotionOperation,
    MotionRequest, PoseTarget)
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import (
    FrozenGeometry, validate_insertion_plan, predict_released_insert_state,
    raise_retreat_for_released_pad_clearance, released_ceiling_clearance_snapshot)

ROOT = Path(__file__).resolve().parents[3]
RETREAT_FRACTIONS = (0., .25, .5, .75)

def strategy_settings():
    return json.loads((ROOT / "data/task_1/realworld/operation_strategy.json").read_text())

def write_report(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

def frozen_geometry(scene):
    """仅把已感知/SDK记录装入几何模型；绝不调用 mj_step。"""
    model = mujoco.MjModel.from_xml_path(str(scene["scene_xml_path"]))
    data = mujoco.MjData(model)
    qpos = np.asarray(scene["snapshot"]["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("当前观测 qpos 与冻结场景不一致")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return FrozenGeometry(model, data, scene["robot_metadata"])

def predicted_scene(scene, frozen, output):
    """显式标记松爪预测，不允许作为任何真机成功证据。"""
    from gene_agent.motion.curobo.scene import export_collision_scene
    data, robot = frozen.data, frozen.robot_metadata
    group = scene["snapshot"]["active_group"]
    snapshot = export_collision_scene(frozen.model, data, robot, group, Path(output))
    snapshot["qpos_snapshot"] = data.qpos.tolist()
    source = dict(scene["provenance"])
    source.update(state_source="microwave_plan_only",
        prediction_origin_sha256=source.get("prediction_origin_sha256",
            scene["snapshot"]["snapshot_sha256"]),
        prediction_step=int(source.get("prediction_step", 0)) + 1,
        collision_snapshot_sha256=snapshot["snapshot_sha256"],
        physics_stepped=False)
    source["joint_positions_by_group_rad"] = {
        g: [float(data.qpos[frozen.model.joint(n).qposadr[0]]) for n in names]
        for g, names in robot["joint_groups"].items()}
    source["gripper_widths_m"] = {
        g: float(sum(data.qpos[frozen.model.joint(n).qposadr[0]]
                     for n in grip["joint_names"]))
        for g, grip in robot["grippers"].items()}
    aligned = dict(source["aligned_object_poses_world"])
    bowl = np.eye(4); bowl[:3, :3] = data.body("bowl").xmat.reshape(3, 3)
    bowl[:3, 3] = data.body("bowl").xpos
    aligned["Bowl/Bowl_0"] = bowl.tolist()
    source["aligned_object_poses_world"] = aligned
    return {**scene, "snapshot": snapshot, "provenance": source}

def held_orientation(tcp_bowl):
    up_frame = np.eye(4)
    up_frame[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    return {"T_tcp_frame": (np.asarray(tcp_bowl) @ up_frame).tolist(),
            "up_world": [0., 0., 1.], "non_terminal_weight": 1.}

@timed_runtime_call("planning", "aligned_strategy", name_parameter="name")
def plan_stage(scene, name, target, *, planner_pool, output, seed=123,
        start=None, spheres=None, tcp_bowl=None, linear=False, orientation=None,
        max_tilt_deg=15., preferred=None, max_attempts=1, optimizer_clearance=None,
        geometry_check=None, joint_target=None):
    """同一运动合同筛选、复核与执行；相同候选轨迹仅在当前场景复核通过后复用。"""
    output = Path(output); started = perf_counter()
    context = prepare_plan_only_context(scene, group="right", planner_pool=planner_pool,
        seed=seed, attachment_spheres_tool=spheres,
        path_orientation_constraint=orientation,
        optimizer_collision_activation_distance=optimizer_clearance)
    start = context["start"] if start is None else start
    operation = MotionOperation.LINEAR_TCP if linear else MotionOperation.PLAN
    request = MotionRequest(f"{name}:observed_{scene['provenance']['source_frame_number']}",
        operation, "right", start, PoseTarget("world_tcp", target),
        context["scene"], CollisionPolicy(), seed=seed, joint_target=joint_target)
    adapter = context["planner"]; raw_planner = adapter.planner
    cached = None
    if preferred is not None:
        from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
        proxy = SimpleNamespace(preferred_trajectory_by_stage={name: preferred},
            endpoint_tolerance_rad=.03, linear_tcp_stages={name} if linear else set())
        cached = CuroboRuntimeBridge._preferred_plan(proxy, name, raw_planner, {
            "joint_names": list(start.joint_names), "joint_positions_rad": list(start.positions_rad),
            "T_world_tcp": raw_planner.fk(start.joint_names, start.positions_rad)}, np.asarray(target))
    if joint_target is not None:
        normalized = adapter.plan(request)
        raw = None
    elif cached is not None and cached["success"]:
        raw = cached
    else:
        method = (raw_planner.plan_linear_tcp if linear else
                  raw_planner.plan_with_orientation if orientation else raw_planner.plan)
        raw = method(start.joint_names, start.positions_rad, np.asarray(target),
            target_frame="world_tcp", max_attempts=max_attempts)
        if cached is not None:
            raw["rejected_cached_trajectory_validation"] = cached["preferred_trajectory_validation"]
    if raw is not None and raw.get("success"):
        if geometry_check is not None:
            check = geometry_check(raw)
        else:
            # 这里的接触只是重建场景的预测检查，永不返回实测夹持结论。
            check = validate_insertion_plan(frozen_geometry(scene),
                "insert" if tcp_bowl is not None else "retreat_1", raw,
                np.eye(4) if tcp_bowl is None else np.asarray(tcp_bowl),
                max_tilt_deg=max_tilt_deg if tcp_bowl is not None else None)
        raw["reconstructed_geometry_postcheck"] = check
        if not check["success"]:
            raw.update(success=False,
                failure_category=("constraint_violation" if check.get("failure_category")
                    == "task_constraint_postcheck_rejected" else "collision_postcheck_rejected"),
                status="reconstructed_geometry_rejected", message=check.get("reason_code"))
    result = normalized if raw is None else adapter._record((request,), (adapter._convert(request, raw),))[0]
    stage = SkillStage(name, "planned" if result.success else "failed",
        request.target, request, result, {"strategy_source": "Gene_Agent_new_simulation",
        "trajectory_reused": bool(cached and cached.get("success")),
        "failure_domain": ("geometry_postcheck" if raw is not None and
            raw.get("status")=="reconstructed_geometry_rejected" else
            "curobo" if not result.success else None),
        "geometry_evidence": "prediction_not_measured_contact"})
    report = {"success": result.success, "stage": stage.to_dict(),
        "stage_collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "planner_cache_hit": context["planner_cache_hit"],
        "planning_wall_s": perf_counter()-started}
    write_report(output / "stage.json", {k:v for k,v in report.items() if k!="stage"} |
        {"stage_name": name, "target_world_tcp": np.asarray(target).tolist(),
         "failure_domain": stage.details["failure_domain"]})
    return {**report, "stage": stage}

def plan_sequence(scene, sequence, *, planner_pool, output, seed=123, spheres=None,
                  tcp_bowl=None, max_tilt_deg=15., held=False, geometry_check=None):
    start = None; stages = {}; rows = []
    for name, target in sequence:
        row = plan_stage(scene, name, target, planner_pool=planner_pool,
            output=Path(output)/name, seed=seed, start=start, spheres=spheres,
            tcp_bowl=tcp_bowl, max_tilt_deg=max_tilt_deg,
            linear=name in {"insert", "retreat_lift", "pre_extract_2", "grasp",
                            "post_extract_1", "post_extract_2", "place", "retreat"},
            orientation=held_orientation(tcp_bowl) if held and name in {"held_turn", "preplace"} else None,
            geometry_check=geometry_check)
        stages[name] = row["stage"]; rows.append({**row, "stage": row["stage"].to_dict()})
        if not row["success"]:
            return {"success": False, "failure_stage": name, "rows": rows, "stages": stages}
        start = row["stage"].result.goal_state
    return {"success": True, "rows": rows, "stages": stages}

def screen_insert(scene, attachment, families, *, planner_pool, output, seed=123):
    """只采纳首个通过持物、松爪包络和开爪撤离全序列的候选。"""
    from gene_agent.realworld.devices.grippers.realman.adapter import bounded_opening_width
    settings = strategy_settings(); rows = []
    frozen = frozen_geometry(scene)
    width_max = frozen.robot_metadata["grippers"]["right"]["width_range_m"][1]
    width = bounded_opening_width(width_max, width_max)
    from gene_agent.realworld.scene import observed_tcp_pose
    tcp_bowl = np.linalg.inv(observed_tcp_pose(scene)) @ np.asarray(
        scene["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"])
    for index, family in enumerate(families):
        folder = Path(output)/f"candidate_{index:03d}"
        sequence = ([("held_turn", family["held_turn"])] if "held_turn" in family else [])
        sequence += [(name, family["targets"][name]) for name in ("preplace", "insert")]
        held = plan_sequence(attachment, sequence, planner_pool=planner_pool,
            output=folder/"held", seed=seed, spheres=attachment["spheres_tool_m"],
            tcp_bowl=tcp_bowl, held=True)
        row = {"index": index, "held": held["rows"], "success": False}
        if not held["success"]:
            rows.append(row); continue
        trajectory = held["stages"]["insert"].result.trajectory.to_dict()
        released, envelope = predict_released_insert_state(frozen, trajectory,
            np.asarray(family["object_goal"]), width)
        row["release_envelope"] = envelope
        if released is None:
            rows.append(row); continue
        release_scene = predicted_scene(scene, released, folder/"release_scene")
        release_snapshot = released_ceiling_clearance_snapshot(
            release_scene["snapshot"], settings["retreat_clearance_margin_m"])
        release_scene["snapshot"] = release_snapshot
        release_scene["provenance"]["collision_snapshot_sha256"] = release_snapshot["snapshot_sha256"]
        retreat_rows = []
        for fraction in RETREAT_FRACTIONS:
            targets, evidence = raise_retreat_for_released_pad_clearance(
                released.model, released.data, released.robot_metadata,
                family["targets"], np.asarray(scene["provenance"]["aligned_object_poses_world"][
                    "Microwave/Microwave_0"])[:3, 1],
                clearance_margin_m=settings["retreat_clearance_margin_m"],
                minimum_lift_m=settings["retreat_minimum_lift_m"], outward_fraction=fraction)
            retreat = plan_sequence(release_scene, [(n, targets[n]) for n in
                ("retreat_lift", "retreat_1", "retreat_2")],
                planner_pool=planner_pool, output=folder/f"retreat_{fraction}", seed=seed)
            retreat_rows.append({"adjustment": evidence, "result": retreat["rows"]})
            if retreat["success"]:
                row.update(success=True, retreat_candidates=retreat_rows)
                rows.append(row)
                summary = [{"index": r["index"], "success": r["success"],
                    "held_stages": [{"name": s["stage"]["name"], "success": s["success"],
                        "duration_s": s.get("planning_wall_s")} for s in r["held"]],
                    "release_envelope_success": r.get("release_envelope",{}).get("success"),
                    "retreat_candidate_count": len(r.get("retreat_candidates",()))} for r in rows]
                write_report(Path(output)/"insert_sequence_screen.json",
                    {"success": True, "selected_candidate_index": index, "candidate_screen": summary})
                return {"success": True, "index": index, "family": family,
                    "held_stages": held["stages"], "retreat_targets": targets,
                    "retreat_evidence": evidence, "candidate_screen": summary}
        row["retreat_candidates"] = retreat_rows; rows.append(row)
    failed_stages = [s for r in rows for s in r["held"] if not s["success"]]
    failed_stages += [s for r in rows for c in r.get("retreat_candidates",())
                     for s in c["result"] if not s["success"]]
    domain = ("curobo" if any(s["stage"]["details"].get("failure_domain")=="curobo"
                             for s in failed_stages) else "geometry_postcheck")
    summary = [{"index":r["index"],"success":False,
        "failure_stages":[{"name":s["stage"]["name"],"failure":s["stage"]["result"].get("failure")}
                          for s in r["held"] if not s["success"]],
        "release_envelope":r.get("release_envelope"),
        "retreat_candidate_count":len(r.get("retreat_candidates",()))} for r in rows]
    write_report(Path(output)/"insert_sequence_screen.json",
        {"success": False, "failure_domain":domain, "candidate_screen":summary})
    return {"success": False, "failure_domain":domain, "candidate_screen":summary}

def without_bowl(scene, output):
    from gene_agent.tasks.policies.microwave_task.planning.extract import allow_placed_object_snapshot
    base = scene["snapshot"]
    snapshot = allow_placed_object_snapshot(base, Path(output), object_body_name="bowl")
    return {**scene, "snapshot": {**base, **snapshot}, "provenance": {
        **scene["provenance"], "collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "derived_from_collision_snapshot_sha256": base["snapshot_sha256"]}}

def extract_contact_postcheck(scene, raw):
    from gene_agent.motion.model import JointTrajectory
    from gene_agent.simulation.planned_contact import validate_planned_target_contacts
    frozen = frozen_geometry(scene)
    dt = float(raw["interpolation_dt_s"])
    trajectory = JointTrajectory(tuple(raw["joint_names"]),
        tuple(i*dt for i in range(len(raw["positions_rad"]))),
        tuple(tuple(q) for q in raw["positions_rad"]))
    return validate_planned_target_contacts(frozen.model, frozen.data,
        frozen.robot_metadata, group="right", object_body="bowl", trajectory=trajectory,
        gripper_joint_positions_m=scene["snapshot"]["gripper_snapshot"]["joint_positions_m"],
        allowed_geom_names=frozen.robot_metadata["grippers"]["right"]["pad_geom_names"])
