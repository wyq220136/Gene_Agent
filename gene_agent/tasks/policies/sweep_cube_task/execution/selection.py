"""在隔离状态副本上筛整组 cuRobo 轨迹；不把预览推进写回物理现场。"""
from __future__ import annotations

import copy
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.sweep import make_sweep_pose_sequences
from gene_agent.motion.mujoco_ik import MujocoPosePlanner
from gene_agent.simulation.runtime import MujocoRuntime

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[5] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs


def advance_preview(preview, plan, grip):
    """只在预览副本中应用规划末关节及刚性持刷关系；不是仿真执行。"""
    for name, value in zip(plan["joint_names"], plan["positions_rad"][-1]):
        preview.data.qpos[preview.model.jnt_qposadr[preview.model.joint(name).id]] = value
    mujoco.mj_forward(preview.model, preview.data)
    tcp = np.asarray(preview.observe()["robots"][preview.group]["T_world_tcp"])
    brush = tcp @ grip
    address = int(preview.model.jnt_qposadr[preview.model.joint("brush_free").id])
    preview.data.qpos[address:address + 3] = brush[:3, 3]
    preview.data.qpos[address + 3:address + 7] = Rotation.from_matrix(brush[:3, :3]).as_quat()[[3, 0, 1, 2]]
    preview.data.qvel[:] = 0.
    mujoco.mj_forward(preview.model, preview.data)


def select_sweep_sequence(run, target_name, stage_label, *, recovery_inside_pan=False,
                          cube_goal_world_xyz=None):
    observation = run.observe()
    original_id = observation["state_id"]
    tcp = np.asarray(observation["robots"][run.group]["T_world_tcp"])
    poses = {n: np.asarray(r["T_world_object"]) for n, r in observation["objects"].items()}
    grip = np.linalg.inv(tcp) @ poses["brush"]
    table = run.model.geom("scene_table")
    table_z = float(run.data.geom_xpos[table.id][2] + table.size[2])
    candidates = make_sweep_pose_sequences(tcp, poses, run.concept_dir,
        target_name=target_name, recovery_inside_pan=recovery_inside_pan,
        table_top_z_m=table_z, initial_brush_z=run.initial_brush_z,
        reference_brush_rotation=getattr(run, "sweep_reference_brush_rotation", None),
        cube_goal_world_xyz=cube_goal_world_xyz,
        goal_inset_extra_m=float(run.scene.get("goal_inset_extra_m", 0.)))
    from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import brush_collision_vertices_object, correct_sequence_table_clearance
    vertices = brush_collision_vertices_object(run.model, run.data)
    candidates = [correct_sequence_table_clearance(row, grip, vertices, table_z) for row in candidates]
    lower_contacts = (*TABLE_CONTACTS, ("brush", target_name))
    sweep_contacts = (*TABLE_CONTACTS, *(("brush", name) for name in ("cube", "cube_red", "cube_blue")),
                      ("brush", "dustpan"))
    steps = (("move_behind_cube", "joint", GRIP_CONTACTS),
             ("lower_to_contact", "linear_tcp", lower_contacts),
             ("sweep", "linear_tcp", sweep_contacts),
             ("retreat", "linear_tcp", sweep_contacts))
    report = {"evidence_mode": "kinematic_preview_only", "start_state_id": original_id,
              "selected_index": None, "candidates": []}
    selected = None
    for index, candidate in enumerate(candidates):
        print(f"清扫序列预检 {stage_label}：候选 {index + 1}/{len(candidates)}"
              "（仅规划，不执行物理动作）", flush=True)
        from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import check_cartesian_brush_pan_walls
        wall_contacts = check_cartesian_brush_pan_walls(run.planner,
            candidate["targets"]["lower_to_contact"], candidate["targets"]["sweep"], grip)
        if wall_contacts:
            report["candidates"].append({**candidate, "stages": {}, "success": False,
                "geometry_rejection": {"failure_category": "brush_pan_wall_sweep_collision",
                                       "contacts": wall_contacts}})
            continue
        from gene_agent.tasks.policies.sweep_cube_task.execution.curobo import inspect_push_contact
        push_contact = inspect_push_contact(run.planner, candidate["targets"]["lower_to_contact"],
                                            candidate["targets"]["sweep"], grip, target_name)
        candidate["push_contact_geometry"] = push_contact
        if not push_contact["success"]:
            report["candidates"].append({**candidate, "stages": {}, "success": False,
                "geometry_rejection": {"failure_category": push_contact["reason"]}})
            continue
        preview = copy.copy(run)
        preview.data = run.runtime.copy_data()
        preview.runtime = MujocoRuntime(run.model, preview.data, run.runtime.metadata)
        preview.runtime._simulation_attachment_evidence = copy.deepcopy(run.runtime._simulation_attachment_evidence)
        preview.planner = MujocoPosePlanner(run.model, preview.data, run.robot, run.group)
        preview._pending_contact_candidates = {}
        preview.output_dir = run.output_dir / "pose_previews" / stage_label / str(index)
        row = {**candidate, "stages": {}, "success": False}
        for name, kind, contacts in steps:
            label = f"{stage_label}__{name}"
            result = preview._move_curobo(label, np.asarray(candidate["targets"][name]), kind,
                                         3., allowed_contacts=contacts, execute=False)
            row["stages"][name] = {"success": result["success"],
                "failure_category": result["plan"].get("failure_category"),
                "plan_path": str(preview.output_dir / "motion_curobo/plans" / f"{label}.json")}
            if not result["success"]:
                break
            advance_preview(preview, result["plan"], grip)
        else:
            row["success"] = True
            selected = candidate
            report["selected_index"] = index
        report["candidates"].append(row)
        if selected is not None:
            break
    if run.observe()["state_id"] != original_id:
        raise RuntimeError("候选预检意外改变了真实状态")
    path = run.output_dir / "knowledge" / f"{stage_label}_sequence_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return selected, report
