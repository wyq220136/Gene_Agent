"""人工持刷的单次清扫 cuRobo 前瞻；预测状态绝不作为实测终态。"""
from __future__ import annotations
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode, configured_motion, current_motion_config

import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.knowledge.sweep import (make_sweep_layout, make_sweep_pose_sequences,
                                      correct_sequence_table_clearance)
from gene_agent.motion import SynchronousPlanner
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.robots.pika_frames import pose_from_mujoco,pose_error
from gene_agent.scene.rigid_geometry import collision_vertices_object
from gene_agent.skills.library.sweep.skill import run_sweep_skill
from .scene import set_free_body

from pathlib import Path as _ContactPath
from gene_agent.skills.library.sweep import SweepContacts
SWEEP_CONTACTS_PROFILE = SweepContacts.load(_ContactPath(__file__).resolve().parents[3] / 'config/tasks/sweep/contacts.json')
GRIP_CONTACTS = SWEEP_CONTACTS_PROFILE.grip_pairs
TABLE_CONTACTS = SWEEP_CONTACTS_PROFILE.table_pairs


class _RecordingPlanner(SynchronousPlanner):
    """让公共 adapter 之外也保留原始 GPU 返回，算法仍调用原实现。"""
    def plan(self,*args,**kwargs):
        self.last_raw=super().plan(*args,**kwargs)
        return self.last_raw

    def plan_linear_tcp(self,*args,**kwargs):
        self.last_raw=super().plan_linear_tcp(*args,**kwargs)
        return self.last_raw


def write_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _set_prediction(model,data,robot,q,grip):
    for name,value in zip(robot["joint_groups"]["left"],q):
        data.qpos[model.joint(name).qposadr[0]]=value
    mujoco.mj_forward(model,data)
    if grip is not None:
        set_free_body(model,data,"brush",pose_from_mujoco(data.site("pika_tcp"))@grip)
    mujoco.mj_forward(model,data)


def check_world_contacts(model,data,robot,positions,grip,allowed_contacts,*,base_table_mount=False):
    """完整 world 几何的密集 FK 后检查；不使用 CPU IK，不移动方块。

cuRobo 阶段粗规划可移除允许刷子接触的障碍；此处仍拒绝其他机器人 link
碰这些障碍，也拒绝刷子穿簸箕侧/后壁。机器人 self collision 由 cuRobo 负责。
"""
    if resolve_collision_mode() is CollisionMode.ROBOT_ONLY:
        return {"success": True, "status": "skipped_by_collision_mode",
                "environment_geometry_checked": False, "physics_stepped": False,
                "self_collision_source": "curobo_spheres"}
    preview=mujoco.MjData(model);mujoco.mj_copyData(preview,model,data)
    allowed={frozenset(pair) for pair in allowed_contacts}
    if base_table_mount:allowed.add(frozenset(("base_link","table")))
    names=robot["joint_groups"]["left"]
    robot_bodies={"base_link",*(f"link_{i}" for i in range(1,8)),
                  "pika_gripper_base_link","pika_gripper_left_link","pika_gripper_right_link"}
    def role(geom):
        name=model.geom(geom).name
        if name=="scene_table":return "table"
        body=int(model.geom_bodyid[geom])
        while body:
            name=model.body(body).name
            if name in {"brush","cube","dustpan"}:return name
            if name in robot_bodies:return name
            body=int(model.body_parentid[body])
        return model.geom(geom).name
    sampled=[positions[0]]
    for start,end in zip(positions[:-1],positions[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(np.asarray(end)-start))/.01)))
        sampled.extend(np.linspace(start,end,count+1)[1:])
    invalid=[];checked=0
    for index,q in enumerate(sampled):
        checked+=1
        _set_prediction(model,preview,robot,q,grip)
        for contact in preview.contact:
            if contact.dist>=-.001:continue
            a,b=role(contact.geom1),role(contact.geom2)
            involved={a,b}
            if not (involved&robot_bodies or "brush" in involved):continue
            if a in robot_bodies and b in robot_bodies:continue
            geom_names=[model.geom(g).name for g in (contact.geom1,contact.geom2)]
            wall=("brush" in involved and any(n.startswith("dustpan_wall_") or n=="dustpan_back_collision" for n in geom_names))
            excessive_table=(involved=={"brush","table"} and contact.dist<-.002)
            if wall or excessive_table or frozenset((a,b)) not in allowed:
                invalid.append({"sample":index,"roles":[a,b],"geoms":geom_names,
                    "penetration_m":-float(contact.dist),"joint_positions_rad":np.asarray(q).tolist(),
                    "qpos":preview.qpos.tolist(),"contact_position_world_m":contact.pos.tolist()})
        if len(invalid)>=12:break
    return {"success":not invalid,"forbidden_contacts":invalid,"sample_count":checked,
            "planned_sample_count":len(sampled),
            "max_joint_step_rad":.01,"source":"full_world_mujoco_fk_geometry_only",
            "self_collision_source":"curobo_spheres","physics_stepped":False}


@configured_motion("site")
def plan_sweep(model,data,robot,grip,site,output:Path)->dict:
    poses={name:pose_from_mujoco(data.body(name)) for name in ("brush","cube","dustpan")}
    initial_qpos=data.qpos.copy()
    initial_tcp=pose_from_mujoco(data.site("pika_tcp"))
    binding_error=pose_error(poses["brush"],initial_tcp@grip)
    if binding_error["translation_m"]>1e-7 or binding_error["rotation_deg"]>1e-5:
        raise ValueError(f"夹持关系与模型初态不一致，拒绝跳动物体后继续规划：{binding_error}")
    table=model.geom("scene_table")
    table_z=float(data.geom_xpos[table.id][2]+table.size[2])
    concept_dir=output/"scene/concepts"
    layout=make_sweep_layout(poses,concept_dir)
    candidates=make_sweep_pose_sequences(initial_tcp,poses,concept_dir,table_top_z_m=table_z,
                                         initial_brush_z=poses["brush"][2,3],orientation_policy="grip_relative")
    vertices=collision_vertices_object(model,data,"brush")
    pool=CuroboPlannerPool(planner_factory=_RecordingPlanner,max_entries=2)
    report={"schema_version":1,"task":"sweep_realworld_plan_only","planner_backend":"curobo",
            "skill_id":"sweep_cube" if site.get("brush_variant")=="old" else "sweep_long_brush",
            "brush_variant":site.get("brush_variant","unspecified"),
            "evidence_mode":"kinematic_plan_preview","execution_enabled":False,"task_success":None,
            "source":site["snapshot_source"],"planning_complete":False,"candidates":[],"selected_index":None,
            "planner_invoked":False,"initial_hold_binding_error":binding_error,
            "hold_registration":"same_observed_brush_registered_to_model_fk_tcp; raw SDK relation in scene/provenance.json",
            "scene_xml_path":str(output/"scene/scene.xml"),"T_tcp_brush":grip.tolist(),
            "world_frame":robot["world_frame"],"gripper_hold":"assumed_rigid_from_snapshot_confirmation",
            "limits":{"max_candidates":site["max_candidates"],"approximate_ik":"disabled"},
            "initial_object_poses_world":{n:p.tolist() for n,p in poses.items()}}
    all_frames=[]
    names=robot["joint_groups"]["left"]
    try:
        start_q=[float(data.qpos[model.joint(n).qposadr[0]]) for n in names]
        initial_check=check_world_contacts(model,data,robot,[start_q],grip,GRIP_CONTACTS,
                                           base_table_mount=site.get("base_table_mounting_contact",False))
        report["initial_state_postcheck"]=initial_check
        if not initial_check["success"]:
            report["status"]="initial_world_collision"
            write_json(output/"preview_frames.json",[{"stage":"initial_only","qpos":initial_qpos.tolist(),"predicted":False}])
            write_json(output/"result.json",report)
            return report
        for index,candidate in enumerate(candidates[:site["max_candidates"]]):
            print(f'清扫候选 {index+1}/{min(len(candidates),site["max_candidates"])}：离线规划尝试，不执行机械臂',flush=True)
            data.qpos[:]=initial_qpos;mujoco.mj_forward(model,data)
            candidate=correct_sequence_table_clearance(candidate,grip,vertices,table_z)
            targets=candidate["targets"]
            row={"index":index,"targets":targets,"candidate_geometry":candidate,
                 "mesh_height_adjustment_m":candidate["mesh_height_adjustment_m"],
                 "stages":[],"success":False}
            predicted_start=site.get("planning_start_source")=="predicted_after_pick"
            frames=[{"stage":"predicted_after_pick" if predicted_start else
                     "initial_measured" if site["snapshot_source"]!="synthetic_fixture" else "initial_synthetic",
                     "qpos":data.qpos.tolist(),"predicted":predicted_start}]

            def move(stage,target,path_kind,duration_s,*,allowed_contacts):
                from .preview_motion import plan_stage
                entry, stage_frames = plan_stage(model, data, robot, pool,
                    output/"candidates"/str(index)/stage, stage, target, path_kind,
                    allowed_contacts, grip,
                    base_table_mount=site.get("base_table_mounting_contact", False))
                entry["duration_hint_s"] = duration_s
                entry["start_source"] = (site.get("planning_start_source","observation")
                    if not row["stages"] else "previous_plan_predicted_fk")
                row["stages"].append(entry)
                report["planner_invoked"] = True
                frames.extend(stage_frames)
                return {"success": entry["success"]}

            def target_for_brush(xyz):
                target=np.asarray(targets["sweep"]).copy()
                target[:3,3]=np.asarray(xyz)-target[:3,:3]@grip[:3,3]
                return target
            skill=run_sweep_skill(layout, contacts=SWEEP_CONTACTS_PROFILE, evidence_mode="prediction", sweep_targets=('cube',),initial_brush_z=poses["brush"][2,3],move_stage=move,
                target_for_brush=target_for_brush,
                current_brush_xyz=lambda: data.body("brush").xpos.copy(),prepared_targets=targets,
                sweep_end_for_current_pose=lambda:(np.asarray(targets["sweep"])@grip)[:3,3],
                retreat_delta_world_m=np.asarray(candidate["retreat_delta_world_m"]))
            row["success"]=skill["success"]
            row["failure_stage"]=skill["failure_stage"]
            report["candidates"].append(row)
            if not skill['success']:
                from .diagnostics import failure_summary
                print(f'候选 {index+1} 未通过：'+failure_summary({'candidates':[row]}),flush=True)
            if len(frames)>len(all_frames):
                all_frames=frames
                report["preview_candidate_index"]=index
            if skill["success"]:
                report["planning_complete"]=True;report["selected_index"]=index;all_frames=frames;break
        if not all_frames:
            data.qpos[:]=initial_qpos;mujoco.mj_forward(model,data)
            all_frames=[{"stage":"initial_only","qpos":initial_qpos.tolist(),"predicted":False}]
        report["preview_frames_path"]=str(output/"preview_frames.json")
        report["status"]="planning_complete" if report["planning_complete"] else "planning_failed"
        write_json(output/"preview_frames.json",all_frames)
        write_json(output/"result.json",report)
        return report
    finally:
        pool.clear()
        data.qpos[:]=initial_qpos;mujoco.mj_forward(model,data)
