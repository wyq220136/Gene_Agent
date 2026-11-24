"""真机 Task1 的完整 plan_only；复用真机各阶段的目标、候选和 cuRobo 入口。"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import traceback

from gene_agent.tasks.policies.microwave_task.policy import (
    MicrowaveTaskPolicy, microwave_task_steps, default_tcp_transition)
from gene_agent.skills.library.door.sequence import door_skill_sequence
from gene_agent.skills import load_grasp_settings
from .prediction import assumption_check
from .plan_only_world import PlanOnlyWorld, PlanningFailed, write_json
from . import door_planning as door
from . import pick_planning as pick
from . import insert_planning as insert
from . import extract_planning as extract
from . import place_planning as place
from . import knob_planning as knob
from .attachment import prepare_observed_bowl_attachment
from .task import _home_tcp_pose

ROOT = Path(__file__).resolve().parents[3]


def _options(fn, options):
    return {k: v for k, v in options.items() if k in inspect.signature(fn).parameters
            and k not in {"output_dir", "planner_pool", "scene_revision"}}


class TaskPreview:
    def __init__(self, world, pool, options):
        self.world, self.pool, self.options = world, pool, options
        self.pick_report = self.grasp_report = None

    def call(self, fn, *args, folder, kind, **kwargs):
        return fn(*args, **{**_options(fn, self.options[kind]), **kwargs},
                  planner_pool=self.pool, output_dir=folder)

    def motion(self, name, report, key="stage", **kwargs):
        stage = report.get(key)
        if stage is None:
            raise PlanningFailed(report.get("failure_category") or "planning_failed",
                                 report.get("message") or name, report)
        if report.get("success") is False:
            raise PlanningFailed(report.get("failure_code") or "motion_planning_failed",
                name, {k:v for k,v in report.items() if k not in {"stage", "lift_stage", "preplace_stage"}})
        self.world.motion(name, stage, **kwargs)

    def door_stage(self, folder, targets, phase, prefix):
        w = self.world
        if phase.motion_kind == "gripper":
            w.event(f"{prefix}/{phase.phase_id}", width=phase.gripper_width_m)
            return
        report = self.call(door.plan_observed_door_stage,
            w.scene(), targets, folder=folder, kind="door",
            phase_id=phase.phase_id, scene_revision=w.revision)
        write_json(folder / "preview_plan_result.json", report)
        if not report["success"]:
            raise PlanningFailed(report.get("failure_category") or "door_planning_failed",
                                 report.get("message") or phase.phase_id, report)
        stage = door.load_planned_door_skill_stage(report["skill_stage_ref"])
        moving = phase.name.startswith(("pull_", "swing_", "push_"))
        w.motion(prefix + "/" + phase.phase_id, stage,
            collision_ref=report["stage_collision_snapshot_ref"],
            articulation=("microwave_door_joint", phase.door_angle_rad) if moving else None)

    def plan_door(self, spec, folder):
        w, options = self.world, self.options["door"]
        for skill in door_skill_sequence(spec.door_part):
            targets = door.build_observed_door_skill_targets(w.scene(), skill_id=skill,
                                                            config_path=options["config_path"])
            for index in range(len(targets["phases"])):
                phase = targets["phases"][index]
                targets = door.rebase_observed_door_skill_targets(targets, w.scene(),
                    max_microwave_translation_m=options["max_microwave_translation_m"],
                    max_microwave_rotation_rad=options["max_microwave_rotation_rad"])
                targets, selected = self.call(door.select_observed_door_phase_candidate,
                    w.scene(), targets, phase_id=phase.phase_id,
                    folder=folder / skill / f"phase_{index:02d}" / "selection",
                    kind="door", scene_revision=w.revision)
                if selected is not None and not selected["success"]:
                    raise PlanningFailed("goal_ik_no_solution", phase.phase_id, selected)
                self.door_stage(folder / skill / f"phase_{index:02d}", targets,
                                targets["phases"][index], spec.name)

    def plan_transition(self, spec, folder):
        w, options = self.world, self.options["door"]
        config_path = Path(options["config_path"]).resolve()
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if config.get("schema_version") != 1:
            raise ValueError("门 Task 过渡配置版本必须为 1")
        pid = ("task/before_reopen/return_default_tcp" if spec.name == "before_reopen_transition"
               else "task/open_to_close/return_default_tcp")
        scene = w.scene()
        override = (config.get("task_transition_overrides_world") or {}).get(pid)
        phase = default_tcp_transition(_home_tcp_pose(scene) if override is None else override,
            phase_id=pid, door_angle_rad=scene["provenance"]["door_angle_rad"],
            gripper_width_m=w.width())
        targets = {"skill_id": "task_transition", "phases": (phase,),
            "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest()}
        self.door_stage(folder, targets, phase, spec.name)
        if spec.name == "before_reopen_transition":
            w.event(spec.name+"/wait", duration_s=5.)

    def plan_pick(self, spec, folder):
        w = self.world
        selection = pick.plan_observed_bowl_grasp_stages(w.scene(),
            output_dir=folder / "selection", seed=self.options["pick"].get("seed", 123))
        if not selection["grasp_prefix_planned"]:
            raise PlanningFailed("grasp_selection_failed", "抓碗候选未通过", selection)
        plan_rows = {r["name"]: r for r in selection["plan"]["stages"]}
        approach = None
        for name in ("open", "pregrasp", "approach", "close"):
            if name in {"open", "close"}:
                w.event(spec.name+"/"+name, width=plan_rows[name]["details"]["width_m"])
            else:
                planned = self.call(pick.plan_observed_bowl_motion_stage,
                    w.scene(), selection, folder=folder/name, kind="pick",
                    stage_name=name, scene_revision=w.revision)
                self.motion(spec.name+"/"+name, planned)
                if name == "approach":
                    approach = planned["stage"]
        w.event(spec.name+"/attach", attach=True,
                assumption="假定闭爪后稳定持碗；规划附着不证明真实夹持")
        grasp = w.hold_report(selection["selected_candidate_id"])
        grasp["stages"] = [{"stage": "approach", "status": "planned",
                            "plan": {"stage": approach.to_dict()}}]
        attachment = prepare_observed_bowl_attachment(w.scene(), grasp,
                                                       output_dir=folder/"attachment")
        lifted = self.call(pick.plan_observed_bowl_lift,
            w.scene(), selection, grasp, attachment, folder=folder/"lift", kind="pick",
            scene_revision=w.revision)
        self.motion(spec.name+"/lift", lifted, key="lift_stage")
        settings = load_grasp_settings(ROOT/"data/task_1/skills/skill_profiles.json",
                                       selection["skill_profile"])
        w.event(spec.name+"/hold", duration_s=settings.hold_duration_s,
                assumption="假定抬升后碗仍被稳定夹持")
        self.pick_report = {**w.hold_report(selection["selected_candidate_id"]), "grasp": grasp}

    def release(self, name, width, duration):
        w = self.world
        w.event(name+"/release", width=width, attach=False,
                assumption="假定松爪后物体留在当前预测位姿；未模拟重力或落稳")
        w.event(name+"/settle", duration_s=duration)
        return w.release_report()

    def plan_insert(self, spec, folder):
        w = self.world
        if self.pick_report is None or w.attachment is None:
            raise ValueError("放碗预览必须接续本次 Pick 的预测持物状态")
        prefix = self.call(insert.plan_observed_bowl_insert_prefix,
            w.scene(), self.pick_report, folder=folder/"preplace", kind="insert",
            current_hold_check=assumption_check(w.scene()).to_dict())
        if not prefix["success"]:
            raise PlanningFailed("insert_prefix_failed", "放入前缀规划失败", prefix["plan"])
        targets = prefix["targets"]
        self.motion(spec.name+"/preplace", prefix, key="preplace_stage")
        linear = self.call(insert.plan_observed_bowl_insert_linear,
            w.scene(), targets, self.pick_report, folder=folder/"insert", kind="insert",
            hold_check=assumption_check(w.scene()).to_dict(), scene_revision=w.revision)
        self.motion(spec.name+"/insert", linear)
        release = self.release(spec.name, w.robot["grippers"][w.group]["width_range_m"][1],
                               self.options["insert"].get("settle_duration_s", .5))
        for name in ("retreat_1", "retreat_2"):
            options = self.options["insert"]
            planned = self.call(insert.plan_observed_bowl_detached_retreat_stage,
                w.scene(), targets, release, folder=folder/name, kind="insert",
                stage_name=name, max_bowl_translation_m=options["max_hold_translation_m"],
                max_bowl_rotation_rad=options["max_hold_rotation_rad"],
                scene_revision=w.revision)
            self.motion(spec.name+"/"+name, planned)

    def plan_extract(self, spec, folder):
        from gene_agent.tasks.policies.microwave_task.operations.bowl import load_object_operation_profile
        w = self.world
        profile = load_object_operation_profile(ROOT/"data/task_1", "Bowl_0")
        w.event(spec.name+"/open", width=profile["grasp_pregrasp_opening_width_m"])
        prefix = self.call(extract.plan_observed_bowl_extract_prefix,
            w.scene(), folder=folder/"prefix", kind="extract")
        if not prefix["success"]:
            raise PlanningFailed("extract_prefix_failed", "取碗前缀规划失败", prefix["plan"])
        targets = prefix["targets"]
        self.motion(spec.name+"/pre_extract_1", prefix, key="pre_extract_1_stage")
        for name in ("pre_extract_2", "grasp"):
            report = self.call(extract.plan_observed_bowl_extract_motion_stage,
                w.scene(), targets, folder=folder/name, kind="extract",
                stage_name=name, scene_revision=w.revision)
            self.motion(spec.name+"/"+name, report)
        w.event(spec.name+"/close", width=profile["grasp_opening_width_m"],
                duration_s=self.options["extract"].get("grasp_settle_time_s", .5))
        w.event(spec.name+"/attach", attach=True,
                assumption="假定炉内取碗闭爪后形成稳定夹持")
        grasp = w.hold_report("fixed_keypose_0")
        tcp_bowl = w.attachment.copy()
        for name in ("post_extract_1", "post_extract_2"):
            report = self.call(extract.plan_observed_bowl_extract_retreat_stage,
                w.scene(), targets, grasp, folder=folder/name, kind="extract",
                stage_name=name, current_hold_check=assumption_check(w.scene()),
                T_tcp_bowl_after_close=tcp_bowl, scene_revision=w.revision)
            self.motion(spec.name+"/"+name, report)
        self.grasp_report = grasp

    def plan_table(self, spec, folder):
        w = self.world
        if self.grasp_report is None or w.attachment is None:
            raise ValueError("桌面放置必须接续本次取碗的预测持物状态")
        prefix = self.call(place.plan_observed_table_place_prefix,
            w.scene(), self.grasp_report, folder=folder/"preplace", kind="table",
            current_hold_check=assumption_check(w.scene()), scene_revision=w.revision)
        if not prefix["success"]:
            raise PlanningFailed("table_prefix_failed", "桌面放置前缀规划失败", prefix["plan"])
        targets = prefix["targets"]
        self.motion(spec.name+"/preplace", prefix, key="preplace_stage")
        descent = self.call(place.plan_observed_table_descent,
            w.scene(), targets, self.grasp_report, folder=folder/"place", kind="table",
            current_hold_check=assumption_check(w.scene()), scene_revision=w.revision)
        self.motion(spec.name+"/place", descent)
        release = self.release(spec.name, w.robot["grippers"][w.group]["width_range_m"][1],
                               self.options["table"].get("settle_duration_s", .5))
        report = self.call(place.plan_observed_table_detached_retreat,
            w.scene(), targets, release, folder=folder/"retreat", kind="table",
            scene_revision=w.revision)
        self.motion(spec.name+"/retreat", report)

    def plan_knob(self, spec, folder):
        w, options = self.world, self.options["knob"]
        args = {} if options.get("task_path") is None else {"task_path": options["task_path"]}
        initial = knob.build_observed_knob_targets(w.scene(), **args)
        w.event(spec.name+"/open", width=initial["settings"].open_width_m)
        selection = self.call(knob.select_observed_knob_grasp,
            w.scene(), folder=folder/"selection", kind="knob", **args)
        if not selection["success"]:
            raise PlanningFailed("knob_selection_failed", "旋钮抓取候选规划失败", {
                k:v for k,v in selection.items() if k not in {"source", "targets"}})
        for target in selection["targets"]:
            name = spec.name+"/"+target.name
            if target.action == "wait":
                w.event(name, duration_s=target.settle_time_s)
            elif target.action.startswith("gripper_"):
                w.event(name, width=target.gripper_width_m)
            elif target.action == "last_joint_rotation":
                # 真机此分支本来是直接 SDK 指令，没有 cuRobo 路径。
                # 预览中明确保留未验证状态，不编造规划轨迹或安全成功。
                raise PlanningFailed("unplanned_direct_joint_command",
                    "旋钮含未经轨迹规划的末关节命令；不能声称完整碰撞规划通过",
                    {"joint_delta_rad": target.joint_delta_rad,
                     "stage": target.name})
            else:
                planned = self.call(knob.plan_observed_knob_motion_stage,
                    w.scene(), selection, folder=folder/target.name, kind="knob",
                    stage_name=target.name, scene_revision=w.revision)
                self.motion(name, planned, articulation=("knob_0_joint", target.end_angle_rad))


def plan_microwave_task1(initial_scene, *, planner_pool, output_dir,
                         subtask_options, include_knob=False):
    """只接收场景和 planner，接口上无法访问设备或执行器。"""
    output = Path(output_dir).resolve()
    policy = MicrowaveTaskPolicy(include_knob=include_knob)
    specs = microwave_task_steps(policy)
    report = {"schema_version": 1, "task": "microwave_task1", "requested_scope": "all",
        "mode": "plan_only", "evidence_mode": "planned_kinematic_preview",
        "execute": False, "status": "planning", "success": False,
        "planning_success": False, "task_success": None, "operation_success": False,
        "include_knob": bool(include_knob), "planner_backend": "curobo",
        "observed_geometry": initial_scene["provenance"].get("observed_geometry"),
        "step_order": [s.name for s in specs], "steps": [], "first_failure": None,
        "claim": "完整预测规划预览；不发送真机命令，不表示夹持、接触或任务物理成功"}
    world = PlanOnlyWorld(initial_scene, output,
        max_start_error_rad=subtask_options["door"]["max_start_error_rad"])
    planner = TaskPreview(world, planner_pool, subtask_options)
    row = None
    try:
        geometry = report.get("observed_geometry") or {}
        if geometry:
            world.event("observed_initial_scene", duration_s=1.)
        if geometry.get("table_world_up_consistent") is False:
            raise PlanningFailed("observed_world_up_mismatch",
                "实测桌面朝上法向与规划世界 +Z 不一致；请核对相机外参及 SDK 坐标。已保存实际场景供回放。",
                geometry)
        for spec in specs:
            row = {"step_id": spec.name, "subtask": spec.kind, "status": "planning",
                   "predicted_start": world.state(), "first_stage": len(world.stages)}
            report["steps"].append(row)
            write_json(output/"task_result.json", report)
            started = perf_counter()
            getattr(planner, "plan_"+spec.kind)(spec, output/spec.name)
            row.update(status="planned", predicted_end=world.state(),
                       last_stage=len(world.stages)-1, wall_time_s=perf_counter()-started)
            world.save_replay()
        report.update(status="planned", success=True, planning_success=True)
    except Exception as error:
        failure = (error.failure if isinstance(error, PlanningFailed) else {
            "code": "planning_exception", "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()})
        if row is not None:
            row.update(status="failed", predicted_end=world.state(),
                       last_stage=len(world.stages)-1, failure=failure)
        details = failure.get("details") or {}
        failed_collision_ref = (details.get("stage_collision_snapshot_ref")
                                if isinstance(details, dict) else None)
        if not failed_collision_ref:
            try:
                failed_scene = world.scene()
                failed_collision_ref = world.collision_ref(failed_scene["snapshot"]["snapshot_sha256"])
            except Exception as evidence_error:
                failure["collision_record_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
        world.stages.append({"name": (row["step_id"] if row else "initial")+"/failed",
            "status": "failed", "operation": "state_event", "predicted": True,
            "failure": failure, "planned_replay_samples": [world.sample(0.)],
            "collision_snapshot_ref": failed_collision_ref})
        report.update(status="planning_failed",
                      first_failure={"step_id": None if row is None else row["step_id"], **failure})
    finally:
        world.save_replay()
        report.update(predicted_final=world.state(), assumptions=world.assumptions,
                      preview_ref=str(output/"plan_preview.json"),
                      completed_steps=sum(r["status"]=="planned" for r in report["steps"]))
        write_json(output/"task_result.json", report)
    return report
