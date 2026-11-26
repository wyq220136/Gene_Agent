"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

from .prediction import is_prediction, predicted_report, reject_assumption_on_live_scene
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode

import hashlib
import json
import math
import time
from gene_agent.feedback.runtime_timing import timed_runtime_call
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase, prepare_grasp_stage_snapshots
from gene_agent.tasks.policies.microwave_task.planning.grasp import (
    active_arm_base_reference, legacy_bowl_proposals, bowl_grasp_geometry_metrics,
    plan_bowl_grasp_stages,
)
from gene_agent.tasks.policies.microwave_task.planning.bowl_candidates import (
    BOWL_AUTO_GRASP_PROFILE,
)
from .bowl_candidates import observed_bowl_candidates
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.model import (
    MotionOperation,
    MotionRequest,
    PlannerMode,
    PoseTarget,
    motion_request_from_dict,
    motion_result_from_dict,
)
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.simulation.planned_contact import validate_planned_target_contacts
from gene_agent.skills import PickSkill, SkillContext, load_grasp_settings
from gene_agent.skills.runtime.planning import SkillPlan, SkillStage

_BOWL_INSTANCE = "Bowl/Bowl_0"
_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_PATH = _ROOT / "data/task_1/skills/skill_profiles.json"


def _affordtoolact_proposals(case):
    """保留已按SDK -X/-Y排序的全部候选，不套用仿真接触代理阈值。

    observed_bowl_candidates 已按 AffordToolAct 生成、排序和限高。
    此处仅转换数据结构；可达性、碰撞、路径和后续抬升由规划层检验。
    """
    return legacy_bowl_proposals(case.candidates)


def _bowl_target(collision_scene: Mapping) -> dict:
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    if (provenance.get("physics_stepped") is not False
            or provenance.get("pose_frame_verified") is not True
            or provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("Bowl 候选需要同帧现场观测及一致的碰撞快照")
    aligned = provenance.get("aligned_object_poses_world")
    if not isinstance(aligned, Mapping) or _BOWL_INSTANCE not in aligned:
        raise ValueError("现场碰撞来源缺少已对齐资产的 Bowl/Bowl_0 pose")
    bowl_pose = validate_pose(aligned[_BOWL_INSTANCE], field_name="T_world_bowl")
    return {
        "object_id": _BOWL_INSTANCE,
        "pose_source": "observed_scene_asset_aligned",
        "source_frame_number": provenance["source_frame_number"],
        "source_timestamp_ms": provenance["source_timestamp_ms"],
        "collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "bowl_pose_world": bowl_pose,
    }


@timed_runtime_call("scene_reconstruction")
def _observed_case(
    collision_scene: Mapping, output_dir: str | Path, *, generate_candidates: bool = True,
) -> tuple[PickBowlCase, dict]:
    """把现场已测 qpos 放进仅用于 FK/碰撞的私有 MuJoCo 数据。"""
    target = _bowl_target(collision_scene)
    snapshot = collision_scene["snapshot"]
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    qpos = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("现场 Bowl 碰撞快照的关节状态与资产模型不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    group = snapshot["active_group"]
    metadata = collision_scene["scene_metadata"]
    if group not in metadata["robot"]["joint_groups"]:
        raise ValueError("Bowl 抓取活动臂与场景机器人元数据不一致")
    candidates = ()
    if generate_candidates:
        if group != "right":
            raise NotImplementedError("真机自动抓碗目前只支持右臂")
        candidates, parameters = observed_bowl_candidates(
            collision_scene, target["bowl_pose_world"], pose_source=target["pose_source"],
        )
        target.update(candidates=candidates, candidate_parameters=parameters)
    case = PickBowlCase(
        Path(output_dir).resolve(), 0, model, data, metadata, snapshot,
        target["bowl_pose_world"], candidates,
    )
    return case, target


def _selection_geometry(case):
    body, position = active_arm_base_reference(case)
    normal = case.data.geom("scene_table").xmat.reshape(3, 3)[:, 2]
    return body, position, normal


def build_observed_bowl_grasp_proposals(collision_scene: Mapping) -> dict:
    """保留 AffordToolAct 的全部160候选及SDK方向优先级。"""
    case, target = _observed_case(collision_scene, ".")
    target["proposals"] = _affordtoolact_proposals(case)
    return target


def build_observed_bowl_grasp_stage_snapshots(
    collision_scene: Mapping, *, open_width_m: float, output_dir: str | Path,
) -> tuple[dict, dict]:
    """按原抓碗函数导出开爪严格碰撞层和允许目标接触层。

    MuJoCo 在这里仅计算现有资产的 FK；只对私有数据副本设置规划
    开口，不推进动力学，也不改变现场场景的实测快照。接触层只移除
    碗作为规划障碍，绝不等于实机可以与其他部位任意接触。
    """
    case, _ = _observed_case(collision_scene, output_dir, generate_candidates=False)
    return prepare_grasp_stage_snapshots(
        case, open_width_m, group=case.snapshot["active_group"],
    )



def _planning_reuse_identity(scene, snapshot):
    if not scene["provenance"].get("alignment_input_sha256"):
        return None  # 旧/注入场景缺标定身份时正常重规划，禁止结果复用。
    from gene_agent.motion.curobo.core import load_realman_motion_config
    settings = load_realman_motion_config(
        scene["snapshot"]["active_group"], scene["robot_metadata"],
        gripper_snapshot=snapshot.get("gripper_snapshot"))
    urdf = Path(settings["robot_config"]["kinematics"]["urdf_path"])
    payload = {"settings": settings, "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
               "alignment": scene["provenance"].get("alignment_input_sha256"),
               "scene_config": scene["provenance"].get("scene_config_sha256")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@timed_runtime_call("stage", "grasp_candidate_selection")
def plan_observed_bowl_grasp_stages(
    collision_scene: Mapping, *, output_dir: str | Path,
    profile_name: str = "pick_bowl_legacy_visual", seed: int = 123,
    frozen_candidates=None, frozen_parameters=None, profile_path=None,
) -> dict:
    """现场抓碗只规划前缀，直接调用原物理探针所用 PickSkill 入口。

    返回的 pregrasp/approach 是预测轨迹。真机实际执行每段后须再读
    现场状态；闭爪、接触、抬升和 hold 尚未执行，因此不能报告 Pick
    成功。当前旧入口仅支持右臂，左臂必须另做等价验收后开放。
    """
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    case, target = _observed_case(collision_scene, output, generate_candidates=frozen_candidates is None)
    if frozen_candidates is not None:
        # 离线复现消费原生成结果，不用当前 Knowledge 参数替换历史输入。
        case.candidates = tuple(frozen_candidates)
        target["candidate_parameters"] = dict(frozen_parameters or {})
    if case.snapshot["active_group"] != "right":
        raise NotImplementedError("原 task_1 抓碗 Skill 规划目前只支持右臂")
    if profile_name != BOWL_AUTO_GRASP_PROFILE:
        raise ValueError("自动候选筛选目前只支持已验证的 pick_bowl_legacy_visual profile")
    effective_profile = Path(profile_path) if profile_path is not None else _PROFILE_PATH
    profile_bytes = effective_profile.read_bytes()
    settings = load_grasp_settings(effective_profile, profile_name)
    settings = replace(
        settings, max_grasp_candidates=len(case.candidates),
        pregrasp_search="first_successful_batch", align_before_approach=True, deduplicate_pregrasp=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    body, position, normal = _selection_geometry(case)
    evidence = {
        "input_state_id": target["collision_snapshot_sha256"],
        "bowl_pose_world": case.bowl_pose.tolist(),
        "candidate_generation_order_policy": target["candidate_parameters"]["candidate_order_policy"],
        "candidate_order_policy": target["candidate_parameters"]["candidate_order_policy"],
        "right_arm_base_reference_body": body,
        "right_arm_base_position_world": position.tolist(),
        "grasp_table_angle_definition": "acute angle(TCP +Z, actual scene_table plane)",
        "grasp_table_angle_constraint_enabled": False,
        "grasp_table_angle_allowed_deg": None,
        "grasp_table_angle_preferred_deg": None,
        "simulation_geometry_pruning_enabled": False,
        "geometry_filter_compatibility": {
            "enabled": [],
            "disabled_simulation_only": [
                "base_side", "table_angle_45_70", "rim_radius",
                "finger_rim_straddle", "rim_height", "contact_tangent",
            ],
            "affordtoolact_specific": [
                "external_axis_sign_not_hard_filtered",
                "tcp_origin_finger_length_height_compensation",
            ],
        },
        "max_grasp_candidates": settings.max_grasp_candidates,
        "candidate_parameters": target["candidate_parameters"],
        "generated_candidate_count": len(case.candidates),
        "pregrasp_search": settings.pregrasp_search,
        "selection_stop_condition": "first_successful_ik_batch_then_checked_prefix",
        "candidate_ranking_scope": "first_successful_pregrasp_batch",
        "pregrasp_request_reuse_scope": "single_grasp_selection_exact_request_except_id",
        "deduplicate_pregrasp": settings.deduplicate_pregrasp,
    }
    evidence_path = output / "candidate_selection.json"
    from gene_agent.feedback.realworld_diagnostics import write_candidate_inventory
    from gene_agent.motion.recording import current_motion_recording
    ordered_proposals = _affordtoolact_proposals(case)
    evidence["geometry_filtered_candidate_count"] = len(ordered_proposals)
    evidence["geometry_rejected_candidate_count"] = len(case.candidates) - len(ordered_proposals)
    candidate_order = [item.candidate_id for item in ordered_proposals]
    inventory_ref = write_candidate_inventory(output, case.candidates, candidate_order=candidate_order)
    evidence["candidate_diagnostics_ref"] = str(inventory_ref)
    recording = current_motion_recording()
    if recording is not None:
        evidence["replay_selection_ref"] = recording.note_selection(
            collision_scene, case.candidates, target["candidate_parameters"], effective_profile, seed)
    try:
        selected = plan_bowl_grasp_stages(
            case, settings, seed=seed, enforce_table_angle=False,
            proposals=ordered_proposals,
        )
    except Exception as error:
        evidence["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_candidate_inventory(output, case.candidates, candidate_order=candidate_order, error=evidence["failure"])
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2,
                                             allow_nan=False) + "\n",
                                 encoding="utf-8")
        raise
    selected_proposal = next((proposal for proposal in selected.proposals
                              if proposal.candidate_id == selected.plan.selected_candidate_id), None)
    selected_geometry = None
    if selected_proposal is not None:
        selected_geometry = bowl_grasp_geometry_metrics(
            selected_proposal, object_pose_world=case.bowl_pose,
            base_position_world=position, table_normal_world=normal,
        )
    write_candidate_inventory(output, case.candidates, selected.plan.to_dict(),
                              candidate_order=[item.candidate_id for item in selected.proposals])
    search_attempts = [row for row in selected.plan.attempts
                       if row.get("stage") in {"pregrasp_ik", "pregrasp"}]
    reuse_count = sum(bool(row.get("result", {}).get("metrics", {}).get("request_reuse"))
                      for row in search_attempts)
    evidence.update(
        ik_batch_screening=[row for row in selected.plan.attempts if row.get("stage") == "ik_batch_screening"],
        pregrasp_search_summary={
            "logical_requests": len(search_attempts),
            "backend_evaluated_requests": len(search_attempts) - reuse_count,
            "reused_requests": reuse_count,
            "expanded_grasp_candidates": len({row["candidate_id"] for row in search_attempts}),
        },
        selected_grasp_geometry=selected_geometry,
        pregrasp_joint_travel_limit_rad=settings.max_pregrasp_joint_travel_rad,
        approach_joint_travel_limit_rad=settings.max_approach_joint_travel_rad,
        candidate_order=[item.candidate_id for item in selected.proposals],
        selected_candidate_id=selected.plan.selected_candidate_id,
        stage_collision_snapshots={
            "pregrasp": selected.strict_snapshot["snapshot_sha256"],
            "approach": selected.contact_snapshot["snapshot_sha256"],
        },
        plan=selected.plan.to_dict(),
    )
    from gene_agent.feedback.curobo_log_policy import write_selection_report
    write_selection_report(evidence_path, evidence)
    planned_prefix = all(
        selected.plan.stage(name).status == "planned"
        for name in ("open", "pregrasp", "approach", "close")
    )
    report = {
        **evidence,
        "schema_version": 1,
        "kind": "observed_bowl_pick_plan_only",
        "object_id": _BOWL_INSTANCE,
        "source_frame_number": target["source_frame_number"],
        "source_timestamp_ms": target["source_timestamp_ms"],
        "base_collision_snapshot_sha256": target["collision_snapshot_sha256"],
        "bowl_pose_world": target["bowl_pose_world"].tolist(),
        "scene_config_sha256": collision_scene["provenance"]["scene_config_sha256"],
        "active_group": case.snapshot["active_group"],
        "world_frame": case.snapshot["world_frame"],
        "stage_collision_snapshot_sha256": {
            "pregrasp": selected.strict_snapshot["snapshot_sha256"],
            "approach": selected.contact_snapshot["snapshot_sha256"],
        },
        "skill_profile": profile_name,
        "skill_profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "candidate_selection_ref": str(evidence_path),
        "grasp_prefix_planned": planned_prefix,
        "physical_pick_success": False,
        "claim": "只规划抓碗前缀；未执行夹爪、物理夹持或抬升",
        "planning_wall_s": time.perf_counter() - started,
        "plan": selected.plan.to_dict(),
    }
    from .planning_reuse import summarize_selection
    from os import getpid
    report["reuse_process_id"] = getpid()
    report["planning_reuse_identity"] = _planning_reuse_identity(collision_scene, selected.strict_snapshot)
    report["planning_work_summary"] = summarize_selection(report["plan"])
    (output / "planning_work_summary.json").write_text(
        json.dumps(report["planning_work_summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_selection_report(output / "selection.json", report)
    return report


_OPERATIONS = {
    "pregrasp": MotionOperation.PLAN,
    "approach": MotionOperation.LINEAR_TCP,
}


def _nonnegative_finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} 必须是非负有限数")
    return number


def _original_request(selection_report: Mapping, stage_name: str) -> MotionRequest:
    if not selection_report.get("grasp_prefix_planned") or not selection_report.get("selected_candidate_id"):
        raise ValueError("原抓碗候选尚未得到完整 pregrasp/approach 前缀")
    matches = [row for row in selection_report["plan"]["stages"]
               if row["name"] == stage_name]
    if len(matches) != 1 or matches[0]["status"] != "planned":
        raise ValueError(f"原抓碗阶段未规划成功：{stage_name}")
    row = matches[0]
    request_data = row.get("request")
    if request_data is None and stage_name == "approach":
        segments = (row.get("details") or {}).get("segments") or ()
        translations = [item.get("request") for item in segments
                        if item.get("phase") == "approach_translate"]
        if len(translations) != 1:
            raise ValueError("组合 approach 缺少唯一的直线平移请求")
        request_data = translations[0]
    request = motion_request_from_dict(request_data)
    if request.operation is not _OPERATIONS[stage_name]:
        raise ValueError(f"原抓碗阶段的路径类型与 {stage_name} 不一致")
    if request.target.frame != "world_tcp":
        raise ValueError("原抓碗阶段目标必须在 world_tcp 坐标系")
    return request


def _check_anchor(
    collision_scene: Mapping, selection_report: Mapping, *,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    max_gripper_width_error_m: float,
) -> tuple[float, dict]:
    translation_limit = _nonnegative_finite(max_bowl_translation_m, name="Bowl 平移容差")
    rotation_limit = _nonnegative_finite(max_bowl_rotation_rad, name="Bowl 转角容差")
    width_limit = _nonnegative_finite(max_gripper_width_error_m, name="夹爪开口容差")
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    if (provenance.get("physics_stepped") is not False
            or provenance.get("pose_frame_verified") is not True
            or provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("抓碗阶段需要同帧现场观测及一致的碰撞快照")
    if provenance.get("scene_config_sha256") != selection_report["scene_config_sha256"]:
        raise ValueError("Bowl 抓取场景资产版本已变化，必须重新选择候选")
    if (snapshot.get("active_group") != selection_report["active_group"]
            or snapshot.get("world_frame") != selection_report["world_frame"]):
        raise ValueError("Bowl 抓取机器人组或 world frame 已变化")
    profile_sha = hashlib.sha256(_PROFILE_PATH.read_bytes()).hexdigest()
    if profile_sha != selection_report["skill_profile_sha256"]:
        raise ValueError("Bowl 抓取 Skill profile 已变化，必须重新选择候选")
    old_pose = validate_pose(selection_report["bowl_pose_world"], field_name="原 Bowl 目标锚点")
    aligned = provenance.get("aligned_object_poses_world")
    if not isinstance(aligned, Mapping) or "Bowl/Bowl_0" not in aligned:
        raise ValueError("现场缺少已对齐资产的 Bowl pose")
    new_pose = validate_pose(aligned["Bowl/Bowl_0"], field_name="当前 Bowl pose")
    translation = float(np.linalg.norm(old_pose[:3, 3] - new_pose[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        old_pose[:3, :3].T @ new_pose[:3, :3],
    ).as_rotvec()))
    if translation > translation_limit + 1e-9 or rotation > rotation_limit + 1e-9:
        raise ValueError(
            f"Bowl pose 已偏离抓取候选：{translation:.4f} m / {rotation:.4f} rad"
        )
    stages = selection_report["plan"]["stages"]
    opens = [row for row in stages if row["name"] == "open"]
    if len(opens) != 1 or opens[0]["status"] != "planned":
        raise ValueError("原抓碗 Skill 缺少已规划的开爪阶段")
    planned_width = float(opens[0]["details"]["width_m"])
    group = snapshot["active_group"]
    measured_width = float(provenance["gripper_widths_m"][group])
    if (not math.isfinite(planned_width) or not math.isfinite(measured_width)
            or abs(measured_width - planned_width) > width_limit + 1e-9):
        raise ValueError(
            f"夹爪实测开口 {measured_width:.4f} m 与原抓碗候选"
            f" {planned_width:.4f} m 不一致"
        )
    return measured_width, {
        "bowl_translation_error_m": translation,
        "bowl_rotation_error_rad": rotation,
        "gripper_width_error_m": abs(measured_width - planned_width),
        "max_bowl_translation_m": translation_limit,
        "max_bowl_rotation_rad": rotation_limit,
        "max_gripper_width_error_m": width_limit,
    }


def _approach_contact_postcheck(collision_scene: Mapping, result, snapshot: Mapping) -> dict:
    """复用原物理抓碗的逐点接触复核；只允许指垫触碰目标碗。"""
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    qpos = np.asarray(collision_scene["snapshot"]["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("现场 Bowl 接触复核的 qpos 与资产模型不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    robot = collision_scene["robot_metadata"]
    group = collision_scene["snapshot"]["active_group"]
    return validate_planned_target_contacts(
        model, data, robot, group=group, object_body="bowl",
        trajectory=result.trajectory,
        gripper_joint_positions_m=snapshot["gripper_snapshot"]["joint_positions_m"],
        allowed_geom_names=robot["grippers"][group]["pad_geom_names"],
    )


def plan_observed_bowl_motion_stage(
    collision_scene: Mapping, selection_report: Mapping, *, stage_name: str,
    planner_pool, output_dir: str | Path,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    max_gripper_width_error_m: float,
    scene_revision: int = 0, planner_mode: PlannerMode | str | None = None,
) -> dict:
    """保留原目标/路径/碰撞策略，只用新观测重设起点和碰撞快照。

    `pregrasp` 用完整障碍，`approach` 仅从规划 world 移除 Bowl；
    两段都依赖真实开爪读数。返回 `SkillStage` 可交真机 runtime 单段
    预检；每段执行后调用方须重新观测，不能把预测终点串为下一段起点。
    """
    if stage_name not in _OPERATIONS:
        raise ValueError(f"不支持的抓碗运动阶段：{stage_name}")
    original = _original_request(selection_report, stage_name)
    width, anchor = _check_anchor(
        collision_scene, selection_report,
        max_bowl_translation_m=max_bowl_translation_m,
        max_bowl_rotation_rad=max_bowl_rotation_rad,
        max_gripper_width_error_m=max_gripper_width_error_m,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    strict, contact = build_observed_bowl_grasp_stage_snapshots(
        collision_scene, open_width_m=width, output_dir=output,
    )
    source = collision_scene["snapshot"]

    def derived_scene(snapshot):
        return {
            **collision_scene,
            "snapshot": snapshot,
            "provenance": {
                **collision_scene["provenance"],
                "collision_snapshot_sha256": snapshot["snapshot_sha256"],
                "derived_from_collision_snapshot_sha256": source["snapshot_sha256"],
                "derived_planning_gripper_width_m": width,
            },
        }

    stage_snapshot = strict if stage_name == "pregrasp" else contact
    stage_scene = derived_scene(stage_snapshot)
    mode = PlannerMode(planner_mode or original.planner_mode)
    context = prepare_plan_only_context(
        stage_scene, group=original.group, planner_pool=planner_pool,
        scene_revision=scene_revision, seed=original.seed, planner_mode=mode,
    )
    request = MotionRequest(
        f"{original.request_id}:observed_{collision_scene['provenance']['source_frame_number']}",
        original.operation, original.group, context["start"], original.target,
        context["scene"], original.collision_policy, seed=original.seed,
        planner_mode=mode, approximate_ik_budget=original.approximate_ik_budget,
        joint_target=original.joint_target,
    )
    from .planning_reuse import reuse_selected_motion
    result, composition, reuse_reason = reuse_selected_motion(
        selection_report, stage_name, request,
        identity=_planning_reuse_identity(collision_scene, strict),
        strict_snapshot_sha256=strict["snapshot_sha256"])
    strict_context = None
    if result is not None:
        pass
    elif stage_name == "approach":
        strict_context = prepare_plan_only_context(
            derived_scene(strict), group=original.group, planner_pool=planner_pool,
            scene_revision=scene_revision, seed=original.seed, planner_mode=mode,
        )
        if strict_context["start"] != context["start"]:
            raise RuntimeError("approach 严格层与接触层的实测起点不一致")
        from gene_agent.skills.library.pick.approach import plan_aligned_approach
        result, composition, _ = plan_aligned_approach(
            strict_context["planner"], context["planner"], request,
            strict_context["scene"],
        )
    else:
        result = context["planner"].plan(request)
        if result.request_id != request.request_id or result.operation is not request.operation:
            raise RuntimeError("抓碗阶段 planner 返回的请求 ID 或路径类型不一致")
    if result.success and (result.trajectory is None or result.goal_state is None):
        raise RuntimeError("抓碗阶段成功结果缺少连续轨迹或终点关节")
    postcheck = None
    if result.success and stage_name == "approach" and resolve_collision_mode(stage_snapshot.get("collision_mode")) is CollisionMode.FULL:
        try:
            postcheck = _approach_contact_postcheck(
                collision_scene, result, stage_snapshot,
            )
        except Exception as error:
            # 接触复核本身失败也不能将轨迹交给真机。保留异常供调试，
            # 不把未完成的检查误报为零非法接触。
            postcheck = {"success": False, "source": "contact_postcheck_error",
                         "error": f"{type(error).__name__}: {error}"}
    accepted = bool(result.success and
                    (postcheck is None or postcheck.get("success") is True))
    stage = SkillStage(
        stage_name, "planned" if accepted else "failed",
        request.target, request, result,
        {"source_selected_candidate_id": selection_report["selected_candidate_id"],
         "anchor_check": anchor,
         "path_kind": request.operation.value,
         "approach_composition": composition,
         "target_contact_postcheck": postcheck,
         "derived_from_collision_snapshot_sha256": source["snapshot_sha256"]},
    )
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_motion_stage_plan_only",
        "stage_name": stage_name,
        "selection_plan_reuse": reuse_reason,
        "success": accepted,
        "planning_success": result.success,
        "failure_code": ("collision_postcheck_rejected" if result.success and not accepted
                         else None),
        "contact_postcheck": postcheck,
        "source_frame_number": collision_scene["provenance"]["source_frame_number"],
        "base_collision_snapshot_sha256": source["snapshot_sha256"],
        "stage_collision_snapshot_sha256": stage_snapshot["snapshot_sha256"],
        "planning_wall_s": time.perf_counter() - started,
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "strict_planner_cache_hit": (None if strict_context is None
                                     else strict_context["planner_cache_hit"]),
        "strict_planner_setup_duration_s": (None if strict_context is None
                                             else strict_context["planner_setup_duration_s"]),
        "anchor_check": anchor,
        "claim": "当前实测起点的单阶段规划；未执行机器人运动",
        "stage": stage.to_dict(),
    }
    (output / "stage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "stage": stage}


def _pose_stage(name: str, row: dict) -> SkillStage:
    target = row.get("target")
    if row.get("status") != "planned" or not isinstance(target, dict):
        raise ValueError(f"原抓碗 {name} 阶段缺少已规划目标")
    return SkillStage(name, "planned", PoseTarget(target["frame"], target["matrix"]))


def plan_observed_bowl_lift(
    collision_scene: dict, selection_report: dict, grasp_report: dict,
    attachment: dict, *, planner_pool, output_dir: str | Path,
    scene_revision: int = 5,
) -> dict:
    """从实际闭爪终态筛旧 post-pick 候选，再做 TCP 直线 lift。

    附着代理必须来自新实测场景和已通过的夹持检查。原抓碗选中目标和
    旋转候选顺序保持不变；唯一替换的是 approach 的预测 goal_state，
    改为闭爪后现场实测关节。这里只规划，不发送机器人命令。
    """
    evidence = grasp_report.get("physical_grasp_check") or {}
    preview = predicted_report(collision_scene, grasp_report, "hold")
    if not preview and (grasp_report.get("execute") is not True
            or grasp_report.get("status") != "executed"
            or grasp_report.get("grasp_success") is not True
            or evidence.get("status") != "pass"):
        raise ValueError("lift 需要已执行且现场确认通过的 Bowl 夹持")
    if attachment.get("physical_hold_evidence") != evidence:
        raise ValueError("附着代理的夹持证据与刚执行的抓碗 Skill 不一致")
    source = collision_scene["snapshot"]
    if attachment.get("base_collision_snapshot_sha256") != source["snapshot_sha256"]:
        raise ValueError("附着代理不属于当前闭爪后的现场碰撞快照")
    attached_snapshot = attachment["snapshot"]
    attached_provenance = attachment["provenance"]
    if attached_provenance.get("collision_snapshot_sha256") != attached_snapshot.get("snapshot_sha256"):
        raise ValueError("附着代理来源与碰撞快照 hash 不一致")
    selected = selection_report.get("selected_candidate_id")
    if not selected or selected != grasp_report.get("selected_candidate_id"):
        raise ValueError("lift 的 Bowl 候选与已执行的抓取不一致")
    if source["active_group"] != "right" or attached_snapshot.get("active_group") != "right":
        raise ValueError("原 task_1 抓碗 lift 当前只支持右臂")
    profile_bytes = _PROFILE_PATH.read_bytes()
    profile_sha = hashlib.sha256(profile_bytes).hexdigest()
    if (selection_report.get("skill_profile_sha256") != profile_sha
            or (attachment.get("planning_attachment_enabled", True)
                and attachment.get("profile_sha256", profile_sha) != profile_sha)):
        raise ValueError("Bowl Skill 或附着球 profile 在规划前已变化")
    profile_name = selection_report["skill_profile"]
    settings = load_grasp_settings(_PROFILE_PATH, profile_name)
    original_rows = {row["name"]: row for row in selection_report["plan"]["stages"]}
    executed_rows = [row for row in grasp_report["stages"] if row["stage"] == "approach"]
    if len(executed_rows) != 1 or executed_rows[0]["status"] != ("planned" if preview else "executed"):
        raise ValueError("lift 缺少已执行的 approach 阶段")
    actual_row = executed_rows[0]["plan"]["stage"]
    if actual_row.get("status") != "planned":
        raise ValueError("已执行 approach 缺少有效规划事实")
    expected = _pose_stage("approach", original_rows["approach"])
    actual = _pose_stage("approach", actual_row)
    if not np.allclose(expected.target.matrix, actual.target.matrix, rtol=0, atol=1e-8):
        raise ValueError("已执行 approach 目标与原选中 Bowl 候选不同")
    old_request = motion_request_from_dict(actual_row["request"])
    old_result = motion_result_from_dict(actual_row["result"])
    if not old_result.success or old_result.goal_state is None or old_request.operation.value != "linear_tcp":
        raise ValueError("已执行 approach 的原始直线规划事实不完整")
    output = Path(output_dir).resolve()
    started = time.perf_counter()
    context = prepare_plan_only_context(
        attachment, group="right", planner_pool=planner_pool,
        scene_revision=scene_revision, seed=old_request.seed,
        planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK,
        attachment_spheres_tool=attachment["spheres_tool_m"],
    )
    # 原 Skill 后续只读取 approach.result.goal_state。这里明确替换为
    # 闭爪后的现场实测关节，不能把预测轨迹终点继续串到 lift。
    measured_result = replace(
        old_result, goal_state=context["start"],
        metrics={**old_result.metrics, "lift_start_source": "measured_after_close"},
    )
    initial = SkillPlan("pick", selected, (
        SkillStage("open", "planned", details=original_rows["open"].get("details", {})),
        _pose_stage("pregrasp", original_rows["pregrasp"]),
        SkillStage("approach", "planned", actual.target,
                   old_request, measured_result,
                   {"goal_state_source": "measured_after_close"}),
        SkillStage("close", "planned", details=original_rows["close"].get("details", {})),
        SkillStage("lift", "not_executed"),
        SkillStage("hold", "not_executed"),
    ), ())
    attached_context = SkillContext(
        "right", context["start"], context["scene"], "bowl", old_request.seed,
    )
    from gene_agent.tasks.policies.microwave_task.planning.bowl_lift import (
        make_bowl_lift_candidates,
    )
    actual_tcp = context["planner"].fk(context["start"])
    lift_candidates = make_bowl_lift_candidates(
        actual_tcp, settings.lift_distance_m,
        selection_report["right_arm_base_position_world"],
    )
    plan = PickSkill().plan_visual_retreat(
        context["planner"], initial, attached_context, settings,
        candidate_poses=lift_candidates,
    )
    lift = plan.stage("lift")
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_lift_plan_only",
        "success": lift.status == "planned",
        "selected_candidate_id": selected,
        "measured_start": context["start"].to_dict(),
        "base_collision_snapshot_sha256": source["snapshot_sha256"],
        "attached_collision_snapshot_sha256": attached_snapshot["snapshot_sha256"],
        "attachment_profile_sha256": profile_sha,
        "physical_hold_evidence": evidence,
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "planning_wall_s": time.perf_counter() - started,
        "claim": "按仿真 Pick 逻辑从实际闭爪状态筛选碗抬升候选；未执行运动或 hold",
        "plan": plan.to_dict(),
        "lift_stage": lift.to_dict(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "lift_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "lift_stage": lift}


__all__ = ['build_observed_bowl_grasp_proposals', 'build_observed_bowl_grasp_stage_snapshots', 'plan_observed_bowl_grasp_stages', 'plan_observed_bowl_motion_stage', 'plan_observed_bowl_lift']
