"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase, prepare_grasp_stage_snapshots
from gene_agent.tasks.policies.microwave_task.planning.grasp import legacy_bowl_proposals, plan_bowl_grasp_stages
from gene_agent.knowledge import bowl_grasp_candidates
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


def build_observed_bowl_grasp_proposals(collision_scene: Mapping) -> dict:
    """用同帧、已对齐资产的现场 pose 生成旧 task_1 的 40 个候选。

    候选数值来自原 Knowledge，世界方位优先级直接复用视觉 Skill
    使用的排序函数。这里不拿名义 MuJoCo pose 代替现场 pose，也不
    把候选可视化当作可达或已抓住的证据。
    """
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
    candidates = bowl_grasp_candidates(
        bowl_pose, pose_source="observed_scene_asset_aligned",
    )
    proposals = legacy_bowl_proposals(candidates, object_pose_world=bowl_pose)
    return {
        "object_id": _BOWL_INSTANCE,
        "pose_source": "observed_scene_asset_aligned",
        "source_frame_number": provenance["source_frame_number"],
        "source_timestamp_ms": provenance["source_timestamp_ms"],
        "collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "bowl_pose_world": bowl_pose,
        "proposals": proposals,
        "candidates": candidates,
    }


def _observed_case(
    collision_scene: Mapping, output_dir: str | Path,
) -> tuple[PickBowlCase, dict]:
    """把现场已测 qpos 放进仅用于 FK/碰撞的私有 MuJoCo 数据。"""
    target = build_observed_bowl_grasp_proposals(collision_scene)
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
    case = PickBowlCase(
        Path(output_dir).resolve(), 0, model, data, metadata, snapshot,
        target["bowl_pose_world"], target["candidates"],
    )
    return case, target


def build_observed_bowl_grasp_stage_snapshots(
    collision_scene: Mapping, *, open_width_m: float, output_dir: str | Path,
) -> tuple[dict, dict]:
    """按原抓碗函数导出开爪严格碰撞层和允许目标接触层。

    MuJoCo 在这里仅计算现有资产的 FK；只对私有数据副本设置规划
    开口，不推进动力学，也不改变现场场景的实测快照。接触层只移除
    碗作为规划障碍，绝不等于实机可以与其他部位任意接触。
    """
    case, _ = _observed_case(collision_scene, output_dir)
    return prepare_grasp_stage_snapshots(
        case, open_width_m, group=case.snapshot["active_group"],
    )


def plan_observed_bowl_grasp_stages(
    collision_scene: Mapping, *, output_dir: str | Path,
    profile_name: str = "pick_bowl_legacy_visual", seed: int = 123,
) -> dict:
    """现场抓碗只规划前缀，直接调用原物理探针所用 PickSkill 入口。

    返回的 pregrasp/approach 是预测轨迹。真机实际执行每段后须再读
    现场状态；闭爪、接触、抬升和 hold 尚未执行，因此不能报告 Pick
    成功。当前旧入口仅支持右臂，左臂必须另做等价验收后开放。
    """
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    case, target = _observed_case(collision_scene, output)
    if case.snapshot["active_group"] != "right":
        raise NotImplementedError("原 task_1 抓碗 Skill 规划目前只支持右臂")
    if profile_name not in {"pick_bowl", "pick_bowl_legacy_visual"}:
        raise ValueError(f"未知抓碗 Skill profile: {profile_name}")
    profile_bytes = _PROFILE_PATH.read_bytes()
    settings = load_grasp_settings(_PROFILE_PATH, profile_name)
    output.mkdir(parents=True, exist_ok=True)
    selected = plan_bowl_grasp_stages(case, settings, seed=seed)
    planned_prefix = all(
        selected.plan.stage(name).status == "planned"
        for name in ("open", "pregrasp", "approach", "close")
    )
    report = {
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
        "candidate_order_policy": "right_arm_base_side_first",
        "right_arm_base_position_world": case.data.body("right_base_link").xpos.tolist(),
        "pregrasp_joint_travel_limit_rad": settings.max_pregrasp_joint_travel_rad,
        "approach_joint_travel_limit_rad": settings.max_approach_joint_travel_rad,
        "candidate_order": [item.candidate_id for item in selected.proposals],
        "selected_candidate_id": selected.plan.selected_candidate_id,
        "grasp_prefix_planned": planned_prefix,
        "physical_pick_success": False,
        "claim": "只规划抓碗前缀；未执行夹爪、物理夹持或抬升",
        "planning_wall_s": time.perf_counter() - started,
        "plan": selected.plan.to_dict(),
    }
    (output / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
    request = motion_request_from_dict(matches[0]["request"])
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
    stage_snapshot = strict if stage_name == "pregrasp" else contact
    source = collision_scene["snapshot"]
    stage_scene = {
        **collision_scene,
        "snapshot": stage_snapshot,
        "provenance": {
            **collision_scene["provenance"],
            "collision_snapshot_sha256": stage_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": source["snapshot_sha256"],
            "derived_planning_gripper_width_m": width,
        },
    }
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
    )
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
         "target_contact_postcheck": postcheck,
         "derived_from_collision_snapshot_sha256": source["snapshot_sha256"]},
    )
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_motion_stage_plan_only",
        "stage_name": stage_name,
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
    if (grasp_report.get("execute") is not True
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
    if len(executed_rows) != 1 or executed_rows[0]["status"] != "executed":
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
    plan = PickSkill().plan_visual_retreat(
        context["planner"], initial, attached_context, settings,
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
        "claim": "仅复用原 PickSkill 规划 post-pick 筛选与直线 lift；未执行运动或 hold",
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
