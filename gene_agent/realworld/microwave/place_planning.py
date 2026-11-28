"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

from .prediction import is_prediction, predicted_report, reject_assumption_on_live_scene

import json
import math
import time
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.planning.extract import LegacyPlaceLinearPlanner, allow_placed_object_snapshot, robot_body_only_snapshot
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    LEGACY_TABLE_CONFIG,
    load_object_operation_profile,
)
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    align_table_tcp_to_surface,
)
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.knowledge.geometry.transforms import pose_delta as _pose_delta
from gene_agent.motion.model import (
    CollisionPolicy,
    MotionOperation,
    MotionRequest,
    PoseTarget,
)
from gene_agent.realworld.microwave.attachment import prepare_observed_bowl_attachment
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.realworld.scene import observed_tcp_pose as _observed_tcp_pose
from gene_agent.skills.runtime.planning import SkillContext, SkillStage
from gene_agent.skills.library.transfer.planning.placement import PlacementTargets, PlaceObjectSkill
from gene_agent.tasks.checks import CheckResult

_ROOT = Path(__file__).resolve().parents[3]
_BOWL = "Bowl/Bowl_0"
_SIMULATION_HOLD_SOURCES = frozenset({
    "simulation_privileged_weld",
    "mujoco_contact_geometry_at_planned_waypoints",
})


def build_observed_table_place_targets(
    collision_scene: Mapping, *, table_height_alignment: Mapping,
    current_hold_check: CheckResult,
) -> dict:
    """按当前夹持关系和桌面几何反算直立碗目标，工作区为显式任务参数。"""
    snapshot = collision_scene["snapshot"]
    source = collision_scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("桌面放置需要同帧现场观测与碰撞快照")
    if snapshot.get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 桌面放置目前只验收右臂")
    if not isinstance(current_hold_check, CheckResult) or not current_hold_check.passed:
        raise ValueError("桌面放置前必须重新确认当前仍持碗")
    reject_assumption_on_live_scene(collision_scene, current_hold_check.to_dict())
    if current_hold_check.source in _SIMULATION_HOLD_SOURCES:
        raise ValueError("仿真夹持证据不能用作真机持碗检查")
    if (current_hold_check.evidence.get("collision_snapshot_sha256")
            != snapshot.get("snapshot_sha256")
            or current_hold_check.evidence.get("source_frame_number")
            != source.get("source_frame_number")):
        raise ValueError("持碗检查不是当前现场快照，必须重新确认")
    if (table_height_alignment.get("accepted_for_planning") is not True
            or not table_height_alignment.get("source")):
        raise ValueError("桌面高度对齐未验收，不能生成真机放置目标")
    bias = float(table_height_alignment["height_bias_m"])
    if not math.isfinite(bias):
        raise ValueError("桌面高度偏差必须是有限值")

    aligned = source.get("aligned_object_poses_world") or {}
    if _BOWL not in aligned or "table_pose_world" not in source:
        raise ValueError("现场快照缺少 Bowl 或桌面 pose")
    bowl = validate_pose(aligned[_BOWL], field_name="T_world_bowl")
    table = validate_pose(source["table_pose_world"], field_name="T_world_table")

    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    qpos = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("桌面放置的现场 qpos 与规划资产不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    geom = data.geom("scene_table")
    actual_table = np.eye(4)
    actual_table[:3, :3] = geom.xmat.reshape(3, 3)
    actual_table[:3, 3] = geom.xpos
    if not np.allclose(actual_table, table, atol=1e-6, rtol=0):
        raise ValueError("桌面现场 pose 与规划碰撞资产不一致")
    bowl_body = data.body("bowl")
    actual_bowl = np.eye(4)
    actual_bowl[:3, :3] = bowl_body.xmat.reshape(3, 3)
    actual_bowl[:3, 3] = bowl_body.xpos
    if not np.allclose(actual_bowl, bowl, atol=1e-6, rtol=0):
        raise ValueError("Bowl 现场 pose 与规划碰撞资产不一致")
    normal = actual_table[:3, 2]
    if normal[2] < .995:
        raise ValueError("桌面倾斜过大，旧 Place 的世界 Z 规则不适用")
    current_top = float((geom.xpos + normal * model.geom("scene_table").size[2])[2])
    old = json.loads(LEGACY_TABLE_CONFIG.read_text(encoding="utf-8"))[
        "legacy_virtual_table"]
    reference_top = float(old["center_world_m"][2] + old["size_xyz_m"][2] / 2)
    profile = load_object_operation_profile(_ROOT / "data/task_1", "Bowl_0")
    original = PoseTarget("world_tcp", profile["table_tcp_pose"])
    table_tcp = align_table_tcp_to_surface(
        original, current_top_z_m=current_top,
        reference_top_z_m=reference_top, height_bias_m=bias,
    )
    world_tcp = _observed_tcp_pose(collision_scene)
    tcp_bowl = validate_pose(np.linalg.inv(world_tcp) @ bowl, field_name="T_tcp_bowl")
    from .strategy_planning import frozen_geometry, strategy_settings
    from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import object_relative_table_target
    target, tcp_bowl, object_goal = object_relative_table_target(
        frozen_geometry(collision_scene), strategy_settings()["table_object_goal"], bowl, world_tcp)
    table_tcp = PoseTarget("world_tcp", target)
    targets = PlacementTargets.from_attachment(table_tcp, tcp_bowl)
    return {
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "hold_evidence": current_hold_check.to_dict(),
        "table_pose_world": table.tolist(),
        "table_alignment": {
            "source": table_height_alignment["source"],
            "height_bias_m": bias,
            "current_surface_top_z_m": current_top,
            "reference_surface_top_z_m": reference_top,
            "policy": "keep_old_world_xy_and_orientation_follow_table_top_z",
        },
        "T_tcp_bowl_from_actual_hold": tcp_bowl.tolist(),
        "original_table_tcp_world": original.matrix.tolist(),
        "keyposes_world_tcp": {
            "preplace": targets.preplace.matrix.tolist(),
            "place": targets.place.matrix.tolist(),
            "retreat": targets.retreat.matrix.tolist(),
        },
        "target_object_pose_world": targets.target_object_pose.tolist(),
        "object_goal": object_goal,
        "target_policy": "upright_object_goal_from_actual_grasp_in_front_table_region",
        "stage_order": ("preplace", "place", "release", "detach", "retreat"),
        "claim": "只生成旧桌面 Place 目标；尚未规划、松爪、执行或验证放置",
    }


def plan_observed_table_place_prefix(
    collision_scene: dict, grasp_report: dict, *,
    current_hold_check: CheckResult, table_height_alignment: dict,
    planner_pool, output_dir: str | Path, seed: int = 123,
    scene_revision: int = 36,
) -> dict:
    """从实测持碗起点做旧 preplace/下降规划，不发送真机命令。

    桌面高度只由已验收的现场表面输入修正。旧 Place 的目标、预放置
    IK/直线筛选、最终下降重试和 robot_body 碰撞范围保持原有顺序。
    若实际执行 preplace，下降前还须由新观测重新规划。
    """
    if collision_scene["snapshot"].get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 桌面 Place 目前只验收右臂")
    targets_report = build_observed_table_place_targets(
        collision_scene, table_height_alignment=table_height_alignment,
        current_hold_check=current_hold_check,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    attachment = prepare_observed_bowl_attachment(
        collision_scene, grasp_report,
        output_dir=output / "attachment",
        hold_evidence=current_hold_check.to_dict(),
        sphere_source="legacy_extract_surface_32",
    )
    if attachment.get("planning_attachment_enabled", True) and len(attachment["spheres_tool_m"]) != 32:
        raise ValueError("旧桌面 Place 的携物碰撞代理必须为 32 球")

    from .strategy_planning import plan_stage, write_report
    row = plan_stage(attachment, "preplace", targets_report["keyposes_world_tcp"]["preplace"],
        planner_pool=planner_pool, output=output/"preplace", seed=seed,
        spheres=attachment["spheres_tool_m"], tcp_bowl=targets_report["T_tcp_bowl_from_actual_hold"],
        max_tilt_deg=None, max_attempts=5)
    report = {**row, "stage": row["stage"].to_dict(), "targets": targets_report,
        "attached_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        "plan": row["stage"].to_dict()}
    write_report(output/"table_plan.json", report)
    return {**report, "preplace_stage": row["stage"]}


def plan_observed_table_descent(
    collision_scene: dict, initial_targets: dict, grasp_report: dict, *,
    current_hold_check: CheckResult, table_height_alignment: dict,
    planner_pool, output_dir: str | Path,
    max_table_height_delta_m: float,
    max_hold_translation_m: float, max_hold_rotation_rad: float,
    seed: int = 123, scene_revision: int = 37,
) -> dict:
    """按仿真同一全场景TCP直线合同下降，不将普通pose规划替代直线。"""
    if collision_scene["snapshot"].get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 桌面 Place 目前只验收右臂")
    limits = (max_table_height_delta_m, max_hold_translation_m,
              max_hold_rotation_rad)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("桌面高度和持碗容差必须为正有限值")
    source = collision_scene["provenance"]
    snapshot = collision_scene["snapshot"]
    if not is_prediction(collision_scene) and int(source["source_frame_number"]) <= int(initial_targets["source_frame_number"]):
        raise ValueError("桌面下降必须使用预放置执行后的新现场观测帧")
    if (not isinstance(current_hold_check, CheckResult)
            or not current_hold_check.passed
            or current_hold_check.evidence.get("collision_snapshot_sha256")
            != snapshot.get("snapshot_sha256")
            or current_hold_check.evidence.get("source_frame_number")
            != source.get("source_frame_number")):
        raise ValueError("桌面下降需要当前观测帧的持碗检查")
    current = build_observed_table_place_targets(
        collision_scene, table_height_alignment=table_height_alignment,
        current_hold_check=current_hold_check,
    )
    top_delta = abs(
        float(current["table_alignment"]["current_surface_top_z_m"])
        - float(initial_targets["table_alignment"]["current_surface_top_z_m"])
    )
    if top_delta > max_table_height_delta_m:
        raise ValueError("桌面顶高在预放置后变化，必须重新确定放置目标")
    hold_error = _pose_delta(
        initial_targets["T_tcp_bowl_from_actual_hold"],
        current["T_tcp_bowl_from_actual_hold"], field="TCP→Bowl 持碗关系",
    )
    if (hold_error[0] > max_hold_translation_m
            or hold_error[1] > max_hold_rotation_rad):
        raise ValueError("Bowl 与 TCP 的持碗关系在预放置后已改变")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    attachment = prepare_observed_bowl_attachment(
        collision_scene, grasp_report,
        output_dir=output / "attachment",
        hold_evidence=current_hold_check.to_dict(),
        sphere_source="legacy_extract_surface_32",
    )
    spheres = attachment["spheres_tool_m"]
    if attachment.get("planning_attachment_enabled", True) and len(spheres) != 32:
        raise ValueError("旧桌面下降的附着碰撞代理必须为 32 球")

    from .strategy_planning import plan_stage, strategy_settings
    return plan_stage(attachment, "place", initial_targets["keyposes_world_tcp"]["place"],
        planner_pool=planner_pool, output=output, seed=seed,
        spheres=spheres, tcp_bowl=current["T_tcp_bowl_from_actual_hold"], linear=True,
        max_tilt_deg=None, max_attempts=5, optimizer_clearance=strategy_settings()[
            "table_object_goal"]["place_optimizer_clearance_m"])


def plan_observed_table_detached_retreat(
    collision_scene: dict, targets_report: dict, release_report: dict, *,
    planner_pool, output_dir: str | Path,
    max_table_translation_m: float, max_table_rotation_rad: float,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    seed: int = 123, scene_revision: int = 38,
) -> dict:
    """只从规划 world 排除已经放下的 Bowl，保留桌面和微波炉。

    真机不执行仿真 detach：松爪与物体离手由现场检查确认。接着导出
    当前实测状态的全场景碰撞，再按旧 Skill 的接触白名单单独排除 Bowl。
    """
    if collision_scene["snapshot"].get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 桌面撤离目前只验收右臂")
    limits = (max_table_translation_m, max_table_rotation_rad,
              max_bowl_translation_m, max_bowl_rotation_rad)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("桌面撤离的位姿容差必须为正有限值")
    check = release_report.get("postcondition") or {}
    if not predicted_report(collision_scene, release_report, "release") and (release_report.get("status") != "executed"
            or release_report.get("release_success") is not True
            or check.get("status") != "pass" or not check.get("source")):
        raise ValueError("桌面松爪后置条件未确认，不能撤离")
    reject_assumption_on_live_scene(collision_scene, check)
    if check["source"] in {
        "simulation_privileged_weld", "mujoco_contact_geometry_at_planned_waypoints",
    }:
        raise ValueError("仿真松爪证据不能用于真机桌面撤离")
    source = collision_scene["provenance"]
    snapshot = collision_scene["snapshot"]
    if not is_prediction(collision_scene) and int(source["source_frame_number"]) <= int(release_report["source_frame_number"]):
        raise ValueError("松爪后必须读取更新的现场观测帧")
    table_error = _pose_delta(
        targets_report["table_pose_world"], source["table_pose_world"],
        field="桌面 pose",
    )
    bowl_error = _pose_delta(
        release_report["bowl_pose_world_after_release"],
        source["aligned_object_poses_world"]["Bowl/Bowl_0"],
        field="松爪后的 Bowl pose",
    )
    if (table_error[0] > max_table_translation_m
            or table_error[1] > max_table_rotation_rad):
        raise ValueError("桌面在松爪后发生移动，不能沿旧目标撤离")
    if (bowl_error[0] > max_bowl_translation_m
            or bowl_error[1] > max_bowl_rotation_rad):
        raise ValueError("Bowl 在松爪后发生移动，不能沿旧目标撤离")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    detached_snapshot = allow_placed_object_snapshot(
        snapshot, output / "detached_world", object_body_name="bowl",
    )
    detached_scene = {
        **collision_scene,
        "snapshot": {**snapshot, **detached_snapshot},
        "provenance": {
            **source,
            "collision_snapshot_sha256": detached_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        },
    }
    context = prepare_plan_only_context(
        detached_scene, group="right", planner_pool=planner_pool,
        scene_revision=scene_revision, seed=seed,
    )
    target = PoseTarget("world_tcp", targets_report["keyposes_world_tcp"]["retreat"])
    request = MotionRequest(
        f"place:retreat:observed_{source['source_frame_number']}",
        MotionOperation.LINEAR_TCP, "right", context["start"], target,
        context["scene"], CollisionPolicy(), seed=seed,
    )
    result = context["planner"].plan(request)
    if result.request_id != request.request_id or result.operation is not request.operation:
        raise RuntimeError("桌面撤离 planner 返回的请求 ID 或路径类型不一致")
    if result.success and (result.trajectory is None or result.goal_state is None):
        raise RuntimeError("成功的桌面撤离缺少连续轨迹或终点关节")
    stage = SkillStage(
        "retreat", "planned" if result.success else "failed",
        target, request, result,
        {"table_pose_error_m_rad": table_error,
         "bowl_pose_error_m_rad": bowl_error,
         "ignored_world_obstacles": detached_snapshot.get("ignored_world_obstacles", [])},
    )
    report = {
        "schema_version": 1,
        "kind": "observed_table_detached_retreat_plan_only",
        "success": bool(result.success),
        "source_frame_number": source["source_frame_number"],
        "base_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "stage_collision_snapshot_sha256": detached_snapshot["snapshot_sha256"],
        "ignored_world_obstacles": detached_snapshot.get("ignored_world_obstacles", []),
        "table_pose_error_m_rad": table_error,
        "bowl_pose_error_m_rad": bowl_error,
        "stage": stage.to_dict(),
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "planning_wall_s": time.perf_counter() - started,
        "claim": "只规划松爪后的直线撤离；未执行或验证 Bowl 稳定留在桌面",
    }
    (output / "stage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "stage": stage}


__all__ = ['build_observed_table_place_targets', 'plan_observed_table_place_prefix', 'plan_observed_table_descent', 'plan_observed_table_detached_retreat']
