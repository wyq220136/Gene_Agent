"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    extract_profile_targets,
)
from gene_agent.tasks.policies.microwave_task.planning.extract import robot_body_only_snapshot
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    load_object_operation_profile,
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
from gene_agent.skills.library.transfer.planning.extraction import (
    ExtractionTargets,
    ExtractObjectSkill,
    _plan_with_legacy_local_fallback,
)
from gene_agent.skills.runtime.planning import SkillContext, SkillStage
from gene_agent.tasks.checks import CheckResult

_ROOT = Path(__file__).resolve().parents[3]
_BOWL = "Bowl/Bowl_0"
_MICROWAVE = "Microwave/Microwave_0"
_NAMES = ("pre_extract_1", "pre_extract_2", "grasp",
          "post_extract_1", "post_extract_2")


def build_observed_bowl_extract_targets(
    collision_scene: dict, *, microwave_frame_alignment: dict,
    grasp_offset_world_m=(0., 0., 0.),
    min_open_angle_rad: float = math.radians(80.),
) -> dict:
    """使用旧 Bowl_0 profile 原值，现场数据只决定父 frame 与初态。

    旧物理探针的诊断偏移仅作用于最后一个 grasp TCP，范围各轴
    ±20 mm；其余四点不随偏移更改。桌面目标仍返回原 profile 世界
    pose，后续放桌面阶段再按当前桌面顶高做单独对齐。
    """
    snapshot = collision_scene["snapshot"]
    source = collision_scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("炉内取碗目标需要同帧现场观测与碰撞快照")
    if snapshot.get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 取出目前只验收右臂")
    if (microwave_frame_alignment.get("accepted_for_planning") is not True
            or not microwave_frame_alignment.get("source")):
        raise ValueError("微波炉 profile frame 对齐尚未验收")
    alignment = validate_pose(
        microwave_frame_alignment["T_scene_asset_profile"],
        field_name="T_scene_asset_profile",
    )
    offset = np.asarray(grasp_offset_world_m, dtype=float)
    if (offset.shape != (3,) or not np.isfinite(offset).all()
            or np.any(np.abs(offset) > .02)):
        raise ValueError("grasp 世界坐标偏移必须是各轴 ±20 mm 内的有限值")
    minimum = float(min_open_angle_rad)
    if not math.isfinite(minimum) or not 0 < minimum <= math.pi:
        raise ValueError("最小开门角必须是 (0, pi] 内的弧度")
    aligned = source.get("aligned_object_poses_world") or {}
    if _MICROWAVE not in aligned or _BOWL not in aligned:
        raise ValueError("取碗目标缺少同帧微波炉或 Bowl pose")
    microwave_pose = validate_pose(aligned[_MICROWAVE], field_name="T_world_microwave")
    bowl_pose = validate_pose(aligned[_BOWL], field_name="T_world_bowl")
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    qpos = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("取碗现场 qpos 与资产模型不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    door_angle = float(qpos[model.joint("microwave_door_joint").qposadr[0]])
    if abs(door_angle - float(source["door_angle_rad"])) > 1e-6:
        raise ValueError("取碗现场门角与同帧资产关节不一致")
    if door_angle < minimum:
        raise ValueError("微波炉门未完全打开，禁止按固定五点取碗")
    for body_name, expected in (("microwave", microwave_pose), ("bowl", bowl_pose)):
        body = data.body(body_name)
        actual = np.eye(4)
        actual[:3, :3] = body.xmat.reshape(3, 3)
        actual[:3, 3] = body.xpos
        if not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError(f"{body_name} 现场 pose 与取碗碰撞资产不一致")
    profile = load_object_operation_profile(_ROOT / "data/task_1", "Bowl_0")
    world_profile = microwave_pose @ alignment
    targets, table_tcp = extract_profile_targets(world_profile, profile)
    keyposes = {name: getattr(targets, name).matrix.copy() for name in _NAMES}
    keyposes["grasp"][:3, 3] += offset
    return {
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "frame_alignment_source": microwave_frame_alignment["source"],
        "T_world_profile_microwave": world_profile.tolist(),
        "bowl_pose_world": bowl_pose.tolist(),
        "door_angle_rad": door_angle,
        "minimum_open_angle_rad": minimum,
        "grasp_offset_world_m": offset.tolist(),
        "keyposes_world_tcp": {name: pose.tolist() for name, pose in keyposes.items()},
        "raw_table_tcp_world": table_tcp.matrix.tolist(),
        "stage_order": ("open", *_NAMES[:3], "close", "attach", *_NAMES[3:]),
        "claim": "只生成旧取碗固定五点；未求 IK、规划、夹持或撤离",
    }


def plan_observed_bowl_extract_prefix(
    collision_scene: dict, *, microwave_frame_alignment: dict,
    planner_pool, output_dir: str | Path, seed: int = 123,
    max_gripper_width_error_m: float = .002,
) -> dict:
    """夹爪已实际打开后，复用 ExtractObjectSkill 规划前三个运动阶段。

    原 Skill 的首点考虑完整场景，后两点仅保留机器人本体障碍；
    不在本函数执行开爪、夹取或物体 attach。真实执行须在每段后
    重新读取状态和碰撞快照，再规划下一段。
    """
    if collision_scene["snapshot"].get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 取碗目前只验收右臂")
    profile = load_object_operation_profile(_ROOT / "data/task_1", "Bowl_0")
    open_width = float(profile["grasp_pregrasp_opening_width_m"])
    close_width = float(profile["grasp_opening_width_m"])
    measured = float(collision_scene["provenance"]["gripper_widths_m"]["right"])
    tolerance = float(max_gripper_width_error_m)
    if (not math.isfinite(measured) or not math.isfinite(tolerance)
            or tolerance < 0 or abs(measured - open_width) > tolerance):
        raise ValueError("取碗规划前的实测夹爪开口与旧 profile 不一致")

    started = time.perf_counter()
    targets_report = build_observed_bowl_extract_targets(
        collision_scene, microwave_frame_alignment=microwave_frame_alignment,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    body_snapshot = robot_body_only_snapshot(
        collision_scene["snapshot"], output / "robot_body_only",
    )
    body_scene = {
        **collision_scene,
        "snapshot": {**collision_scene["snapshot"], **body_snapshot},
        "provenance": {
            **collision_scene["provenance"],
            "collision_snapshot_sha256": body_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": collision_scene[
                "snapshot"]["snapshot_sha256"],
        },
    }
    world = prepare_plan_only_context(
        collision_scene, group="right", planner_pool=planner_pool,
        scene_revision=30, seed=seed,
    )
    body = prepare_plan_only_context(
        body_scene, group="right", planner_pool=planner_pool,
        scene_revision=31, seed=seed,
    )
    if world["start"] != body["start"]:
        raise ValueError("完整场景与本体碰撞层的取碗实测起点不一致")
    keyposes = targets_report["keyposes_world_tcp"]
    targets = ExtractionTargets(*(
        PoseTarget("world_tcp", keyposes[name]) for name in (
            "pre_extract_1", "pre_extract_2", "grasp",
            "post_extract_1", "post_extract_2",
        )
    ))
    plan = ExtractObjectSkill().plan_grasp(
        world["planner"], body["planner"],
        SkillContext("right", world["start"], world["scene"], "bowl", seed),
        body["scene"], targets,
        pregrasp_width_m=open_width, grasp_width_m=close_width,
    )
    success = all(plan.stage(name).status == "planned"
                  for name in ("pre_extract_1", "pre_extract_2", "grasp"))
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_extract_prefix_plan_only",
        "success": success,
        "source_collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
        "robot_body_collision_snapshot_sha256": body_snapshot["snapshot_sha256"],
        "ignored_world_obstacles": body_snapshot.get("ignored_world_obstacles", []),
        "targets": targets_report,
        "measured_start": world["start"].to_dict(),
        "measured_open_width_m": measured,
        "profile_open_width_m": open_width,
        "profile_close_width_m": close_width,
        "planner_cache_hit": {
            "world": world["planner_cache_hit"],
            "robot_body": body["planner_cache_hit"],
        },
        "planner_setup_duration_s": (
            world["planner_setup_duration_s"] + body["planner_setup_duration_s"]
        ),
        "planning_wall_s": time.perf_counter() - started,
        "plan": plan.to_dict(),
        "claim": "仅从实测开爪状态规划旧取碗前三个运动阶段；未执行夹取或附着",
    }
    (output / "extract_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "pre_extract_1_stage": plan.stage("pre_extract_1")}


_OPERATIONS = {
    "pre_extract_1": MotionOperation.PLAN,
    "pre_extract_2": MotionOperation.LINEAR_TCP,
    "grasp": MotionOperation.LINEAR_TCP,
}


def plan_observed_bowl_extract_motion_stage(
    collision_scene: dict, initial_targets: dict, *, stage_name: str,
    microwave_frame_alignment: dict, planner_pool, output_dir: str | Path,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    seed: int = 123, scene_revision: int = 32,
) -> dict:
    """从实测关节、夹爪和 Bowl 状态规划一段，不串接预测终点。

    首段使用完整碰撞场景；接触附近两段保留原版 robot_body 和
    机器人自碰撞，明确列出停用的普通 world 障碍。执行许可与目标
    接触检查仍由上层现场子任务负责，规划成功不等于安全执行。
    """
    if stage_name not in _OPERATIONS:
        raise ValueError(f"不支持的取碗运动阶段：{stage_name}")
    if collision_scene["snapshot"].get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 取碗目前只验收右臂")
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_bowl_translation_m, max_bowl_rotation_rad)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("取碗锚点容差必须为正有限值")
    source = collision_scene["provenance"]
    profile = load_object_operation_profile(_ROOT / "data/task_1", "Bowl_0")
    expected_width = float(profile["grasp_pregrasp_opening_width_m"])
    measured_width = float(source["gripper_widths_m"]["right"])
    if (not math.isfinite(measured_width)
            or abs(measured_width - expected_width) > .002):
        raise ValueError("取碗接触前的实测夹爪开口与旧 profile 不一致")
    if (stage_name != "pre_extract_1" and
            int(source["source_frame_number"]) <= int(initial_targets["source_frame_number"])):
        raise ValueError("取碗后续阶段必须使用上一阶段执行后的新观测帧")
    current = build_observed_bowl_extract_targets(
        collision_scene, microwave_frame_alignment=microwave_frame_alignment,
    )
    microwave_error = _pose_delta(
        initial_targets["T_world_profile_microwave"],
        current["T_world_profile_microwave"], field="微波炉 profile frame",
    )
    bowl_error = _pose_delta(
        initial_targets["bowl_pose_world"], current["bowl_pose_world"],
        field="Bowl 取碗锚点",
    )
    if (microwave_error[0] > max_microwave_translation_m
            or microwave_error[1] > max_microwave_rotation_rad):
        raise ValueError("微波炉 pose 已偏离原取碗 keypose 锚点")
    if (bowl_error[0] > max_bowl_translation_m
            or bowl_error[1] > max_bowl_rotation_rad):
        raise ValueError("Bowl pose 已偏离原取碗 keypose 锚点")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = collision_scene["snapshot"]
    if stage_name == "pre_extract_1":
        planning_scene = collision_scene
        ignored = []
    else:
        body_snapshot = robot_body_only_snapshot(base, output / "robot_body_only")
        planning_scene = {
            **collision_scene,
            "snapshot": {**base, **body_snapshot},
            "provenance": {
                **source,
                "collision_snapshot_sha256": body_snapshot["snapshot_sha256"],
                "derived_from_collision_snapshot_sha256": base["snapshot_sha256"],
            },
        }
        ignored = body_snapshot.get("ignored_world_obstacles", [])
    context = prepare_plan_only_context(
        planning_scene, group="right", planner_pool=planner_pool,
        scene_revision=scene_revision, seed=seed,
    )
    target = PoseTarget("world_tcp", current["keyposes_world_tcp"][stage_name])
    request = MotionRequest(
        f"extract:{stage_name}:observed_{source['source_frame_number']}",
        _OPERATIONS[stage_name], "right", context["start"], target,
        context["scene"], CollisionPolicy(), seed=seed,
    )
    result = context["planner"].plan(request)
    if result.request_id != request.request_id or result.operation is not request.operation:
        raise RuntimeError("取碗阶段 planner 返回的请求 ID 或路径类型不一致")
    if result.success and (result.trajectory is None or result.goal_state is None):
        raise RuntimeError("成功的取碗运动阶段缺少连续轨迹或终点关节")
    stage = SkillStage(
        stage_name, "planned" if result.success else "failed",
        target, request, result,
        {"microwave_anchor_error_m_rad": microwave_error,
         "bowl_anchor_error_m_rad": bowl_error,
         "ignored_world_obstacles": ignored},
    )
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_extract_motion_stage_plan_only",
        "success": bool(result.success),
        "stage_name": stage_name,
        "source_frame_number": source["source_frame_number"],
        "base_collision_snapshot_sha256": base["snapshot_sha256"],
        "stage_collision_snapshot_sha256": planning_scene["snapshot"]["snapshot_sha256"],
        "ignored_world_obstacles": ignored,
        "microwave_anchor_error_m_rad": microwave_error,
        "bowl_anchor_error_m_rad": bowl_error,
        "stage": stage.to_dict(),
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "planning_wall_s": time.perf_counter() - started,
        "claim": "仅从实测状态规划当前取碗阶段；未执行或验证 Bowl 夹持",
    }
    (output / "stage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "stage": stage}


_RETREAT_NAMES = ("post_extract_1", "post_extract_2")
_SIMULATION_HOLD_SOURCES = frozenset({
    "simulation_privileged_weld",
    "mujoco_contact_geometry_at_planned_waypoints",
})


def plan_observed_bowl_extract_retreat_stage(
    collision_scene: dict, initial_targets: dict, grasp_report: dict, *,
    stage_name: str, current_hold_check: CheckResult,
    T_tcp_bowl_after_close, microwave_frame_alignment: dict,
    planner_pool, output_dir: str | Path,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_hold_translation_m: float, max_hold_rotation_rad: float,
    seed: int = 123, scene_revision: int = 34,
) -> dict:
    """用旧 32 球附着代理与 robot_body 场景，从实测关节规划一段。

    只有现场夹持检查通过才构造规划 attachment。该代理不会让真实 Bowl
    物理粘连；执行后仍须重新确认夹持和 TCP→Bowl 关系。
    """
    if stage_name not in _RETREAT_NAMES:
        raise ValueError(f"不支持的取碗携物撤离阶段：{stage_name}")
    if collision_scene["snapshot"].get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 取碗目前只验收右臂")
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_hold_translation_m, max_hold_rotation_rad)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in limits):
        raise ValueError("取碗撤离的锚点及持碗容差必须为正有限值")
    source = collision_scene["provenance"]
    snapshot = collision_scene["snapshot"]
    if int(source["source_frame_number"]) <= int(initial_targets["source_frame_number"]):
        raise ValueError("携物撤离必须使用闭爪后的新现场观测帧")
    if (grasp_report.get("execute") is not True
            or grasp_report.get("status") != "executed"
            or grasp_report.get("grasp_success") is not True):
        raise ValueError("取碗撤离需要已执行且确认夹持的闭爪报告")
    if (not isinstance(current_hold_check, CheckResult)
            or not current_hold_check.passed):
        raise ValueError("取碗撤离前必须确认当前仍持碗")
    if current_hold_check.source in _SIMULATION_HOLD_SOURCES:
        raise ValueError("仿真夹持证据不能用于真机取碗撤离")
    if (current_hold_check.evidence.get("collision_snapshot_sha256")
            != snapshot.get("snapshot_sha256")
            or current_hold_check.evidence.get("source_frame_number")
            != source.get("source_frame_number")):
        raise ValueError("持碗检查不是当前现场快照，必须重新确认")

    current = build_observed_bowl_extract_targets(
        collision_scene, microwave_frame_alignment=microwave_frame_alignment,
    )
    microwave_error = _pose_delta(
        initial_targets["T_world_profile_microwave"],
        current["T_world_profile_microwave"], field="微波炉 profile frame",
    )
    if (microwave_error[0] > max_microwave_translation_m
            or microwave_error[1] > max_microwave_rotation_rad):
        raise ValueError("微波炉 pose 已偏离原取碗 keypose 锚点")
    bowl = validate_pose(
        source["aligned_object_poses_world"]["Bowl/Bowl_0"],
        field_name="T_world_bowl",
    )
    actual_tcp_bowl = validate_pose(
        np.linalg.inv(_observed_tcp_pose(collision_scene)) @ bowl,
        field_name="T_tcp_bowl",
    )
    hold_error = _pose_delta(
        T_tcp_bowl_after_close, actual_tcp_bowl, field="闭爪后的 TCP→Bowl",
    )
    if (hold_error[0] > max_hold_translation_m
            or hold_error[1] > max_hold_rotation_rad):
        raise ValueError("Bowl 与 TCP 的持碗关系已变化，不能继续撤离")

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
        raise ValueError("旧取碗撤离的附着碰撞代理必须为 32 球")
    body_snapshot = robot_body_only_snapshot(
        attachment["snapshot"], output / "attached_robot_body",
        attached_object_id="bowl",
    )
    body_scene = {
        **attachment,
        "snapshot": {**attachment["snapshot"], **body_snapshot},
        "provenance": {
            **attachment["provenance"],
            "collision_snapshot_sha256": body_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": attachment[
                "snapshot"]["snapshot_sha256"],
        },
    }
    context = prepare_plan_only_context(
        body_scene, group="right", planner_pool=planner_pool,
        scene_revision=scene_revision, seed=seed,
        attachment_spheres_tool=spheres,
    )
    target = PoseTarget("world_tcp", current["keyposes_world_tcp"][stage_name])
    request = MotionRequest(
        f"extract:{stage_name}:observed_{source['source_frame_number']}",
        MotionOperation.LINEAR_TCP, "right", context["start"], target,
        context["scene"], CollisionPolicy(), seed=seed,
    )
    result = context["planner"].plan(request)
    if result.request_id != request.request_id or result.operation is not request.operation:
        raise RuntimeError("取碗撤离 planner 返回的请求 ID 或路径类型不一致")
    if result.success and (result.trajectory is None or result.goal_state is None):
        raise RuntimeError("成功的取碗撤离缺少连续轨迹或终点关节")
    attempts = [{"stage": stage_name, "request": request.to_dict(),
                 "result": result.to_dict()}]
    nominal = SkillStage(stage_name, "planned" if result.success else "failed",
                         target, request, result)
    stage = nominal if result.success else _plan_with_legacy_local_fallback(
        context["planner"], nominal, attempts, limit_deg=10., step_deg=2.5,
    )
    stage = SkillStage(
        stage.name, stage.status, stage.target, stage.request, stage.result,
        {**stage.details, "microwave_anchor_error_m_rad": microwave_error,
         "hold_relation_error_m_rad": hold_error,
         "ignored_world_obstacles": body_snapshot.get("ignored_world_obstacles", [])},
    )
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_extract_retreat_stage_plan_only",
        "success": stage.status == "planned",
        "stage_name": stage_name,
        "source_frame_number": source["source_frame_number"],
        "base_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "attached_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        "stage_collision_snapshot_sha256": body_snapshot["snapshot_sha256"],
        "ignored_world_obstacles": body_snapshot.get("ignored_world_obstacles", []),
        "microwave_anchor_error_m_rad": microwave_error,
        "hold_relation_error_m_rad": hold_error,
        "T_tcp_bowl_from_actual_hold": actual_tcp_bowl.tolist(),
        "attachment_ref": str(output / "attachment/attachment.json"),
        "stage": stage.to_dict(),
        "attempts": attempts,
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "planning_wall_s": time.perf_counter() - started,
        "claim": "只规划当前携物直线撤离；未执行或验证 Bowl 已取出",
    }
    (output / "stage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "stage": stage}


__all__ = ['build_observed_bowl_extract_targets', 'plan_observed_bowl_extract_prefix', 'plan_observed_bowl_extract_motion_stage', 'plan_observed_bowl_extract_retreat_stage']
