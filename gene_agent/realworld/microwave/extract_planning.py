"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

from .prediction import is_prediction, predicted_report, reject_assumption_on_live_scene

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
    """按当前感知碗缘、进炉方向和指垫支点生成目标；旧profile仅作参考。"""
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
    _, table_tcp = extract_profile_targets(world_profile, profile)
    from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import (
        actual_bowl_extract_targets, actual_hold_retreat_targets)
    tcp = _observed_tcp_pose(collision_scene)
    pads = collision_scene["robot_metadata"]["grippers"]["right"]["pad_geom_names"]
    centers = np.asarray([data.geom(name).xpos for name in pads])
    keyposes, target_details = actual_bowl_extract_targets(
        bowl_pose, tcp, centers.mean(axis=0), microwave_pose[:3, 1])
    keyposes.update(actual_hold_retreat_targets(tcp, microwave_pose[:3, 1]))
    keyposes["grasp"][:3, 3] += offset
    return {
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "frame_alignment_source": microwave_frame_alignment["source"],
        "T_world_profile_microwave": world_profile.tolist(),
        "bowl_pose_world": bowl_pose.tolist(),
        "target_derivation": target_details,
        "target_policy": "actual_bowl_rim_contact_pivot",
        "door_angle_rad": door_angle,
        "minimum_open_angle_rad": minimum,
        "grasp_offset_world_m": offset.tolist(),
        "keyposes_world_tcp": {name: pose.tolist() for name, pose in keyposes.items()},
        "raw_table_tcp_world": table_tcp.matrix.tolist(),
        "stage_order": ("open", *_NAMES[:3], "close", "attach", *_NAMES[3:]),
        "claim": "按实际碗缘与夹持姿态生成目标；未规划或执行",
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

    from .strategy_planning import plan_stage, without_bowl, extract_contact_postcheck, write_report
    rows, stages, start = [], {}, None
    for name in ("pre_extract_1", "pre_extract_2", "grasp"):
        current_scene = without_bowl(collision_scene, output/"grasp_world") if name=="grasp" else collision_scene
        planned = plan_stage(current_scene, name, targets_report["keyposes_world_tcp"][name],
            planner_pool=planner_pool, output=output/name, seed=seed, start=start,
            linear=name!="pre_extract_1",
            geometry_check=(lambda raw: extract_contact_postcheck(collision_scene, raw))
                if name=="grasp" else (lambda raw: {"success": True,
                    "source": "curobo_full_world_validation"}))
        stages[name] = planned["stage"]
        rows.append({**planned, "stage": planned["stage"].to_dict()})
        if not planned["success"]:
            break
        start = planned["stage"].result.goal_state
    report = {"success": len(stages)==3 and all(s.status=="planned" for s in stages.values()),
        "targets": targets_report, "plan": {"stages": rows},
        "source_collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
        "planning_wall_s": time.perf_counter()-started}
    targets_report["screened_prefix_trajectories"] = {
        n: s.result.trajectory.to_dict() for n,s in stages.items() if s.status=="planned"}
    write_report(output/"extract_plan.json", report)
    return {**report, "pre_extract_1_stage": stages.get("pre_extract_1")}


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
    if (not is_prediction(collision_scene) and stage_name != "pre_extract_1" and
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

    from .strategy_planning import plan_stage, without_bowl, extract_contact_postcheck
    planned_scene = without_bowl(collision_scene, output/"grasp_world") if stage_name=="grasp" else collision_scene
    return plan_stage(planned_scene, stage_name, initial_targets["keyposes_world_tcp"][stage_name],
        planner_pool=planner_pool, output=output, seed=seed, linear=stage_name!="pre_extract_1",
        preferred=(initial_targets.get("screened_prefix_trajectories") or {}).get(stage_name),
        geometry_check=(lambda raw: extract_contact_postcheck(collision_scene, raw))
            if stage_name=="grasp" else (lambda raw: {"success": True,
                "source": "curobo_full_world_validation"}))


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
    """从实际夹持姿态抬升并完全撤出，保留完整场景和持物碰撞。"""
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
    if not is_prediction(collision_scene) and int(source["source_frame_number"]) <= int(initial_targets["source_frame_number"]):
        raise ValueError("携物撤离必须使用闭爪后的新现场观测帧")
    if not predicted_report(collision_scene, grasp_report, "hold") and (grasp_report.get("execute") is not True
            or grasp_report.get("status") != "executed"
            or grasp_report.get("grasp_success") is not True):
        raise ValueError("取碗撤离需要已执行且确认夹持的闭爪报告")
    if (not isinstance(current_hold_check, CheckResult)
            or not current_hold_check.passed):
        raise ValueError("取碗撤离前必须确认当前仍持碗")
    reject_assumption_on_live_scene(collision_scene, current_hold_check.to_dict())
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

    from .strategy_planning import plan_stage, frozen_geometry
    from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import actual_hold_retreat_targets
    from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import actual_exterior_retreat_target
    current_tcp = _observed_tcp_pose(collision_scene)
    inward = np.asarray(source["aligned_object_poses_world"][_MICROWAVE])[:3,1]
    if stage_name=="post_extract_1":
        initial_targets["actual_hold_retreat_targets"] = {
            n: p.tolist() for n,p in actual_hold_retreat_targets(current_tcp, inward).items()}
    targets = initial_targets.get("actual_hold_retreat_targets")
    if targets is None:
        raise ValueError("取碗撤离必须从闭爪后的实测持碗状态生成")
    target = np.asarray(targets[stage_name])
    if stage_name=="post_extract_2":
        target, derivation = actual_exterior_retreat_target(frozen_geometry(collision_scene),
            {"robots":{"right":{"T_world_tcp": current_tcp}}, "state_id": snapshot["snapshot_sha256"]},
            target, targets["post_extract_1"], group="right", required_clearance_m=.005)
        initial_targets["exterior_target_derivation"] = derivation
    return plan_stage(attachment, stage_name, target,
        planner_pool=planner_pool, output=output, seed=seed, linear=True,
        spheres=spheres, tcp_bowl=actual_tcp_bowl, max_tilt_deg=15.)


__all__ = ['build_observed_bowl_extract_targets', 'plan_observed_bowl_extract_prefix', 'plan_observed_bowl_extract_motion_stage', 'plan_observed_bowl_extract_retreat_stage']
