"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    fixed_place_targets,
    load_object_operation_profile,
)
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    adjust_insert_targets,
)
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.knowledge.geometry.transforms import pose_delta as _pose_delta
from gene_agent.motion.curobo.scene import export_self_collision_only_snapshot
from gene_agent.motion.model import (
    CollisionPolicy,
    MotionOperation,
    MotionRequest,
    PoseTarget,
)
from gene_agent.realworld.microwave.attachment import (
    prepare_observed_bowl_attachment,
    require_current_hold_check,
    same_observation_as_hold_report,
    validate_manual_held_bowl_report,
)
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.realworld.scene import observed_tcp_pose as _observed_tcp_pose
from gene_agent.skills.library.transfer.planning.insertion import InsertionTargets, InsertObjectSkill
from gene_agent.skills.runtime.planning import SkillContext, SkillStage

_ROOT = Path(__file__).resolve().parents[3]
_BOWL = "Bowl/Bowl_0"
_MICROWAVE = "Microwave/Microwave_0"


def build_observed_bowl_insert_targets(
    scene: Mapping, pick_report: Mapping, *,
    microwave_frame_alignment: Mapping,
    current_hold_check: Mapping | None = None,
    insert_depth_offset_m: float = 0.,
    retreat_rotation_local_deg=(0., 0., 0.),
) -> dict:
    """保留原 keypose 数值规则，只把输入换成现场状态和已验收 frame。

    使用刚完成 Pick 后的同一碰撞快照；其他现场帧需要先重新确认
    持碗状态，不能把之前的人工确认无限期沿用。实验性的旧资产叠图
    没有 `accepted_for_planning`，因此不能误作真机标定。
    """
    snapshot = scene["snapshot"]
    source = scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("放碗目标需要同帧现场观测及一致的碰撞快照")
    manual = pick_report.get("source_format") == "manual_prepared_held_bowl"
    if manual:
        check = validate_manual_held_bowl_report(
            scene, pick_report, current_hold_check=current_hold_check,
        )
    elif (pick_report.get("execute") is not True
            or pick_report.get("status") != "executed"
            or pick_report.get("pick_success") is not True
            or not pick_report.get("post_lift_scene_snapshot_sha256")):
        raise ValueError("放碗目标需要已执行且确认持碗的 Pick 结果")
    if not manual:
        pick_frame = pick_report.get("post_lift_source_frame_number")
        pick_timestamp = pick_report.get("post_lift_source_timestamp_ms")
        if pick_frame is None or pick_timestamp is None:
            raise ValueError("Pick 缺少最终帧来源，不能沿用旧持碗证据")
        same_observation = same_observation_as_hold_report(
            scene, snapshot_sha256=pick_report["post_lift_scene_snapshot_sha256"],
            frame_number=pick_frame, timestamp_ms=pick_timestamp,
        )
        if not same_observation and current_hold_check is None:
            raise ValueError("更新后的放碗观测必须重新确认当前仍持碗")
        check = (current_hold_check if current_hold_check is not None
                 else pick_report.get("postcondition") or {})
        if current_hold_check is not None:
            require_current_hold_check(scene, check)
    if check.get("status") != "pass" or not check.get("source"):
        raise ValueError("放碗目标缺少有来源的现场持碗后置条件")
    if check["source"] in {
        "simulation_privileged_weld", "mujoco_contact_geometry_at_planned_waypoints",
    }:
        raise ValueError("仿真夹持证据不能作为真机放碗前置条件")
    if (microwave_frame_alignment.get("accepted_for_planning") is not True
            or not microwave_frame_alignment.get("source")):
        raise ValueError("微波炉 profile frame 对齐尚未验收，不能规划真机放碗")
    T_scene_profile = validate_pose(
        microwave_frame_alignment["T_scene_asset_profile"],
        field_name="T_scene_asset_profile",
    )
    aligned = source["aligned_object_poses_world"]
    T_world_microwave = validate_pose(aligned[_MICROWAVE], field_name="T_world_microwave")
    T_world_bowl = validate_pose(aligned[_BOWL], field_name="T_world_bowl")
    T_world_tcp = _observed_tcp_pose(scene)
    T_tcp_bowl = validate_pose(
        np.linalg.inv(T_world_tcp) @ T_world_bowl, field_name="T_tcp_bowl",
    )
    profile = load_object_operation_profile(_ROOT / "data/task_1", "Bowl_0")
    old = fixed_place_targets(T_world_microwave @ T_scene_profile, profile, T_tcp_bowl)
    original = {
        "preplace": old.pre_place_tcp, "insert": old.place_tcp,
        "retreat_1": old.post_place_1_tcp, "retreat_2": old.post_place_2_tcp,
    }
    angles = np.asarray(retreat_rotation_local_deg, dtype=float)
    targets, object_goal = adjust_insert_targets(
        old, T_world_microwave, depth_offset_m=insert_depth_offset_m,
        retreat_rotation_local_deg=angles)
    return {
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "frame_alignment_source": microwave_frame_alignment["source"],
        "hold_evidence_source": check["source"],
        "hold_origin": ("manual_prepared_held_bowl" if manual else "executed_pick"),
        "T_world_profile_microwave": (T_world_microwave @ T_scene_profile).tolist(),
        "T_tcp_bowl_from_actual_hold": T_tcp_bowl.tolist(),
        "keyposes_world_tcp": {name: pose.tolist() for name, pose in targets.items()},
        "original_keyposes_world_tcp": {name: pose.tolist() for name, pose in original.items()},
        "desired_placed_object_world": object_goal.tolist(),
        "insert_depth_offset_microwave_y_m": float(insert_depth_offset_m),
        "retreat_rotation_tcp_local_xyz_deg": angles.tolist(),
        "claim": "仅生成旧放入 Skill 目标；尚未规划、执行或验证物体已放入",
    }


def plan_observed_bowl_insert_prefix(
    collision_scene: dict, pick_report: dict, *,
    microwave_frame_alignment: dict, planner_pool,
    output_dir: str | Path, seed: int = 123,
    current_hold_check: dict | None = None,
    insert_depth_offset_m: float = 0.,
    retreat_rotation_local_deg=(0., 0., 0.),
) -> dict:
    """从 Pick 已验收的实测终态筛 preplace，再规划直线 insert。

    这是旧 Skill 的预测规划前缀。执行 preplace 后必须从新实测关节和
    新碰撞快照重规划 insert；不能直接把本结果预测终点串成真机动作。
    原 Skill 的 insert 段使用“仅自碰撞”规划层，该层明确列出被停用
    的 world 障碍，调用方不得把它说成完整场景避障。
    """
    output = Path(output_dir).resolve()
    started = time.perf_counter()
    targets_report = build_observed_bowl_insert_targets(
        collision_scene, pick_report,
        microwave_frame_alignment=microwave_frame_alignment,
        current_hold_check=current_hold_check,
        insert_depth_offset_m=insert_depth_offset_m,
        retreat_rotation_local_deg=retreat_rotation_local_deg,
    )
    if collision_scene["snapshot"]["active_group"] != "right":
        raise NotImplementedError("原 task_1 固定 Bowl 放入动作目前只验收右臂")
    grasp = (pick_report if pick_report.get("source_format") == "manual_prepared_held_bowl"
             else pick_report["grasp"])
    attachment = prepare_observed_bowl_attachment(
        collision_scene, grasp, output_dir=output / "attachment",
        hold_evidence=current_hold_check,
    )
    world = prepare_plan_only_context(
        attachment, group="right", planner_pool=planner_pool,
        scene_revision=10, seed=seed,
        attachment_spheres_tool=attachment["spheres_tool_m"],
    )
    self_snapshot = export_self_collision_only_snapshot(
        attachment["snapshot"], output / "attached_self_only",
        attached_object_id="bowl",
    )
    # 原导出器只写规划 world 的差异字段。以下来源、关节与夹爪字段
    # 仍属于同一实测帧；实际被停用的 world 障碍在 self_snapshot 中列出。
    self_scene = {
        **attachment,
        "snapshot": {**attachment["snapshot"], **self_snapshot},
        "provenance": {
            **attachment["provenance"],
            "collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        },
    }
    self_only = prepare_plan_only_context(
        self_scene, group="right", planner_pool=planner_pool,
        scene_revision=11, seed=seed,
        attachment_spheres_tool=attachment["spheres_tool_m"],
    )
    if self_only["start"] != world["start"]:
        raise ValueError("完整场景与仅自碰撞规划层的实测起点不一致")
    keyposes = targets_report["keyposes_world_tcp"]
    targets = InsertionTargets(*(
        PoseTarget("world_tcp", keyposes[name]) for name in (
            "preplace", "insert", "retreat_1", "retreat_2",
        )
    ))
    plan = InsertObjectSkill().plan_attached(
        world["planner"], self_only["planner"],
        SkillContext("right", world["start"], world["scene"], "bowl", seed),
        self_only["scene"], targets,
    )
    report = {
        "schema_version": 1,
        "kind": "observed_bowl_insert_prefix_plan_only",
        "success": all(plan.stage(name).status == "planned"
                       for name in ("preplace", "insert")),
        "source_collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
        "attached_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        "self_only_collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
        "ignored_world_obstacles": self_snapshot.get("ignored_world_obstacles", []),
        "attachment_ref": str(output / "attachment/attachment.json"),
        "targets": targets_report,
        "measured_start": world["start"].to_dict(),
        "planner_cache_hit": {
            "world": world["planner_cache_hit"],
            "self_only": self_only["planner_cache_hit"],
        },
        "planner_setup_duration_s": (
            world["planner_setup_duration_s"] + self_only["planner_setup_duration_s"]
        ),
        "planning_wall_s": time.perf_counter() - started,
        "claim": "仅规划旧放入 Skill 前缀；未执行 preplace/insert，也未验证 Bowl 放置",
        "plan": plan.to_dict(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "insert_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "preplace_stage": plan.stage("preplace")}


def plan_observed_bowl_insert_linear(
    collision_scene: dict, initial_targets: dict, pick_report: dict, *,
    microwave_frame_alignment: dict, hold_check: dict,
    planner_pool, output_dir: str | Path,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_hold_translation_m: float, max_hold_rotation_rad: float,
    seed: int = 123, scene_revision: int = 12,
) -> dict:
    """复用旧版 place 普通路径筛选和 TCP 直线动作，但以实测关节为起点。

    场景 world 在该段按旧策略停用；规划报告会列出停用障碍。
    是否允许真实机器人执行这段仍须外部现场安全检查，plan-only
    成功绝不等于 Bowl 已放入或轨迹对周围物体安全。
    """
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_hold_translation_m, max_hold_rotation_rad)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError("放碗锚点容差必须为正有限值")
    source = collision_scene["provenance"]
    if int(source["source_frame_number"]) <= int(initial_targets["source_frame_number"]):
        raise ValueError("insert 必须使用 preplace 执行后的新现场观测帧")
    current = build_observed_bowl_insert_targets(
        collision_scene, pick_report,
        microwave_frame_alignment=microwave_frame_alignment,
        current_hold_check=hold_check,
        insert_depth_offset_m=initial_targets["insert_depth_offset_microwave_y_m"],
        retreat_rotation_local_deg=initial_targets["retreat_rotation_tcp_local_xyz_deg"],
    )
    microwave_delta = _pose_delta(
        initial_targets["T_world_profile_microwave"],
        current["T_world_profile_microwave"], field="微波炉 profile frame",
    )
    if (microwave_delta[0] > max_microwave_translation_m
            or microwave_delta[1] > max_microwave_rotation_rad):
        raise ValueError("微波炉在 preplace 后超出原 keypose 锚点容差")
    hold_delta = _pose_delta(
        initial_targets["T_tcp_bowl_from_actual_hold"],
        current["T_tcp_bowl_from_actual_hold"], field="TCP→Bowl 持碗关系",
    )
    if (hold_delta[0] > max_hold_translation_m
            or hold_delta[1] > max_hold_rotation_rad):
        raise ValueError("Bowl 与 TCP 的持碗关系在 preplace 后已改变")
    if collision_scene["snapshot"]["active_group"] != "right":
        raise NotImplementedError("原 task_1 Bowl insert 当前只验收右臂")
    output = Path(output_dir).resolve()
    started = time.perf_counter()
    grasp = (pick_report if pick_report.get("source_format") == "manual_prepared_held_bowl"
             else pick_report["grasp"])
    attachment = prepare_observed_bowl_attachment(
        collision_scene, grasp,
        output_dir=output / "attachment", hold_evidence=hold_check,
    )
    self_snapshot = export_self_collision_only_snapshot(
        attachment["snapshot"], output / "attached_self_only",
        attached_object_id="bowl",
    )
    self_scene = {
        **attachment,
        "snapshot": {**attachment["snapshot"], **self_snapshot},
        "provenance": {
            **attachment["provenance"],
            "collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        },
    }
    context = prepare_plan_only_context(
        self_scene, group="right", planner_pool=planner_pool,
        scene_revision=scene_revision, seed=seed,
        attachment_spheres_tool=attachment["spheres_tool_m"],
    )
    target = PoseTarget("world_tcp", current["keyposes_world_tcp"]["insert"])
    screen = MotionRequest(
        f"insert:place_screen:observed_{source['source_frame_number']}",
        MotionOperation.PLAN, "right", context["start"], target,
        context["scene"], CollisionPolicy(), seed=seed,
    )
    screened = context["planner"].plan(screen)
    if screened.request_id != screen.request_id or screened.operation is not MotionOperation.PLAN:
        raise RuntimeError("insert 普通路径筛选结果与请求不一致")
    linear = None
    result = None
    if screened.success and screened.trajectory is not None:
        linear = MotionRequest(
            f"insert:place_linear:observed_{source['source_frame_number']}",
            MotionOperation.LINEAR_TCP, "right", context["start"], target,
            context["scene"], CollisionPolicy(), seed=seed,
        )
        result = context["planner"].plan(linear)
        if result.request_id != linear.request_id or result.operation is not MotionOperation.LINEAR_TCP:
            raise RuntimeError("insert 直线规划结果与请求不一致")
        if result.success and (result.trajectory is None or result.goal_state is None):
            raise RuntimeError("成功的 insert 直线缺少轨迹或终点关节")
    success = bool(result is not None and result.success)
    stage = SkillStage(
        "insert", "planned" if success else "failed", target,
        linear, result,
        {"operation": "linear_tcp", "screen_success": screened.success,
         "microwave_anchor_error_m_rad": microwave_delta,
         "hold_relation_error_m_rad": hold_delta,
         "ignored_world_obstacles": self_snapshot.get("ignored_world_obstacles", [])},
    )
    report = {
        "schema_version": 1, "kind": "observed_bowl_insert_linear_plan_only",
        "success": success,
        "source_frame_number": source["source_frame_number"],
        "base_collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
        "attached_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        "stage_collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
        "ignored_world_obstacles": self_snapshot.get("ignored_world_obstacles", []),
        "microwave_anchor_error_m_rad": microwave_delta,
        "hold_relation_error_m_rad": hold_delta,
        "screen_request": screen.to_dict(), "screen_result": screened.to_dict(),
        "linear_request": None if linear is None else linear.to_dict(),
        "linear_result": None if result is None else result.to_dict(),
        "stage": stage.to_dict(),
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "planning_wall_s": time.perf_counter() - started,
        "claim": "只从 preplace 实测终态筛选并规划直线 insert；未执行或验证 Bowl 放置",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "stage": stage}


def plan_observed_bowl_detached_retreat_stage(
    collision_scene: dict, targets_report: dict, release_report: dict, *,
    stage_name: str, microwave_frame_alignment: dict, planner_pool,
    output_dir: str | Path,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    seed: int = 123, scene_revision: int = 13,
) -> dict:
    """按旧 Skill 的 LINEAR_TCP + 仅自碰撞策略规划一个撤离段。

    release_report 必须说明夹爪命令已执行、Bowl 已确认离爪，并提供
    松爪后 Bowl pose。另一段撤离仍重新读取实际现场，不从此段目标
    关节状态预测下一段的起点。
    """
    if stage_name not in {"retreat_1", "retreat_2"}:
        raise ValueError(f"未知的 Bowl 放碗撤离阶段：{stage_name}")
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_bowl_translation_m, max_bowl_rotation_rad)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError("撤离锚点容差必须为正有限值")
    check = release_report.get("postcondition") or {}
    if (release_report.get("status") != "executed"
            or release_report.get("release_success") is not True
            or check.get("status") != "pass" or not check.get("source")):
        raise ValueError("撤离前必须有已执行松爪及 Bowl 离爪的现场通过证据")
    if check["source"] in {"simulation_privileged_weld",
                           "mujoco_contact_geometry_at_planned_waypoints"}:
        raise ValueError("仿真松爪证据不能用于真机撤离")
    snapshot = collision_scene["snapshot"]
    source = collision_scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("撤离需要同帧现场观测及一致的碰撞快照")
    if (int(source["source_frame_number"]) <= int(release_report["source_frame_number"])
            or float(source["source_timestamp_ms"]) <= float(release_report["source_timestamp_ms"])):
        raise ValueError("撤离必须使用松爪后更新的现场帧")
    if snapshot["active_group"] != "right":
        raise NotImplementedError("原 task_1 Bowl 撤离当前只验收右臂")
    if (microwave_frame_alignment.get("accepted_for_planning") is not True
            or not microwave_frame_alignment.get("source")):
        raise ValueError("微波炉 profile frame 对齐未经现场验收")
    aligned = source["aligned_object_poses_world"]
    T_world_microwave = validate_pose(
        aligned["Microwave/Microwave_0"], field_name="T_world_microwave",
    )
    T_scene_profile = validate_pose(
        microwave_frame_alignment["T_scene_asset_profile"],
        field_name="T_scene_asset_profile",
    )
    microwave_delta = _pose_delta(
        targets_report["T_world_profile_microwave"],
        T_world_microwave @ T_scene_profile, field="微波炉 profile frame",
    )
    if (microwave_delta[0] > max_microwave_translation_m
            or microwave_delta[1] > max_microwave_rotation_rad):
        raise ValueError("撤离前微波炉已偏离原 keypose 锚点")
    bowl_delta = _pose_delta(
        release_report["bowl_pose_world_after_release"],
        aligned["Bowl/Bowl_0"], field="已放置 Bowl",
    )
    if (bowl_delta[0] > max_bowl_translation_m
            or bowl_delta[1] > max_bowl_rotation_rad):
        raise ValueError("Bowl 在松爪后发生明显位移，停止撤离")
    output = Path(output_dir).resolve()
    started = time.perf_counter()
    self_snapshot = export_self_collision_only_snapshot(
        snapshot, output / "detached_self_only", attached_object_id=None,
    )
    self_scene = {
        **collision_scene,
        "snapshot": {**snapshot, **self_snapshot},
        "provenance": {
            **source,
            "collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        },
    }
    context = prepare_plan_only_context(
        self_scene, group="right", planner_pool=planner_pool,
        scene_revision=scene_revision, seed=seed,
    )
    target = PoseTarget("world_tcp", targets_report["keyposes_world_tcp"][stage_name])
    request = MotionRequest(
        f"insert:{stage_name}:observed_{source['source_frame_number']}",
        MotionOperation.LINEAR_TCP, "right", context["start"], target,
        context["scene"], CollisionPolicy(), seed=seed,
    )
    result = context["planner"].plan(request)
    if result.request_id != request.request_id or result.operation is not MotionOperation.LINEAR_TCP:
        raise RuntimeError("Bowl 撤离直线结果与请求不一致")
    if result.success and (result.trajectory is None or result.goal_state is None):
        raise RuntimeError("成功的 Bowl 撤离直线缺少轨迹或终点关节")
    stage = SkillStage(
        stage_name, "planned" if result.success else "failed",
        target, request, result,
        {"microwave_anchor_error_m_rad": microwave_delta,
         "bowl_after_release_error_m_rad": bowl_delta,
         "ignored_world_obstacles": self_snapshot.get("ignored_world_obstacles", [])},
    )
    report = {
        "schema_version": 1, "kind": "observed_bowl_detached_retreat_plan_only",
        "success": result.success, "stage_name": stage_name,
        "source_frame_number": source["source_frame_number"],
        "base_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "stage_collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
        "ignored_world_obstacles": self_snapshot.get("ignored_world_obstacles", []),
        "microwave_anchor_error_m_rad": microwave_delta,
        "bowl_after_release_error_m_rad": bowl_delta,
        "planning_wall_s": time.perf_counter() - started,
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "stage": stage.to_dict(),
        "claim": "只规划一个实测起点的 TCP 直线撤离段；未执行或验证 Bowl 稳定放置",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "stage": stage}


__all__ = ['build_observed_bowl_insert_targets', 'plan_observed_bowl_insert_prefix', 'plan_observed_bowl_insert_linear', 'plan_observed_bowl_detached_retreat_stage']
