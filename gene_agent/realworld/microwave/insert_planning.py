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
    insert_depth_offset_m: float = .02,
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
    preview = predicted_report(scene, pick_report, "hold")
    manual = pick_report.get("source_format") == "manual_prepared_held_bowl"
    if preview:
        check = current_hold_check or pick_report["postcondition"]
    elif manual:
        check = validate_manual_held_bowl_report(
            scene, pick_report, current_hold_check=current_hold_check,
        )
    elif (pick_report.get("execute") is not True
            or pick_report.get("status") != "executed"
            or pick_report.get("pick_success") is not True
            or not pick_report.get("post_lift_scene_snapshot_sha256")):
        raise ValueError("放碗目标需要已执行且确认持碗的 Pick 结果")
    if not manual and not preview:
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
    reject_assumption_on_live_scene(scene, check)
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
    from .strategy_planning import frozen_geometry, strategy_settings
    from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import insert_candidate_families
    settings = strategy_settings()
    families = insert_candidate_families(frozen_geometry(scene), T_world_bowl, T_tcp_bowl,
        insert_depth_offset_m=insert_depth_offset_m,
        insert_tcp_table_angle_max_deg=settings["insert_tcp_table_angle_max_deg"],
        insert_max_bowl_tilt_deg=settings["insert_max_bowl_tilt_deg"],
        retreat_clearance_margin_m=settings["retreat_clearance_margin_m"],
        retreat_rotation_local_deg=angles)
    targets, object_goal = families[0]["targets"], families[0]["object_goal"]
    return {
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "frame_alignment_source": microwave_frame_alignment["source"],
        "hold_evidence_source": check["source"],
        "hold_origin": ("planned_pick_assumption" if preview else
                        "manual_prepared_held_bowl" if manual else "executed_pick"),
        "T_world_profile_microwave": (T_world_microwave @ T_scene_profile).tolist(),
        "T_tcp_bowl_from_actual_hold": T_tcp_bowl.tolist(),
        "target_policy": "full_insert_release_envelope_and_released_retreat_screen",
        "candidate_families": [{k: ({n: p.tolist() for n, p in v.items()} if k == "targets"
            else v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in family.items()} for family in families],
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
    insert_depth_offset_m: float = .02,
    retreat_rotation_local_deg=(0., 0., 0.),
) -> dict:
    """筛持物进入、完整松爪包络、开爪撤离全序列；预测不作为执行事实。"""
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

    from .strategy_planning import screen_insert, write_report
    families = [{**family, "targets": {n: np.asarray(p) for n, p in family["targets"].items()},
        "object_goal": np.asarray(family["object_goal"])} for family in targets_report["candidate_families"]]
    screen = screen_insert(collision_scene, attachment, families,
        planner_pool=planner_pool, output=output/"sequence_screen", seed=seed)
    report = {"schema_version": 2, "kind": "observed_bowl_insert_full_sequence_screen",
        "success": screen["success"], "failure_domain": screen.get("failure_domain"),
        "source_collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
        "attached_collision_snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
        "targets": targets_report, "plan": {"candidate_screen": screen["candidate_screen"]},
        "planning_wall_s": time.perf_counter()-started}
    stages = {}
    if screen["success"]:
        from gene_agent.motion.recording import current_motion_recording
        recording = current_motion_recording()
        logs = getattr(recording, "curobo_logs", None)
        if logs is not None:
            logs.resolve_successful_selection({"success": True,
                "screen_contract": "full_insert_release_retreat"})
        family = screen["family"]; stages = screen["held_stages"]
        targets_report.update(selected_candidate_index=screen["index"],
            keyposes_world_tcp={n: np.asarray(p).tolist() for n, p in family["targets"].items()},
            desired_placed_object_world=np.asarray(family["object_goal"]).tolist(),
            held_turn_world_tcp=(np.asarray(family["held_turn"]).tolist()
                                if "held_turn" in family else None),
            screened_held_trajectories={n: s.result.trajectory.to_dict() for n, s in stages.items()},
            release_retreat_preflight=screen["retreat_evidence"])
    write_report(output/"insert_plan.json", report)
    return {**report, "preplace_stage": stages.get("preplace"),
            "held_turn_stage": stages.get("held_turn")}


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
    if not is_prediction(collision_scene) and int(source["source_frame_number"]) <= int(initial_targets["source_frame_number"]):
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

    from .strategy_planning import plan_stage
    planned = plan_stage(attachment, "insert", initial_targets["keyposes_world_tcp"]["insert"],
        planner_pool=planner_pool, output=output, seed=seed,
        spheres=attachment["spheres_tool_m"],
        tcp_bowl=current["T_tcp_bowl_from_actual_hold"], linear=True,
        preferred=(initial_targets.get("screened_held_trajectories") or {}).get("insert"))
    return {**planned, "microwave_anchor_error_m_rad": microwave_delta,
            "hold_relation_error_m_rad": hold_delta}


def plan_observed_bowl_detached_retreat_stage(
    collision_scene: dict, targets_report: dict, release_report: dict, *,
    stage_name: str, microwave_frame_alignment: dict, planner_pool,
    output_dir: str | Path,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_bowl_translation_m: float, max_bowl_rotation_rad: float,
    seed: int = 123, scene_revision: int = 13,
) -> dict:
    """松爪落稳后按实际感知几何筛三段撤离；全场景，净空抬升保持TCP直线。"""
    if stage_name not in {"retreat_lift", "retreat_1", "retreat_2"}:
        raise ValueError(f"未知的 Bowl 放碗撤离阶段：{stage_name}")
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_bowl_translation_m, max_bowl_rotation_rad)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError("撤离锚点容差必须为正有限值")
    check = release_report.get("postcondition") or {}
    if not predicted_report(collision_scene, release_report, "release") and (release_report.get("status") != "executed"
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
    if (not is_prediction(collision_scene) and (int(source["source_frame_number"]) <= int(release_report["source_frame_number"])
            or float(source["source_timestamp_ms"]) <= float(release_report["source_timestamp_ms"]))):
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

    from .strategy_planning import (plan_stage, plan_sequence, frozen_geometry,
        strategy_settings, RETREAT_FRACTIONS, write_report)
    from gene_agent.tasks.policies.microwave_task.operations.transfer_scene import raise_retreat_for_released_pad_clearance
    settings = strategy_settings()
    if stage_name == "retreat_lift":
        frozen = frozen_geometry(collision_scene); candidates = []
        nominal = {n: np.asarray(p) for n, p in targets_report["keyposes_world_tcp"].items()
                   if n in {"preplace", "insert", "retreat_1", "retreat_2"}}
        for fraction in RETREAT_FRACTIONS:
            poses, evidence = raise_retreat_for_released_pad_clearance(
                frozen.model, frozen.data, frozen.robot_metadata, nominal,
                T_world_microwave[:3, 1],
                clearance_margin_m=settings["retreat_clearance_margin_m"],
                minimum_lift_m=settings["retreat_minimum_lift_m"], outward_fraction=fraction)
            screen = plan_sequence(collision_scene, [(n, poses[n]) for n in
                ("retreat_lift", "retreat_1", "retreat_2")], planner_pool=planner_pool,
                output=output/f"candidate_{fraction}", seed=seed)
            candidates.append({"adjustment": evidence, "stages": screen["rows"]})
            if screen["success"]:
                targets_report["actual_released_retreat"] = {
                    "keyposes_world_tcp": {n: p.tolist() for n, p in poses.items()},
                    "trajectories": {n: s.result.trajectory.to_dict() for n, s in screen["stages"].items()}}
                break
        write_report(output/"actual_released_retreat_sequence_screen.json",
            {"success": bool(candidates and screen["success"]), "candidates": candidates,
             "source": "observed_state_after_release_detach_settle"})
        if not candidates or not screen["success"]:
            return {"success": False, "stage": next(s for s in reversed(
                list(screen["stages"].values())) if s.status != "planned"),
                "stage_collision_snapshot_sha256": snapshot["snapshot_sha256"]}
    actual = targets_report.get("actual_released_retreat")
    if actual is None:
        raise ValueError("必须先按实际松爪落稳状态筛选三段撤离，不能使用预视轨迹")
    return plan_stage(collision_scene, stage_name, actual["keyposes_world_tcp"][stage_name],
        planner_pool=planner_pool, output=output, seed=seed, linear=stage_name=="retreat_lift",
        preferred=actual["trajectories"][stage_name])


__all__ = ['build_observed_bowl_insert_targets', 'plan_observed_bowl_insert_prefix', 'plan_observed_bowl_insert_linear', 'plan_observed_bowl_detached_retreat_stage']
