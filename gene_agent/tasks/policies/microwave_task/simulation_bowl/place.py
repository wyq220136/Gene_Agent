"""从实际持碗终态反算桌面 Place TCP，并用 MuJoCo 验证落稳。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco
import trimesh
from benchmarks.microwave_task.evaluation import table_place_postcondition

from gene_agent.feedback.state import model_fingerprint, save_state
from gene_agent.motion.curobo.attachment import build_attachment_surface_spheres_from_mesh
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.tasks.policies.microwave_task.operations.bowl import align_table_tcp_to_surface
from gene_agent.motion.model import PoseTarget
from gene_agent.skills.library.transfer.planning.placement import PlacementTargets
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.skills.library.transfer.skill import run_transfer_flow
from gene_agent.tasks.checks import CheckResult
from gene_agent.visualization.trajectory import TrajectoryRecorder

from gene_agent.tasks.policies.microwave_task.planning.attachment import BOWL_MESH
from gene_agent.tasks.policies.microwave_task.planning.extract import _world_legacy_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    load_object_operation_profile, extract_profile_targets, LEGACY_TABLE_CONFIG,
)
from gene_agent.tasks.policies.microwave_task.simulation_bowl.extract import _tilt_deg, _write_json
from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import (
    insertion_contact_check, validate_insertion_plan,
)
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase


ROOT = Path(__file__).resolve().parents[5]
TABLE_PLACE_STAGES = ("preplace", "place", "release", "detach", "settle", "retreat")
PHYSICAL_PLACE_CONFIG = ROOT / "data/task_1/simulation/physical_table_place.json"


def _physical_table_target(runtime, original: PoseTarget) -> tuple[PoseTarget, dict]:
    """读取 task_1 放置参数与当前 MuJoCo 桌面顶面，形成可审计目标。"""
    config = json.loads(PHYSICAL_PLACE_CONFIG.read_text(encoding="utf-8"))
    if (config.get("schema_version") != 1 or config.get("object_id") != "Bowl_0"
            or config.get("height_policy") !=
            "keep_old_world_xy_and_orientation_follow_table_top_z"):
        raise ValueError("物理桌面放置配置格式不匹配")
    geom_name = config["surface_geom_name"]
    geom = runtime.data.geom(geom_name)
    normal = np.asarray(geom.xmat).reshape(3, 3)[:, 2]
    if float(normal[2]) < .995:
        raise ValueError("当前桌面倾斜过大，旧 Place 的世界 Z 接近不适用")
    center = np.asarray(geom.xpos)
    half_thickness = float(runtime.model.geom(geom_name).size[2])
    current_top = float((center + normal * half_thickness)[2])
    old = json.loads(LEGACY_TABLE_CONFIG.read_text(encoding="utf-8"))[
        "legacy_virtual_table"]
    reference_top = float(old["center_world_m"][2]
                          + old["size_xyz_m"][2] / 2)
    bias = float(config["height_bias_m"])
    target = align_table_tcp_to_surface(
        original, current_top_z_m=current_top,
        reference_top_z_m=reference_top, height_bias_m=bias)
    return target, {
        "source_config_ref": "data/task_1/simulation/physical_table_place.json",
        "surface_geom_name": geom_name,
        "current_surface_top_z_m": current_top,
        "reference_surface_top_z_m": reference_top,
        "height_bias_m": bias,
        "policy": config["height_policy"],
        "old_tcp_world": original.matrix.tolist(),
        "physical_tcp_world": target.matrix.tolist(),
    }


def _table_support(runtime) -> dict:
    """只接受 Bowl 碰撞体与指定桌面碰撞体的实际接触。"""
    contacts = query_diagnostics(runtime, body_names=("bowl",))["contacts"]
    matching = [row for row in contacts
                if "scene_table" in row.get("geom_names", ())
                and any(name.startswith("bowl_") for name in row.get("geom_names", ()))]
    return {"supported": bool(matching), "contact_count": len(matching),
            "support_geom": "scene_table"}


def _object_relative_table_target(runtime, old_tcp, old_extract, world_legacy,
                                  profile, bowl, tcp):
    """在炉前桌面区域放下直立的碗；由真实夹持关系反算腕部姿态。"""
    options = json.loads(PHYSICAL_PLACE_CONFIG.read_text(encoding='utf-8'))['object_goal']
    nominal_bowl = world_legacy @ np.asarray(profile['microwave_from_placed_object'])
    old_goal = old_tcp.matrix @ np.linalg.inv(old_extract.grasp.matrix) @ nominal_bowl
    up = np.array([0., 0., 1.])
    x = bowl[:3, 0] - up * bowl[2, 0]
    x /= np.linalg.norm(x)
    rotation = np.column_stack((x, up, np.cross(x, up)))
    yaw = np.deg2rad(float(options['yaw_offset_deg']))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]]) @ rotation
    bottom = []
    body_id = runtime.model.body('bowl').id
    for g in range(runtime.model.ngeom):
        if (runtime.model.geom_bodyid[g] != body_id or
                not (runtime.model.geom_contype[g] or runtime.model.geom_conaffinity[g])):
            continue
        local = np.empty(9)
        mujoco.mju_quat2Mat(local, runtime.model.geom_quat[g])
        r = rotation @ local.reshape(3, 3)
        size = runtime.model.geom_size[g]
        if runtime.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0] * np.linalg.norm(r[2, :2]) + size[1] * abs(r[2, 2])
        else:
            extent = np.sum(np.abs(r[2]) * size)
        bottom.append(float((rotation @ runtime.model.geom_pos[g])[2] - extent))
    if not bottom:
        raise ValueError('桌面放置需要实际碗碰撞几何')
    table = runtime.data.geom('scene_table')
    normal = table.xmat.reshape(3, 3)[:, 2]
    top = float((table.xpos + normal * runtime.model.geom('scene_table').size[2])[2])
    goal = np.eye(4)
    goal[:3, :3] = rotation
    # 碗绕直立轴转动不会改变任务语义；调整heading避开腕部折回基座。
    # 这是显式任务工作区参数，不从历史成功轨迹/seed读取关节或目标。
    goal[:2, 3] = np.asarray(options['table_xy_world_m'], dtype=float)
    clearance = float(options['release_clearance_m'])
    if not np.isfinite(goal[:2, 3]).all() or not 0 < clearance <= .005:
        raise ValueError('桌面放置区域及松爪高度配置无效')
    goal[2, 3] = top - min(bottom) + clearance
    attachment = np.linalg.inv(tcp) @ bowl
    target = PoseTarget('world_tcp', goal @ np.linalg.inv(attachment))
    return target, attachment, {
        'policy': 'upright_object_goal_from_actual_grasp_in_front_table_region',
        'xy_source': 'physical_table_place.object_goal.table_xy_world_m',
        'yaw_offset_deg': float(options['yaw_offset_deg']),
        'legacy_nominal_bowl_goal': old_goal.tolist(),
        'desired_bowl_goal': goal.tolist(),
        'T_tcp_bowl_from_actual_hold': attachment.tolist(),
        'collision_bottom_local_z_m': min(bottom),
        'table_top_world_z_m': top,
        'release_clearance_m': clearance,
    }


def _place_contact_check(model, data, metadata):
    check = insertion_contact_check(model, data, metadata)
    # 与碗的指垫接触已由共享检查放行；桌面不可与活动机械臂接触。
    root = model.body(metadata['group_root_bodies']['right']).id
    def active(body):
        while body:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return False
    for contact in data.contact:
        if contact.dist > 0:
            continue
        ids = (int(contact.geom1), int(contact.geom2))
        if (model.geom('scene_table').id in ids
                and any(active(int(model.geom_bodyid[g])) for g in ids)):
            check['forbidden_contacts'].append({
                'geom_names': [model.geom(g).name for g in ids],
                'penetration_m': max(0., -float(contact.dist)),
            })
    check['success'] = not check['forbidden_contacts']
    return check


def run_assisted_table_place(
    runtime, output_dir: Path, extract_report: dict, *,
    group: str = "right", object_id: str = "Bowl_0",
    max_tilt_deg: float = 15., max_settle_translation_m: float = .005,
    record_video: bool = False, stop_after=None,
) -> dict:
    """按实际 attachment 执行：携物预放置、下降、松爪、撤离。

    入口必须接在同一 MuJoCo 模型的辅助取碗终态。显式工作区定义放碗区域，
    碗直立且底部位于实际桌面上方，由实际抓法反算 TCP。仿真 weld 只在
    先前真实双侧接触后存在，并在松指后取消；放置成功由桌面接触、
    稳定性、倾角和最终撤离共同决定，不由规划器成功代替。
    """
    if group != "right" or object_id != "Bowl_0" \
            or not extract_report.get("assisted_extract_success") \
            or extract_report.get("group") != group \
            or extract_report.get("object_id") != object_id:
        raise ValueError("桌面放置必须接在同组同 Bowl 的辅助取碗成功终态后")
    if not np.isfinite(max_tilt_deg) or not 0 < max_tilt_deg < 90 \
            or not np.isfinite(max_settle_translation_m) \
            or max_settle_translation_m < 0:
        raise ValueError("放置倾角与稳定移动阈值必须有效")
    current = runtime.query(robot_groups=(group,), object_names=("bowl",))
    stages = extract_report.get("stages") or ()
    tail = stages[-1] if stages else {}
    if tail.get("name") != "post_extract_2" \
            or tail.get("end_state_id") != current["state_id"]:
        raise ValueError("取碗撤离报告终态与当前 MuJoCo 状态不一致")
    if extract_report.get("model_fingerprint") != model_fingerprint(runtime.model):
        raise ValueError("取碗撤离报告与当前模型指纹不一致")
    active = runtime.simulation_attachments()
    if len(active) != 1 or active[0].get("object_instance_id") != "bowl" \
            or not (active[0].get("activation_contact") or {}).get("bilateral_pad_contact"):
        raise ValueError("当前状态缺少此前真实双侧接触后启用的仿真固定")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bowl_start = np.asarray(current["objects"]["bowl"]["T_world_object"])
    tcp_start = np.asarray(current["robots"][group]["T_world_tcp"])
    case = PickBowlCase(output, runtime.scene_revision, runtime.model,
                        runtime.data, runtime.metadata, {}, bowl_start, [])
    world_legacy, alignment = _world_legacy_frame(case)
    profile = load_object_operation_profile(ROOT / "data/task_1", object_id)
    old_extract, original_table_tcp = extract_profile_targets(world_legacy, profile)
    old_table_tcp, table_alignment = _physical_table_target(runtime, original_table_tcp)
    table_tcp, tcp_bowl, object_goal = _object_relative_table_target(
        runtime, old_table_tcp, old_extract, world_legacy, profile, bowl_start, tcp_start)
    targets = PlacementTargets.from_attachment(table_tcp, tcp_bowl)
    mesh = trimesh.load_mesh(BOWL_MESH, process=False)
    centers_object, radii, removed = build_attachment_surface_spheres_from_mesh(
        mesh, num_spheres=32, radius=.002)
    report = {
        "schema_version": 1, "skill": "place_extracted_bowl_on_table",
        "mode": "physical_assisted", "physical_table_place_success": False,
        "full_task_success": False, "group": group, "object_id": object_id,
        "model_fingerprint": model_fingerprint(runtime.model),
        "source_extract_state_id": current["state_id"],
        "stage_order": list(TABLE_PLACE_STAGES), "stages": [],
        "profile_ref": "data/task_1/knowledge/microwave_object_manipulation.json",
        "frame_alignment": alignment,
        "table_alignment": table_alignment,
        "object_goal": object_goal,
        "table_tcp_world": table_tcp.matrix.tolist(),
        "targets_world_tcp": {
            "preplace": targets.preplace.matrix.tolist(),
            "place": targets.place.matrix.tolist(),
            "retreat": targets.retreat.matrix.tolist(),
        },
        "attached_proxy": {"source": "old_visual_32_surface_spheres",
                           "sphere_count": len(radii),
                           "removed_mesh_component": removed},
    }
    recorder = None
    try:
        if record_video:
            recorder = TrajectoryRecorder(
                output / "visual", target_body="bowl", width=640, height=480,
                fps=10, sample_interval_s=.1, combine_mp4=True,
                detail_view="gripper_side", extra_view="contact_side",
                overview_camera={"lookat": [.24, .33, -.04], "distance": .77,
                                 "azimuth": -55., "elevation": -25.},
                stage_camera_overrides={
                    "place": {"gripper_side": {"lookat": [.13, .34, -.11],
                                                "distance": .38, "azimuth": -45.,
                                                "elevation": -17.}},
                    "release": {"gripper_side": {"lookat": [.13, .34, -.16],
                                                  "distance": .32, "azimuth": -45.,
                                                  "elevation": -12.}},
                    "retreat": {"gripper_side": {"lookat": [.13, .34, -.15],
                                                  "distance": .40, "azimuth": -45.,
                                                  "elevation": -16.}},
                },
                robot_metadata=runtime.robot_metadata, active_group=group,
            )

        def attached_spheres(_stage: str, _observation: dict) -> dict:
            attachment = runtime.simulation_attachments()
            if len(attachment) != 1 or attachment[0]["object_instance_id"] != "bowl":
                raise ValueError("携物放置阶段缺少有效仿真附着")
            bowl = np.asarray(runtime.query(object_names=("bowl",))["objects"][
                "bowl"]["T_world_object"])
            centers = (bowl[:3, :3] @ centers_object.T).T + bowl[:3, 3]
            return {
                "object_instance_id": "bowl",
                "spheres_world_m": np.column_stack((centers, radii)).tolist(),
                "coverage": "old_visual_32_surface_spheres",
                "physical_hold_evidence": attachment[0],
            }

        attached_bridge = CuroboRuntimeBridge(
            runtime, output / "motion_attached", group=group,
            visual_recorder=recorder, attachment_for_stage=attached_spheres,
            # 炉外到桌面上方属于自由搬运，最终下降才要求TCP直线。
            linear_tcp_stages=("place",),
            plan_postcheck_for_stage=lambda stage, plan, _: validate_insertion_plan(
                runtime, 'insert', plan, tcp_bowl, max_tilt_deg=None),
            execution_stop_condition_for_stage=lambda stage, model, data, _: (
                None if (check := _place_contact_check(model, data, runtime.robot_metadata))['success']
                else {'failure_type': 'forbidden_table_place_contact', **check}),
            execution_time_scale_by_stage={'preplace': 2., 'place': 3.},
            # 支撑接触需要接近桌面；只减小优化软代价的激活距离，
            # 完整world、自碰撞及MuJoCo逐点硬碰撞检查仍不变。
            optimizer_collision_activation_distance_by_stage={
                'place': json.loads(PHYSICAL_PLACE_CONFIG.read_text(encoding='utf-8'))[
                    'object_goal']['place_optimizer_clearance_m']},
            max_attempts_by_stage={'preplace': 5, 'place': 5},
            endpoint_tolerance_rad=.03, settle_timeout_s=1.5,
            replay_object_names=("bowl",),
        )

        def observe(stage: str) -> dict:
            state = runtime.query(robot_groups=(group,), object_names=("bowl",))
            save_state(runtime.model, runtime.copy_data(),
                       output / "states" / f"{stage}.npz")
            return state

        def stop(name: str, *, outcome: dict | None = None, error=None) -> dict:
            row = {"name": name, "success": False}
            if outcome is not None:
                row.update({"planning_failure_category": (
                    (outcome.get("plan") or {}).get("failure_category")),
                    "execution_failure_type": (
                        (outcome.get("execution") or {}).get("failure_type"))})
            if error is not None:
                row.update({"exception_type": type(error).__name__,
                            "message": str(error)})
            report["stages"].append(row)
            report["failure_stage"] = name
            return report

        observed = OperationState(current['state_id'], 'mujoco', current)
        bowl_detach = bowl_settled = settled_support = None
        def perform(name):
            nonlocal current, observed, bowl_detach, bowl_settled, settled_support
            if name in {'preplace', 'place'}:
                target = targets.preplace.matrix if name == 'preplace' else targets.place.matrix
                try:
                    outcome = attached_bridge(name, target, current["state_id"])
                except Exception as error:
                    return stop(name, error=error)
                operation = "linear_tcp" if name == 'place' else 'plan'
                if not outcome.get("success"):
                    return stop(name, outcome=outcome)
                current = observe(name)
                report["stages"].append({
                    "name": name, "success": True, "operation": operation,
                    "planning_collision_scope": "full_world_with_bowl_attachment",
                    "collision_snapshot_ref": (
                        f"motion_attached/collision/{name}/collision_snapshot.json"),
                    "attachment_ref": (
                        f"motion_attached/attachments/{name}.json"),
                    "planning_duration_s": (outcome.get("plan") or {}).get("duration_s"),
                    "end_state_id": current["state_id"],
                })
                _write_json(output / "result.json", report)
                observed = OperationState(current['state_id'], 'mujoco', current)
                return {'success': True}
            if name == 'hold_check':
                active = runtime.simulation_attachments()
                valid = bool(len(active) == 1 and active[0].get('object_instance_id') == 'bowl'
                    and (active[0].get('activation_contact') or {}).get('bilateral_pad_contact'))
                return {'success': True, 'check': CheckResult(
                    'pass' if valid else 'fail', 'mujoco_recorded_assistance',
                    {'simulation_attachments': active})}
            if name == 'release':
                width = float(runtime.robot_metadata["grippers"][group]["width_range_m"][1])
                if recorder is not None:
                    recorder.begin_stage("release", runtime.model, runtime.data)
                try:
                    opened = runtime.execute_gripper_command(
                        group=group, width_m=width, duration_s=.6,
                        planned_state_id=current["state_id"], timeout_s=1.0,
                        replay_object_names=("bowl",),
                        step_observer=(None if recorder is None else recorder.observer("release")),
                    )
                except Exception as error:
                    return stop("release", error=error)
                finally:
                    if recorder is not None:
                        recorder.end_stage("release", runtime.model, runtime.data)
                if not opened.get("success"):
                    return stop("release", outcome={"execution": opened})
                current = observe("release")
                report["stages"].append({"name": "release", "success": True,
                                         "width_m": width, "end_state_id": current["state_id"]})
                observed = OperationState(current['state_id'], 'mujoco', current)
                return {'success': True}
            if name == 'detach':
                if recorder is not None:
                    recorder.begin_stage("detach", runtime.model, runtime.data)
                try:
                    runtime.release_simulation_object(group, "bowl")
                except Exception as error:
                    return stop("detach", error=error)
                finally:
                    if recorder is not None:
                        recorder.end_stage("detach", runtime.model, runtime.data)
                current = observe("detach")
                report["stages"].append({"name": "detach", "success": True,
                                         "end_state_id": current["state_id"]})
                bowl_detach = np.asarray(current["objects"]["bowl"]["T_world_object"])
                observed = OperationState(current['state_id'], 'mujoco', current)
                return {'success': True}
            if name == 'settle':
                if recorder is not None:
                    recorder.begin_stage("settle", runtime.model, runtime.data)
                try:
                    held = runtime.execute_hold(
                        duration_s=.5, planned_state_id=current["state_id"],
                        replay_object_names=("bowl",), replay_group=group,
                        step_observer=(None if recorder is None else recorder.observer("settle")),
                    )
                except Exception as error:
                    return stop("settle", error=error)
                finally:
                    if recorder is not None:
                        recorder.end_stage("settle", runtime.model, runtime.data)
                if not held.get("success"):
                    return stop("settle", outcome={"execution": held})
                current = observe("settle")
                report["stages"].append({"name": "settle", "success": True,
                                         "end_state_id": current["state_id"]})
                bowl_settled = np.asarray(current["objects"]["bowl"]["T_world_object"])
                settled_support = _table_support(runtime)
                observed = OperationState(current['state_id'], 'mujoco', current)
                return {'success': True}
            if name == 'release_check':
                detached = not runtime.simulation_attachments()
                return {'success': True, 'check': CheckResult(
                    'pass' if detached and settled_support['supported'] else 'fail',
                    'mujoco_contact_and_attachment',
                    {'settled_table_support': settled_support,
                     'simulation_attachment_released': detached})}
            if name == 'retreat':
                detached_bridge = CuroboRuntimeBridge(
                    runtime, output / "motion_detached", group=group,
                    visual_recorder=recorder,
                    disabled_body_names_for_stage=lambda _stage, _obs: ("bowl",),
                    plan_postcheck_for_stage=lambda stage, plan, _: validate_insertion_plan(
                        runtime, 'retreat', plan, tcp_bowl),
                    execution_stop_condition_for_stage=lambda stage, model, data, _: (
                        None if (check := _place_contact_check(model, data, runtime.robot_metadata))['success']
                        else {'failure_type': 'forbidden_table_retreat_contact', **check}),
                    linear_tcp_stages=("retreat",), endpoint_tolerance_rad=.03,
                    settle_timeout_s=1.5, replay_object_names=("bowl",),
                )
                try:
                    outcome = detached_bridge("retreat", targets.retreat.matrix,
                                              current["state_id"])
                except Exception as error:
                    return stop("retreat", error=error)
                if not outcome.get("success"):
                    return stop("retreat", outcome=outcome)
                current = observe("retreat")
                report["stages"].append({
                    "name": "retreat", "success": True, "operation": "linear_tcp",
                    "collision_snapshot_ref": "motion_detached/collision/retreat/collision_snapshot.json",
                    "planning_duration_s": (outcome.get("plan") or {}).get("duration_s"),
                    "end_state_id": current["state_id"],
                })
                observed = OperationState(current['state_id'], 'mujoco', current)
                return {'success': True}
            if name == 'final_check':
                bowl_final = np.asarray(current["objects"]["bowl"]["T_world_object"])
                final_support = _table_support(runtime)
                settle_translation = float(np.linalg.norm(
                    bowl_settled[:3, 3] - bowl_detach[:3, 3]))
                retreat_translation = float(np.linalg.norm(
                    bowl_final[:3, 3] - bowl_settled[:3, 3]))
                final_tilt = _tilt_deg(bowl_final)
                checks = {
                    "settled_table_support": settled_support,
                    "final_table_support": final_support,
                    "settle_translation_m": settle_translation,
                    "retreat_translation_m": retreat_translation,
                    "maximum_allowed_translation_m": max_settle_translation_m,
                    "final_bowl_tilt_deg": final_tilt,
                    "maximum_allowed_tilt_deg": max_tilt_deg,
                    "simulation_attachment_released": not runtime.simulation_attachments(),
                }
                report["physical_place_check"] = checks
                report["physical_table_place_success"] = table_place_postcondition(checks)
                report["failure_stage"] = None if report["physical_table_place_success"] \
                    else "retreat"
                return {'success': True, 'check': CheckResult(
                    'pass' if report['physical_table_place_success'] else 'fail',
                    'actual_mujoco_contact_and_pose', checks)}
            raise ValueError(f'未知桌面放置阶段：{name}')

        def guarded_perform(name):
            nonlocal observed
            try:
                result = perform(name)
                if result is not report:
                    return result
            except Exception as error:
                stop(name, error=error)
            # 错误发生在执行之后时，保存当前物理状态，避免回退到上一步。
            try:
                current_failure = observe(f'{name}_failure')
                observed = OperationState(current_failure['state_id'], 'mujoco', current_failure)
            except Exception as error:
                observed = OperationState(f'unavailable:{name}', 'unavailable',
                    {'state_unavailable': True, 'reason': str(error)})
            return {'success': False}
        flow = run_transfer_flow(CallbackOperationBackend(
            observe=lambda: observed, perform=guarded_perform, evidence_mode='mujoco', failure_details=lambda stage: report.get('first_failure')),
            operation='place', approve_path=False, stop_after=stop_after)
        report['shared_flow'] = flow.to_dict()
        report['requested_scope'] = flow.requested_scope
        report['operation_success'] = flow.success
        report['actual_final'] = flow.to_dict()['end_state']
        if stop_after and flow.success:
            report['physical_table_place_success'] = None
        if not flow.success:
            report['physical_table_place_success'] = False
            if report.get('failure_stage') is None:
                report['failure_stage'] = flow.failure_stage
        return report
    finally:
        if recorder is not None:
            try:
                recorder.finalize()
                report["video_ref"] = "visual/skill_execution.mp4"
            except Exception as error:
                report["render_error"] = f"{type(error).__name__}: {error}"
        _write_json(output / "result.json", report)


__all__ = ["TABLE_PLACE_STAGES", "align_table_tcp_to_surface",
           "run_assisted_table_place"]
