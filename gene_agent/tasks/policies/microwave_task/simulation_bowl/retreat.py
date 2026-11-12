"""从真实闭爪状态保持实际夹持姿态，先抬升再向炉口外撤碗。"""

from __future__ import annotations
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode

from pathlib import Path
import traceback

import mujoco
import numpy as np
import trimesh

from gene_agent.skills.library.transfer.skill import run_extract_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult
from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.feedback.state import model_fingerprint, save_state
from gene_agent.motion.curobo.attachment import build_attachment_surface_spheres_from_mesh
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.visualization.trajectory import TrajectoryRecorder

from gene_agent.tasks.policies.microwave_task.planning.attachment import BOWL_MESH
from gene_agent.tasks.policies.microwave_task.planning.extract import _world_legacy_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    load_object_operation_profile, extract_profile_targets,
)
from gene_agent.tasks.policies.microwave_task.simulation_bowl.extract import _tilt_deg, _write_json, build_bowl_motion_trace
from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import cavity_footprint_margins
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase
from gene_agent.tasks.policies.microwave_task.operations.extract import actual_hold_retreat_targets
from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import validate_insertion_plan, insertion_contact_check


ROOT = Path(__file__).resolve().parents[5]
RETREAT_STAGE_ORDER = ("attach", "post_extract_1", "post_extract_2")


def bowl_exterior_clearance(runtime) -> dict:
    """用 Bowl 全部碰撞体的保守外接球检查它是否完全越过炉门前沿。

    前沿由关门姿态的门板与炉腔地板确定，只在 MuJoCo 副本中计算；
    不改变实际门角。正余量表示整个碰撞代理已退出，而非仅碗中心退出。
    """
    model, data = runtime.model, runtime.data
    bowl_id = model.body("bowl").id
    geom_ids = [index for index in range(model.ngeom)
                if int(model.geom_bodyid[index]) == bowl_id
                and int(model.geom_contype[index]) != 0]
    if not geom_ids:
        raise ValueError("取碗终点缺少 Bowl 碰撞体")
    cavity = cavity_footprint_margins(model, data, geom_ids)
    floor_id = model.geom("microwave_floor_collision").id
    axis = int(cavity["front_axis"])
    sign = float(cavity["front_sign"])
    boundary = float(cavity["front_boundary_local_m"])
    floor_rotation = np.asarray(data.geom_xmat[floor_id]).reshape(3, 3)
    floor_origin = np.asarray(data.geom_xpos[floor_id])
    projections = [float(floor_rotation[:, axis] @
                         (data.geom_xpos[index] - floor_origin))
                   for index in geom_ids]
    if sign < 0:
        nearest_to_cavity = max(
            value + float(model.geom_rbound[index])
            for index, value in zip(geom_ids, projections))
        clearance = boundary - nearest_to_cavity
    else:
        nearest_to_cavity = min(
            value - float(model.geom_rbound[index])
            for index, value in zip(geom_ids, projections))
        clearance = nearest_to_cavity - boundary
    return {
        "clearance_m": float(clearance),
        "front_axis": axis,
        "front_sign": sign,
        "front_boundary_local_m": boundary,
        "nearest_bowl_proxy_local_m": nearest_to_cavity,
        "bowl_collision_geom_count": len(geom_ids),
        "method": "all_bowl_collision_geom_bounding_spheres_vs_closed_door_front",
    }


def run_assisted_extract_retreat(
    runtime, output_dir: Path, grasp_report: dict, *,
    group: str = "right", object_id: str = "Bowl_0",
    max_tilt_deg: float = 15., min_exterior_clearance_m: float = .005,
    record_video: bool = False,
) -> dict:
    """仅在真实双侧夹持后显式启用仿真固定，并连续撤离。

    这是 assisted 模式：weld 只保证仿真碗不会因代理误差滑落，不能
    冒充纯物理夹持。两段保持实际 TCP 朝向，以完整场景避碰；
    所有目标在实际前一阶段状态上重新规划。
    """
    if object_id != "Bowl_0" or grasp_report.get("group") != group \
            or grasp_report.get("object_id") != object_id \
            or not grasp_report.get("physical_grasp_prefix_success"):
        raise ValueError("撤离必须接在同组同 Bowl 的真实抓取前缀成功状态后")
    if not np.isfinite(max_tilt_deg) or not 0 < max_tilt_deg < 90 \
            or not np.isfinite(min_exterior_clearance_m) \
            or min_exterior_clearance_m < 0:
        raise ValueError("取碗撤离的倾角与完全退出余量必须有效")
    close = next((row for row in grasp_report.get("stages", ())
                  if row.get("name") == "close"), None)
    current = runtime.query(robot_groups=(group,), object_names=("bowl",))
    if close is None or close.get("end_state_id") != current["state_id"]:
        raise ValueError("抓取报告终态与当前 MuJoCo 状态不一致")
    if grasp_report.get("model_fingerprint") != model_fingerprint(runtime.model):
        raise ValueError("抓取报告与当前 MuJoCo 模型指纹不一致")
    contacts = query_diagnostics(runtime, body_names=("bowl",))["contacts"]
    evidence = classify_grasp_contacts(
        contacts, runtime.robot_metadata["grippers"][group], object_body="bowl")
    if not evidence["success"]:
        raise ValueError("当前状态已失去有效双侧指垫接触，禁止辅助 attach")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bowl_initial = np.asarray(current["objects"]["bowl"]["T_world_object"])
    case = PickBowlCase(output, runtime.scene_revision, runtime.model,
                        runtime.data, runtime.metadata, {}, bowl_initial, [])
    world_legacy, alignment = _world_legacy_frame(case)
    profile = load_object_operation_profile(ROOT / "data/task_1", object_id)
    actual_tcp = np.asarray(current["robots"][group]["T_world_tcp"])
    tcp_bowl = np.linalg.inv(actual_tcp) @ bowl_initial
    inward = np.asarray(runtime.copy_data().body("microwave").xmat).reshape(3, 3)[:, 1]
    target_poses = actual_hold_retreat_targets(actual_tcp, inward)
    mesh = (trimesh.load_mesh(BOWL_MESH, process=False)
            if resolve_collision_mode() is CollisionMode.FULL else None)
    centers_object, radii, removed = build_attachment_surface_spheres_from_mesh(
        mesh, num_spheres=32, radius=.002)
    report = {
        "schema_version": 1,
        "skill": "extract_bowl_assisted_retreat",
        "mode": "physical_assisted",
        "natural_grasp_success": False,
        "assisted_extract_success": False,
        "full_task_success": False,
        "group": group, "object_id": object_id,
        "source_close_state_id": current["state_id"],
        "source_model_fingerprint": model_fingerprint(runtime.model),
        "frame_alignment": alignment,
        "target_policy": "actual_hold_lift_then_outward",
        "T_tcp_bowl_from_actual_hold": tcp_bowl.tolist(),
        "profile_ref": "data/task_1/knowledge/microwave_object_manipulation.json",
        "stage_order": list(RETREAT_STAGE_ORDER),
        "targets_world_tcp": {name: pose.tolist()
                              for name, pose in target_poses.items()},
        "attached_proxy": {
            "source": "old_visual_32_surface_spheres",
            "sphere_count": len(radii),
            "removed_mesh_component": removed,
        },
        "stages": [],
    }
    recorder = None
    try:
        if record_video:
            recorder = TrajectoryRecorder(
                output / "visual", target_body="bowl", width=640, height=480,
                fps=10, sample_interval_s=.1, combine_mp4=True,
                detail_view="gripper_side", extra_view="contact_side",
                overview_camera={"lookat": [.46, .38, .17], "distance": .72,
                                 "azimuth": -12., "elevation": -22.},
                stage_camera_overrides={
                    "attach": {"gripper_side": {
                        "lookat": [.52, .40, .15], "distance": .48,
                        "azimuth": -20., "elevation": -14.}},
                    "post_extract_1": {"gripper_side": {
                        "lookat": [.50, .39, .16], "distance": .48,
                        "azimuth": -20., "elevation": -14.}},
                    "post_extract_2": {"gripper_side": {
                        "lookat": [.38, .39, .16], "distance": .55,
                        "azimuth": -20., "elevation": -14.}},
                },
                robot_metadata=runtime.robot_metadata, active_group=group,
            )
            recorder.begin_stage("attach", runtime.model, runtime.data)
        def attachment_for_stage(_stage: str, _observation: dict) -> dict:
            active = runtime.simulation_attachments()
            if len(active) != 1 or active[0]["object_instance_id"] != "bowl":
                raise ValueError("撤离阶段缺少显式碗附着事实")
            bowl = np.asarray(runtime.query(object_names=("bowl",))["objects"][
                "bowl"]["T_world_object"])
            centers = (bowl[:3, :3] @ centers_object.T).T + bowl[:3, 3]
            return {
                "object_instance_id": "bowl",
                "spheres_world_m": np.column_stack((centers, radii)).tolist(),
                "coverage": "old_visual_32_surface_spheres",
                "physical_hold_evidence": {
                    "mode": "simulation_privileged_weld",
                    "activation_contact": active[0]["activation_contact"],
                },
            }

        bridge = CuroboRuntimeBridge(
            runtime, output / "motion", group=group,
            visual_recorder=recorder,
            attachment_for_stage=attachment_for_stage,
            linear_tcp_stages=RETREAT_STAGE_ORDER[1:],
            plan_postcheck_for_stage=lambda stage, plan, snapshot: (
                validate_insertion_plan(runtime, "insert", plan, tcp_bowl)),
            execution_stop_condition_for_stage=lambda stage, model, data, sample: (
                None if (check := insertion_contact_check(model, data, runtime.robot_metadata))["success"]
                else {"failure_type": "forbidden_contact_during_extract", **check}),
            endpoint_tolerance_rad=.03, settle_timeout_s=1.5,
            replay_object_names=("bowl",),
        )
        maximum_path_tilt = _tilt_deg(bowl_initial)
        def operation_state():
            actual = runtime.query(robot_groups=(group,), object_names=('bowl',))
            return OperationState(actual['state_id'], 'mujoco_execution', actual)

        def perform(name):
            nonlocal current, maximum_path_tilt
            if name == 'grasp_check':
                return {'success': True, 'check': CheckResult(
                    'pass' if evidence['success'] else 'fail', 'mujoco_contact', evidence)}
            if name.endswith('_hold_check'):
                active = runtime.simulation_attachments()
                held = len(active) == 1 and active[0]['object_instance_id'] == 'bowl'
                return {'success': True, 'check': CheckResult(
                    'pass' if held else 'fail', 'simulation_privileged_weld',
                    {'active_attachment_count': len(active),
                     'physical_hold_verified': False})}
            if name == 'attach':
                attached = runtime.attach_simulation_object(
                    group, "bowl", contact_evidence=evidence)
                if recorder is not None:
                    recorder.end_stage("attach", runtime.model, runtime.data)
                current = runtime.query(robot_groups=(group,), object_names=("bowl",))
                save_state(runtime.model, runtime.copy_data(), output / "states/attach.npz")
                report["stages"].append({
                    "name": "attach", "success": True, "end_state_id": current["state_id"],
                    "assistance": attached,
                    "contact_count": evidence["contact_count"],
                })

                return {'success': True}
            if name == 'extract_check':
                clearance = bowl_exterior_clearance(runtime)
                clearance["required_clearance_m"] = float(min_exterior_clearance_m)
                clearance["success"] = clearance["clearance_m"] >= min_exterior_clearance_m
                report["exterior_clearance_check"] = clearance
                report["maximum_path_tilt_deg"] = maximum_path_tilt
                report["assisted_extract_success"] = bool(
                    clearance["success"] and maximum_path_tilt <= max_tilt_deg
                    and runtime.simulation_attachments())
                report["failure_stage"] = None if report["assisted_extract_success"] \
                    else "post_extract_2"
                return {'success': True, 'check': CheckResult(
                    'pass' if report['assisted_extract_success'] else 'fail',
                    'mujoco_assisted_execution', {'clearance': clearance,
                    'maximum_path_tilt_deg': maximum_path_tilt})}
            if name in target_poses:
                start = current
                try:
                    outcome = bridge(name, target_poses[name], start["state_id"])
                except Exception as error:
                    report["stages"].append({
                        "name": name, "success": False,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    })
                    report["failure_stage"] = name
                    return {'success': False}
                row = {
                    "name": name, "success": bool(outcome["success"]),
                    "operation": "linear_tcp",
                    "planning_collision_scope": "world_and_self_with_actual_attachment",
                    "collision_snapshot_ref": (
                        f"motion/collision/{name}/collision_snapshot.json"),
                    "attachment_ref": f"motion/attachments/{name}.json",
                    "planning_duration_s": (outcome.get("plan") or {}).get("duration_s"),
                    "planning_failure_category": (
                        outcome.get("plan") or {}).get("failure_category"),
                    "execution_failure_type": (
                        outcome.get("execution") or {}).get("failure_type"),
                }
                execution = outcome.get("execution") or {}
                if execution.get("replay_samples"):
                    trace = build_bowl_motion_trace(
                        execution["replay_samples"], bowl_initial, group=group)
                    ref = f"bowl_motion/{name}.json"
                    _write_json(output / ref, trace)
                    row["bowl_motion_ref"] = ref
                    row["maximum_bowl_tilt_deg"] = trace["max_tilt_deg"]
                    # 展示用轨迹保留两位小数，成功判定仍使用原始姿态，
                    # 不能把 15.004° 四舍五入成 15.00° 后误报通过。
                    maximum_path_tilt = max(maximum_path_tilt, *(
                        _tilt_deg(np.asarray(sample["objects"]["bowl"]["T_world_object"]))
                        for sample in execution["replay_samples"]
                    ))
                if outcome["success"]:
                    current = runtime.query(robot_groups=(group,), object_names=("bowl",))
                    save_state(runtime.model, runtime.copy_data(),
                               output / "states" / f"{name}.npz")
                    row["end_state_id"] = current["state_id"]
                    row["bowl_tilt_deg"] = _tilt_deg(np.asarray(
                        current["objects"]["bowl"]["T_world_object"]))
                report["stages"].append(row)
                _write_json(output / "result.json", report)
                if not outcome["success"]:
                    report["failure_stage"] = name
                    return {'success': False}
                return {'success': True}
            raise ValueError(f'未知取出撤离阶段 {name}')

        observed = operation_state()

        def guarded_perform(name):
            nonlocal observed
            try:
                outcome = perform(name)
            except Exception as error:
                report['stages'].append({
                    'name': name, 'success': False,
                    'exception_type': type(error).__name__, 'message': str(error),
                    'traceback': traceback.format_exc()})
                outcome = {'success': False}
            # 即使动作已部分执行后抛异常，也重新读取终态；不能退回前一阶段状态。
            try:
                observed = operation_state()
            except Exception as error:
                observed = OperationState(f'unavailable:{name}', 'unavailable', {
                    'state_unavailable': True, 'reason': str(error)})
                report['stages'].append({
                    'name': name, 'success': False, 'message': str(error),
                    'traceback': traceback.format_exc(),
                    'failure_type': 'end_observation_failed'})
                outcome = {'success': False}
            if not outcome['success']:
                report['failure_stage'] = name
                row = next((row for row in reversed(report['stages'])
                            if row.get('name') == name), None)
                if row is not None:
                    row['end_state_id'] = observed.state_id
                try:
                    save_state(runtime.model, runtime.copy_data(),
                               output / 'states' / f'{name}_failure.npz')
                except Exception as error:
                    report['failure_state_save_error'] = f'{type(error).__name__}: {error}'
            return outcome

        flow = run_extract_flow(CallbackOperationBackend(
            observe=lambda: observed, perform=guarded_perform, evidence_mode='execution', failure_details=lambda stage: report.get('first_failure')),
            resume_after_grasp=True)
        report['shared_flow'] = flow.to_dict()
        report['assisted_extract_success'] = flow.success
        if not flow.success and not report.get('failure_stage'):
            report['failure_stage'] = flow.failure_stage
        return report
    finally:
        if "bridge" in locals():
            bridge.close()
        # attach 会把当前 TCP→Bowl 相对位姿写入模型的 eq_data，
        # 因而阶段结束的模型指纹必须重新计算，供 Place 接力校验。
        report["model_fingerprint"] = model_fingerprint(runtime.model)
        if recorder is not None:
            try:
                recorder.finalize()
                report["video_ref"] = "visual/skill_execution.mp4"
            except Exception as error:
                report["render_error"] = f"{type(error).__name__}: {error}"
        _write_json(output / "result.json", report)


__all__ = ["RETREAT_STAGE_ORDER", "bowl_exterior_clearance",
           "run_assisted_extract_retreat"]
