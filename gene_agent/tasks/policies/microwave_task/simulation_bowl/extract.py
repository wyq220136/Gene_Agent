"""从当前 MuJoCo 状态按实际碗姿态执行炉内取碗的抓取前缀。

本模块组合碗缘 Knowledge、真实指垫支点与现有规划器。第一次物理
验收止于闭爪：运动轨迹通过不代表已抓住碗，接触和碗口姿态必须单独
检查。后续携物撤离与桌面放置仍由同一个 Runtime 的后续阶段接入。
"""

from __future__ import annotations

import json
from pathlib import Path
import traceback

import numpy as np

from gene_agent.skills.library.transfer.skill import run_extract_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult
from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.feedback.state import model_fingerprint, save_state
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.motion.model import JointTrajectory
from gene_agent.simulation.planned_contact import validate_planned_target_contacts
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.visualization.trajectory import TrajectoryRecorder

from gene_agent.tasks.policies.microwave_task.planning.extract import _world_legacy_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    load_object_operation_profile, extract_profile_targets,
)
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase
from gene_agent.tasks.policies.microwave_task.operations.extract import actual_bowl_extract_targets
from gene_agent.tasks.policies.microwave_task.simulation_bowl.pick import close_bowl_on_contact
from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import insertion_contact_check


ROOT = Path(__file__).resolve().parents[5]
EXTRACT_GRASP_STAGE_ORDER = (
    "open", "pre_extract_1", "pre_extract_2", "grasp", "close",
)
# 炉内取碗时，通用随夹爪侧视相机会落到门板后面。此相机只影响本探针
# 的录像，从炉腔侧前方看清两指与碗沿；操作目标和仿真物理均不改变。
_CAVITY_GRIPPER_SIDE_CAMERA = {
    "lookat": [.52, .40, .135], "distance": .36,
    "azimuth": 50., "elevation": -14.,
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tilt_deg(pose_world: np.ndarray) -> float:
    """Bowl_0 的局部 +Y 是碗口方向，用世界 +Z 衡量倾斜。"""
    pose = np.asarray(pose_world, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("碗口姿态检查需要有限的 4x4 世界 pose")
    return float(np.rad2deg(np.arccos(np.clip(pose[2, 1], -1., 1.))))


def build_bowl_motion_trace(
    replay_samples: list[dict], initial_bowl_pose_world: np.ndarray,
    *, group: str,
) -> dict:
    """从执行器已有的重放采样定位碗先滑动、后倾斜的时间。

    只读取原本就会记录的实际状态，不额外触发规划、物理积分或录像。
    每个越限点同时保留 TCP，方便把数值与同一阶段的录像帧对齐。
    """
    initial = np.asarray(initial_bowl_pose_world, dtype=float)
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("碗运动轨迹需要有限的 4x4 初始 pose")
    rows = []
    first_time = float(replay_samples[0]["sim_time_s"]) if replay_samples else 0.
    for index, sample in enumerate(replay_samples):
        pose = np.asarray(sample["objects"]["bowl"]["T_world_object"], dtype=float)
        robot = sample["robots"][group]
        tcp = np.asarray(robot["T_world_tcp"], dtype=float)
        row = {
            "index": index,
            "sim_time_s": round(float(sample["sim_time_s"]), 6),
            "time_from_stage_start_s": round(
                float(sample["sim_time_s"]) - first_time, 3),
            "bowl_tilt_deg": round(_tilt_deg(pose), 2),
            "bowl_displacement_mm": round(float(np.linalg.norm(
                pose[:3, 3] - initial[:3, 3])) * 1000., 2),
            "bowl_position_world_m": np.round(pose[:3, 3], 5).tolist(),
            "tcp_position_world_m": np.round(tcp[:3, 3], 5).tolist(),
            "gripper_width_m": round(float(robot["gripper"]["width_m"]), 5),
        }
        rows.append(row)

    def first_over(field: str, threshold: float) -> dict | None:
        return next((row for row in rows if row[field] > threshold), None)

    return {
        "sample_count": len(rows),
        "max_tilt_deg": max((row["bowl_tilt_deg"] for row in rows), default=0.),
        "max_displacement_mm": max(
            (row["bowl_displacement_mm"] for row in rows), default=0.),
        "first_displacement_over_5mm": first_over("bowl_displacement_mm", 5.),
        "first_tilt_over_5deg": first_over("bowl_tilt_deg", 5.),
        "first_tilt_over_15deg": first_over("bowl_tilt_deg", 15.),
        "samples": rows,
    }


def evaluate_extract_grasp(
    initial_bowl_pose_world: np.ndarray,
    after_close_bowl_pose_world: np.ndarray,
    contact: dict,
    *,
    close_execution_success: bool,
    max_tilt_deg: float = 15.,
    max_displacement_m: float = .02,
) -> dict:
    """把真实闭爪、双侧指垫、姿态和位移作为独立验收事实。

    只在抓取前缀验收；它不证明后续携物不会滑落。阈值明示在报告中，
    调参时不得只改结果标签或把单侧接触当作有效夹持。
    """
    initial = np.asarray(initial_bowl_pose_world, dtype=float)
    final = np.asarray(after_close_bowl_pose_world, dtype=float)
    if (initial.shape != (4, 4) or final.shape != (4, 4)
            or not np.isfinite(initial).all() or not np.isfinite(final).all()
            or not np.isfinite(max_tilt_deg) or not 0 < max_tilt_deg < 90
            or not np.isfinite(max_displacement_m) or max_displacement_m <= 0):
        raise ValueError("抓取验收需要有效的碗 pose、倾角和位移阈值")
    tilt = _tilt_deg(final)
    displacement = float(np.linalg.norm(final[:3, 3] - initial[:3, 3]))
    undesired = contact.get("undesired_gripper_contacts") or []
    pad_contact = bool(contact.get("success")
                       and contact.get("bilateral_pad_contact")
                       and not undesired)
    return {
        "success": bool(close_execution_success and pad_contact
                        and tilt <= max_tilt_deg
                        and displacement <= max_displacement_m),
        "close_execution_success": bool(close_execution_success),
        "bilateral_pad_contact": bool(contact.get("bilateral_pad_contact")),
        "valid_pad_contact": pad_contact,
        "undesired_gripper_contact_count": len(undesired),
        "bowl_tilt_deg": tilt,
        "maximum_allowed_tilt_deg": float(max_tilt_deg),
        "bowl_displacement_m": displacement,
        "maximum_allowed_displacement_m": float(max_displacement_m),
        "reason_codes": [reason for reason, failed in (
            ("close_execution_failed", not close_execution_success),
            ("invalid_pad_contact", not pad_contact),
            ("bowl_tilt_exceeded", tilt > max_tilt_deg),
            ("bowl_displacement_exceeded", displacement > max_displacement_m),
        ) if failed],
    }


def run_physical_extract_grasp(
    runtime,
    output_dir: Path,
    *,
    group: str = "right",
    object_id: str = "Bowl_0",
    grasp_offset_world_m=(0., 0., 0.),
    max_tilt_deg: float = 15.,
    max_displacement_m: float = .02,
    record_video: bool = False,
) -> dict:
    """消费在线实际碗状态，逐段重规划碗缘接近关键点并执行至闭爪。

    `grasp_offset_world_m` 是诊断覆盖量，只移动最后一个 TCP 目标，
    不写回旧 Profile；相机/规划碰撞与接触记录都属于同一实际状态链。
    调用方负责装配来源 Runtime，并确认门已物理打开及碗已落稳。
    """
    if object_id != "Bowl_0":
        raise ValueError("当前物理取出探针仅有 Bowl_0 仿真资产")
    offset = np.asarray(grasp_offset_world_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() \
            or np.any(np.abs(offset) > .02):
        raise ValueError("grasp 世界坐标偏移必须是各轴 ±20 mm 内的有限值")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    initial = runtime.query(robot_groups=(group,), object_names=("bowl",))
    bowl_initial = np.asarray(initial["objects"]["bowl"]["T_world_object"],
                              dtype=float)
    case = PickBowlCase(
        output, runtime.scene_revision, runtime.model, runtime.data,
        runtime.metadata, {}, bowl_initial, [],
    )
    world_legacy, alignment = _world_legacy_frame(case)
    profile = load_object_operation_profile(ROOT / "data/task_1", object_id)
    tcp_initial = np.asarray(initial["robots"][group]["T_world_tcp"])
    pads = [runtime.model.geom(name).id for name in
            runtime.robot_metadata["grippers"][group]["pad_geom_names"]]
    pad_center = runtime.copy_data().geom_xpos[pads].mean(axis=0)
    # 当前炉体局部+Y为进炉方向；旧资产frame轴向不同，不能混用。
    inward = np.asarray(runtime.copy_data().body("microwave").xmat).reshape(3, 3)[:, 1]
    target_poses, target_details = actual_bowl_extract_targets(
        bowl_initial, tcp_initial, pad_center, inward)
    target_poses["grasp"][:3, 3] += offset
    recorder = None
    if record_video:
        recorder = TrajectoryRecorder(
            output / "visual", target_body="bowl", width=640, height=480,
            fps=10, sample_interval_s=.1, combine_mp4=True,
            detail_view="gripper_side", extra_view="contact_side",
            overview_camera={"lookat": [.53, .39, .15], "distance": .6,
                             "azimuth": -12., "elevation": -22.},
            stage_camera_overrides={
                stage: {"gripper_side": _CAVITY_GRIPPER_SIDE_CAMERA}
                for stage in ("initial", *EXTRACT_GRASP_STAGE_ORDER)
            },
            robot_metadata=runtime.robot_metadata, active_group=group,
        )
        # 即使第一段规划失败也保存实际初态，用户可看清失败发生的位置。
        recorder.begin_stage("initial", runtime.model, runtime.data)
        recorder.end_stage("initial", runtime.model, runtime.data)
    def contact_postcheck(stage, plan, snapshot):
        if stage != "grasp":
            return None
        trajectory = JointTrajectory(
            tuple(plan["joint_names"]),
            tuple(i * plan["interpolation_dt_s"]
                  for i in range(len(plan["positions_rad"]))),
            tuple(tuple(row) for row in plan["positions_rad"]),
        )
        return validate_planned_target_contacts(
            runtime.model, runtime.data, runtime.robot_metadata,
            group=group, object_body="bowl", trajectory=trajectory,
            gripper_joint_positions_m=snapshot["gripper_snapshot"]["joint_positions_m"],
            allowed_geom_names=runtime.robot_metadata["grippers"][group]["pad_geom_names"],
        )

    bridge = CuroboRuntimeBridge(
        runtime, output / "motion", group=group, visual_recorder=recorder,
        linear_tcp_stages=("pre_extract_2", "grasp"),
        disabled_body_names_for_stage=lambda stage, observation: (
            ("bowl",) if stage == "grasp" else ()),
        plan_postcheck_for_stage=contact_postcheck,
        execution_stop_condition_for_stage=lambda stage, model, data, sample: (
            None if (check := insertion_contact_check(model, data, runtime.robot_metadata))["success"]
            else {"failure_type": "forbidden_contact_during_extract", **check}),
        endpoint_tolerance_rad=.03, settle_timeout_s=1.5,
        replay_object_names=("bowl",),
    )
    report = {
        "schema_version": 1,
        "skill": "extract_bowl_grasp_prefix",
        "mode": "physical",
        "source_state_id": initial["state_id"],
        "model_fingerprint": model_fingerprint(runtime.model),
        "group": group,
        "object_id": object_id,
        "grasp_offset_world_m": offset.tolist(),
        "profile_ref": "data/task_1/knowledge/microwave_object_manipulation.json",
        "frame_alignment": alignment,
        "target_generation": target_details,
        "stage_order": list(EXTRACT_GRASP_STAGE_ORDER),
        "targets_world_tcp": {name: target.tolist()
                              for name, target in target_poses.items()},
        "stages": [],
        "physical_grasp_prefix_success": False,
        "full_extract_success": False,
    }

    def save_progress() -> None:
        _write_json(output / "result.json", report)

    def observe(stage: str) -> dict:
        current = runtime.query(robot_groups=(group,), object_names=("bowl",))
        save_state(runtime.model, runtime.copy_data(),
                   output / "states" / f"{stage}.npz")
        diagnostics = query_diagnostics(runtime, body_names=("bowl", "microwave"))
        _write_json(output / "contacts" / f"{stage}.json", diagnostics)
        return current

    current = initial

    def operation_state():
        actual = runtime.query(robot_groups=(group,), object_names=('bowl',))
        return OperationState(actual['state_id'], 'mujoco_execution', actual)

    def perform(name):
        nonlocal current
        if name == 'grasp_check':
            check = report['physical_grasp_check']
            return {'success': True, 'check': CheckResult(
                'pass' if check['success'] else 'fail', 'mujoco_contact', check)}
        if name == 'open':
            width = float(profile["grasp_pregrasp_opening_width_m"])
            command = dict(group=group, width_m=width, duration_s=.6,
                           planned_state_id=initial["state_id"],
                           replay_object_names=("bowl",))
            if recorder is not None:
                recorder.begin_stage("open", runtime.model, runtime.data)
                command["step_observer"] = recorder.observer("open")
            try:
                opened = runtime.execute_gripper_command(**command)
            except Exception as error:
                report["stages"].append({
                    "name": "open", "success": False,
                    "exception_type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(),
                })
                report["failure_stage"] = "open"
                return {'success': False}
            finally:
                if recorder is not None:
                    recorder.end_stage("open", runtime.model, runtime.data)
            report["stages"].append({"name": "open", "success": opened["success"],
                                     "execution": opened})
            save_progress()
            if not opened["success"]:
                report["failure_stage"] = "open"
                return {'success': False}
            current = observe("open")

            return {'success': True}
        if name in target_poses:
            start = current
            try:
                outcome = bridge(name, target_poses[name], start["state_id"])
            except Exception as error:
                report["stages"].append({
                    "name": name, "success": False,
                    "start_state_id": start["state_id"],
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                })
                report["failure_stage"] = name
                return {'success': False}
            row = {
                "name": name,
                "success": bool(outcome["success"]),
                "start_state_id": start["state_id"],
                "requested_target_world_tcp": target_poses[name].tolist(),
                "operation": ("plan" if name == "pre_extract_1"
                              else "linear_tcp"),
                "planning_collision_scope": (
                    "world_and_self_with_verified_pad_bowl_contact"),
                "collision_snapshot_ref": (
                    f"motion/collision/{name}"
                    "/collision_snapshot.json"),
                "planning_duration_s": (
                    outcome.get("plan") or {}).get("duration_s"),
                "simulation_duration_s": (
                    outcome.get("execution") or {}).get("sim_duration_s"),
                "planning_failure_category": (
                    outcome.get("plan") or {}).get("failure_category"),
                "execution_failure_type": (
                    outcome.get("execution") or {}).get("failure_type"),
            }
            execution = outcome.get("execution") or {}
            if execution.get("replay_samples"):
                trace = build_bowl_motion_trace(
                    execution["replay_samples"], bowl_initial, group=group)
                trace_ref = f"bowl_motion/{name}.json"
                _write_json(output / trace_ref, trace)
                row["bowl_motion_ref"] = trace_ref
                row["bowl_motion"] = {key: value for key, value in trace.items()
                                      if key != "samples"}
            if outcome["success"]:
                current = observe(name)
                row["end_state_id"] = current["state_id"]
                # 规划成功只说明目标有轨迹；物理执行的 TCP 仍可能因
                # 控制跟踪或接触偏离目标，抓取阶段必须显式报告偏差。
                actual_tcp = np.asarray(
                    current["robots"][group]["T_world_tcp"], dtype=float)
                tcp_error_mm = (actual_tcp[:3, 3]
                                - target_poses[name][:3, 3]) * 1000.
                row["actual_target_tcp_position_error_mm"] = (
                    np.round(tcp_error_mm, 2).tolist())
                row["actual_target_tcp_position_error_norm_mm"] = round(
                    float(np.linalg.norm(tcp_error_mm)), 2)
                row["bowl_pose_world"] = current["objects"]["bowl"][
                    "T_world_object"]
                row["bowl_tilt_deg"] = _tilt_deg(row["bowl_pose_world"])
            report["stages"].append(row)
            save_progress()
            if not outcome["success"]:
                report["failure_stage"] = name
                return {'success': False}

            return {'success': True}
        if name == 'close':
            close_command = dict(
                group=group, width_m=float(profile["grasp_opening_width_m"]),
                duration_s=1.5, planned_state_id=current["state_id"],
                replay_object_names=("bowl",),
            )
            if recorder is not None:
                recorder.begin_stage("close", runtime.model, runtime.data)
                close_command["step_observer"] = recorder.observer("close")
            try:
                closed = close_bowl_on_contact(runtime, **close_command)
            except Exception as error:
                report["stages"].append({
                    "name": "close", "success": False,
                    "exception_type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(),
                })
                report["failure_stage"] = "close"
                return {'success': False}
            finally:
                if recorder is not None:
                    recorder.end_stage("close", runtime.model, runtime.data)
            current = observe("close")
            if closed.get("replay_samples"):
                trace = build_bowl_motion_trace(
                    closed["replay_samples"], bowl_initial, group=group)
                trace_ref = "bowl_motion/close.json"
                _write_json(output / trace_ref, trace)
                report["close_bowl_motion_ref"] = trace_ref
                report["close_bowl_motion"] = {
                    key: value for key, value in trace.items() if key != "samples"
                }
            contacts = query_diagnostics(runtime, body_names=("bowl",))["contacts"]
            contact = classify_grasp_contacts(
                contacts, runtime.robot_metadata["grippers"][group],
                object_body="bowl",
            )
            check = evaluate_extract_grasp(
                bowl_initial, current["objects"]["bowl"]["T_world_object"],
                contact, close_execution_success=closed["success"],
                max_tilt_deg=max_tilt_deg,
                max_displacement_m=max_displacement_m,
            )
            report["stages"].append({
                "name": "close", "success": bool(closed["success"]),
                "end_state_id": current["state_id"],
                "bowl_pose_world": current["objects"]["bowl"]["T_world_object"],
                "gripper_width_m": float(profile["grasp_opening_width_m"]),
                "contact": {key: contact[key] for key in (
                    "success", "bilateral_pad_contact", "pad_sides",
                    "contact_count", "undesired_gripper_contacts")},
                "execution": closed,
            })
            report["physical_grasp_check"] = check
            report["physical_grasp_prefix_success"] = check["success"]
            report["failure_stage"] = None if check["success"] else "close"
            return {'success': bool(closed['success'])}
        raise ValueError(f'未知取出阶段 {name}')

    try:
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
            stop_after='grasp')
        report['shared_flow'] = flow.to_dict()
        report['physical_grasp_prefix_success'] = flow.success
        if not flow.success and not report.get('failure_stage'):
            report['failure_stage'] = flow.failure_stage
        return report
    finally:
        bridge.close()
        if recorder is not None:
            try:
                recorder.finalize()
                report["video_ref"] = "visual/skill_execution.mp4"
            except Exception as error:
                report["render_error"] = f"{type(error).__name__}: {error}"
        save_progress()


__all__ = [
    "EXTRACT_GRASP_STAGE_ORDER", "build_bowl_motion_trace",
    "evaluate_extract_grasp",
    "run_physical_extract_grasp",
]
