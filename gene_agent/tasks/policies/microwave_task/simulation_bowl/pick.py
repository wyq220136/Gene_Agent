"""微波炉抓碗的 MuJoCo 装配；独立诊断和完整 Task 调用同一实现。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np


from gene_agent.feedback.frames import query_frame_diagnostics
from gene_agent.feedback.failure import capture_failure
from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.feedback.state import save_state
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.motion.model import JointTrajectory
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.simulation.planned_contact import (
    classify_target_contacts,
    validate_planned_target_contacts,
)
from gene_agent.visualization import render_views
from gene_agent.skills.library.pick.model import GraspSettings, PICK_STAGE_OPERATIONS, load_grasp_settings


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


from gene_agent.skills.library.pick.planning import make_pregrasp_pose
from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill


PICK_EXECUTION_POSITION_KP_SCALE_BY_STAGE = {
    "pregrasp": 1.0,
    "approach": 1.0,
    "lift": 1.0,
}
PICK_CLOSE_POSITION_KP_SCALE = 1.0
BOWL_GRASP_TABLE_ANGLE_RANGE_DEG = (45.0, 70.0)
BOWL_GRASP_PREFERRED_TABLE_ANGLE_DEG = 55.0


def close_bowl_on_contact(runtime, **command) -> dict:
    """小步闭爪，双侧指垫已接触便保持当前目标，不继续挤压碗沿。"""
    group = command['group']
    start = runtime.query(robot_groups=(group,))
    if start['state_id'] != command['planned_state_id']:
        raise ValueError('闭爪收到过期 planned_state_id')
    initial_width = start['robots'][group]['gripper']['width_m']
    target = float(command['width_m'])
    widths = np.arange(initial_width - .001, target, -.001).tolist() + [target]
    result = None
    attempts, replay, samples = [], [], []
    elapsed = 0.
    for width in widths:
        current = runtime.query(robot_groups=(group,))
        row = runtime.execute_gripper_command(**{
            **command, 'width_m': width, 'duration_s': .15,
            'planned_state_id': current['state_id'], 'timeout_s': .45})
        contact = classify_grasp_contacts(
            query_diagnostics(runtime, body_names=('bowl',))['contacts'],
            runtime.robot_metadata['grippers'][group], object_body='bowl')
        attempts.append({'target_width_m': width, 'success': row['success'],
                         'bilateral_pad_contact': contact['bilateral_pad_contact'],
                         'undesired_gripper_contacts': contact['undesired_gripper_contacts']})
        replay.extend(row.get('replay_samples', []))
        samples.extend({**sample, 'time_s': sample['time_s'] + elapsed}
                       for sample in row.get('samples', []))
        elapsed += row.get('sim_duration_s', .15)
        result = row
        if not row['success'] or contact['success'] or contact['undesired_gripper_contacts']:
            break
    return {**result, 'start_state_id': start['state_id'], 'sim_duration_s': elapsed,
            'samples': samples, 'replay_samples': replay, 'close_attempts': attempts,
            'close_policy': 'stop_after_verified_bilateral_pad_contact',
            'close_contact_verified': bool(contact['success']),
            'contact_feedback_source': 'mujoco_actual_pad_contacts',
            'max_width_error_m': max((abs(s['reference_width_m'] - s['actual_width_m'])
                                     for s in samples), default=0.)}


def run_probe(
    runtime, output_dir, *, candidate: dict, motion_stage, diagnostics=None,
    evidence_writer=None, failure_handler=None, open_width_m: float = .08,
    close_width_m: float = 0.0, retreat_m: float = .08, lift_m: float = 0.0,
    settings: GraspSettings | None = None, execute_lift: bool = True,
    visual_recorder=None, after_contact_verified=None,
    postpick_candidate_motion=None, selected_pregrasp_pose=None, stop_after="pick",
    postpick_translation_direction_world=None,
    close_gripper_command=None, close_position_kp_scale: float | None = None,
    hold_position_kp_scale: float | None = None,
) -> dict:
    """旧实验入口只装配任务实例与证据；阶段状态机已迁入 Skill 模块。"""
    output = Path(output_dir)
    diagnostics = diagnostics or (lambda: query_diagnostics(runtime, body_names=("bowl",)))
    evidence_writer = evidence_writer or (
        lambda stage, observation, target_pose: _write_stage_evidence(
            runtime, output, stage, observation, target_pose,
        )
    )
    return run_pick_skill(
        runtime, output, candidate=candidate, motion_stage=motion_stage, stop_after=stop_after,
        object_id="bowl", group="right", diagnostics=diagnostics,
        evidence_writer=evidence_writer, failure_handler=failure_handler,
        open_width_m=open_width_m, close_width_m=close_width_m,
        retreat_m=retreat_m, lift_m=lift_m, settings=settings,
        execute_lift=execute_lift, visual_recorder=visual_recorder,
        after_contact_verified=after_contact_verified,
        close_gripper_command=close_gripper_command,
        close_position_kp_scale=close_position_kp_scale,
        postpick_candidate_motion=postpick_candidate_motion,
        postpick_translation_direction_world=postpick_translation_direction_world,
        selected_pregrasp_pose=selected_pregrasp_pose,
        hold_position_kp_scale=hold_position_kp_scale,
    )


def _write_stage_evidence(
    runtime, output: Path, stage: str, observation: dict, target_pose=None,
) -> None:
    stage_dir = output / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    _write_json(stage_dir / "observation.json", observation)
    frames = query_frame_diagnostics(
        runtime, object_name="bowl", group="right", target_pose=target_pose,
    )
    _write_json(stage_dir / "frames.json", frames)
    diagnostics = query_diagnostics(runtime, body_names=("bowl",))
    _write_json(stage_dir / "diagnostics.json", diagnostics)
    save_state(runtime.model, runtime.copy_data(), stage_dir / "state.npz")
    render_views(
        runtime.model, runtime.data, stage_dir / "images",
        target_body="bowl", width=640, height=480,
        view_names=("overview", "contact", "contact_side"), collision_overlay=True,
        contact_debug=True,
        debug_frames={
            name: value["T_world_frame"] for name, value in frames["frames"].items()
        },
        frame_diagnostics=frames,
        frame_axis_length_m=.08,
    )


def run_experiment(
    output_dir: Path,
    *,
    candidate_id: str = "bowl_ring_00_00",
    depth_m: float | None = None,
    retreat_m: float | None = None,
    seed: int | None = None,
    perturb: bool = False,
    record_video: bool = False,
    linear_approach: bool = True,
    gravity_compensated_arm: bool = False,
    lift_m: float | None = None,
    simulation_weld_after_contact: bool = False,
    scene_config_path: Path | None = None,
    arm_kp_scale: float = 1.0,
    skill_profile: str = "pick_bowl",
    after_pick=None,
    video_stage_camera_overrides: dict[str, dict[str, dict]] | None = None,
    runtime=None,
    stop_after="pick",
) -> dict:
    from gene_agent.knowledge import bowl_grasp_candidates
    from gene_agent.tasks.policies.microwave_task.planning.bowl import apply_microwave_joint_overrides
    from gene_agent.simulation import MujocoRuntime, build_scene
    from gene_agent.simulation.execution import scale_position_actuator_kp

    if stop_after not in {'pick', 'pregrasp', 'approach', 'grasp'}:
        raise ValueError('未知 Pick 停止范围')
    if stop_after != 'pick' and after_pick is not None:
        raise ValueError('局部 Pick 不能触发完整操作的后续回调')
    if not linear_approach:
        raise ValueError("抓碗 Skill 的 approach 阶段必须使用 TCP 直线规划")
    # 手动探针保留既有 12 mm 默认；自动模式必须与 Viser 的 Bowl
    # Knowledge 默认 15 mm 一致。显式 --depth-m 始终优先。
    effective_depth_m = float(depth_m) if depth_m is not None else (
        .015 if candidate_id == "auto" else .012
    )
    profile_path = Path(__file__).resolve().parents[5] / "data/task_1/skills/skill_profiles.json"
    if skill_profile not in {"pick_bowl", "pick_bowl_legacy_visual"}:
        raise ValueError(f"未知抓碗 Skill profile: {skill_profile}")
    settings = load_grasp_settings(profile_path, skill_profile)
    if retreat_m is not None:
        settings = replace(settings, approach_distance_m=float(retreat_m))
    execute_lift = lift_m is None or float(lift_m) > 0.0
    if lift_m is not None and float(lift_m) < 0.0:
        raise ValueError("lift_m 不能为负")
    if lift_m is not None and float(lift_m) > 0.0:
        settings = replace(settings, lift_distance_m=float(lift_m))
    actual_lift_m = settings.lift_distance_m if execute_lift else 0.0

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # 物理探针可以直接消费 Viser 保存的同一场景，避免默认 Bowl 摆位和
    # 关门初态覆盖视觉调试结果。未指定时保持旧入口的 seed=0 行为。
    scene_config = (json.loads(Path(scene_config_path).read_text(encoding="utf-8"))
                    if scene_config_path is not None else {})
    effective_seed = int(seed if seed is not None else scene_config.get("seed", 0))
    resumed_runtime = runtime is not None
    if resumed_runtime:
        # Task 串接只消费开门后的实际积分状态；不能重新设置门角、机器人
        # 或 Bowl 的 qpos，也不能再执行一轮初始落稳。
        if perturb:
            raise ValueError("已有 Runtime 不能在中途重新采样场景扰动")
        model, data, metadata = runtime.model, runtime.data, runtime.metadata
        if (simulation_weld_after_contact and
                metadata.get("simulation_attachment_pairs", {}).get("right") != "bowl"):
            raise ValueError("门场景未预编译右臂到 Bowl 的仿真辅助约束")
    else:
        pose_overrides = {name: np.asarray(pose, dtype=float)
                          for name, pose in scene_config.get("object_poses_world", {}).items()}
        model, data, metadata = build_scene(
            output / "model", seed=effective_seed, perturb=perturb,
            pose_overrides=pose_overrides,
            table_size_xy_m=scene_config.get("table_size_xy_m"),
            table_config=scene_config.get('table'),
            gravity_compensated_groups=("right",) if gravity_compensated_arm else (),
            simulation_attachment_pairs={"right": "bowl"} if simulation_weld_after_contact else None,
        )
        apply_microwave_joint_overrides(
            model, data, scene_config.get("object_joint_positions_rad")
        )
        if arm_kp_scale != 1.0:
            scale_position_actuator_kp(
                model, metadata["robot"]["joint_groups"]["right"], arm_kp_scale,
            )
        mujoco.mj_forward(model, data)
        runtime = MujocoRuntime(model, data, metadata)
        # 仅独立 Skill 入口需要确定性初始落稳；Task 从上一段真实状态接续。
        for _ in range(300):
            mujoco.mj_step(model, data)
    if resumed_runtime and arm_kp_scale != 1.0:
        scale_position_actuator_kp(
            model, metadata["robot"]["joint_groups"]["right"], arm_kp_scale,
        )
    selection_input = runtime.query(object_names=("bowl",), robot_groups=("right",))
    effective_scene = {
        **scene_config, "seed": effective_seed, "arm_kp_scale": arm_kp_scale,
        "skill_profile_name": skill_profile, "grasp_depth_m": effective_depth_m,
        "source_config_ref": (str(Path(scene_config_path).resolve())
                              if scene_config_path is not None else None),
        "reused_runtime_without_reset": resumed_runtime,
        "input_state_id": selection_input["state_id"],
        "gravity_compensated_groups": metadata.get("gravity_compensated_groups", []),
        "simulation_attachment_pairs": metadata.get("simulation_attachment_pairs", {}),
    }
    if resumed_runtime:
        effective_scene["source_door_run_ref"] = metadata.get("source_door_run_ref")
        effective_scene["source_model_ref"] = metadata.get("xml_path")
    if resumed_runtime and hasattr(model, "joint"):
        # 来源配置的 85° 只用于独立放碗探针；Task 续接必须在报告中
        # 写入物理开门后的真实角度，而不是把旧配置当成实际状态。
        door_joint = model.joint("microwave_door_joint")
        effective_scene["object_joint_positions_rad"] = {
            **scene_config.get("object_joint_positions_rad", {}),
            "microwave_door_joint": float(data.qpos[door_joint.qposadr[0]]),
        }
        effective_scene["input_robot_joint_positions_rad"] = (
            selection_input["robots"]["right"]["joint_positions_rad"])
    _write_json(output / "scene_config_effective.json", effective_scene)
    bowl = selection_input["objects"]["bowl"]["T_world_object"]
    candidate_parameters = {"depth_m": effective_depth_m}
    if candidate_id == "auto":
        # 从碗沿外侧斜向接近，让掌部留在碗外；近竖直抓取虽能抬碗，
        # 却会在碗保持直立入炉时把腕部送到炉顶。负倾角使掌部远离碗心。
        candidate_parameters.update({
            # 指垫中点落在碗沿厚度中心。结合 7 mm 闭爪宽度，两侧
            # 接触面分别越过内外壁，避免两根指爪同时伸入碗腔。
            "radial_offset_m": 0.0,
            # TCP +Z 与桌面平面保持约 45..70 度。候选留出 1 度余量，
            # 避免碗落稳后的微小倾斜把端点数值推到范围之外。
            "tip_y_rad": np.deg2rad([-35.0, -40.0, -30.0, -45.0, -25.0, -21.0]).tolist(),
            "max_abs_tip_y_rad": np.deg2rad(50.0),
            # 斜抓时内侧指杆比指垫更靠近碗沿；上移3mm为抬升留出净空。
            "tcp_world_offset_m": [0., 0., -.012],
        })
        # TCP 在指尖，实际接触代理中心在其后方；倾斜必须绕接触区旋转，
        # 否则同一个 TCP 位置会让内侧指垫提前压碗、闭爪时将碗掀斜。
        current_tcp = np.asarray(selection_input['robots']['right']['T_world_tcp'])
        pad_ids = [model.geom(name).id for name in
                   runtime.robot_metadata['grippers']['right']['pad_geom_names']]
        pad_center = runtime.copy_data().geom_xpos[pad_ids].mean(axis=0)
        candidate_parameters['tilt_pivot_tcp_m'] = (
            current_tcp[:3, :3].T @ (pad_center - current_tcp[:3, 3])).tolist()
    candidates = bowl_grasp_candidates(np.asarray(bowl), parameters=candidate_parameters)
    selected_pregrasp_pose = None
    selected_pregrasp_trajectory = None
    selected_approach_trajectory = None
    if candidate_id == "auto":
        if skill_profile != "pick_bowl_legacy_visual":
            raise ValueError("自动候选筛选目前只支持已验证的 pick_bowl_legacy_visual profile")
        from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase
        from gene_agent.tasks.policies.microwave_task.planning.grasp import (
            active_arm_base_reference,
            bowl_grasp_geometry_metrics,
            plan_bowl_grasp_stages,
        )

        selection_case = PickBowlCase(
            output / "candidate_selection_scene", 0, model, data, metadata,
            {}, np.asarray(bowl, dtype=float), candidates,
        )
        selected = plan_bowl_grasp_stages(selection_case, settings, seed=effective_seed)
        selected_plan = selected.plan
        arm_base_body, arm_base_position = active_arm_base_reference(selection_case)
        selected_candidate = next((item for item in selected.proposals
                                   if item.candidate_id == selected_plan.selected_candidate_id), None)
        selected_geometry = None
        if selected_candidate is not None:
            table_normal = np.asarray(data.geom("scene_table").xmat, dtype=float).reshape(3, 3)[:, 2]
            selected_geometry = bowl_grasp_geometry_metrics(
                selected_candidate,
                object_pose_world=np.asarray(bowl, dtype=float),
                base_position_world=arm_base_position,
                table_normal_world=table_normal,
            )
        selection_summary = {
            "input_state_id": selection_input["state_id"],
            "bowl_pose_world": bowl,
            "candidate_order_policy": "right_arm_base_side_external_rim_then_45_to_70_deg_table_angle",
            "right_arm_base_reference_body": arm_base_body,
            "right_arm_base_position_world": arm_base_position.tolist(),
            "grasp_table_angle_definition": "acute angle(TCP +Z, actual scene_table plane)",
            "grasp_table_angle_allowed_deg": list(BOWL_GRASP_TABLE_ANGLE_RANGE_DEG),
            "grasp_table_angle_preferred_deg": BOWL_GRASP_PREFERRED_TABLE_ANGLE_DEG,
            "selected_grasp_geometry": selected_geometry,
            "pregrasp_joint_travel_limit_rad": settings.max_pregrasp_joint_travel_rad,
            "approach_joint_travel_limit_rad": settings.max_approach_joint_travel_rad,
            "candidate_order": [item.candidate_id for item in selected.proposals],
            "selected_candidate_id": selected_plan.selected_candidate_id,
            "stage_collision_snapshots": {
                "pregrasp": selected.strict_snapshot["snapshot_sha256"],
                "approach": selected.contact_snapshot["snapshot_sha256"],
            },
            "plan": selected_plan.to_dict(),
        }
        _write_json(output / "candidate_selection.json", selection_summary)
        if runtime.query(object_names=("bowl",), robot_groups=("right",))["state_id"] != selection_input["state_id"]:
            raise RuntimeError("候选筛选不应改变在线 MuJoCo 状态")
        if (selected_plan.selected_candidate_id is None
                or selected_plan.stage("pregrasp").status != "planned"
                or selected_plan.stage("approach").status != "planned"):
            result = {
                "success": False, "status": "failed",
                "failure_stage": "candidate_selection",
                "candidate_selection_ref": "candidate_selection.json",
            }
            _write_json(output / "result.json", result)
            return result
        candidate_id = selected_plan.selected_candidate_id
        selected_pregrasp_pose = selected_plan.stage("pregrasp").target.matrix
        selected_pregrasp_trajectory = selected_plan.stage("pregrasp").result.trajectory
        selected_approach_trajectory = selected_plan.stage("approach").result.trajectory
    try:
        candidate = next(item for item in candidates if item["candidate_id"] == candidate_id)
    except StopIteration as error:
        raise ValueError(f"未知抓碗候选: {candidate_id}") from error
    _write_json(output / "candidate.json", candidate)
    def on_stage_failure(stage: str, kind: str, details: dict) -> None:
        if kind in {"planner_failed", "collision_postcheck_rejected",
                    "joint_travel_postcheck_rejected"}:
            source_path = f"plans/{stage}.json"
        elif kind == "postcondition_failed":
            source_path = "diagnostics_closed.json"
        else:
            source_path = f"executions/{stage}.json" if stage != "open" else "result.json"
        capture_failure(
            runtime, output / "failures" / stage,
            stage_id=stage, failure_type=kind,
            message=str(details.get("message") or details.get("status") or kind),
            object_names=("bowl",), robot_groups=("right",), target_body="bowl",
            details={
                "status": details.get("status"),
                "failure_category": details.get("failure_category"),
                "source_path": source_path,
            },
        )

    recorder = None
    if record_video:
        from gene_agent.visualization.trajectory import TrajectoryRecorder
        recorder = TrajectoryRecorder(
            output / "visual", target_body="bowl", width=640, height=480,
            fps=10, sample_interval_s=.1, combine_mp4=True,
            detail_view="gripper_side", extra_view="contact",
            # 从微波炉正前上方同时看到桌面碗和腔内碗；已经用初始及落稳状态检查。
            overview_camera={"lookat": [.42, .26, .13], "distance": .85,
                             "azimuth": -10., "elevation": -25.},
            stage_camera_overrides=video_stage_camera_overrides,
            robot_metadata=runtime.robot_metadata, active_group="right",
        )
    attachment_profile = json.loads(profile_path.read_text(encoding="utf-8"))[
                                         "pick_bowl_attachment_proxy"]

    def attachment_for_stage(stage: str, _observation: dict):
        if stage != "lift":
            return None
        active_weld = runtime.simulation_attachments()
        contact = classify_grasp_contacts(
            query_diagnostics(runtime, body_names=("bowl",))["contacts"],
            runtime.robot_metadata["grippers"]["right"], object_body="bowl",
        )
        if not contact["success"] and not active_weld:
            raise ValueError("lift不能建立附着代理：当前没有有效双侧指垫接触或已验证仿真固定")
        object_pose = np.asarray(runtime.query(object_names=("bowl",))["objects"]["bowl"]["T_world_object"])
        local = np.asarray(attachment_profile["sphere_centers_object_m"], dtype=float)
        centers = (object_pose[:3, :3] @ local.T).T + object_pose[:3, 3]
        radius = float(attachment_profile["sphere_radius_m"])
        spheres = np.column_stack((centers, np.full(len(centers), radius))).tolist()
        return {"object_instance_id": "bowl", "spheres_world_m": spheres,
                "physical_hold_evidence": (
                    {"mode": "simulation_privileged_weld",
                     "activation_contact": active_weld[0]["activation_contact"]}
                    if active_weld else
                    {"bilateral_pad_contact": True,
                     "contact_count": contact["contact_count"]}
                ),
                "coverage": attachment_profile["coverage"]}

    def disabled_body_names_for_stage(stage: str, _observation: dict) -> tuple[str, ...]:
        # Viser 中的旧抓取规则：pregrasp 对碗严格避碰，最后直线接近
        # 允许指垫接触碗。这里仅从 cuRobo 世界移除 Bowl；MuJoCo 真实
        # 接触几何仍保留，轨迹须在执行前逐点复核。
        return ("bowl",) if stage == "approach" else ()

    def plan_postcheck_for_stage(stage: str, plan: dict, snapshot: dict) -> dict | None:
        if stage != "approach":
            return None
        positions = tuple(tuple(float(value) for value in row)
                          for row in plan["positions_rad"])
        dt = float(plan["interpolation_dt_s"])
        trajectory = JointTrajectory(
            tuple(plan["joint_names"]),
            tuple(index * dt for index in range(len(positions))),
            positions,
        )
        return validate_planned_target_contacts(
            runtime.model, runtime.data, runtime.robot_metadata,
            group="right", object_body="bowl", trajectory=trajectory,
            gripper_joint_positions_m=snapshot["gripper_snapshot"]["joint_positions_m"],
            allowed_geom_names=runtime.robot_metadata["grippers"]["right"]["pad_geom_names"],
        )

    def execution_stop_condition_for_stage(stage: str, model, data, sample: dict):
        if stage not in {"pregrasp", "approach", "lift"}:
            return None
        allowed = (() if stage == "pregrasp" else
                   runtime.robot_metadata["grippers"]["right"]["pad_geom_names"])
        evidence = classify_target_contacts(
            model, data, runtime.robot_metadata,
            group="right", object_body="bowl", allowed_geom_names=allowed,
        )
        if evidence["success"]:
            return None
        return {
            "failure_type": "forbidden_target_contact_during_execution",
            "stage": stage,
            "sample_time_s": sample.get("time_s"),
            **evidence,
        }

    stages = CuroboRuntimeBridge(
        runtime, output, on_failure=on_stage_failure, visual_recorder=recorder,
        linear_tcp_stages=tuple(
            name for name in (("approach", "lift") if execute_lift else ("approach",))
            if PICK_STAGE_OPERATIONS[name] == "linear_tcp"
        ),
        endpoint_tolerance_rad=.03,
        # Task 在抓碗前已从门控制器切换到配置声明的负载控制器；各阶段
        # 只使用 1.0 倍，避免在负载增益上再次叠乘。独立探针需要更强控制
        # 时应通过 arm_kp_scale 显式声明，而不是在阶段内部隐藏第二套增益。
        execution_position_kp_scale_by_stage={
            **PICK_EXECUTION_POSITION_KP_SCALE_BY_STAGE,
        },
        max_joint_travel_rad_by_stage=(
            {"approach": settings.max_approach_joint_travel_rad}
            if settings.max_approach_joint_travel_rad is not None else None
        ),
        settle_timeout_s=1.5,
        attachment_for_stage=attachment_for_stage if execute_lift else None,
        replay_object_names=("bowl",),
        disabled_body_names_for_stage=disabled_body_names_for_stage,
        plan_postcheck_for_stage=plan_postcheck_for_stage,
        execution_stop_condition_for_stage=execution_stop_condition_for_stage,
        preferred_trajectory_by_stage=(
            {
                "pregrasp": selected_pregrasp_trajectory,
                "approach": selected_approach_trajectory,
            }
            if selected_pregrasp_trajectory is not None
            and selected_approach_trajectory is not None else None
        ),
    )
    def activate_weld(contact: dict):
        return runtime.attach_simulation_object(
            "right", "bowl", contact_evidence=contact,
        )

    probe_options = dict(candidate=candidate, motion_stage=stages, stop_after=stop_after,
                         retreat_m=settings.approach_distance_m, lift_m=actual_lift_m,
                         settings=settings, execute_lift=execute_lift,
                         selected_pregrasp_pose=selected_pregrasp_pose,
                         failure_handler=on_stage_failure,
                         # 闭爪期间保持当前负载控制器，不恢复到门控制器。
                         close_position_kp_scale=PICK_CLOSE_POSITION_KP_SCALE,
                         hold_position_kp_scale=1.0,
                         after_contact_verified=activate_weld if simulation_weld_after_contact else None)
    if selected_pregrasp_pose is not None:
        probe_options['close_gripper_command'] = lambda **command: close_bowl_on_contact(
            runtime, **command)
    if skill_profile == "pick_bowl_legacy_visual":
        probe_options["postpick_candidate_motion"] = stages.plan_and_execute_candidates
        # 抓取接近轴接近水平后，沿原路撤离不再等价于抬升。这里保持抓取
        # 姿态并沿世界 +Z 生成 17 个小旋转候选，保留原碰撞和高度判据。
        probe_options["postpick_translation_direction_world"] = [0.0, 0.0, 1.0]
    if recorder is not None:
        probe_options["visual_recorder"] = recorder
    try:
        result = run_probe(runtime, output, **probe_options)
        if result.get("success") and after_pick is not None:
            # 子任务接续必须消费同一个 Runtime 的实际 held 状态；失败的
            # Pick 不能靠重置或手工放置物体来伪造后续放入成功。
            result["continuation"] = after_pick(runtime, output, recorder, result)
    finally:
        evidence = recorder.finalize() if recorder is not None else None
    if after_pick is not None:
        # Pick 的 success 仅代表抓碗阶段；组合结果必须等待放置物理判定。
        result["compound_success"] = bool(
            result.get("success")
            and (result.get("continuation") or {}).get("physical_place_success")
        )
        result["fully_physical_compound_success"] = bool(
            result.get("physical_pick_success")
            and (result.get("continuation") or {}).get("physical_place_success")
        )
        result["compound_mode"] = (
            "simulation_assisted" if simulation_weld_after_contact else "physical"
        )
    if evidence is not None:
        result["visual_evidence"] = evidence
        result["video_scope"] = (
            "记录实际执行阶段；approach 为"
            + "TCP直线接近"
            + (f"，并含直线lift及{settings.hold_duration_s:g}秒hold"
               if execute_lift else "，未尝试物理抬升")
        )
        _write_json(output / "result.json", result)
    result["simulation_weld_after_contact"] = simulation_weld_after_contact
    result["linear_approach"] = linear_approach
    result["skill_profile"] = str(profile_path)
    result["skill_profile_name"] = skill_profile
    result["gravity_compensated_groups"] = metadata.get("gravity_compensated_groups", [])
    result["scene_config_ref"] = (str(Path(scene_config_path).resolve())
                                  if scene_config_path is not None else None)
    result["candidate_selection_ref"] = (
        "candidate_selection.json" if selected_pregrasp_pose is not None else None
    )
    result["arm_kp_scale"] = arm_kp_scale
    _write_json(output / "result.json", result)
    return result
