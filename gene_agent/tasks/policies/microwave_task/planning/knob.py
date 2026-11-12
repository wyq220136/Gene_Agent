"""标定微波炉 knob_0 的可复现规划与 MuJoCo 运动学预览记录。"""

from __future__ import annotations
from gene_agent.motion.curobo.session import CuroboPlannerPool

import json
import hashlib
from dataclasses import replace
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

from gene_agent.knowledge.geometry import load_concept
from gene_agent.knowledge.grasp import generate_grasps, load_grasp_profiles
from gene_agent.knowledge.operation import AxisSpec
from gene_agent.motion import (
    CuroboPlannerAdapter, JointState, MotionOperation, MotionRequest, PlannerMode, PlanningSceneRef,
    PoseTarget, SynchronousPlanner, load_realman_motion_config,
)
from gene_agent.motion.curobo.scene import (
    export_collision_scene, export_self_collision_only_snapshot,
)
from gene_agent.simulation import build_scene
from gene_agent.skills.library.knob.planning import KnobSkillSettings, RotateKnobSkill

if TYPE_CHECKING:
    from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase


ROOT = Path(__file__).resolve().parents[5]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(position, rotation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def _joint_angle(model, data, name: str) -> float:
    return float(data.qpos[model.joint(name).qposadr[0]])


def _set_gripper_width(model, data, descriptor: dict, width_m: float) -> None:
    lower, upper = descriptor["width_range_m"]
    if not lower <= width_m <= upper:
        raise ValueError(f"夹爪目标开口 {width_m} m 超出 [{lower}, {upper}] m")
    for name in descriptor["joint_names"]:
        data.qpos[model.joint(name).qposadr[0]] = width_m / 2


def _planned_sample(model, data, joint_names: tuple[str, ...], time_s: float,
                    *, group: str, robot: dict, angle_rad: float,
                    joint_name: str) -> dict:
    mujoco.mj_forward(model, data)
    tcp = data.site(robot["tcp_sites"][group])
    knob = data.body("knob_0")
    return {
        "time_s": float(time_s), "qpos": data.qpos.tolist(),
        "robot_group": group,
        "joint_names": list(joint_names),
        "joint_positions_rad": [float(data.qpos[model.joint(name).qposadr[0]])
                                for name in joint_names],
        "gripper_width_m": sum(float(data.qpos[model.joint(name).qposadr[0]])
                               for name in robot["grippers"][group]["joint_names"]),
        "articulation_joint": joint_name, "articulation_angle_rad": float(angle_rad),
        "tcp_pose_world": _pose(tcp.xpos, tcp.xmat).tolist(),
        "object_poses_world": {"knob_0": _pose(knob.xpos, knob.xmat).tolist()},
    }


def _rotation_path_metrics(model, data, names: tuple[str, ...], positions,
                           axis_point: np.ndarray, axis_direction: np.ndarray,
                           tcp_site: str) -> tuple[dict, list[float]]:
    """检查真实 FK 样本的轴向漂移和单调旋转，防止端点规划冒充绕轴段。"""
    points = []
    rotations = []
    for row in positions:
        for name, value in zip(names, row):
            data.qpos[model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(model, data)
        site = data.site(tcp_site)
        points.append(np.asarray(site.xpos).copy())
        rotations.append(np.asarray(site.xmat).reshape(3, 3).copy())
    points = np.asarray(points)
    starts = rotations[0]
    turns = []
    off_axis = []
    for matrix in rotations:
        vector = Rotation.from_matrix(matrix @ starts.T).as_rotvec()
        signed = float(np.dot(vector, axis_direction))
        turns.append(signed)
        off_axis.append(float(np.linalg.norm(vector - signed * axis_direction)))
    relative = points - axis_point
    radial = relative - np.outer(relative @ axis_direction, axis_direction)
    radius = np.linalg.norm(radial, axis=1)
    axial = relative @ axis_direction
    metrics = {
        "max_axis_radius_m": float(radius.max()),
        "max_axis_radius_change_m": float(np.max(np.abs(radius - radius[0]))),
        "max_axial_drift_m": float(np.max(np.abs(axial - axial[0]))),
        "max_rotation_off_axis_rad": float(max(off_axis)),
        "max_angle_regression_rad": float(max(0., -np.min(np.diff(turns)))),
        "tcp_rotation_rad": float(turns[-1]),
    }
    return metrics, turns


def _last_joint_waypoints(start_positions, delta_rad: float, *,
                          speed_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """复现旧 Viser 的末关节直接命令，供纯视觉逐帧回放。"""
    start = np.asarray(start_positions, dtype=float)
    delta = float(delta_rad)
    speed = float(speed_scale)
    if start.ndim != 1 or start.size < 1 or not np.all(np.isfinite(start)):
        raise ValueError("末关节命令需要有限的一维关节起点")
    if not np.isfinite(delta) or not np.isfinite(speed) or speed <= 0:
        raise ValueError("末关节角度和速度比例必须有效")
    duration = max(0.05, abs(delta) / (np.deg2rad(90.0) * speed))
    times = np.linspace(0.0, duration, 21)
    positions = np.repeat(start[None, :], len(times), axis=0)
    positions[:, -1] += np.linspace(0.0, delta, len(times))
    return times, positions


def _select_legacy_grasp(proposals, settings: KnobSkillSettings, *, group: str,
                         start: JointState, grasp_adapter, full_adapter,
                         grasp_scene: PlanningSceneRef, full_scene: PlanningSceneRef,
                         seed: int, screen_pregrasp_path: bool = False):
    """先筛最终 grasp，再按旧顺序检查 pregrasp 终点及完整路径。

    纯 IK 只证明终点存在某个关节解。仿真完整任务显式启用路径筛选，避免
    从当前冗余关节分支前往该终点时让 link_5 扫过头部；真机调用方可继续
    使用自身的分阶段规划流程。
    """
    proposals = tuple(proposals)
    if not proposals:
        raise ValueError("旋钮 grasp 候选不能为空")
    requests = tuple(MotionRequest(
        f"{item.candidate_id}:final_ik", MotionOperation.IK, group, start,
        PoseTarget("world_tcp", item.tcp_pose_world), grasp_scene, seed=seed,
    ) for item in proposals)
    results = tuple(grasp_adapter.evaluate_candidates(requests))
    if len(results) != len(requests):
        raise RuntimeError("最终 grasp batch IK 返回数量不一致")
    attempts = []
    eligible = []
    for item, request, result in zip(proposals, requests, results, strict=True):
        if result.request_id != request.request_id or result.operation is not MotionOperation.IK:
            raise RuntimeError("最终 grasp batch IK 返回顺序不一致")
        attempts.append({"candidate_id": item.candidate_id, "stage": "final_grasp_ik",
                         "request": request.to_dict(), "result": result.to_dict()})
        if result.success:
            eligible.append(item)
    records = tuple((item, index, pose) for item in eligible
                    for index, pose in enumerate(RotateKnobSkill.pregrasp_candidates(
                        item.tcp_pose_world, distance_m=settings.approach_distance_m,
                    )))
    # robot_only 下两种快照可复用同一实例；恢复原先独立 pregrasp
    # 求解器的采样起点。仅重置一次，后续 128 候选分块继续同一序列。
    grasp_planner = getattr(grasp_adapter, "planner", None)
    if records and grasp_planner is not None and grasp_planner is getattr(full_adapter, "planner", None):
        grasp_planner.reset_seed()
    for offset in range(0, len(records), 128):
        batch = records[offset:offset + 128]
        pre_requests = tuple(MotionRequest(
            f"{item.candidate_id}:pregrasp_ik_{index:02d}", MotionOperation.IK,
            group, start, PoseTarget("world_tcp", pose), full_scene, seed=seed,
        ) for item, index, pose in batch)
        pre_results = tuple(full_adapter.evaluate_candidates(pre_requests))
        if len(pre_results) != len(pre_requests):
            raise RuntimeError("pregrasp batch IK 返回数量不一致")
        selected = None
        for (item, _index, pose), request, result in zip(
            batch, pre_requests, pre_results, strict=True,
        ):
            if result.request_id != request.request_id or result.operation is not MotionOperation.IK:
                raise RuntimeError("pregrasp batch IK 返回顺序不一致")
            attempts.append({"candidate_id": item.candidate_id, "stage": "pregrasp_ik",
                             "request": request.to_dict(), "result": result.to_dict()})
            if not result.success or selected is not None:
                continue
            if not screen_pregrasp_path:
                selected = (item, pose)
                continue
            planner = getattr(full_adapter, "planner", None)
            if planner is not None:
                planner.reset_seed()
            path_request = MotionRequest(
                f"{item.candidate_id}:pregrasp_path_{_index:02d}",
                MotionOperation.PLAN, group, start,
                PoseTarget("world_tcp", pose), full_scene, seed=seed,
            )
            path_result = full_adapter.plan(path_request)
            attempts.append({
                "candidate_id": item.candidate_id,
                "stage": "pregrasp_full_path",
                "request": path_request.to_dict(),
                "result": path_result.to_dict(),
            })
            if path_result.success and path_result.trajectory is not None:
                selected = (item, pose)
        if selected is not None:
            return selected[0], selected[1], attempts, len(eligible)
    return None, None, attempts, len(eligible)


def _plan_stage_with_fallback(stage, request: MotionRequest, adapter):
    """旧撤离先尝试原始直线约束及三个比例，失败后才执行普通 pose 兜底。"""
    if not stage.retry_non_terminal_scales:
        result = adapter.plan(request)
        return result, [{"operation": request.operation.value,
                         "request": request.to_dict(), "result": result.to_dict()}]
    invalid = adapter._validate(request, MotionOperation.LINEAR_TCP)
    if invalid is not None:
        return invalid, [{"operation": request.operation.value,
                          "request": request.to_dict(), "result": invalid.to_dict()}]
    attempts = []
    result = None
    for scale in (1.0, *stage.retry_non_terminal_scales):
        raw = adapter.planner.plan_linear_tcp(
            request.start.joint_names, request.start.positions_rad,
            request.target.matrix, target_frame=request.target.frame,
            non_terminal_scale=scale,
        )
        result = adapter._convert(request, raw)
        attempts.append({"operation": "linear_tcp", "non_terminal_scale": scale,
                         "request": request.to_dict(), "result": result.to_dict()})
        if result.success:
            return result, attempts
    if stage.non_linear_fallback:
        fallback_request = replace(
            request, request_id=f"{request.request_id}:pose_fallback",
            operation=MotionOperation.PLAN,
        )
        result = adapter.plan(fallback_request)
        attempts.append({"operation": "plan", "request": fallback_request.to_dict(),
                         "result": result.to_dict()})
    return result, attempts


def run_rotate_knob_0_planning(output_dir: str | Path, *,
                               task_path: str | Path = ROOT / "data/task_1/skills/knob_0_task.json",
                               case: PickBowlCase | None = None) -> dict:
    """同一旋钮 Skill 的细分段共享求解器；结束及异常都释放 GPU 资源。"""
    with CuroboPlannerPool(planner_factory=SynchronousPlanner, max_entries=4) as pool:
        return _run_rotate_knob_0_planning(output_dir,task_path=task_path,case=case,pool=pool)


def _run_rotate_knob_0_planning(output_dir, *, task_path, case, pool) -> dict:
    """逐阶段冻结碰撞、规划并保存可播放的预期运动；不声称真实接触执行。

    每个旋转细分段沿用上段机器人终点；旋钮视觉关节只在全部旋转
    完成后更新。全量 qpos 是运动学预览，不是 MuJoCo 动力学积分采样。
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_source = Path(task_path).resolve()
    task = json.loads(task_source.read_text(encoding="utf-8"))
    if task.get("schema_version") != 1 or task.get("instance_id") != "knob_0":
        raise ValueError("旋钮任务配置格式或实例 ID 不正确")
    group = task["robot_group"]
    joint_name = task["joint_name"]
    site_name = task["site_name"]
    seed = int(task["seed"])
    if case is None:
        microwave_pose = np.asarray(task["microwave_pose_world"], dtype=float)
        model, data, metadata = build_scene(
            output / "source_scene", seed=seed,
            pose_overrides={"microwave": microwave_pose},
        )
        scene_source = "standalone_scene"
        model_ref = "source_scene/scene.xml"
    else:
        # Task1 已有唯一的场景和已规划终态；这里直接复用 model/data，
        # 不能从配置重新 build_scene，否则机器人、Bowl 和门会瞬移。
        model, data, metadata = case.model, case.data, case.metadata
        mujoco.mj_forward(model, data)
        microwave_body = data.body("microwave")
        microwave_pose = _pose(microwave_body.xpos, microwave_body.xmat)
        scene_source = "provided_case"
        model_ref = str(Path(metadata["xml_path"]).resolve())
    input_qpos = data.qpos.copy()
    robot = metadata["robot"]
    names = tuple(robot["joint_groups"][group])
    descriptor = robot["grippers"][group]
    _set_gripper_width(model, data, descriptor, float(task["open_width_m"]))
    mujoco.mj_forward(model, data)
    concept_path = ROOT / task["concept_path"]
    profile_path = ROOT / task["grasp_profile_path"]
    concept = load_concept(concept_path, part_name="knob_0")
    profile = load_grasp_profiles(profile_path)[task["grasp_profile_name"]]
    proposals = generate_grasps(concept.concept, microwave_pose, profile).proposals
    if not proposals:
        raise ValueError("标定旋钮 profile 未生成 grasp 候选")
    site = data.site(site_name)
    site_pose = _pose(site.xpos, site.xmat)
    concept_pose = microwave_pose @ concept.concept.pose_in_object
    center_error = float(np.linalg.norm(site_pose[:3, 3] - concept_pose[:3, 3]))
    axis_error = float(np.arccos(np.clip(
        np.dot(site_pose[:3, 2], concept_pose[:3, 2]), -1., 1.
    )))
    if center_error > .001 or axis_error > .01:
        raise ValueError(f"标定 Concept 与 MuJoCo knob site 失配：{center_error:.4f} m / {axis_error:.3f} rad")
    joint = model.joint(joint_name)
    axis_world = data.body("knob_0").xmat.reshape(3, 3) @ model.jnt_axis[joint.id]
    if float(np.dot(axis_world, site_pose[:3, 2])) < .999:
        raise ValueError("knob site +Z 与实际关节正方向不一致")
    start_angle = _joint_angle(model, data, joint_name)
    target_angle = float(task["target_angle_rad"])
    lo, hi = map(float, model.jnt_range[joint.id])
    if not lo <= target_angle <= hi:
        raise ValueError("旋钮目标角超出编译后关节限位")
    axis = AxisSpec(site_pose[:3, 3], axis_world, "world", "compiled_mujoco_knob_joint")
    settings = KnobSkillSettings(
        float(task["approach_distance_m"]), target_angle,
        float(task["rotation_step_rad"]), float(task["open_width_m"]),
        float(proposals[0].closing_width_m), float(task["release_width_m"]),
        tip_rotations_x_rad=tuple(dict.fromkeys(
            float(item.parameters.get("tip_x_rad", 0.0)) for item in proposals
        )),
    )
    # 两个 IK 场景分别对应旧版最终 grasp 的 world 忽略和 pregrasp 的完整碰撞。
    selection_dir = output / "collision" / "grasp_selection"
    full_snapshot = export_collision_scene(
        model, data, robot, group, selection_dir / "full_scene",
        disabled_body_names=("knob_0",),
    )
    grasp_snapshot = export_self_collision_only_snapshot(
        full_snapshot, selection_dir / "self_only",
    )
    selection_start = JointState(
        names, tuple(_joint_angle(model, data, name) for name in names),
    )
    selection_cache = []
    def selection_adapter(snapshot, revision):
        motion_config = load_realman_motion_config(
            group, robot, gripper_snapshot=snapshot.get("gripper_snapshot"),
        )
        lease = pool.acquire(motion_config, snapshot,
            planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed)
        # 旧每段冷构造从同一采样起点开始；复用图时显式保持此语义。
        lease.planner.reset_seed()
        selection_cache.append(lease.to_dict())
        return CuroboPlannerAdapter(lease.planner, group=group, scene_revision=revision)

    grasp_adapter = selection_adapter(grasp_snapshot, 0)
    full_adapter = selection_adapter(full_snapshot, 1)
    candidate, selected_pregrasp, selection_attempts, grasp_reachable = _select_legacy_grasp(
        proposals, settings, group=group, start=selection_start,
        grasp_adapter=grasp_adapter, full_adapter=full_adapter,
        grasp_scene=PlanningSceneRef(0, grasp_snapshot["snapshot_sha256"], grasp_snapshot["scope"]),
        full_scene=PlanningSceneRef(1, full_snapshot["snapshot_sha256"], full_snapshot["scope"]),
        seed=seed, screen_pregrasp_path=True,
    )
    targets = (
        RotateKnobSkill().targets(
            candidate.tcp_pose_world, axis, start_angle_rad=start_angle,
            settings=settings, selected_pregrasp_pose_world=selected_pregrasp,
        ) if candidate is not None else ()
    )
    report = {
        "schema_version": 1, "skill": "rotate_knob_0", "instance_id": "knob_0",
        "selection_planner_cache": selection_cache,
        "source_format": "kinematic_plan", "success": False,
        "claim": "TCP 运动由 cuRobo 规划，零 tip X 的末关节旋转为直接关节命令；旋钮仅作运动学视觉更新，未执行接触物理",
        "scene_revision": 0, "seed": seed, "robot_group": group,
        "model_ref": model_ref, "scene_source": scene_source,
        "input_qpos": input_qpos.tolist(),
        "effective_microwave_pose_world": microwave_pose.tolist(),
        "input_sha256": {
            "task": _sha256(task_source), "concept": _sha256(concept_path),
            "grasp_profile": _sha256(profile_path),
        },
        "selected_candidate_id": candidate.candidate_id if candidate is not None else None,
        "candidate_count": len(proposals), "task_config_ref": "task_config.json",
        "grasp_reachable_count": grasp_reachable,
        "pregrasp_path_screening_enabled": True,
        "grasp_selection_attempts": selection_attempts,
        "grasp_selection_collision_refs": {
            "final_grasp_ik": "collision/grasp_selection/self_only/collision_snapshot.json",
            "pregrasp_ik": "collision/grasp_selection/full_scene/collision_snapshot.json",
        },
        "effective_tip_rotations_x_rad": list(settings.tip_rotations_x_rad),
        "rotation_mode": "last_joint" if settings.use_last_joint_rotation else "linear_tcp",
        "concept_site_error_m": center_error,
        "concept_site_axis_error_rad": axis_error,
        "axis_point_world": axis.point.tolist(),
        "axis_direction_world": axis.direction.tolist(),
        "joint_name": joint_name, "joint_start_angle_rad": start_angle,
        "joint_target_angle_rad": target_angle,
        "stage_contract": [{"name": stage.name, "action": stage.action,
                            "operation": stage.operation.value if stage.operation else None}
                           for stage in targets],
        "stages": [],
    }
    shutil.copyfile(task_source, output / "task_config.json")
    _write(output / "grasp_candidates.json", {
        "source_profile": str(profile_path),
        "resolved_profile": profile.to_dict(),
        "selected_candidate_id": candidate.candidate_id if candidate is not None else None,
        "candidates": [{"candidate_id": item.candidate_id,
                        "tcp_pose_world": item.tcp_pose_world.tolist(),
                        "closing_width_m": item.closing_width_m}
                       for item in proposals],
    })
    if candidate is None:
        report["failure_stage"] = "grasp_selection"
        report["failure_reason"] = (
            "全部最终 grasp IK 不可达" if grasp_reachable == 0
            else ("pregrasp 终点存在 IK，但从实际起点没有完整无碰撞路径"
                  if any(row.get("stage") == "pregrasp_full_path"
                         for row in selection_attempts)
                  else "最终 grasp 可达，但五个 pregrasp IK 均不可达")
        )
        # 即使 IK 阶段就停止，也提供可播放的起始帧与碰撞快照，
        # 使失败会话可以在同一 Viser 时间轴中检查。
        report["stage_contract"] = [{"name": "grasp_selection", "action": "batch_ik",
                                     "operation": "ik"}]
        report["stages"].append({
            "name": "grasp_selection", "status": "failed", "action": "batch_ik",
            "operation": "ik", "collision_scope": "self_only",
            "collision_snapshot_ref": "collision/grasp_selection/self_only/collision_snapshot.json",
            "start_angle_rad": start_angle, "end_angle_rad": start_angle,
            "gripper_width_m": settings.open_width_m,
            "request": None, "result": None,
            "planned_replay_samples": [_planned_sample(
                model, data, names, 0.0, group=group, robot=robot,
                angle_rad=start_angle, joint_name=joint_name,
            )],
        })
        report["final_joint_angle_rad"] = start_angle
        report["completion_basis"] = "planned_kinematic_target_only"
        _write(output / "skill_rotate_knob_0_result.json", report)
        return report
    tcp_site = robot["tcp_sites"][group]
    current_width = settings.open_width_m
    failed = False
    for index, stage in enumerate(targets):
        stage_dir = output / "collision" / stage.name
        # 即使该阶段不规划，也冻结可查看的场景。旋钮始终是视觉目标，
        # 不作为被附着的 world 障碍；旧接近/旋转只检查机器人自身碰撞。
        source_dir = stage_dir / "full_scene" if stage.collision_scope == "self_only" else stage_dir
        full_snapshot = export_collision_scene(
            model, data, robot, group, source_dir,
            disabled_body_names=("knob_0",),
        )
        snapshot = (
            export_self_collision_only_snapshot(full_snapshot, stage_dir)
            if stage.collision_scope == "self_only" else full_snapshot
        )
        scene_ref = PlanningSceneRef(index, snapshot["snapshot_sha256"], snapshot["scope"])
        row = {
            "name": stage.name,
            "status": "not_executed" if failed else "planned",
            "action": stage.action,
            "operation": stage.operation.value if stage.operation else None,
            "collision_scope": stage.collision_scope,
            "gripper_is_grasping": stage.gripper_is_grasping,
            "target_collision_enabled": stage.target_collision_enabled,
            "attach_object": stage.attach_object,
            "collision_snapshot_ref": f"collision/{stage.name}/collision_snapshot.json",
            "start_angle_rad": stage.start_angle_rad,
            "end_angle_rad": stage.end_angle_rad,
            "gripper_width_m": stage.gripper_width_m,
            "target": (PoseTarget("world_tcp", stage.tcp_pose_world).to_dict()
                       if stage.tcp_pose_world is not None else None),
            "request": None, "result": None,
            "planned_replay_samples": [],
        }
        if failed:
            report["stages"].append(row)
            continue
        start_qpos = data.qpos.copy()
        if stage.action in {"gripper_open", "gripper_grasp", "gripper_release", "wait"}:
            # 旧 Viser 的 grasp/release 是立即设定开口及 is_grasping；
            # settle 只等待，不让旋钮提前转动。
            if stage.action != "wait":
                _set_gripper_width(model, data, descriptor, stage.gripper_width_m)
                current_width = stage.gripper_width_m
            row["settle_time_s"] = stage.settle_time_s
            row["planned_replay_samples"].append(_planned_sample(
                model, data, names, stage.settle_time_s, group=group, robot=robot,
                angle_rad=_joint_angle(model, data, joint_name), joint_name=joint_name,
            ))
        elif stage.action == "last_joint_rotation":
            start_positions = tuple(_joint_angle(model, data, name) for name in names)
            times, positions = _last_joint_waypoints(
                start_positions, float(stage.joint_delta_rad),
            )
            # 与旧同步 Viser 一样，这个分支直接给末关节命令，不调用 cuRobo。
            for time_s, positions_row in zip(times, positions, strict=True):
                for name, value in zip(names, positions_row, strict=True):
                    data.qpos[model.joint(name).qposadr[0]] = float(value)
                row["planned_replay_samples"].append(_planned_sample(
                    model, data, names, time_s, group=group, robot=robot,
                    angle_rad=_joint_angle(model, data, joint_name), joint_name=joint_name,
                ))
            # 旧运行时给回放保留连续角度，但后续 cuRobo 撤离使用等效的
            # [-pi, pi] 起点，以免周期关节的累计表示越界。
            last_joint_adr = model.joint(names[-1]).qposadr[0]
            data.qpos[last_joint_adr] = np.arctan2(
                np.sin(data.qpos[last_joint_adr]), np.cos(data.qpos[last_joint_adr]),
            )
            mujoco.mj_forward(model, data)
            row["direct_joint_command"] = {
                "joint_name": names[-1], "delta_rad": stage.joint_delta_rad,
                "planner_used": False,
            }
        else:
            settings_dict = load_realman_motion_config(
                group, robot, gripper_snapshot=snapshot.get("gripper_snapshot"),
            )
            lease = pool.acquire(settings_dict, snapshot,
                planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed)
            planner = lease.planner
            planner.reset_seed()
            row["planner_cache"] = lease.to_dict()
            adapter = CuroboPlannerAdapter(planner, group=group, scene_revision=index)
            start = JointState(names, tuple(_joint_angle(model, data, name) for name in names))
            request = MotionRequest(
                f"{candidate.candidate_id}:{stage.name}", stage.operation,
                group, start, PoseTarget("world_tcp", stage.tcp_pose_world),
                scene_ref, seed=seed,
            )
            result, attempts = _plan_stage_with_fallback(stage, request, adapter)
            # 若普通 pose 兜底被调用，阶段结果必须与最终一次请求的
            # operation 一致，否则只读 Skill 会话会拒绝这份报告。
            row["request"], row["result"] = (
                attempts[-1]["request"], attempts[-1]["result"],
            )
            row["operation"] = attempts[-1]["operation"]
            row["planning_attempts"] = attempts
            if not result.success or result.trajectory is None:
                row["status"] = "failed"
                failed = True
                data.qpos[:] = start_qpos
                mujoco.mj_forward(model, data)
            else:
                trajectory = result.trajectory
                positions = trajectory.positions_rad
                turns = None
                if stage.name.startswith("rotate_"):
                    metrics, turns = _rotation_path_metrics(
                        model, data, trajectory.joint_names, positions,
                        axis.point, axis.direction, tcp_site,
                    )
                    row["axis_path_validation"] = metrics
                    if (metrics["max_axis_radius_change_m"] > .005
                            or metrics["max_axial_drift_m"] > .005
                            or metrics["max_rotation_off_axis_rad"] > .05
                            or metrics["max_angle_regression_rad"] > .03
                            or abs(metrics["tcp_rotation_rad"] -
                                   (stage.end_angle_rad - stage.start_angle_rad)) > .05):
                        row["status"] = "failed"
                        row["axis_path_failure"] = "规划轨迹偏离旋钮轴或旋转方向"
                        failed = True
                        data.qpos[:] = start_qpos
                        mujoco.mj_forward(model, data)
                if not failed:
                    for sample_index, (time_s, positions_row) in enumerate(zip(
                            trajectory.times_s, positions)):
                        for name, value in zip(trajectory.joint_names, positions_row):
                            data.qpos[model.joint(name).qposadr[0]] = float(value)
                        row["planned_replay_samples"].append(_planned_sample(
                            model, data, names, time_s, group=group, robot=robot,
                            angle_rad=_joint_angle(model, data, joint_name),
                            joint_name=joint_name,
                        ))
                    mujoco.mj_forward(model, data)
        if not failed and stage.visual_angle_commit:
            # 旧版只在全部旋转命令执行完后更新旋钮物体。额外的一帧把机器人
            # 终点与旋钮新角度一起写入，先前所有运动帧的旋钮角保持旧值。
            data.qpos[joint.qposadr[0]] = stage.end_angle_rad
            commit_time = (
                row["planned_replay_samples"][-1]["time_s"] + 1e-6
                if row["planned_replay_samples"] else 0.0
            )
            row["planned_replay_samples"].append(_planned_sample(
                model, data, names, commit_time, group=group, robot=robot,
                angle_rad=stage.end_angle_rad, joint_name=joint_name,
            ))
            row["visual_angle_commit"] = True
        if not failed and stage.name.startswith("rotate_"):
            inner_sites = (
                descriptor.get("positive_inner_site_name") or
                descriptor.get("fingers", {}).get("positive", {}).get("inner_site_name"),
                descriptor.get("negative_inner_site_name") or
                descriptor.get("fingers", {}).get("negative", {}).get("inner_site_name"),
            )
            if all(name is not None for name in inner_sites):
                knob_center = np.asarray(data.site(site_name).xpos)
                row["kinematic_gripper_knob_geometry"] = {
                    "inner_site_to_knob_center_m": [float(np.linalg.norm(
                        data.site(name).xpos - knob_center)) for name in inner_sites],
                    "knob_radius_xy_m": concept.concept.radius_xy_m.tolist(),
                    "evidence_kind": "planned_qpos_static_geometry_not_physical_contact",
                }
        report["stages"].append(row)
        _write(output / "skill_rotate_knob_0_result.json", report)
    report["success"] = not failed and abs(_joint_angle(model, data, joint_name) - target_angle) <= 1e-8
    report["final_joint_angle_rad"] = _joint_angle(model, data, joint_name)
    report["completion_basis"] = "planned_kinematic_target_only"
    rotations = [row for row in report["stages"] if row["name"].startswith("rotate_")]
    if rotations:
        report["last_rotate_gripper_knob_geometry"] = rotations[-1].get(
            "kinematic_gripper_knob_geometry")
    _write(output / "skill_rotate_knob_0_result.json", report)
    return report


__all__ = ["run_rotate_knob_0_planning"]
