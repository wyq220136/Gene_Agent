"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.tasks.policies.microwave_task.planning.knob import _plan_stage_with_fallback, _rotation_path_metrics, _select_legacy_grasp
from gene_agent.knowledge.geometry import load_concept, validate_pose
from gene_agent.knowledge.geometry.transforms import pose_delta as _pose_delta
from gene_agent.knowledge.grasp import generate_grasps, load_grasp_profiles
from gene_agent.knowledge.operation import AxisSpec
from gene_agent.motion.curobo.scene import (
    export_collision_scene,
    export_self_collision_only_snapshot,
)
from gene_agent.motion.model import (
    CollisionPolicy,
    MotionRequest,
    PoseTarget,
    motion_request_from_dict,
)
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.skills.library.knob.planning import KnobSkillSettings, RotateKnobSkill

_ROOT = Path(__file__).resolve().parents[3]
# 真机默认沿用已在 MuJoCo 连续任务验收的 210°/10° grasp 参数；
# 原六候选 profile 与 Viser/MuJoCo 旧入口保持原文件不变，仍可显式传入。
_DEFAULT_TASK = _ROOT / "data/task_1/realworld/knob_0_task_realworld_tuned.json"
_MICROWAVE = "Microwave/Microwave_0"


def build_observed_knob_targets(
    collision_scene: dict, *, task_path: str | Path = _DEFAULT_TASK,
) -> dict:
    """复用原任务配置、Concept 与 grasp profile，不选候选或假定 IK 成功。

    现场微波炉 pose、旋钮角和机器人 qpos 来自同一观测快照。MuJoCo
    在私有数据中只求 FK 和关节限位，不推进物理；场景里的已验证资产
    是坐标系核对依据，不是视觉或真实接触成功证据。
    """
    snapshot = collision_scene["snapshot"]
    source = collision_scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("旋钮目标需要同帧现场观测及一致的碰撞快照")
    if snapshot.get("active_group") != "right":
        raise NotImplementedError("原 task_1 knob_0 当前只验收右臂")
    aligned = source.get("aligned_object_poses_world") or {}
    if _MICROWAVE not in aligned:
        raise ValueError("旋钮目标缺少已对齐资产的微波炉 pose")
    microwave_pose = validate_pose(aligned[_MICROWAVE], field_name="T_world_microwave")

    config_path = Path(task_path).resolve()
    task_bytes = config_path.read_bytes()
    task = json.loads(task_bytes)
    if (task.get("schema_version") != 1 or task.get("instance_id") != "knob_0"
            or task.get("robot_group") != "right"
            or task.get("joint_name") != "knob_0_joint"
            or task.get("site_name") != "knob_0_site"):
        raise ValueError("当前只支持旧 task_1 的 right/knob_0 配置")
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    qpos = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("旋钮现场 qpos 与资产模型不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    joint = model.joint(task["joint_name"])
    start_angle = float(qpos[joint.qposadr[0]])
    observed_angle = float(source["knob_angle_rad"])
    if not np.isfinite(observed_angle) or abs(start_angle - observed_angle) > 1e-6:
        raise ValueError("现场旋钮角与同帧关节 qpos 不一致")
    target_angle = float(task["target_angle_rad"])
    lower, upper = map(float, model.jnt_range[joint.id])
    if not lower <= start_angle <= upper or not lower <= target_angle <= upper:
        raise ValueError("旋钮实测角或目标角超出编译后关节限位")

    concept_path = (_ROOT / task["concept_path"]).resolve()
    profile_path = (_ROOT / task["grasp_profile_path"]).resolve()
    concept = load_concept(concept_path, part_name="knob_0")
    profile = load_grasp_profiles(profile_path)[task["grasp_profile_name"]]
    proposals = generate_grasps(concept.concept, microwave_pose, profile).proposals
    if not proposals:
        raise ValueError("当前旋钮 profile 未生成 grasp 候选")
    site = data.site(task["site_name"])
    site_rotation = site.xmat.reshape(3, 3)
    concept_pose = microwave_pose @ concept.concept.pose_in_object
    center_error = float(np.linalg.norm(site.xpos - concept_pose[:3, 3]))
    axis_error = float(np.arccos(np.clip(
        np.dot(site_rotation[:, 2], concept_pose[:3, 2]), -1., 1.,
    )))
    if center_error > .001 or axis_error > .01:
        raise ValueError("现场微波炉 Concept 与 knob site 坐标系不一致")
    axis_world = data.body("knob_0").xmat.reshape(3, 3) @ model.jnt_axis[joint.id]
    if float(np.dot(axis_world, site_rotation[:, 2])) < .999:
        raise ValueError("knob site +Z 与关节正方向不一致")
    axis = AxisSpec(site.xpos.copy(), axis_world, "world", "compiled_mujoco_knob_joint")
    settings = KnobSkillSettings(
        float(task["approach_distance_m"]), target_angle,
        float(task["rotation_step_rad"]), float(task["open_width_m"]),
        float(proposals[0].closing_width_m), float(task["release_width_m"]),
        tip_rotations_x_rad=tuple(dict.fromkeys(
            float(item.parameters.get("tip_x_rad", 0.)) for item in proposals
        )),
    )
    return {
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "microwave_pose_world": microwave_pose,
        "start_angle_rad": start_angle,
        "target_angle_rad": target_angle,
        "axis": axis,
        "settings": settings,
        "proposals": proposals,
        "concept_site_error_m": center_error,
        "concept_site_axis_error_rad": axis_error,
        "task_config_path": str(config_path),
        "task_config_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "concept_sha256": hashlib.sha256(concept_path.read_bytes()).hexdigest(),
        "grasp_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "claim": "仅生成同帧现场旋钮 Knowledge/Skill 输入；未筛 IK、规划或执行",
    }


def _derived_scene(scene: dict, snapshot: dict) -> dict:
    """改变规划碰撞层时保留同一现场观测来源，不修改实测机器人状态。"""
    return {
        **scene,
        "snapshot": snapshot,
        "provenance": {
            **scene["provenance"],
            "collision_snapshot_sha256": snapshot["snapshot_sha256"],
            "derived_from_collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        },
    }


def select_observed_knob_grasp(
    collision_scene: dict, *, planner_pool, output_dir: str | Path,
    task_path: str | Path | None = None, seed: int = 123,
) -> dict:
    """先用仅自碰撞筛最终 grasp，再用完整场景筛五个 pregrasp。

    调用前必须实际开爪并取得新观测。原视觉规划器在预筛前把夹爪
    设为打开；真机不能把这个预测动作当作实测，因此这里显式拒绝
    尚未完成开爪的状态。当前只选择候选，不规划运动或发送 SDK 命令。
    """
    kwargs = {} if task_path is None else {"task_path": task_path}
    source = build_observed_knob_targets(collision_scene, **kwargs)
    snapshot = collision_scene["snapshot"]
    group = snapshot["active_group"]
    reported_width = collision_scene["provenance"].get("gripper_widths_m", {}).get(group)
    if reported_width is None:
        raise ValueError("旋钮候选筛选缺少实测夹爪开口")
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    mujoco.mj_forward(model, data)
    descriptor = collision_scene["robot_metadata"]["grippers"][group]
    actual_width = sum(float(data.qpos[model.joint(name).qposadr[0]])
                       for name in descriptor["joint_names"])
    open_width = source["settings"].open_width_m
    if (abs(float(reported_width) - actual_width) > 1e-5
            or abs(actual_width - open_width) > 1e-3):
        raise ValueError("旋钮候选筛选前必须实际开爪并重新观测夹爪开口")
    output = Path(output_dir).resolve()
    full = export_collision_scene(
        model, data, collision_scene["robot_metadata"], group,
        output / "collision/full_scene", disabled_body_names=("knob_0",),
    )
    self_only = export_self_collision_only_snapshot(
        full, output / "collision/self_only",
    )
    self_snapshot = {**full, **self_only}
    full_context = prepare_plan_only_context(
        _derived_scene(collision_scene, full), group=group,
        planner_pool=planner_pool, scene_revision=0, seed=seed,
    )
    grasp_context = prepare_plan_only_context(
        _derived_scene(collision_scene, self_snapshot), group=group,
        planner_pool=planner_pool, scene_revision=1, seed=seed,
    )
    if full_context["start"] != grasp_context["start"]:
        raise ValueError("旋钮两层碰撞场景的实测机器人起点不一致")
    candidate, pregrasp, attempts, reachable = _select_legacy_grasp(
        source["proposals"], source["settings"], group=group,
        start=full_context["start"],
        grasp_adapter=grasp_context["planner"],
        full_adapter=full_context["planner"],
        grasp_scene=grasp_context["scene"],
        full_scene=full_context["scene"], seed=seed,
    )
    targets = (
        RotateKnobSkill().targets(
            candidate.tcp_pose_world, source["axis"],
            start_angle_rad=source["start_angle_rad"],
            settings=source["settings"],
            selected_pregrasp_pose_world=pregrasp,
        ) if candidate is not None else ()
    )
    record = {
        "schema_version": 1, "kind": "observed_knob_grasp_selection_plan_only",
        "success": candidate is not None,
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "full_collision_snapshot_sha256": full["snapshot_sha256"],
        "self_collision_snapshot_sha256": self_snapshot["snapshot_sha256"],
        "task_config_sha256": source["task_config_sha256"],
        "grasp_profile_sha256": source["grasp_profile_sha256"],
        "candidate_count": len(source["proposals"]),
        "grasp_reachable_count": reachable,
        "selected_candidate_id": None if candidate is None else candidate.candidate_id,
        "selected_pregrasp_pose_world": None if pregrasp is None else pregrasp.tolist(),
        "selection_attempts": attempts,
        "stage_contract": [
            {"name": stage.name, "action": stage.action,
             "operation": None if stage.operation is None else stage.operation.value,
             "collision_scope": stage.collision_scope,
             "tcp_pose_world": (None if stage.tcp_pose_world is None
                                else stage.tcp_pose_world.tolist())}
            for stage in targets
        ],
        "planner_cache_hit": {
            "full": full_context["planner_cache_hit"],
            "self_only": grasp_context["planner_cache_hit"],
        },
        "failure_reason": (
            None if candidate is not None else
            "最终 grasp IK 均不可达" if reachable == 0 else
            "最终 grasp 可达，但五个 pregrasp IK 均不可达"
        ),
        "claim": "只复用旧策略筛候选并生成 Skill 目标；未规划轨迹或转动旋钮",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**record, "source": source, "targets": targets}


def plan_observed_knob_motion_stage(
    collision_scene: dict, selection: dict, *, stage_name: str,
    planner_pool, output_dir: str | Path,
    max_microwave_translation_m: float, max_microwave_rotation_rad: float,
    max_knob_angle_error_rad: float,
    seed: int = 123, scene_revision: int = 0,
) -> dict:
    """从当前实测关节重新规划选定阶段，不沿预测轨迹串下一段。

    旋钮从规划 world 排除；接近和绕轴段沿用旧仅自碰撞层。
    撤离调用旧直线非终端约束重试及普通 pose 兜底。规划成功仍
    不说明旋钮受力转动，执行层必须用现场角度验证。
    """
    limits = (max_microwave_translation_m, max_microwave_rotation_rad,
              max_knob_angle_error_rad)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits):
        raise ValueError("旋钮阶段容差必须为正有限值")
    if not selection.get("selected_candidate_id"):
        raise ValueError("旋钮阶段缺少已筛选 grasp 候选")
    matches = [stage for stage in selection["targets"] if stage.name == stage_name]
    if len(matches) != 1:
        raise ValueError(f"旋钮阶段名称不存在或重复：{stage_name}")
    stage_target = matches[0]
    if stage_target.operation is None or stage_target.tcp_pose_world is None:
        raise ValueError("夹爪、等待或末关节命令不是 TCP 轨迹规划阶段")
    current = build_observed_knob_targets(
        collision_scene, task_path=selection["source"]["task_config_path"],
    )
    original = selection["source"]
    if (current["task_config_sha256"] != original["task_config_sha256"]
            or current["grasp_profile_sha256"] != original["grasp_profile_sha256"]):
        raise ValueError("旋钮任务或 grasp profile 已变化，必须重新筛选候选")
    microwave_delta = _pose_delta(
        original["microwave_pose_world"], current["microwave_pose_world"],
        field="微波炉现场 pose",
    )
    if (microwave_delta[0] > max_microwave_translation_m
            or microwave_delta[1] > max_microwave_rotation_rad):
        raise ValueError("旋钮阶段微波炉 pose 超出选候选时的锚点容差")
    angle_error = abs(current["start_angle_rad"] - stage_target.start_angle_rad)
    if angle_error > max_knob_angle_error_rad:
        raise ValueError("旋钮角与当前阶段预期起点不一致，不能沿用旧目标")
    snapshot = collision_scene["snapshot"]
    group = snapshot["active_group"]
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    mujoco.mj_forward(model, data)
    output = Path(output_dir).resolve()
    started = time.perf_counter()
    full = export_collision_scene(
        model, data, collision_scene["robot_metadata"], group,
        output / "collision/full_scene", disabled_body_names=("knob_0",),
    )
    if stage_target.collision_scope == "self_only":
        self_only = export_self_collision_only_snapshot(
            full, output / "collision/self_only",
        )
        planning_snapshot = {**full, **self_only}
    elif stage_target.collision_scope == "full_scene":
        planning_snapshot = full
    else:
        raise ValueError(f"旋钮 TCP 阶段没有可规划碰撞层：{stage_target.collision_scope}")
    context = prepare_plan_only_context(
        _derived_scene(collision_scene, planning_snapshot), group=group,
        planner_pool=planner_pool, scene_revision=scene_revision, seed=seed,
    )
    target = PoseTarget("world_tcp", stage_target.tcp_pose_world)
    request = MotionRequest(
        f"knob:{selection['selected_candidate_id']}:{stage_name}:{current['source_frame_number']}",
        stage_target.operation, group, context["start"], target,
        context["scene"], CollisionPolicy(), seed=seed,
    )
    result, attempts = _plan_stage_with_fallback(
        stage_target, request, context["planner"],
    )
    effective_request = (
        motion_request_from_dict(attempts[-1]["request"]) if attempts else request
    )
    if result.request_id != effective_request.request_id or result.operation != effective_request.operation:
        raise RuntimeError("旋钮阶段规划结果与实际最后一次请求不一致")
    success = bool(result.success and result.trajectory is not None
                   and result.goal_state is not None)
    metrics = None
    if success and stage_name.startswith("rotate_"):
        metrics, _ = _rotation_path_metrics(
            model, data, result.trajectory.joint_names,
            result.trajectory.positions_rad,
            current["axis"].point, current["axis"].direction,
            collision_scene["robot_metadata"]["tcp_sites"][group],
        )
        expected = stage_target.end_angle_rad - stage_target.start_angle_rad
        if (metrics["max_axis_radius_change_m"] > .005
                or metrics["max_axial_drift_m"] > .005
                or metrics["max_rotation_off_axis_rad"] > .05
                or metrics["max_angle_regression_rad"] > .03
                or abs(metrics["tcp_rotation_rad"] - expected) > .05):
            success = False
    stage = SkillStage(
        stage_name, "planned" if success else "failed", target,
        effective_request, result,
        {"collision_scope": stage_target.collision_scope,
         "ignored_world_obstacles": planning_snapshot.get("ignored_world_obstacles", []),
         "microwave_anchor_error_m_rad": microwave_delta,
         "knob_start_angle_error_rad": angle_error,
         "axis_path_validation": metrics,
         "retreat_fallback_used": effective_request.operation != stage_target.operation},
    )
    record = {
        "schema_version": 1, "kind": "observed_knob_motion_stage_plan_only",
        "stage_name": stage_name, "success": success,
        "source_frame_number": current["source_frame_number"],
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "stage_collision_snapshot_sha256": planning_snapshot["snapshot_sha256"],
        "collision_scope": stage_target.collision_scope,
        "ignored_world_obstacles": planning_snapshot.get("ignored_world_obstacles", []),
        "microwave_anchor_error_m_rad": microwave_delta,
        "knob_start_angle_error_rad": angle_error,
        "axis_path_validation": metrics,
        "planning_attempts": attempts,
        "stage": stage.to_dict(),
        "planner_cache_hit": context["planner_cache_hit"],
        "planner_setup_duration_s": context["planner_setup_duration_s"],
        "planning_wall_s": time.perf_counter() - started,
        "claim": "只规划当前实测起点的旋钮阶段轨迹；未执行或证明旋钮受力转动",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**record, "stage": stage}


__all__ = ['build_observed_knob_targets', 'select_observed_knob_grasp', 'plan_observed_knob_motion_stage']
