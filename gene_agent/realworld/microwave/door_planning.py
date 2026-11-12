"""现场状态适配；操作几何与规划算法复用现有 Knowledge、Skill 和 motion。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.operations.door import apply_selected_push_candidate, select_legacy_push_candidate
from gene_agent.tasks.policies.microwave_task.operations.door import build_door_knowledge, load_push_open_profile
from gene_agent.tasks.policies.microwave_task.operations.door import approximate_ik_budget_for, local_rotation_search_for
from gene_agent.tasks.policies.microwave_task.planning.door_scene import export_observed_door_stage_scene
from gene_agent.feedback.motion_diagnostics import summarize_motion_failure
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion import PlannerMode
from gene_agent.motion.curobo.approximate import (
    ApproximateIKBudget,
    plan_with_approximate_ik,
)
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.motion.model import (
    JointState,
    JointTrajectory,
    MotionOperation,
    MotionRequest,
    MotionResult,
    PoseTarget,
    motion_request_from_dict,
    motion_result_from_dict,
)
from gene_agent.realworld.planning import prepare_plan_only_context
from gene_agent.skills.library.door.sequence import door_phases
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.skills.runtime.target_selection import plan_with_local_rotation_search

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _ROOT / "data/task_1/skills/door_skill_debug.json"
_SKILLS = frozenset({"handle_pull", "push_open", "push_close"})


def _microwave_anchor(collision_scene: dict) -> dict:
    """固定门 Skill 目标所依赖的微波炉刚体 pose 与资产版本。"""
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    if (provenance.get("physics_stepped") is not False
            or provenance.get("pose_frame_verified") is not True
            or provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("门阶段需要同帧现场观测及一致的碰撞快照")
    pose = validate_pose(
        provenance["aligned_object_poses_world"]["Microwave/Microwave_0"],
        field_name="T_world_microwave",
    )
    return {
        "microwave_pose_world": pose.tolist(),
        "scene_config_sha256": provenance["scene_config_sha256"],
        "active_group": snapshot["active_group"],
        "world_frame": snapshot["world_frame"],
    }


def rebase_observed_door_skill_targets(
    targets: dict, collision_scene: dict, *,
    max_microwave_translation_m: float,
    max_microwave_rotation_rad: float,
) -> dict:
    """同一门 Skill 内只更新实测起点和碰撞，不重生逐角 Knowledge。

    门角、活动臂关节和夹爪开口可随执行改变。微波炉本体 pose 若偏离
    原目标锚点超过调用方显式给出的误差界，则必须中止该 Skill；不能
    用已打开的门重新生成原拉门序列。把手 Skill 结束后，新推门 Skill
    应从实测门角单独生成目标。
    """
    limits = (float(max_microwave_translation_m),
              float(max_microwave_rotation_rad))
    if any(not math.isfinite(value) or value < 0 for value in limits):
        raise ValueError("微波炉目标锚点容差必须为非负有限数")
    original = targets["target_anchor"]
    current = _microwave_anchor(collision_scene)
    for field in ("scene_config_sha256", "active_group", "world_frame"):
        if current[field] != original[field]:
            raise ValueError(f"门 Skill 目标锚点的 {field} 已变化，必须停止并重新检查")
    old_pose = validate_pose(original["microwave_pose_world"],
                             field_name="原微波炉目标锚点")
    new_pose = validate_pose(current["microwave_pose_world"],
                             field_name="当前微波炉目标锚点")
    translation = float(np.linalg.norm(old_pose[:3, 3] - new_pose[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        old_pose[:3, :3].T @ new_pose[:3, :3],
    ).as_rotvec()))
    if translation > limits[0] + 1e-9 or rotation > limits[1] + 1e-9:
        raise ValueError(
            "微波炉 pose 已偏离当前门 Skill 目标："
            f"{translation:.4f} m / {rotation:.4f} rad，"
            f"容差 {limits[0]:.4f} m / {limits[1]:.4f} rad"
        )
    return {
        **targets,
        "collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
        "target_rebase": {
            "original_collision_snapshot_sha256": targets["collision_snapshot_sha256"],
            "current_collision_snapshot_sha256": collision_scene["snapshot"]["snapshot_sha256"],
            "microwave_translation_error_m": translation,
            "microwave_rotation_error_rad": rotation,
            "max_microwave_translation_m": limits[0],
            "max_microwave_rotation_rad": limits[1],
        },
    }


def _load_tuning(root: Path, config: dict) -> dict[str, dict]:
    states = {}
    for kind, reference in config.get("knowledge_states", {}).items():
        path = Path(reference).expanduser()
        path = path if path.is_absolute() else root / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("input_valid") is False:
            raise ValueError(f"Knowledge 状态含有未通过校验的输入：{path}")
        states[kind] = dict(payload.get("tuner_values") or {})
    return states


def build_observed_door_skill_targets(
    collision_scene: dict, *, skill_id: str,
    config_path: str | Path = _DEFAULT_CONFIG,
) -> dict:
    """输出一个门 Skill 的原有阶段、路径类型及当前观测来源。

    `handle_pull` 用于已关门场景；`push_open`/`push_close` 的起始角
    从同一碰撞快照读取。现场动作执行后须重新观测门角并重新规划，
    不能继续沿用这里预测的终态。
    """
    if skill_id not in _SKILLS:
        raise ValueError(f"未知门 Skill：{skill_id}")
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    if provenance.get("physics_stepped") is not False or provenance.get("pose_frame_verified") is not True:
        raise ValueError("门目标需要同帧现场观测，不能使用 MuJoCo 动力学结果")
    if provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise ValueError("门目标的现场来源与碰撞快照 hash 不一致")
    metadata = collision_scene["scene_metadata"]
    if snapshot.get("active_group") not in metadata["robot"]["joint_groups"]:
        raise ValueError("活动臂与场景机器人元数据不一致")
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    data = mujoco.MjData(model)
    recorded = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if recorded.shape != (model.nq,) or not np.isfinite(recorded).all():
        raise ValueError("现场碰撞快照的 qpos 与资产模型不一致")
    data.qpos[:] = recorded
    mujoco.mj_forward(model, data)
    door_joint = model.joint(metadata["objects"]["door_joint"])
    observed_angle = float(data.qpos[door_joint.qposadr[0]])
    if not np.isclose(observed_angle, provenance["door_angle_rad"], atol=1e-7, rtol=0):
        raise ValueError("现场记录门角与碰撞快照关节状态不一致")

    config_source = Path(config_path).expanduser().resolve()
    config_bytes = config_source.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != 1:
        raise ValueError("门 Skill 配置版本必须为 1")
    if config.get("stage_overrides_microwave"):
        raise NotImplementedError("现场门目标暂不支持覆盖 Knowledge 阶段，避免静默偏离 Viewer")
    if skill_id == "push_open":
        profile = load_push_open_profile(_ROOT, config, snapshot["active_group"])
        if profile is not None and "reposition_after_40" in profile:
            raise NotImplementedError("逐角推门换点需接入原完整 Skill 规则后才能用于现场")

    # 原 Viewer 在 rebuild 前把门重置为零角。这里只改私有 data；
    # snapshot 的门角及碰撞 geom 完全保持实际观测状态。
    data.qpos[door_joint.qposadr[0]] = 0.0
    mujoco.mj_forward(model, data)
    case = SimpleNamespace(model=model, data=data, metadata=metadata, snapshot=snapshot)
    tuning = _load_tuning(_ROOT, config)
    from gene_agent.skills.library.door.sequence import door_skill_angle_range
    start_deg, target_deg = door_skill_angle_range(skill_id, config, observed_angle)
    knowledge = build_door_knowledge(
        _ROOT, case, config, tuning, kind=skill_id,
        start_deg=float(start_deg), target_deg=float(target_deg),
    )
    phases = door_phases(
        handle=knowledge if skill_id == "handle_pull" else None,
        push_open=knowledge if skill_id == "push_open" else None,
        push_close=knowledge if skill_id == "push_close" else None,
    )
    return {
        "skill_id": skill_id,
        "phases": phases,
        "knowledge": knowledge,
        "observed_door_angle_rad": observed_angle,
        "knowledge_geometry_door_angle_rad": 0.0,
        "collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "config_path": str(config_source),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "target_anchor": _microwave_anchor(collision_scene),
    }


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法保存的规划结果类型：{type(value).__name__}")


def load_planned_door_skill_stage(path: str | Path) -> SkillStage:
    """把保存的单阶段公共契约交给真机 runtime 做二次实测预检。"""
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    if row.get("status") != "planned":
        raise ValueError("只允许加载已成功规划的门阶段")
    return SkillStage(
        name=row["name"], status="planned",
        target=PoseTarget(row["target"]["frame"], row["target"]["matrix"]),
        request=motion_request_from_dict(row["request"]),
        result=motion_result_from_dict(row["result"]),
        details=row.get("details", {}),
    )


def select_observed_door_phase_candidate(
    collision_scene: dict, targets: dict, *, phase_id: str,
    planner_pool: CuroboPlannerPool, output_dir: str | Path,
    scene_revision: int = 0, seed: int = 123,
    planner_mode: PlannerMode | str | None = None,
) -> tuple[dict, dict | None]:
    """按旧 Task1 两批 IK 选择当前门角接触候选，并改写当前 Skill 目标。

    预接触候选选择时暂从静态规划 world 移出可动门板；真正的
    pre-contact 轨迹仍由 ``plan_observed_door_stage`` 在门碰撞开启的
    快照里求解。当前函数只改返回的 targets 副本，不改 Knowledge。
    """
    snapshot = collision_scene["snapshot"]
    if targets["collision_snapshot_sha256"] != snapshot["snapshot_sha256"]:
        raise ValueError("门候选选择的目标与当前现场快照不一致")
    phases = list(targets["phases"])
    matches = [index for index, item in enumerate(phases) if item.phase_id == phase_id]
    if len(matches) != 1:
        raise ValueError(f"未知或重复的门阶段：{phase_id}")
    phase_index = matches[0]
    phase = phases[phase_index]
    if phase.skill_id not in {"push_open", "push_close"}:
        return targets, None
    if phase.name == "pre_contact":
        candidate_index = 0
    elif phase.name.startswith("push_"):
        candidate_index = int(phase.name.removeprefix("push_"))
    else:
        return targets, None
    knowledge = targets["knowledge"]
    candidate_set = knowledge.door_candidate_sets[candidate_index]
    config_path = Path(targets["config_path"])
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != targets["config_sha256"]:
        raise ValueError("门候选选择时配置已变化，必须重新生成目标")
    config = json.loads(config_bytes)
    mode = PlannerMode(planner_mode or config.get(
        "planner_mode", PlannerMode.CUDA_GRAPH_BENCHMARK.value,
    ))
    output = Path(output_dir).resolve()
    # 旧 Task1 的候选 IK 暂移出活动门板。接触阶段和 pre-contact
    # 轨迹分别使用自己的规划快照，不能把此筛选场景当作轨迹碰撞场景。
    selection_phase = replace(phase, contact_policy="door_contact")
    candidate_scene = export_observed_door_stage_scene(
        collision_scene, selection_phase, output_dir=output,
    )
    context = prepare_plan_only_context(
        candidate_scene, group=snapshot["active_group"],
        planner_pool=planner_pool, scene_revision=scene_revision,
        seed=seed, planner_mode=mode,
    )
    start = context["start"]
    chosen, facts = select_legacy_push_candidate(
        context["planner"].planner, start.joint_names,
        start.positions_rad, candidate_set.candidates,
    )
    facts.update(
        phase_id=phase.phase_id,
        angle_deg=float(np.rad2deg(candidate_set.door_angle_rad)),
        collision_snapshot_ref=str(output / "collision" / "collision_snapshot.json"),
        snapshot_sha256=candidate_scene["snapshot"]["snapshot_sha256"],
        planner_cache_hit=context["planner_cache_hit"],
        planner_setup_duration_s=context["planner_setup_duration_s"],
    )
    if chosen is None:
        return targets, {"success": False, **facts}
    updated = apply_selected_push_candidate(
        targets["phases"], knowledge, phase_index, chosen,
    )
    return {**targets, "phases": updated}, {"success": True, **facts}


def plan_observed_door_stage(
    collision_scene: dict, targets: dict, *, phase_id: str,
    planner_pool: CuroboPlannerPool, output_dir: str | Path,
    scene_revision: int = 0, seed: int = 123,
    planner_mode: PlannerMode | str | None = None,
) -> dict:
    """从同帧实测起点规划一个既有门阶段，保留原 Viewer 的候选策略。

    调用方先用 ``build_observed_door_skill_targets`` 生成目标。每次真实
    执行后必须重新观测并重新生成目标和碰撞快照；本函数不会把预测终点
    当成下一阶段的实测起点，也不会调用 SDK。
    """
    base = collision_scene["snapshot"]
    if targets["collision_snapshot_sha256"] != base["snapshot_sha256"]:
        raise ValueError("门阶段目标与现场碰撞快照不一致，必须重新观测")
    matches = [phase for phase in targets["phases"] if phase.phase_id == phase_id]
    if len(matches) != 1 or matches[0].skill_id != targets["skill_id"]:
        raise ValueError(f"未知或重复的门阶段：{phase_id}")
    phase = matches[0]
    if phase.motion_kind == "gripper":
        raise ValueError("夹爪开合是执行事件，不是 cuRobo 轨迹规划阶段")
    config_path = Path(targets["config_path"])
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != targets["config_sha256"]:
        raise ValueError("门阶段配置在目标生成后改变，必须重新生成目标")
    config = json.loads(config_bytes)
    if config.get("schema_version") != 1:
        raise ValueError("门 Skill 配置版本必须为 1")
    mode = PlannerMode(planner_mode or config.get(
        "planner_mode", PlannerMode.CUDA_GRAPH_BENCHMARK.value,
    ))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage_started = time.perf_counter()
    stage_scene = export_observed_door_stage_scene(
        collision_scene, phase, output_dir=output,
    )
    exported_s = time.perf_counter() - stage_started
    context = prepare_plan_only_context(
        stage_scene, group=base["active_group"], planner_pool=planner_pool,
        scene_revision=scene_revision, seed=seed, planner_mode=mode,
    )
    planner = context["planner"].planner
    start = context["start"]
    names, q = start.joint_names, start.positions_rad
    budget_data = approximate_ik_budget_for(config, phase)
    budget = ApproximateIKBudget(**budget_data) if budget_data is not None else None
    local_search = local_rotation_search_for(config, phase)
    solve_started = time.perf_counter()
    if local_search:
        raw = plan_with_local_rotation_search(
            planner, names, q, phase.tcp_pose_world, budget=budget,
            x_degrees=local_search.get("x_degrees", (0.,)),
            y_degrees=local_search.get("y_degrees", (0.,)),
            target_frame="world_tcp", linear_tcp=phase.motion_kind == "linear_tcp",
        )
    else:
        raw = plan_with_approximate_ik(
            planner, names, q, phase.tcp_pose_world, budget=budget,
            target_frame="world_tcp", linear_tcp=phase.motion_kind == "linear_tcp",
        )
    fallback = None
    actual_operation = phase.motion_kind
    effective_target = phase.tcp_pose_world
    fixed_close_pose = config.get("close_pre_contact_fallback_pose_world")
    if (not raw.get("success") and phase.skill_id == "push_close"
            and phase.name == "pre_contact" and fixed_close_pose is not None):
        # 仅复制旧 Viewer 已有的关门预接触兜底目标。第一次失败仍在
        # fallback 和原始规划记录中可查，不能悄悄替换 Knowledge。
        first_failure = raw
        effective_target = validate_pose(
            np.asarray(fixed_close_pose, dtype=float),
            field_name="close_pre_contact_fallback_pose_world",
        )
        raw = plan_with_approximate_ik(
            planner, names, q, effective_target, budget=budget,
            target_frame="world_tcp", linear_tcp=False,
        )
        actual_operation = "plan"
        fallback = {
            "source_status": first_failure.get("status"),
            "source_failure_category": first_failure.get("failure_category"),
            "source_message": first_failure.get("message"),
            "source_target_tcp_pose_world": phase.tcp_pose_world.tolist(),
            "retry_target_tcp_pose_world": effective_target.tolist(),
            "success": bool(raw.get("success")),
        }
    if (not raw.get("success") and phase.phase_id ==
            "task/open_to_close/return_default_tcp"
            and config.get("return_default_tcp_non_linear_fallback", False)):
        # 旧 Task1 仅在门边撤离失败时允许普通规划；原始直线失败
        # 和真实执行路径类型都必须进入现场报告与 SkillStage 契约。
        first_failure = raw
        raw = planner.plan(names, q, effective_target, target_frame="world_tcp")
        actual_operation = "plan"
        fallback = {
            "from_operation": phase.motion_kind,
            "to_operation": actual_operation,
            "source_status": first_failure.get("status"),
            "source_failure_category": first_failure.get("failure_category"),
            "source_message": first_failure.get("message"),
            "success": bool(raw.get("success")),
        }
    solve_s = time.perf_counter() - solve_started
    success = bool(raw.get("success"))
    failure_category = raw.get("failure_category")
    positions = None
    sample_dt = None
    if success:
        try:
            positions = np.asarray(raw.get("positions_rad"), dtype=float)
            sample_dt = float(raw.get("interpolation_dt_s"))
            valid = (positions.ndim == 2 and positions.shape[1] == len(names)
                     and len(positions) >= 2 and np.isfinite(positions).all()
                     and np.isfinite(sample_dt) and sample_dt > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            success = False
            failure_category = "invalid_trajectory"
    result = {
        "kind": "observed_door_stage_plan_only",
        "phase_id": phase.phase_id,
        "skill_id": phase.skill_id,
        "success": success,
        "failure_category": failure_category,
        "status": raw.get("status"),
        "message": raw.get("message"),
        "requested_operation": phase.motion_kind,
        "operation": actual_operation,
        "requested_tcp_pose_world": phase.tcp_pose_world.tolist(),
        "effective_tcp_pose_world": effective_target.tolist(),
        "fallback": fallback,
        "approximate_ik": raw.get("approximate_ik"),
        "target_selection": raw.get("target_selection"),
        "start_joint_names": list(names),
        "start_joint_positions_rad": list(q),
        "base_collision_snapshot_sha256": base["snapshot_sha256"],
        "target_rebase": targets.get("target_rebase"),
        "stage_collision_snapshot_sha256": stage_scene["snapshot"]["snapshot_sha256"],
        "stage_collision_snapshot_ref": str(output / "collision" / "collision_snapshot.json"),
        "planner_mode": mode.value,
        "timing": {
            "collision_export_s": exported_s,
            "planner_setup_s": context["planner_setup_duration_s"],
            "planner_cache_hit": context["planner_cache_hit"],
            "candidate_and_plan_s": solve_s,
            "backend_s": raw.get("duration_s"),
            "stage_wall_s": time.perf_counter() - stage_started,
        },
    }
    if not success and failure_category != "invalid_trajectory":
        result["failure_summary"] = summarize_motion_failure(
            raw, stage_scene["snapshot"],
        ).to_dict()
    if success:
        operation = MotionOperation(actual_operation)
        request = MotionRequest(
            request_id=f"door:{phase.phase_id}:{stage_scene['snapshot']['snapshot_sha256'][:12]}",
            operation=operation, group=base["active_group"], start=start,
            target=PoseTarget("world_tcp", effective_target),
            scene=context["scene"], seed=seed, planner_mode=mode,
            approximate_ik_budget=None if budget is None else budget.to_dict(),
        )
        trajectory = JointTrajectory(
            names, tuple(index * sample_dt for index in range(len(positions))),
            tuple(tuple(row) for row in positions),
        )
        motion_result = MotionResult(
            request_id=request.request_id, operation=operation,
            success=True, backend="curobo", planner_mode=mode,
            duration_s=float(raw.get("duration_s", 0.0)),
            goal_state=JointState(names, tuple(positions[-1])),
            trajectory=trajectory,
            metrics={"scene_snapshot_sha256": stage_scene["snapshot"]["snapshot_sha256"]},
        )
        stage = SkillStage(
            name=phase.phase_id, status="planned", target=request.target,
            request=request, result=motion_result,
            details={"original_tcp_pose_world": phase.tcp_pose_world.tolist(),
                     "effective_tcp_pose_world": effective_target.tolist(),
                     "fallback": fallback,
                     "approximate_ik": raw.get("approximate_ik"),
                     "target_selection": raw.get("target_selection"),
                     "claim": "仅规划；真机 runtime 仍需从实测起点二次预检"},
        )
        contract_path = output / "skill_stage.json"
        contract_path.write_text(
            json.dumps(stage.to_dict(), ensure_ascii=False, indent=2,
                       default=_json_value) + "\n", encoding="utf-8",
        )
        result["skill_stage_ref"] = str(contract_path)
    (output / "plan.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = []
