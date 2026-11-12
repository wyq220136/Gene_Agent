"""同一 MuJoCo GT 状态下的碗候选、cuRobo 碰撞与规划事实。"""

from __future__ import annotations
from gene_agent.motion.curobo.scene import planner_scene_options

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from typing import Sequence

import mujoco
import numpy as np

from gene_agent.knowledge import bowl_grasp_candidates
from gene_agent.motion import (
    CollisionPolicy, CuroboPlannerAdapter, JointState, MotionOperation,
    MotionRequest, PlannerMode, PlanningSceneRef, PoseTarget,
    SynchronousPlanner, load_realman_motion_config,
)
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.simulation import build_scene, set_joint_positions


MICROWAVE_EDITABLE_JOINTS = frozenset({
    "microwave_door_joint", "knob_0_joint", "knob_1_joint",
})


@dataclass
class PickBowlCase:
    output_dir: Path
    revision: int
    model: Any
    data: Any
    metadata: dict[str, Any]
    snapshot: dict[str, Any]
    bowl_pose: np.ndarray
    candidates: list[dict[str, Any]]
    report: dict[str, Any] | None = None


def _body_pose(data: Any, name: str) -> np.ndarray:
    body = data.body(name)
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose


def apply_microwave_joint_overrides(
    model: Any, data: Any, overrides: dict[str, float] | None,
) -> dict[str, float]:
    """一次性检查微波炉关节编辑，再提交到 MuJoCo 状态。

    只允许微波炉的三个标量关节；范围读取编译后的模型，避免 URDF 与
    MuJoCo 转换后的边界不一致。任何一个输入无效时保持原状态不变。
    """
    if not overrides:
        return {}
    checked: dict[str, float] = {}
    for name, raw in overrides.items():
        if name not in MICROWAVE_EDITABLE_JOINTS:
            raise ValueError(f"不支持编辑的微波炉关节: {name}")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"微波炉旋转关节不存在: {name}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"关节值必须是有限数字: {name}") from exc
        if not np.isfinite(value):
            raise ValueError(f"关节值必须是有限数字: {name}")
        lower, upper = map(float, model.jnt_range[joint_id])
        if bool(model.jnt_limited[joint_id]) and not lower <= value <= upper:
            raise ValueError(f"{name} 超出关节范围 [{lower}, {upper}]: {value}")
        checked[name] = value
    set_joint_positions(model, data, checked)
    return checked


def prepare_pick_bowl_scene(
    output_dir: Path, *, revision: int = 0, seed: int = 0,
    pose_overrides: dict[str, np.ndarray] | None = None,
    joint_overrides: dict[str, float] | None = None,
    table_size_xy_m: tuple[float, float] | None = None,
) -> PickBowlCase:
    """构建唯一的 GT 场景；编辑必须重建模型和碰撞快照。"""
    output = Path(output_dir).resolve()
    if revision < 0:
        raise ValueError("scene revision 必须非负")
    output.mkdir(parents=True, exist_ok=False)
    model, data, metadata = build_scene(
        output / "source_scene", seed=seed, pose_overrides=pose_overrides,
        table_size_xy_m=table_size_xy_m,
    )
    joints = apply_microwave_joint_overrides(model, data, joint_overrides)
    mujoco.mj_forward(model, data)
    snapshot = export_collision_scene(
        model, data, metadata["robot"], "right", output / "collision"
    )
    bowl_pose = _body_pose(data, "bowl")
    candidates = bowl_grasp_candidates(bowl_pose, pose_source="mujoco_gt")
    (output / "case_meta.json").write_text(
        json.dumps({"revision": revision, "seed": seed, "joint_overrides": joints}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PickBowlCase(
        output, revision, model, data, metadata, snapshot, bowl_pose, candidates
    )


def prepare_grasp_stage_snapshots(
    case: PickBowlCase, open_width_m: float, *, group: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """以 Skill 将执行的开口冻结 pregrasp 与接触阶段的规划碰撞。

    只修改 MuJoCo 数据副本来表达待执行的开爪状态，不推进仿真，也不改
    场景里的机器人或碗。pregrasp 保留全部障碍；最终 grasp IK 和直线
    approach 沿用旧版的目标物体接触规则，只从规划世界排除目标碗。
    物理执行仍须核对实际开口以及非指垫接触，不能把规划豁免当作物理成功。
    """
    descriptor = case.metadata["robot"]["grippers"][group]
    if descriptor["mode"] != "parallel_slide_proxy":
        raise NotImplementedError("静态抓碗规划快照暂只支持 parallel_slide_proxy")
    width = float(open_width_m)
    lower, upper = map(float, descriptor["width_range_m"])
    if not np.isfinite(width) or not lower <= width <= upper:
        raise ValueError(f"规划夹爪开口须在 [{lower}, {upper}] m 范围内")
    opened = mujoco.MjData(case.model)
    mujoco.mj_copyData(opened, case.model, case.data)
    for name in descriptor["joint_names"]:
        opened.qpos[case.model.joint(name).qposadr[0]] = width / 2.0
    mujoco.mj_forward(case.model, opened)
    strict = export_collision_scene(
        case.model, opened, case.metadata["robot"], group,
        case.output_dir / "collision_grasp_open" / "strict",
    )
    contact = export_collision_scene(
        case.model, opened, case.metadata["robot"], group,
        case.output_dir / "collision_grasp_open" / "target_contact",
        disabled_body_names=("bowl",),
    )
    return strict, contact


def load_pick_bowl_case(output_dir: Path) -> PickBowlCase:
    """从已保存场景重开 Viser；初次查看不重复占用 GPU 规划。"""
    output = Path(output_dir).resolve()
    info = json.loads((output / "case_meta.json").read_text(encoding="utf-8"))
    scene_dir = output / "source_scene"
    robot = json.loads((scene_dir / "robot" / "robot_metadata.json").read_text(encoding="utf-8"))
    objects = json.loads((scene_dir / "objects" / "scene_metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "xml_path": str(scene_dir / "scene.xml"), "robot": robot, "objects": objects,
        "world_frame": robot["world_frame"], "seed": info["seed"],
    }
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    data = mujoco.MjData(model)
    set_joint_positions(model, data, robot["home_joint_positions"])
    apply_microwave_joint_overrides(model, data, info.get("joint_overrides"))
    snapshot = json.loads((output / "collision" / "collision_snapshot.json").read_text(encoding="utf-8"))
    report_path = output / "pick_bowl_result.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    if report is not None:
        if report["snapshot_sha256"] != snapshot["snapshot_sha256"]:
            raise ValueError("规划报告与碰撞快照哈希不一致")
        snapshot["active_robot_spheres"] = report.get("active_robot_spheres", [])
    bowl_pose = _body_pose(data, "bowl")
    candidates = bowl_grasp_candidates(bowl_pose, pose_source="mujoco_gt")
    return PickBowlCase(
        output, int(info["revision"]), model, data, metadata, snapshot,
        bowl_pose, candidates, report,
    )


def _robot_spheres_world(planner: SynchronousPlanner, names, q) -> list[dict[str, Any]]:
    """把实际 cuRobo 活动组碰撞球映回 world，供碰撞叠图核对。"""
    state = planner.kinematics.compute_kinematics(planner._state(names, q))
    spheres = state.robot_spheres.detach().cpu().numpy().reshape(-1, 4)
    params = planner.kinematics.config.kinematics_config
    link_ids = params.link_sphere_idx_map.detach().cpu().numpy()
    link_names = {value: key for key, value in params.link_name_to_idx_map.items()}
    transform = np.asarray(planner.world_from_base)
    result = []
    for index, sphere in enumerate(spheres):
        if sphere[3] <= 0:
            continue
        center = transform[:3, :3] @ sphere[:3] + transform[:3, 3]
        result.append({
            "sphere_index": index,
            "link": link_names[int(link_ids[index])],
            "center_world_m": center.tolist(),
            "radius_m": float(sphere[3]),
        })
    return result


def tcp_path_for_positions(case: PickBowlCase, names, positions) -> list[list[float]]:
    """用同一 MuJoCo 机器人 FK 生成回放线，绝不改动 GT 起始 data。"""
    preview = mujoco.MjData(case.model)
    mujoco.mj_copyData(preview, case.model, case.data)
    site_name = case.metadata["robot"]["tcp_sites"]["right"]
    path = []
    for row in positions:
        if len(row) != len(names):
            raise ValueError("轨迹关节数与规划组不一致")
        for name, value in zip(names, row):
            preview.qpos[case.model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(case.model, preview)
        path.append(preview.site(site_name).xpos.tolist())
    return path


def summarize_ik_diagnostics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 IK 失败；起点状态来自每次求解前已有的碰撞预检。"""
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    obstacles: dict[str, int] = {}
    ambiguous = 0
    start_failures = []
    for attempt in attempts:
        result = attempt["result"]
        if not result.get("success"):
            failure = result.get("failure") or {}
            kind = failure.get("kind") or "unknown"
            counts[kind] = counts.get(kind, 0) + 1
            status = failure.get("backend_status") or ""
            if status in {"start_collision", "start_joint_limit", "start_invalid"} or kind == "start_collision":
                start_failures.append(failure)
            diagnostic = (failure.get("details") or {}).get("solver_diagnostic") or {}
            reason = diagnostic.get("reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            elif kind in {"constraint_violation", "unknown"} and not status.startswith("start_"):
                ambiguous += 1
            collision = diagnostic.get("collision_check") or {}
            for name in set(collision.get("colliding_obstacles") or []):
                obstacles[name] = obstacles.get(name, 0) + 1
    if start_failures:
        valid: bool | None = False
        summaries = list(dict.fromkeys(str(failure.get("summary") or failure.get("backend_status")) for failure in start_failures))
        evidence = "cuRobo 起始姿态严格复核失败：" + "；".join(summaries)
    elif any(item["result"].get("success") for item in attempts) or reasons or counts.get("constraint_violation"):
        valid = True
        evidence = "起始姿态严格碰撞复核通过；目标无解仍可能是多种原因"
    else:
        valid = None
        evidence = "缺少足够结果确认起始姿态"
    return {
        "start_state": {"valid": valid, "evidence": evidence},
        "failure_counts": counts,
        "reason_counts": reasons,
        "colliding_obstacle_counts": obstacles,
        "ambiguous_solver_failures": ambiguous,
    }


def evaluate_ik_requests(
    adapter: CuroboPlannerAdapter, requests: Sequence[MotionRequest], *, mode: str,
) -> tuple[Any, ...]:
    """按显式模式筛选 IK；batch 失败不会在底层悄悄改用串行结果。"""
    if mode == "batch":
        return adapter.evaluate_candidates(requests)
    if mode == "serial":
        return tuple(adapter.solve_ik(request) for request in requests)
    raise ValueError(f"未知 IK 筛选模式: {mode}")


def run_pick_bowl_planning(
    case: PickBowlCase, *, seed: int = 123, ik_mode: str = "serial",
) -> dict[str, Any]:
    """对每个 final grasp 先做 IK，再只对首个可达候选做轨迹规划。"""
    metadata = case.metadata
    snapshot = case.snapshot
    settings = load_realman_motion_config(
        "right", metadata["robot"], gripper_snapshot=snapshot.get("gripper_snapshot")
    )
    planner = SynchronousPlanner(
        **settings,
        **planner_scene_options(snapshot),
        planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK,
        random_seed=seed,
    )
    names = tuple(metadata["robot"]["joint_groups"]["right"])
    q = tuple(float(case.data.qpos[case.model.joint(name).qposadr[0]]) for name in names)
    state = JointState(names, q)
    scene_ref = PlanningSceneRef(case.revision, snapshot["snapshot_sha256"], snapshot["scope"])
    adapter = CuroboPlannerAdapter(planner, group="right", scene_revision=case.revision)
    snapshot["active_robot_spheres"] = _robot_spheres_world(planner, names, q)
    requests = tuple(MotionRequest(
        f"r{case.revision}-{candidate['candidate_id']}-ik", MotionOperation.IK,
        "right", state, PoseTarget("world_tcp", np.asarray(candidate["T_world_tcp"])),
        scene_ref, CollisionPolicy(), seed=seed,
    ) for candidate in case.candidates)
    results = evaluate_ik_requests(adapter, requests, mode=ik_mode)
    if len(results) != len(requests):
        raise RuntimeError("IK 筛选返回数量与候选数不一致")
    attempts: list[dict[str, Any]] = []
    chosen = None
    total_duration = 0.0
    for candidate, request, result in zip(case.candidates, requests, results):
        identifier = candidate["candidate_id"]
        if result.request_id != request.request_id or result.operation is not MotionOperation.IK:
            raise RuntimeError("IK 筛选结果顺序或 operation 与请求不一致")
        total_duration += result.duration_s
        attempts.append({
            "candidate_id": identifier, "stage": "final_grasp_ik",
            "request": request.to_dict(), "result": result.to_dict(),
        })
        if chosen is None and result.success:
            chosen = identifier
    plan = None
    if chosen is not None:
        candidate = next(item for item in case.candidates if item["candidate_id"] == chosen)
        request = MotionRequest(
            f"r{case.revision}-{chosen}-plan", MotionOperation.PLAN, "right", state,
            PoseTarget("world_tcp", np.asarray(candidate["T_world_tcp"])),
            scene_ref, CollisionPolicy(), seed=seed,
        )
        result = adapter.plan(request)
        total_duration += result.duration_s
        plan = {"candidate_id": chosen, "stage": "home_to_final_grasp",
                "request": request.to_dict(), "result": result.to_dict()}
        if result.success and result.trajectory is not None:
            plan["tcp_path_world_m"] = tcp_path_for_positions(
                case, names, result.trajectory.positions_rad
            )
    report = {
        "schema_version": 1,
        "task": "pick_bowl_planning_inspection",
        "claim": "IK/轨迹检查；不是夹爪闭合、抬碗或物理抓取成功",
        "scene_revision": case.revision,
        "seed": seed,
        "world_frame": metadata["world_frame"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "scene_scope": snapshot["scope"],
        "collision_policy": "strict",
        "ik_screening_mode": ik_mode,
        "ik_screening_duration_s": sum(result.duration_s for result in results),
        "mounting_contact_policy": snapshot["mounting_contact_policy"],
        "obstacle_count": {kind: len(items) for kind, items in snapshot["scene_model"].items()},
        "obstacle_records": snapshot["records"],
        "excluded_geometries": snapshot["excluded"],
        "active_robot_spheres": snapshot["active_robot_spheres"],
        "source_assets": {
            "robot_xml": metadata["robot"]["xml_path"],
            "scene_xml": metadata["xml_path"],
            "microwave_urdf": metadata["objects"]["resolved_config"]["microwave_asset"]["urdf"],
            "bowl_visual": "assets/objects/bowl_visual.obj",
            "robot_collision_config": "assets/robots/realman/config/realman_right.yml",
            "robot_collision_sphere_count": len(snapshot["active_robot_spheres"]),
            "gripper_snapshot_id": snapshot.get("gripper_snapshot_id"),
        },
        "object_poses_world": metadata["objects"]["sample_poses"],
        "joint_names": list(names),
        "start_joints_rad": list(q),
        "candidates": case.candidates,
        "ik_attempts": attempts,
        "ik_diagnostics": summarize_ik_diagnostics(attempts),
        "selected_candidate_id": chosen,
        "plan_attempt": plan,
        "total_planning_duration_s": total_duration,
        "ik_success_count": sum(item["result"]["success"] for item in attempts),
        "plan_success": bool(plan and plan["result"]["success"]),
    }
    report_path = case.output_dir / "pick_bowl_result.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    case.report = report
    return report


def plan_pick_bowl_candidate(
    case: PickBowlCase, candidate_id: str, *, seed: int = 123,
) -> dict[str, Any]:
    """只规划指定候选，并把结果另存于候选轨迹表。

    IK 筛选结果仅用于解释，不拦截用户显式要求的轨迹规划。规划器每次
    以本 case 冻结的碰撞快照构建，避免复用编辑前的障碍或关节状态。
    原始首选候选的 plan_attempt 保持不变，便于复查自动选择结果。
    """
    candidate = next(
        (item for item in case.candidates if item["candidate_id"] == candidate_id), None
    )
    if candidate is None:
        raise ValueError(f"未知 grasp 候选: {candidate_id}")
    report = case.report
    if report is None:
        raise ValueError("尚无初始规划报告，请先运行 run_pick_bowl_planning")
    snapshot = case.snapshot
    if report["snapshot_sha256"] != snapshot["snapshot_sha256"]:
        raise ValueError("规划报告与当前碰撞快照哈希不一致")
    settings = load_realman_motion_config(
        "right", case.metadata["robot"],
        gripper_snapshot=snapshot.get("gripper_snapshot"),
    )
    planner = SynchronousPlanner(
        **settings,
        **planner_scene_options(snapshot),
        planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK,
        random_seed=seed,
    )
    names = tuple(case.metadata["robot"]["joint_groups"]["right"])
    q = tuple(float(case.data.qpos[case.model.joint(name).qposadr[0]]) for name in names)
    state = JointState(names, q)
    scene_ref = PlanningSceneRef(case.revision, snapshot["snapshot_sha256"], snapshot["scope"])
    adapter = CuroboPlannerAdapter(planner, group="right", scene_revision=case.revision)
    request = MotionRequest(
        f"r{case.revision}-{candidate_id}-plan", MotionOperation.PLAN, "right", state,
        PoseTarget("world_tcp", np.asarray(candidate["T_world_tcp"])),
        scene_ref, CollisionPolicy(), seed=seed,
    )
    result = adapter.plan(request)
    plan = {
        "candidate_id": candidate_id,
        "stage": "home_to_final_grasp",
        "request": request.to_dict(),
        "result": result.to_dict(),
    }
    prior_ik = next(
        (item for item in report.get("ik_attempts", [])
         if item["candidate_id"] == candidate_id), None,
    )
    if prior_ik is not None:
        plan["prior_ik_success"] = bool(prior_ik["result"]["success"])
        if not plan["prior_ik_success"]:
            plan["prior_ik_failure"] = prior_ik["result"].get("failure")
            plan["note"] = "用户显式要求规划；先前 IK 筛选失败不阻止本次独立规划"
    if result.success and result.trajectory is not None:
        plan["tcp_path_world_m"] = tcp_path_for_positions(
            case, names, result.trajectory.positions_rad,
        )
    report.setdefault("candidate_plans", {})[candidate_id] = plan
    (case.output_dir / "pick_bowl_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return plan


__all__ = [
    "PickBowlCase", "prepare_pick_bowl_scene", "load_pick_bowl_case",
    "run_pick_bowl_planning", "tcp_path_for_positions",
    "summarize_ik_diagnostics", "plan_pick_bowl_candidate",
    "apply_microwave_joint_overrides", "evaluate_ik_requests",
]
