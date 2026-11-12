"""把 task_1 已验证的碗候选接到后端无关 PickSkill，并保存阶段事实。"""

from __future__ import annotations
from gene_agent.motion.curobo.scene import planner_scene_options
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode

import json
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np

from gene_agent.knowledge.geometry import load_concept, validate_pose
from gene_agent.knowledge.grasp.model import GraspProposal
from gene_agent.motion import (
    CuroboPlannerAdapter, JointState, PlannerMode, PlanningSceneRef,
    SynchronousPlanner, load_realman_motion_config,
)
from gene_agent.skills import PickSkill, SkillContext, load_grasp_settings
from gene_agent.skills.runtime.planning import SkillPlan
from gene_agent.simulation.planned_contact import validate_planned_target_contacts

from gene_agent.tasks.policies.microwave_task.planning.attachment import build_bowl_attachment_snapshot
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase, prepare_grasp_stage_snapshots, tcp_path_for_positions


ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class BowlGraspStageSelection:
    """同一 Bowl GT 状态下，视觉和在线物理共用的候选筛选事实。"""

    plan: SkillPlan
    adapter: CuroboPlannerAdapter
    context: SkillContext
    proposals: tuple[GraspProposal, ...]
    strict_snapshot: dict[str, Any]
    contact_snapshot: dict[str, Any]


def apply_approach_contact_postcheck(plan: SkillPlan, evidence: dict[str, Any]) -> SkillPlan:
    """把目标接触的几何复核写回阶段事实；失败时不继续闭爪。"""
    approach = plan.stage("approach")
    if approach.status != "planned":
        raise ValueError("只可复核已规划的 approach 阶段")
    passed = bool(evidence.get("success"))
    stages = tuple(
        replace(stage, status=("planned" if passed else "failed"),
                details={**stage.details,
                         "target_contact_postcheck": evidence,
                         **({} if passed else {"reason": "规划接近轨迹存在非允许部位与目标物接触"})})
        if stage.name == "approach" else
        replace(stage, status="not_executed")
        if stage.name == "close" and not passed else stage
        for stage in plan.stages
    )
    return replace(plan, stages=stages, attempts=plan.attempts + ({
        "candidate_id": plan.selected_candidate_id,
        "stage": "approach_contact_postcheck", "result": evidence,
    },))


def legacy_bowl_proposals(
    candidates: Sequence[dict[str, Any]], *,
    object_pose_world: np.ndarray | None = None,
) -> tuple[GraspProposal, ...]:
    """保留旧 TCP 数值和 Bowl 的全局方位优先级。

    旧 ``build_object_ring_grasps`` 不是按 azimuth 枚举顺序直接把候选交给
    IK；它以 Ring Concept 中心为基准，先偏好世界 -X，再偏好世界 -Y，
    同方位保持生成顺序。新 Knowledge 的 40 个 pose 数值已与旧版相同，
    因此这里只恢复旧排序，不重新生成抓取或微调操作参数。

    ``object_pose_world=None`` 仅供调用方已经排好序时转换数据结构；
    在线 task_1 路径必须传入当前 GT 物体 pose。
    """
    proposals = tuple(GraspProposal(
        candidate_id=item["candidate_id"], generator="ring", part_name="bowl_rim",
        tcp_pose_world=np.asarray(item["T_world_tcp"], dtype=float),
        closing_width_m=float(item["parameters"]["opening_width_m"]),
        parameters=item["parameters"],
    ) for item in candidates)
    if object_pose_world is None:
        return proposals
    object_pose = validate_pose(object_pose_world, field_name="object_pose_world")
    ring = load_concept(
        ROOT / "assets/objects/bowl_concept.json", part_name="bowl_rim",
    ).concept
    center = (object_pose @ ring.pose_in_object)[:3, 3]

    def old_side_key(item: GraspProposal, index: int) -> tuple[float, float, int]:
        delta = item.tcp_pose_world[:3, 3] - center
        if 'tilt_pivot_tcp_m' in item.parameters.get('explicit_overrides', ()):
            # 接触支点补偿会略改 TCP 方位，不能据此把同一碗沿位置的
            # 小倾角排到大倾角前面；使用原始 ring 方位，保留倾角生成顺序。
            azimuth = item.parameters['azimuth_rad']
            delta = (object_pose @ ring.pose_in_object)[:3, :3] @ np.array(
                [np.cos(azimuth), np.sin(azimuth), 0.])
        planar_delta = delta[:2]
        radius = float(np.linalg.norm(planar_delta))
        if radius <= 1e-9 or not np.isfinite(radius):
            return (float("inf"), float("inf"), index)
        angle = float(np.arctan2(-planar_delta[0], planar_delta[1]))
        targets = (np.pi / 2.0, np.pi)
        distances = tuple(
            abs((angle - target + np.pi) % (2.0 * np.pi) - np.pi)
            for target in targets
        )
        return (*distances, index)

    return tuple(item for index, item in sorted(
        enumerate(proposals), key=lambda pair: old_side_key(pair[1], pair[0]),
    ))


def base_side_bowl_proposals(
    candidates: Sequence[dict[str, Any]], *, object_pose_world: np.ndarray,
    base_position_world: np.ndarray, table_normal_world: np.ndarray | None = None,
    min_base_alignment: float = 0.0,
    min_table_angle_deg: float = 45.0,
    max_table_angle_deg: float = 70.0,
    preferred_table_angle_deg: float = 55.0,
    min_external_approach_alignment: float = 0.3,
    rim_radial_tolerance_m: float = 0.001,
    rim_height_tolerance_m: float = 0.002,
) -> tuple[GraspProposal, ...]:
    """筛选右臂基座侧、从外向内接近且接触点位于碗沿的候选。

    抓取角定义为 TCP +Z 接近轴与实际桌面平面的锐角，而不是与 Bowl
    中轴的夹角。后者会把近水平伸入碗内的姿态误认为合格钝角。
    """
    proposals = legacy_bowl_proposals(candidates)
    object_pose = validate_pose(object_pose_world, field_name="object_pose_world")
    base = np.asarray(base_position_world, dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("活动臂基座位置必须是有限三维向量")
    ring = load_concept(
        ROOT / "assets/objects/bowl_concept.json", part_name="bowl_rim",
    ).concept
    ring_world = object_pose @ ring.pose_in_object
    center = ring_world[:3, 3]
    toward_base = base[:2] - center[:2]
    norm = float(np.linalg.norm(toward_base))
    if norm <= 1e-9:
        raise ValueError("Bowl 中心与活动臂基座平面位置重合，无法定义近基座侧")
    toward_base /= norm

    normal = np.asarray(
        [0.0, 0.0, 1.0] if table_normal_world is None else table_normal_world,
        dtype=float,
    )
    normal_norm = float(np.linalg.norm(normal))
    if normal.shape != (3,) or not np.isfinite(normal).all() or normal_norm <= 1e-9:
        raise ValueError("桌面法向必须是有限非零三维向量")
    normal /= normal_norm

    lower = float(min_table_angle_deg)
    upper = float(max_table_angle_deg)
    preferred = float(preferred_table_angle_deg)
    minimum_alignment = float(min_base_alignment)
    minimum_external = float(min_external_approach_alignment)
    radial_tolerance = float(rim_radial_tolerance_m)
    height_tolerance = float(rim_height_tolerance_m)
    if (not np.isfinite([
            lower, upper, preferred, minimum_alignment, minimum_external,
            radial_tolerance, height_tolerance,
        ]).all()
            or not 0.0 < lower <= preferred <= upper < 90.0
            or not -1.0 <= minimum_alignment <= 1.0):
        raise ValueError("Bowl 桌面抓取角或近右臂基座侧阈值非法")
    if not 0.0 <= minimum_external <= 1.0:
        raise ValueError("Bowl 外侧接近阈值必须位于 [0, 1]")
    if radial_tolerance < 0.0 or height_tolerance < 0.0:
        raise ValueError("Bowl 碗沿容差不能为负")

    outer_radius = float(ring.outer_radius_m)
    inner_radius = float(ring.inner_radius_m)
    ring_height = float(ring.height_m)
    ring_axis = ring_world[:3, 2]

    def candidate_metrics(item: GraspProposal) -> dict[str, float]:
        """返回筛选与日志共享的抓取几何量。"""
        azimuth = float(item.parameters["azimuth_rad"])
        radial = ring_world[:3, :3] @ np.array(
            [np.cos(azimuth), np.sin(azimuth), 0.0], dtype=float,
        )
        radial_norm = float(np.linalg.norm(radial))
        approach = np.asarray(item.tcp_pose_world[:3, 2], dtype=float).copy()
        approach_norm = float(np.linalg.norm(approach))
        if (radial_norm <= 1e-9 or approach_norm <= 1e-9
                or not np.isfinite([radial_norm, approach_norm]).all()):
            return {key: float("nan") for key in (
                "base_alignment", "table_angle_deg", "external_approach_alignment",
                "contact_radius_m", "contact_height_m", "contact_tangent_error_m",
                "inner_pad_contact_radius_m", "outer_pad_contact_radius_m",
            )}
        radial /= radial_norm
        approach /= approach_norm
        planar = radial[:2]
        planar_norm = float(np.linalg.norm(planar))
        if planar_norm <= 1e-9:
            return {key: float("nan") for key in (
                "base_alignment", "table_angle_deg", "external_approach_alignment",
                "contact_radius_m", "contact_height_m", "contact_tangent_error_m",
                "inner_pad_contact_radius_m", "outer_pad_contact_radius_m",
            )}
        pivot = np.asarray(item.parameters.get("tilt_pivot_tcp_m", [0.0, 0.0, 0.0]),
                           dtype=float)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError("Bowl 候选接触支点必须是有限三维向量")
        contact = item.tcp_pose_world[:3, 3] + item.tcp_pose_world[:3, :3] @ pivot
        contact_delta = contact - center
        opening_axis = np.asarray(item.tcp_pose_world[:3, 0], dtype=float).copy()
        opening_axis /= np.linalg.norm(opening_axis)
        half_width = float(item.closing_width_m) / 2.0
        pad_contact_radii = sorted((
            float(np.dot(contact_delta - opening_axis * half_width, radial)),
            float(np.dot(contact_delta + opening_axis * half_width, radial)),
        ))
        tangent = np.cross(ring_axis, radial)
        tangent_norm = float(np.linalg.norm(tangent))
        tangent = tangent / tangent_norm if tangent_norm > 1e-9 else tangent
        return {
            "base_alignment": float(np.dot(planar / planar_norm, toward_base)),
            "table_angle_deg": float(np.rad2deg(np.arcsin(np.clip(
                abs(np.dot(approach, normal)), 0.0, 1.0,
            )))),
            "external_approach_alignment": float(-np.dot(approach, radial)),
            "contact_radius_m": float(np.dot(contact_delta, radial)),
            "contact_height_m": float(np.dot(contact_delta, ring_axis)),
            "contact_tangent_error_m": float(abs(np.dot(contact_delta, tangent))),
            # 用闭爪宽度投影两侧指垫接触面。合格抓取必须跨过碗沿
            # 内外边界，不能让两根指爪同时落在碗腔内或同时在碗外。
            "inner_pad_contact_radius_m": pad_contact_radii[0],
            "outer_pad_contact_radius_m": pad_contact_radii[1],
        }

    eligible: list[tuple[int, GraspProposal, dict[str, float]]] = []
    for index, item in enumerate(proposals):
        metrics = candidate_metrics(item)
        if (np.isfinite(list(metrics.values())).all()
                and metrics["base_alignment"] >= minimum_alignment - 1e-9
                and lower - 1e-9 <= metrics["table_angle_deg"] <= upper + 1e-9
                and metrics["external_approach_alignment"] >= minimum_external - 1e-9
                and inner_radius - radial_tolerance <= metrics["contact_radius_m"]
                <= outer_radius + radial_tolerance
                and metrics["inner_pad_contact_radius_m"] <= inner_radius + radial_tolerance
                and metrics["outer_pad_contact_radius_m"] >= outer_radius - radial_tolerance
                and abs(metrics["contact_height_m"]) <= ring_height / 2.0 + height_tolerance
                and metrics["contact_tangent_error_m"] <= radial_tolerance):
            eligible.append((index, item, metrics))
    if not eligible:
        raise ValueError("没有满足右臂基座侧、外侧碗沿和桌面角度约束的 Bowl 候选")

    return tuple(item for index, item, metrics in sorted(
        eligible,
        # 先选最靠近右臂肩部基座的碗沿位置；同方位优先约 55 度。
        key=lambda row: (
            -row[2]["base_alignment"],
            abs(row[2]["table_angle_deg"] - preferred),
            -row[2]["external_approach_alignment"],
            row[0],
        ),
    ))


def bowl_grasp_geometry_metrics(
    candidate: GraspProposal, *, object_pose_world: np.ndarray,
    base_position_world: np.ndarray, table_normal_world: np.ndarray,
) -> dict[str, float | bool]:
    """公开单候选几何证据，和筛选器使用相同定义。"""
    selected = base_side_bowl_proposals(
        [{
            "candidate_id": candidate.candidate_id,
            "T_world_tcp": candidate.tcp_pose_world.tolist(),
            "parameters": candidate.parameters,
        }],
        object_pose_world=object_pose_world,
        base_position_world=base_position_world,
        table_normal_world=table_normal_world,
        min_base_alignment=-1.0,
        min_table_angle_deg=1e-6,
        max_table_angle_deg=89.999999,
        preferred_table_angle_deg=45.0,
        min_external_approach_alignment=0.0,
        rim_radial_tolerance_m=1.0,
        rim_height_tolerance_m=1.0,
    )
    if not selected:
        raise ValueError("无法计算 Bowl 抓取几何证据")
    object_pose = validate_pose(object_pose_world, field_name="object_pose_world")
    ring = load_concept(ROOT / "assets/objects/bowl_concept.json", part_name="bowl_rim").concept
    ring_world = object_pose @ ring.pose_in_object
    center = ring_world[:3, 3]
    azimuth = float(candidate.parameters["azimuth_rad"])
    radial = ring_world[:3, :3] @ np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    radial /= np.linalg.norm(radial)
    approach = candidate.tcp_pose_world[:3, 2].copy()
    approach /= np.linalg.norm(approach)
    normal = np.asarray(table_normal_world, dtype=float).copy()
    normal /= np.linalg.norm(normal)
    base_delta = np.asarray(base_position_world, dtype=float)[:2] - center[:2]
    base_delta /= np.linalg.norm(base_delta)
    pivot = np.asarray(candidate.parameters.get("tilt_pivot_tcp_m", [0.0, 0.0, 0.0]))
    contact = candidate.tcp_pose_world[:3, 3] + candidate.tcp_pose_world[:3, :3] @ pivot
    delta = contact - center
    opening_axis = candidate.tcp_pose_world[:3, 0].copy()
    opening_axis /= np.linalg.norm(opening_axis)
    half_width = float(candidate.closing_width_m) / 2.0
    pad_contact_radii = sorted((
        float((delta - opening_axis * half_width) @ radial),
        float((delta + opening_axis * half_width) @ radial),
    ))
    ring_axis = ring_world[:3, 2]
    tangent = np.cross(ring_axis, radial)
    tangent /= np.linalg.norm(tangent)
    return {
        "base_alignment": float(np.dot(radial[:2] / np.linalg.norm(radial[:2]), base_delta)),
        "table_angle_deg": float(np.rad2deg(np.arcsin(np.clip(abs(approach @ normal), 0., 1.)))),
        "external_approach_alignment": float(-approach @ radial),
        "contact_radius_m": float(delta @ radial),
        "contact_height_m": float(delta @ ring_axis),
        "contact_tangent_error_m": float(abs(delta @ tangent)),
        "inner_pad_contact_radius_m": pad_contact_radii[0],
        "outer_pad_contact_radius_m": pad_contact_radii[1],
        "rim_straddled": bool(
            pad_contact_radii[0] <= float(ring.inner_radius_m) + 0.001
            and pad_contact_radii[1] >= float(ring.outer_radius_m) - 0.001
        ),
        "rim_inner_radius_m": float(ring.inner_radius_m),
        "rim_outer_radius_m": float(ring.outer_radius_m),
    }


def active_arm_base_reference(case: PickBowlCase, group: str = "right") -> tuple[str, np.ndarray]:
    """返回活动臂第一关节所在 body 的世界位置，即肩部/手臂基座。"""
    joint_names = tuple(case.metadata["robot"]["joint_groups"][group])
    if not joint_names:
        raise ValueError(f"机器人组 {group} 没有活动关节")
    joint = case.model.joint(joint_names[0])
    body_id = int(case.model.jnt_bodyid[joint.id])
    body_name = case.model.body(body_id).name
    position = np.asarray(case.data.xpos[body_id], dtype=float).copy()
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("活动臂肩部基座位置必须是有限三维向量")
    return body_name, position


def plan_bowl_grasp_stages(
    case: PickBowlCase, profile, *, seed: int = 123,
) -> BowlGraspStageSelection:
    """从当前 GT、实际开口快照按旧候选顺序筛 grasp 和 pregrasp。

    只规划到 approach，不执行机械臂，也不做视觉 attach。在线物理
    入口随后必须从执行后的真实状态重新规划各段并验证接触。
    """
    robot = case.metadata["robot"]
    names = tuple(robot["joint_groups"]["right"])
    start = JointState(names, tuple(
        float(case.data.qpos[case.model.joint(name).qposadr[0]]) for name in names
    ))
    _, base_position = active_arm_base_reference(case, "right")
    table = case.data.geom("scene_table")
    table_normal = np.asarray(table.xmat, dtype=float).reshape(3, 3)[:, 2].copy()
    proposals = base_side_bowl_proposals(
        case.candidates, object_pose_world=case.bowl_pose,
        base_position_world=base_position,
        table_normal_world=table_normal,
    )
    # Bowl 的当前 Knowledge 候选共享同一个 pregrasp 开口。若未来的
    # 候选各用不同开口，就必须分批冻结对应快照，不能用第一项代替全部。
    openings = {float(item.parameters.get("pregrasp_opening_width_m", profile.open_width_m))
                for item in proposals[:profile.max_grasp_candidates]}
    if len(openings) != 1:
        raise ValueError("Bowl 候选开口不一致，不能共用同一抓取碰撞快照")
    strict_snapshot, contact_snapshot = prepare_grasp_stage_snapshots(
        case, openings.pop(), group="right",
    )

    def stage_adapter(stage_snapshot: dict[str, Any]) -> CuroboPlannerAdapter:
        settings = load_realman_motion_config(
            "right", robot, gripper_snapshot=stage_snapshot.get("gripper_snapshot"),
        )
        planner = SynchronousPlanner(
            **settings, **planner_scene_options(stage_snapshot),
            planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed,
        )
        return CuroboPlannerAdapter(
            planner, group="right", scene_revision=case.revision,
        )

    adapter = stage_adapter(strict_snapshot)
    contact_adapter = stage_adapter(contact_snapshot)
    context = SkillContext(
        group="right", start=start,
        scene=PlanningSceneRef(
            case.revision, strict_snapshot["snapshot_sha256"], strict_snapshot["scope"],
        ),
        object_id="bowl", seed=seed,
    )
    contact_scene = PlanningSceneRef(
        case.revision, contact_snapshot["snapshot_sha256"], contact_snapshot["scope"],
    )
    result = PickSkill().plan(
        adapter, proposals, context, profile,
        contact_planner=contact_adapter, contact_scene=contact_scene,
    )
    return BowlGraspStageSelection(
        result, adapter, context, proposals, strict_snapshot, contact_snapshot,
    )


def run_pick_bowl_skill_planning(case: PickBowlCase, *, seed: int = 123) -> dict[str, Any]:
    """按旧 Pick 顺序规划抓碗、闭爪后 attach 和直线撤离。

    attach 是特权视觉状态转移，不代表接触力或物理夹持已验证。
    """
    snapshot = case.snapshot
    robot = case.metadata["robot"]
    profile_path = ROOT / "data/task_1/skills/skill_profiles.json"
    # 旧 Viser 的 50 mm pregrasp / 100 mm post-pick 属于纯视觉迁移配置。
    # 原 pick_bowl 配置仍供现有 MuJoCo 物理探针使用，避免改变已验证的实验。
    profile = load_grasp_settings(profile_path, "pick_bowl_legacy_visual")
    selected = plan_bowl_grasp_stages(case, profile, seed=seed)
    result = selected.plan
    adapter = selected.adapter
    context = selected.context
    proposals = selected.proposals
    strict_snapshot = selected.strict_snapshot
    contact_snapshot = selected.contact_snapshot
    skill = PickSkill()
    attachment = None
    approach = result.stage("approach")
    contact_postcheck = None
    if (approach.status == "planned" and approach.result is not None
            and approach.result.trajectory is not None
            and resolve_collision_mode() is CollisionMode.FULL):
        contact_postcheck = validate_planned_target_contacts(
            case.model, case.data, robot, group="right", object_body="bowl",
            trajectory=approach.result.trajectory,
            gripper_joint_positions_m=strict_snapshot["gripper_snapshot"]["joint_positions_m"],
            allowed_geom_names=robot["grippers"]["right"]["pad_geom_names"],
        )
        result = apply_approach_contact_postcheck(result, contact_postcheck)
        approach = result.stage("approach")
    if (approach.status == "planned" and approach.result is not None
            and approach.result.goal_state is not None
            and result.stage("close").status == "planned"):
        # 旧 Viser 在实际 approach 终点闭爪，然后才将 Bowl 从 world
        # 碰撞障碍转移为 link_7 上的 32 个小球。不能使用标称 grasp TCP
        # 反算 attach 位姿，否则 IK 误差会直接变成碗的视觉跳变。
        end_state = approach.result.goal_state
        actual_tcp = adapter.fk(end_state)
        attachment = build_bowl_attachment_snapshot(
            case, end_state, actual_tcp,
            close_width_m=float(result.stage("close").details["width_m"]),
            output_dir=case.output_dir / "visual_attachment",
        )
        attached_scene = attachment["snapshot"]
        attached_settings = load_realman_motion_config(
            "right", robot, gripper_snapshot=attached_scene.get("gripper_snapshot"),
        )
        attached_planner = SynchronousPlanner(
            **attached_settings,
            **planner_scene_options(attached_scene),
            attachment_spheres_tool=attachment["spheres_link_m"],
            planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed,
        )
        attached_adapter = CuroboPlannerAdapter(
            attached_planner, group="right", scene_revision=case.revision + 1,
        )
        attached_context = SkillContext(
            group="right", start=end_state,
            scene=PlanningSceneRef(
                case.revision + 1, attached_scene["snapshot_sha256"],
                attached_scene["scope"],
            ),
            object_id="bowl", seed=seed,
        )
        result = skill.plan_visual_retreat(
            attached_adapter, result, attached_context, profile,
        )
        (case.output_dir / "visual_attachment/attachment_state.json").write_text(
            json.dumps({
                "object_instance_id": "bowl",
                "T_tcp_object": attachment["T_tcp_bowl"],
                "collision_snapshot_ref": attachment["collision_snapshot_ref"],
                "privileged_visual": True,
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    report = {
        "schema_version": 1,
        "task": "pick_bowl_skill_planning",
        "instance_id": context.object_id,
        "scene_revision": case.revision,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "seed": seed,
        "candidate_source": "legacy_task_1_bowl_40",
        "candidate_order": [item.candidate_id for item in proposals],
        "skill_profile": str(profile_path),
        "skill_settings": asdict(profile),
        "stage_collision_snapshots": {
            name: {
                "collision_snapshot_ref": f"collision_grasp_open/{variant}/collision_snapshot.json",
                "snapshot_sha256": stage_snapshot["snapshot_sha256"],
                "gripper_width_m": stage_snapshot["gripper_snapshot"]["width_m"],
                "disabled_body_names": stage_snapshot["disabled_body_names"],
            }
            for name, variant, stage_snapshot in (
                ("pregrasp", "strict", strict_snapshot),
                ("approach", "target_contact", contact_snapshot),
            )
        },
        "approach_contact_postcheck": contact_postcheck,
        "visual_attachment": None if attachment is None else {
            "collision_snapshot_ref": attachment["collision_snapshot_ref"],
            "scene_revision": case.revision + 1,
            "snapshot_sha256": attachment["snapshot"]["snapshot_sha256"],
            "sphere_count": len(attachment["spheres_link_m"]),
            "T_tcp_bowl": attachment["T_tcp_bowl"],
            "T_world_tcp_at_attach": actual_tcp.tolist(),
            "gripper_width_m": attachment["gripper_width_m"],
            "privileged_visual": True,
        },
        **result.to_dict(),
    }
    for stage in report["stages"]:
        if stage["name"] == "pregrasp":
            stage["collision_snapshot_ref"] = report["stage_collision_snapshots"]["pregrasp"]["collision_snapshot_ref"]
        elif stage["name"] in {"approach", "close"}:
            stage["collision_snapshot_ref"] = report["stage_collision_snapshots"]["approach"]["collision_snapshot_ref"]
        trajectory = ((stage.get("result") or {}).get("trajectory") or {})
        if trajectory.get("positions_rad"):
            stage["tcp_path_world_m"] = tcp_path_for_positions(
                case, trajectory["joint_names"], trajectory["positions_rad"],
            )
    (case.output_dir / "skill_pick_bowl_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["legacy_bowl_proposals", "run_pick_bowl_skill_planning"]
