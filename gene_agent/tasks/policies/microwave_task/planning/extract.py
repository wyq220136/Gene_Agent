"""旧版微波炉取出到桌面放置的视觉规划适配层。

炉内关键位姿与桌面 TCP 均直接读取旧 profile。场景碰撞只按旧 Skill
阶段切换：完整 world、只保留 robot_body 的 world、以及 detach 后允许
接触刚放下物体的 world。这里的 attach 和 release 都是视觉状态事件，
规划成功不表示夹爪真实抓住或放稳物体。
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.curobo.scene import (
    export_collision_scene, filter_collision_snapshot,
)
from gene_agent.motion.model import JointState, PoseTarget
from gene_agent.motion import SynchronousPlanner
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.skills.library.transfer.planning.extraction import ExtractObjectSkill, ExtractionTargets
from gene_agent.skills.runtime.planning import SkillContext
from gene_agent.skills.library.transfer.planning.placement import PlaceObjectSkill, PlacementTargets

from gene_agent.tasks.policies.microwave_task.planning.attachment import build_bowl_attachment_snapshot
from gene_agent.scene.microwave_frame import estimate_legacy_microwave_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import extract_profile_targets, fixed_extract_targets, load_object_operation_profile, LEGACY_TABLE_CONFIG
from gene_agent.tasks.policies.microwave_task.planning.insert import _planner, _scene_ref
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase, tcp_path_for_positions


ROOT = Path(__file__).resolve().parents[5]


class LegacyPlaceLinearPlanner:
    """仅在最终桌面下降按旧 PlaceSkill 的非终端权重顺序重试。

    候选 pre-place 的线性筛选保持默认权重；只有最终 place 使用
    ``5, 2, 3, 1``。当前 MotionRequest 不携带该旧版数值，因此在应用
    边界调用公开的 raw 规划入口，再用适配器转换成统一 MotionResult。
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.linear_attempts: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def plan(self, request):
        if request.request_id != "place:place_linear":
            return self.adapter.plan(request)
        result = None
        for scale in (5.0, 2.0, 3.0, 1.0):
            raw = self.adapter.planner.plan_linear_tcp(
                request.start.joint_names, request.start.positions_rad,
                request.target.matrix, target_frame=request.target.frame,
                non_terminal_scale=scale,
            )
            result = self.adapter._convert(request, raw)
            self.linear_attempts.append({
                "request_id": request.request_id,
                "non_terminal_scale": scale,
                "result": result.to_dict() if hasattr(result, "to_dict") else {"success": result.success},
            })
            if result.success:
                break
        return result


def table_tcp_after_scene_edit(case: PickBowlCase, target: PoseTarget) -> PoseTarget:
    """把旧世界桌面 TCP 随用户桌面摆位刚性移动，不改 profile 数值。

    无编辑时旧 virtual_table 的中心与姿态就是基准，结果逐项等于原
    profile。尺寸变化不会重新标定 TCP；仅桌面 pose 编辑改变目标。
    """
    legacy = json.loads(LEGACY_TABLE_CONFIG.read_text(encoding="utf-8"))["legacy_virtual_table"]
    old = np.eye(4)
    old[:3, 3] = legacy["center_world_m"]
    geom = case.data.geom("scene_table")
    new = np.eye(4)
    new[:3, :3] = geom.xmat.reshape(3, 3)
    new[:3, 3] = geom.xpos
    return PoseTarget("world_tcp", validate_pose(new @ np.linalg.inv(old) @ target.matrix,
                                                  field_name="edited_table_tcp"))


def released_or_calibrated_object_pose(
    world_legacy_microwave: np.ndarray,
    profile: dict,
    actual_insert_release: np.ndarray | None,
) -> np.ndarray:
    """有连续放入结果时使用实际 detach pose；独立取出才用旧标定位姿。"""
    if actual_insert_release is not None:
        return validate_pose(actual_insert_release, field_name="actual_insert_release")
    local = validate_pose(
        profile["microwave_from_placed_object"],
        field_name="microwave_from_placed_object",
    )
    return validate_pose(
        validate_pose(world_legacy_microwave, field_name="world_legacy_microwave") @ local,
        field_name="world_object",
    )


def extract_start_from_insert(
    insert_report: dict, *, scene_revision: int,
) -> tuple[JointState, np.ndarray]:
    """连续任务从实际放入撤离终点出发，防止跨场景或标称 pose 冒充状态。"""
    if (not insert_report.get("success") or insert_report.get("instance_id") != "bowl"
            or insert_report.get("scene_revision") != scene_revision):
        raise ValueError("连续取出需要同一 revision 的成功 Bowl 放入规划")
    pose = insert_report.get("actual_released_object_world")
    if pose is None:
        raise ValueError("放入报告缺少实际 release pose")
    stage = next((row for row in insert_report.get("stages", [])
                  if row.get("name") == "retreat_2"), None)
    if stage is None or stage.get("status") != "planned":
        raise ValueError("放入报告缺少实际 retreat_2 终点")
    goal = stage["result"]["goal_state"]
    return (JointState(tuple(goal["joint_names"]), tuple(goal["positions_rad"])),
            validate_pose(pose, field_name="actual_released_object_world"))


def prepare_legacy_table_case(
    source: PickBowlCase, output_dir: Path,
) -> tuple[PickBowlCase, dict]:
    """克隆同一 MuJoCo 场景，把旧桌面几何同时用于视觉和碰撞导出。

    新场景默认桌面比旧 ``virtual_table`` 高约 8.6 cm，且 XY 中心与面积
    不同；直接把旧桌面 TCP 用于新桌面会在 detach 后造成起点穿透。
    克隆后的 MJB 保留修正的桌面 visual/geom，revision 加一；原场景、
    放入报告及原 XML 均保持不变。后续碰撞快照从这份克隆模型导出。
    """
    config = json.loads(LEGACY_TABLE_CONFIG.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("不支持的旧桌面几何配置版本")
    expected = config["source_scene_table"]
    legacy = config["legacy_virtual_table"]
    original = source.model.geom("scene_table")
    current_center = np.asarray(original.pos, dtype=float)
    current_size = 2.0 * np.asarray(original.size, dtype=float)
    if (not np.allclose(current_center, expected["center_world_m"], atol=1e-7)
            or not np.allclose(current_size, expected["size_xyz_m"], atol=1e-7)):
        raise ValueError("当前 MuJoCo 桌面与已审计的 task_1 源配置不符，拒绝套用修正")
    model = mujoco.MjModel.from_xml_path(source.metadata["xml_path"])
    table = model.geom("scene_table")
    table.pos[:] = np.asarray(legacy["center_world_m"], dtype=float)
    table.size[:] = np.asarray(legacy["size_xyz_m"], dtype=float) / 2.0
    data = mujoco.MjData(model)
    data.qpos[:] = source.data.qpos
    data.qvel[:] = source.data.qvel
    data.time = source.data.time
    mujoco.mj_forward(model, data)
    output = Path(output_dir).resolve() / "visual_extract"
    output.mkdir(parents=True, exist_ok=True)
    model_ref = output / "corrected_scene.mjb"
    mujoco.mj_saveModel(model, str(model_ref))
    qpos_ref = output / "corrected_initial_qpos.json"
    facts = {
        "source_scene_revision": source.revision,
        "corrected_scene_revision": source.revision + 1,
        "config_ref": str(LEGACY_TABLE_CONFIG),
        "model_ref": str(model_ref),
        "initial_qpos_ref": str(qpos_ref),
        "source_table_center_world_m": current_center.tolist(),
        "source_table_size_xyz_m": current_size.tolist(),
        "source_table_top_z_m": float(current_center[2] + current_size[2] / 2.0),
        "legacy_table_center_world_m": legacy["center_world_m"],
        "legacy_table_size_xyz_m": legacy["size_xyz_m"],
        "legacy_table_top_z_m": float(
            legacy["center_world_m"][2] + legacy["size_xyz_m"][2] / 2.0
        ),
        "legacy_source": config["source"],
    }
    return replace(source, revision=source.revision + 1, model=model, data=data), facts


def apply_legacy_table_geometry_in_place(case: PickBowlCase, output_dir: Path) -> dict:
    """在 Task1 首阶段前对共享 model/data 修正桌面，并立刻重导快照。

    后续门、旋钮、取放复用同一 model 对象与修正后的桌面；不能在取出
    中途才克隆或移动桌面。调用方应随后重新采样 Task1 initial_state。
    """
    config = json.loads(LEGACY_TABLE_CONFIG.read_text(encoding="utf-8"))
    expected = config["source_scene_table"]
    legacy = config["legacy_virtual_table"]
    table = case.model.geom("scene_table")
    center = np.asarray(table.pos, dtype=float)
    size = 2.0 * np.asarray(table.size, dtype=float)
    legacy_center = np.asarray(legacy["center_world_m"], dtype=float)
    legacy_size = np.asarray(legacy["size_xyz_m"], dtype=float)
    already = (np.allclose(center, legacy_center, atol=1e-7)
               and np.allclose(size, legacy_size, atol=1e-7))
    if not already and (not np.allclose(center, expected["center_world_m"], atol=1e-7)
                        or not np.allclose(size, expected["size_xyz_m"], atol=1e-7)):
        raise ValueError("当前 Task1 桌面与源配置或旧配置均不符，拒绝原位修正")
    if not already:
        table.pos[:] = legacy_center
        table.size[:] = legacy_size / 2.0
        case.revision += 1
    mujoco.mj_forward(case.model, case.data)
    case.snapshot = export_collision_scene(
        case.model, case.data, case.metadata["robot"], "right",
        Path(output_dir) / "task1_collision/shared_legacy_table",
    )
    return {
        "status": ("legacy_table_already_in_shared_task_scene" if already else
                   "legacy_table_applied_to_shared_task_scene"),
        "scene_revision": case.revision,
        "model_identity_preserved": True,
        "source_table_center_world_m": expected["center_world_m"],
        "source_table_size_xyz_m": expected["size_xyz_m"],
        "legacy_table_center_world_m": legacy["center_world_m"],
        "legacy_table_size_xyz_m": legacy["size_xyz_m"],
        "collision_snapshot_sha256": case.snapshot["snapshot_sha256"],
        "config_ref": str(LEGACY_TABLE_CONFIG),
    }


def state_after_extract_report(start, report: dict):
    """只从实际 release 与最终 retreat 的关节末帧构造 Task1 交接状态。"""
    if not report.get("success") or start.attachments:
        raise ValueError("取出必须成功规划且起始 Bowl 未附着")
    extract_rows = {row["name"]: row for row in report["extract"]["stages"]}
    place_rows = {row["name"]: row for row in report["place"]["stages"]}
    if any(extract_rows[name]["status"] != "planned" for name in ("attach", "post_extract_2")):
        raise ValueError("取出报告缺少实际 attach 与撤离")
    if any(place_rows[name]["status"] != "planned" for name in ("release", "detach", "retreat")):
        raise ValueError("放置报告缺少实际 release、detach 与撤离")
    goal = place_rows["retreat"]["result"]["goal_state"]
    if tuple(goal["joint_names"]) != start.joint_names:
        raise ValueError("取出终点机器人关节顺序不同")
    poses = dict(start.object_poses_world)
    if "bowl" not in poses:
        raise ValueError("当前 Task1 场景缺少 Bowl")
    poses["bowl"] = validate_pose(
        report["actual_released_object_world"], field_name="actual_released_object_world",
    )
    return replace(
        start,
        joint_positions_rad=tuple(float(x) for x in goal["positions_rad"]),
        gripper_width_m=float(place_rows["release"]["details"]["width_m"]),
        object_poses_world=poses, attachments=(),
    )


def save_extraction_initial_state(data, output_dir: Path) -> Path:
    """保存已放入炉内的 Bowl 和连续机器人起点，而非源场景桌面 Bowl。"""
    qpos = np.asarray(data.qpos, dtype=float)
    if qpos.ndim != 1 or not np.all(np.isfinite(qpos)):
        raise ValueError("视觉起点 qpos 必须是一维有限数组")
    path = Path(output_dir) / "visual_extract/corrected_initial_qpos.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qpos.tolist(), allow_nan=False) + "\n", encoding="utf-8")
    return path


def snapshot_current_task_case(case: PickBowlCase, output_dir: Path) -> dict:
    """在 gripper open 事件前冻结 Task1 实际 qpos 与完整碰撞 world。"""
    snapshot = export_collision_scene(
        case.model, case.data, case.metadata["robot"], "right",
        Path(output_dir) / "visual_extract/current_world",
    )
    case.snapshot = snapshot
    return snapshot


def robot_body_only_snapshot(
    source: dict, output_dir: Path, *, attached_object_id: str | None = None,
) -> dict:
    """实现旧 ``ignore_world=True``：只忽略普通物体，保留 robot_body。"""
    return filter_collision_snapshot(
        source, output_dir, keep_body_names={"robot_body"},
        rejected_body_names=None, attached_object_id=attached_object_id,
        scope="旧视觉 Skill 忽略普通场景，仅保留 robot_body；保留机器人自碰撞",
    )


def allow_placed_object_snapshot(
    source: dict, output_dir: Path, *, object_body_name: str,
) -> dict:
    """detach 撤离时只排除新放下物体，保留桌面、炉体和 robot_body。"""
    if source.get("collision_mode") != "robot_only" and not any(row["body_name"] == object_body_name for row in source["records"]):
        raise ValueError(f"完整场景没有待允许接触物体 {object_body_name}")
    return filter_collision_snapshot(
        source, output_dir, keep_body_names=None,
        rejected_body_names={object_body_name}, attached_object_id=None,
        scope=f"完整场景；仅允许接触已放下物体 {object_body_name}",
    )


def _frozen_data(
    case: PickBowlCase, *, arm: JointState, bowl_pose: np.ndarray,
    gripper_width_m: float,
):
    """只复制 GT 状态供本阶段导出；源 run 的 MuJoCo data 不被规划修改。"""
    data = mujoco.MjData(case.model)
    mujoco.mj_copyData(data, case.model, case.data)
    for name, value in zip(arm.joint_names, arm.positions_rad):
        data.qpos[case.model.joint(name).qposadr[0]] = float(value)
    gripper = case.metadata["robot"]["grippers"]["right"]
    for name in gripper["joint_names"]:
        data.qpos[case.model.joint(name).qposadr[0]] = gripper_width_m / 2.0
    free = case.model.joint("bowl_free")
    pose = validate_pose(bowl_pose, field_name="bowl_pose")
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    address = free.qposadr[0]
    data.qpos[address:address + 3] = pose[:3, 3]
    data.qpos[address + 3:address + 7] = (q[3], q[0], q[1], q[2])
    mujoco.mj_forward(case.model, data)
    return data


def _world_legacy_frame(case: PickBowlCase) -> tuple[np.ndarray, dict]:
    closed = mujoco.MjData(case.model)
    mujoco.mj_copyData(closed, case.model, case.data)
    closed.qpos[case.model.joint("microwave_door_joint").qposadr[0]] = 0.0
    mujoco.mj_forward(case.model, closed)
    alignment = estimate_legacy_microwave_frame(case.model, closed, ROOT / "data/task_1")
    body = case.data.body("microwave")
    world_calibrated = np.eye(4)
    world_calibrated[:3, :3] = body.xmat.reshape(3, 3)
    world_calibrated[:3, 3] = body.xpos
    return world_calibrated @ alignment.T_calibrated_legacy, {
        "status": "experimental_visual_alignment_not_real_calibration",
        "T_calibrated_legacy": alignment.T_calibrated_legacy.tolist(),
        "handle_center_residual_m": alignment.handle_center_residual_m,
        "door_size_max_residual_m": alignment.door_size_max_residual_m,
        "source_profile": str(ROOT / "data/task_1/knowledge/microwave_object_manipulation.json"),
    }


def _annotate_stages(
    case: PickBowlCase, stages: list[dict], snapshot_refs: dict[str, str],
    *, attachment_ref: str | None, object_pose_world: np.ndarray | None,
) -> None:
    for stage in stages:
        name = stage["name"]
        if name in snapshot_refs:
            stage["collision_snapshot_ref"] = snapshot_refs[name]
        if attachment_ref is not None and name in {
            "attach", "post_extract_1", "post_extract_2", "preplace", "place",
        }:
            stage["attachment_ref"] = attachment_ref
        if object_pose_world is not None and name in {"release", "detach", "retreat"}:
            stage["object_pose_world"] = object_pose_world.tolist()
        trajectory = ((stage.get("result") or {}).get("trajectory") or {})
        if trajectory.get("positions_rad"):
            stage["tcp_path_world_m"] = tcp_path_for_positions(
                case, trajectory["joint_names"], trajectory["positions_rad"],
            )


def run_extract_bowl_skill_planning(
    case: PickBowlCase, *, seed: int = 123,
    object_id: str = "Bowl_0", output_dir: Path | None = None,
    insert_report: dict | None = None,
    current_scene: bool = False,
    stop_after_extract: bool = False,
    follow_table_edits: bool = False,
    pool: CuroboPlannerPool | None = None,
) -> dict:
    """从旧固定五点取 Bowl，可在 attachment 撤离后停止或继续桌面放置。

    ``current_scene=True`` 是 Task1 在线入口：直接使用共享 model/data 的
    当前机器人、门、旋钮、Bowl pose，不克隆模型或沿用旧 insert 的关节
    状态。独立探针仍可从同一 run 的放入报告继续，并创建旧桌面克隆。
    当前场景只有 Bowl mesh；Plate 仅能做 profile 数值校验。
    """
    if pool is None:
        with CuroboPlannerPool(planner_factory=SynchronousPlanner) as owned:
            return run_extract_bowl_skill_planning(
                case, seed=seed, object_id=object_id, output_dir=output_dir,
                insert_report=insert_report, current_scene=current_scene,
                stop_after_extract=stop_after_extract, follow_table_edits=follow_table_edits,
                pool=owned,
            )
    if object_id != "Bowl_0":
        raise ValueError("当前 MuJoCo 场景与附着球仅提供 Bowl，Plate 缺少资产")
    output = Path(output_dir if output_dir is not None else case.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_revision = case.revision
    if current_scene:
        # Task1 前序 Skill 已在同一 model 中完成；此处只保存当前显示模型，
        # 绝不于中途替换桌面或将机器人关节重设为旧 insert 的终点。
        visual_dir = output / "visual_extract"
        visual_dir.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(case.model, str(visual_dir / "current_scene.mjb"))
        model_ref = "visual_extract/current_scene.mjb"
        table = case.model.geom("scene_table")
        table_alignment = {
            "status": "existing_shared_task_scene",
            "table_center_world_m": np.asarray(table.pos, dtype=float).tolist(),
            "table_size_xyz_m": (2.0 * np.asarray(table.size, dtype=float)).tolist(),
            "config_ref": str(LEGACY_TABLE_CONFIG),
        }
        save_extraction_initial_state(case.data, output)
        current_snapshot = snapshot_current_task_case(case, output)
    else:
        case, table_alignment = prepare_legacy_table_case(case, output)
        model_ref = "visual_extract/corrected_scene.mjb"
        current_snapshot = None
    robot = case.metadata["robot"]
    profile = load_object_operation_profile(ROOT / "data/task_1", object_id)
    world_legacy, alignment = _world_legacy_frame(case)
    targets, table_tcp = extract_profile_targets(world_legacy, profile)
    if follow_table_edits:
        table_tcp = table_tcp_after_scene_edit(case, table_tcp)
    prior_path = case.output_dir / "skill_insert_bowl_result.json"
    if not current_scene and insert_report is None and prior_path.exists():
        insert_report = json.loads(prior_path.read_text(encoding="utf-8"))
    if current_scene:
        if insert_report is not None:
            raise ValueError("Task1 当前场景取出不能重新注入旧 insert 报告")
        names = tuple(robot["joint_groups"]["right"])
        start = JointState(names, tuple(
            float(case.data.qpos[case.model.joint(name).qposadr[0]]) for name in names
        ))
        body = case.data.body("bowl")
        object_pose = np.eye(4)
        object_pose[:3, :3] = body.xmat.reshape(3, 3)
        object_pose[:3, 3] = body.xpos
        source_pose = "current_task_scene"
        inserted_pose = None
    elif insert_report is None:
        names = tuple(robot["joint_groups"]["right"])
        start = JointState(names, tuple(
            float(case.data.qpos[case.model.joint(name).qposadr[0]]) for name in names
        ))
        source_pose = "legacy_calibrated_independent_extract"
        inserted_pose = None
    else:
        start, inserted_pose = extract_start_from_insert(
            insert_report, scene_revision=source_revision,
        )
        source_pose = "actual_prior_insert_release"
    if not current_scene:
        object_pose = released_or_calibrated_object_pose(world_legacy, profile, inserted_pose)
    open_width = float(profile["grasp_pregrasp_opening_width_m"])
    close_width = float(profile["grasp_opening_width_m"])
    frozen = _frozen_data(case, arm=start, bowl_pose=object_pose,
                          gripper_width_m=open_width)
    if not current_scene:
        save_extraction_initial_state(frozen, output)
    current_case = replace(case, data=frozen)
    free_world = export_collision_scene(
        case.model, frozen, robot, "right", output / "visual_extract/free_world",
    )
    free_body = robot_body_only_snapshot(
        free_world, output / "visual_extract/free_robot_body",
    )
    free_planner = _planner(robot, free_world, sphere_link_m=None,
                            seed=seed, revision=case.revision + 1, pool=pool)
    free_ignore = _planner(robot, free_body, sphere_link_m=None,
                           seed=seed, revision=case.revision + 2, pool=pool)
    skill = ExtractObjectSkill()
    extracted = skill.plan_grasp(
        free_planner, free_ignore,
        SkillContext("right", start, _scene_ref(free_world, case.revision + 1),
                     "bowl", seed),
        _scene_ref(free_body, case.revision + 2), targets,
        pregrasp_width_m=open_width, grasp_width_m=close_width,
    )
    attachment = None
    placed = None
    attached_body = None
    detached_allowed = None
    actual_release = None
    place_retry_attempts = []
    if extracted.stage("grasp").status == "planned":
        grasp_end = extracted.stage("grasp").result.goal_state
        grasp_tcp = free_ignore.fk(grasp_end)
        attachment = build_bowl_attachment_snapshot(
            current_case, grasp_end, grasp_tcp, close_width_m=close_width,
            output_dir=output / "visual_extract/attached_world",
        )
        attached_body = robot_body_only_snapshot(
            attachment["snapshot"], output / "visual_extract/attached_robot_body",
            attached_object_id="bowl",
        )
        attached_planner = _planner(
            robot, attached_body, sphere_link_m=attachment["spheres_link_m"],
            seed=seed, revision=case.revision + 3, pool=pool,
        )
        extracted = skill.plan_attached_retreat(
            attached_planner, extracted,
            SkillContext("right", grasp_end,
                         _scene_ref(attached_body, case.revision + 3), "bowl", seed),
            targets, grasp_width_m=close_width, local_pose_fallback=True,
        )
        if not stop_after_extract and extracted.stage("post_extract_2").status == "planned":
            tcp_object = np.asarray(attachment["T_tcp_bowl"], dtype=float)
            place_targets = PlacementTargets.from_attachment(table_tcp, tcp_object)
            place_backend = LegacyPlaceLinearPlanner(attached_planner)
            place_skill = PlaceObjectSkill()
            placed = place_skill.plan_attached(
                place_backend,
                SkillContext("right", extracted.stage("post_extract_2").result.goal_state,
                             _scene_ref(attached_body, case.revision + 3), "bowl", seed),
                _scene_ref(attached_body, case.revision + 3), place_targets,
            )
            place_retry_attempts = place_backend.linear_attempts
            if placed.stage("place").status == "planned":
                place_end = placed.stage("place").result.goal_state
                actual_release = attached_planner.fk(place_end) @ tcp_object
                release_width = float(robot["grippers"]["right"]["width_range_m"][1])
                detached_data = _frozen_data(
                    case, arm=place_end, bowl_pose=actual_release,
                    gripper_width_m=release_width,
                )
                detached_world = export_collision_scene(
                    case.model, detached_data, robot, "right",
                    output / "visual_extract/detached_world",
                )
                detached_allowed = allow_placed_object_snapshot(
                    detached_world, output / "visual_extract/detached_allow_bowl",
                    object_body_name="bowl",
                )
                detached_planner = _planner(
                    robot, detached_allowed, sphere_link_m=None,
                    seed=seed, revision=case.revision + 4, pool=pool,
                )
                placed = place_skill.plan_visual_release_and_retreat(
                    detached_planner, placed,
                    SkillContext("right", place_end,
                                 _scene_ref(detached_allowed, case.revision + 4),
                                 "bowl", seed),
                    place_targets, release_width_m=release_width,
                )
    report = {
        "schema_version": 1,
        "task": ("extract_bowl_skill_planning" if stop_after_extract else
                 "extract_bowl_to_table_skill_planning"),
        "instance_id": "bowl",
        "object_profile_id": object_id,
        "scene_revision": case.revision,
        "source_scene_revision": source_revision,
        "table_alignment": table_alignment,
        "model_ref": model_ref,
        "initial_qpos_ref": "visual_extract/corrected_initial_qpos.json",
        "initial_collision_snapshot_sha256": (
            None if current_snapshot is None else current_snapshot["snapshot_sha256"]
        ),
        "source_pose": source_pose,
        "source_insert_report": str(prior_path) if insert_report is not None else None,
        "frame_alignment": alignment,
        "initial_object_world": object_pose.tolist(),
        "actual_released_object_world": None if actual_release is None else actual_release.tolist(),
        "T_tcp_bowl": None if attachment is None else attachment["T_tcp_bowl"],
        "target_table_tcp_world": table_tcp.matrix.tolist(),
        "visual_only": True,
        "planner_cache": {
            "pre_extract": free_planner.cache_evidence,
            "grasp": free_ignore.cache_evidence,
            "attached_retreat": None if attachment is None else attached_planner.cache_evidence,
            "detached_retreat": None if actual_release is None else detached_planner.cache_evidence,
        },
        "success": extracted.success and (
            stop_after_extract or (placed is not None and placed.success)),
        "extract": extracted.to_dict(),
        "place": None if placed is None else placed.to_dict(),
        "place_linear_retry_attempts": place_retry_attempts,
        "limitations": [
            "当前计划不表示真实夹持或物理放置成功。",
            "本适配层尚未覆盖旧版局部旋转 pose fallback 与多候选 pre-place 顺序筛选。",
            "旧 allowed_collision_links=(attached_object,) 尚未由新碰撞 API 表达。",
            "当前资产对齐为实验视觉映射，非真机标定。",
            ("Task1 复用当前共享 model/data；桌面必须在首阶段前对齐，"
             "本段不更改场景几何。" if current_scene else
             "桌面 visual、MuJoCo geom 与碰撞快照均使用本次克隆的旧 task_1 几何；源 run 的放入规划使用原桌面。"),
        ],
    }
    extract_refs = {
        "open": "visual_extract/free_world/collision_snapshot.json",
        "pre_extract_1": "visual_extract/free_world/collision_snapshot.json",
        "pre_extract_2": "visual_extract/free_robot_body/collision_snapshot.json",
        "grasp": "visual_extract/free_robot_body/collision_snapshot.json",
        "close": "visual_extract/free_robot_body/collision_snapshot.json",
    }
    if attached_body is not None:
        extract_refs.update({name: "visual_extract/attached_robot_body/collision_snapshot.json"
                             for name in ("attach", "post_extract_1", "post_extract_2")})
    _annotate_stages(
        case, report["extract"]["stages"], extract_refs,
        attachment_ref=(None if attachment is None else
                        "visual_extract/attached_world/collision_snapshot.json"),
        object_pose_world=None,
    )
    if report["place"] is not None:
        place_refs = {name: "visual_extract/attached_robot_body/collision_snapshot.json"
                      for name in ("preplace", "place")}
        if detached_allowed is not None:
            place_refs.update({name: "visual_extract/detached_allow_bowl/collision_snapshot.json"
                               for name in ("release", "detach", "retreat")})
        _annotate_stages(
            case, report["place"]["stages"], place_refs,
            attachment_ref="visual_extract/attached_world/collision_snapshot.json",
            object_pose_world=actual_release,
        )
    (output / "skill_extract_bowl_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_extract_bowl_from_current_case(
    case: PickBowlCase, incoming, *, seed: int,
    output_dir: Path | None = None,
):
    """Task1 在线状态入口：只在完整成功后提交同一 MuJoCo 场景终态。

    调用方可先 ``apply_task_state(case, incoming)``；这里仍再次验证实际
    ``case.data`` 与交接状态一致。失败仅产出报告，不把已规划的部分轨迹
    当作已经执行的状态。成功后重新导出完整 world，供下一次关门使用。
    """
    from gene_agent.tasks.policies.microwave_task.visual import TaskStepResult
    from gene_agent.visualization.microwave.catalog import apply_task_state, capture_task_state

    step = "extract_place_bowl"
    if incoming.attachments:
        return TaskStepResult(step, incoming, incoming, False, (), {
            "code": "incoming_attachment_present",
            "message": "炉内取出要求 Bowl 在场景中为独立物体",
        })
    actual = capture_task_state(case)
    discontinuity = incoming.discontinuity(actual)
    if discontinuity is not None:
        return TaskStepResult(step, incoming, incoming, False, (), {
            "code": "current_scene_discontinuity", "message": discontinuity,
        })
    try:
        report = run_extract_bowl_skill_planning(
            case, seed=seed, output_dir=output_dir, current_scene=True,
        )
    except Exception as error:
        return TaskStepResult(step, actual, actual, False, (), {
            "code": "planning_exception",
            "message": f"{type(error).__name__}: {error}",
        })
    report_root = Path(output_dir or case.output_dir).resolve()

    def viewer_row(skill: str, stage: dict) -> dict:
        """Task viewer 使用可直接读取的绝对引用；磁盘报告仍保留相对路径。"""
        row = {"skill": skill, **stage}
        for field in ("collision_snapshot_ref", "attachment_ref"):
            if row.get(field):
                path = Path(row[field])
                row[field] = str(path if path.is_absolute() else report_root / path)
        return row

    rows = tuple(
        viewer_row(skill, stage)
        for skill, part in (("extract_bowl", report.get("extract")),
                            ("place_bowl", report.get("place")))
        if part is not None
        for stage in part.get("stages", ())
    )
    if not report.get("success"):
        first = next((row for row in rows if row.get("status") == "failed"), {})
        failure = ((first.get("result") or {}).get("failure") or {})
        return TaskStepResult(step, actual, actual, False, rows, {
            "code": failure.get("kind") or "plan_incomplete",
            "message": failure.get("summary") or "取出或桌面放置未完成规划",
            "stage": first.get("name"),
            "skill": first.get("skill"),
            "report_ref": str(Path(output_dir or case.output_dir) /
                              "skill_extract_bowl_result.json"),
        })
    proposed = state_after_extract_report(actual, report)
    previous_qpos = case.data.qpos.copy()
    previous_snapshot = case.snapshot
    try:
        apply_task_state(case, proposed)
        end = capture_task_state(case)
        mismatch = proposed.discontinuity(end)
        if mismatch is not None:
            raise RuntimeError(f"取出提交后的 MuJoCo 状态不连续：{mismatch}")
        case.snapshot = export_collision_scene(
            case.model, case.data, case.metadata["robot"], "right",
            Path(output_dir or case.output_dir) / "task1_collision/after_extract",
        )
    except Exception:
        case.data.qpos[:] = previous_qpos
        mujoco.mj_forward(case.model, case.data)
        case.snapshot = previous_snapshot
        raise
    return TaskStepResult(step, actual, end, True, rows)


def extract_visual_run(report: dict, output_dir: Path):
    """将真实取放阶段转换成在线 Viser 共用 VisualRun，保留失败前缀。"""
    from gene_agent.visualization.viser_sim.model import VisualRun
    from gene_agent.visualization.microwave.bowl_session import object_visual_run

    output = Path(output_dir)
    extract = dict(report["extract"])
    extract.update(scene_revision=report["scene_revision"],
                   instance_id=report["instance_id"])
    first = object_visual_run(extract, "extract_bowl", output)
    stages = list(first.stages)
    if report.get("place") is not None and first.success:
        place = dict(report["place"])
        place.update(scene_revision=report["scene_revision"],
                     instance_id=report["instance_id"])
        second = object_visual_run(
            place, "place_bowl", output, initial_attachments=("bowl",),
        )
        stages.extend(second.stages)
    run = VisualRun("extract_place_bowl", int(report["scene_revision"]), tuple(stages))
    if run.success != bool(report.get("success")):
        raise RuntimeError("取放视觉阶段与规划报告总结果不一致")
    return run


def extract_visual_playback(report: dict, output_dir: Path, case: PickBowlCase):
    """从同一保存模型和完整 qpos 构造可任意 seek 的取放视觉时间轴。"""
    from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState

    output = Path(output_dir)
    run = extract_visual_run(report, output)
    model = mujoco.MjModel.from_binary_path(str(output / report["model_ref"]))
    qpos = np.asarray(json.loads((output / report["initial_qpos_ref"]).read_text(
        encoding="utf-8",
    )), dtype=float)
    if qpos.shape != (model.nq,):
        raise ValueError("取放视觉初态 qpos 与保存模型不匹配")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    robot = case.metadata["robot"]
    names = tuple(robot["joint_groups"]["right"])
    joints = np.asarray([data.qpos[model.joint(name).qposadr[0]] for name in names])
    gripper = robot["grippers"]["right"]["joint_names"]
    width = sum(float(data.qpos[model.joint(name).qposadr[0]]) for name in gripper)
    body = data.body("bowl")
    bowl = np.eye(4)
    bowl[:3, :3] = body.xmat.reshape(3, 3)
    bowl[:3, 3] = body.xpos
    articulation = {
        name: float(data.qpos[model.joint(name).qposadr[0]])
        for name in ("knob_0_joint", "knob_1_joint")
    }
    initial = VisualState(
        names, joints, width, {"bowl": bowl},
        float(data.qpos[model.joint("microwave_door_joint").qposadr[0]]),
        (), articulation,
    )
    scratch = mujoco.MjData(model)
    site_name = robot["tcp_sites"]["right"]

    def tcp_pose(joint_names: tuple[str, ...], positions: np.ndarray) -> np.ndarray:
        scratch.qpos[:] = qpos
        for name, position in zip(joint_names, positions, strict=True):
            scratch.qpos[model.joint(name).qposadr[0]] = float(position)
        mujoco.mj_forward(model, scratch)
        site = scratch.site(site_name)
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return pose

    return VisualPlayback(run, initial, tcp_pose)


__all__ = [
    "LegacyPlaceLinearPlanner",
    "extract_profile_targets", "table_tcp_after_scene_edit",
    "released_or_calibrated_object_pose",
    "extract_start_from_insert", "robot_body_only_snapshot",
    "allow_placed_object_snapshot", "prepare_legacy_table_case",
    "apply_legacy_table_geometry_in_place", "state_after_extract_report",
    "save_extraction_initial_state", "run_extract_bowl_skill_planning",
    "snapshot_current_task_case",
    "run_extract_bowl_from_current_case",
    "extract_visual_run", "extract_visual_playback",
]
