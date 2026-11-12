"""微波炉 Task1 适配器共用的 MuJoCo 状态边界。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.visualization.viser_sim.model import SceneEdits

from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase
from gene_agent.tasks.policies.microwave_task.visual import TaskState, TaskStepResult
from gene_agent.tasks.policies.microwave_task.operations.door import apply_selected_push_candidate, select_legacy_push_candidate


ROOT = Path(__file__).resolve().parents[3]


def _free_body_joints(model) -> dict[str, int]:
    """只取真正由 free joint 驱动的对象；门/旋钮由标量关节单列。"""
    result = {}
    for body_id in range(1, model.nbody):
        body = model.body(body_id)
        if int(body.jntnum[0]) != 1:
            continue
        joint_id = int(body.jntadr[0])
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        result[name] = joint_id
    return result


def _pose(data, name: str) -> np.ndarray:
    body = data.body(name)
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(body.xmat).reshape(3, 3)
    pose[:3, 3] = np.asarray(body.xpos)
    return pose


def _sync_runtime_table_metadata(case: PickBowlCase) -> None:
    """源 XML 被旧桌面修正后，使场景元数据指向实际可视化几何。"""
    table = case.model.geom("scene_table")
    world = case.data.geom("scene_table")
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(world.xmat).reshape(3, 3)
    pose[:3, 3] = np.asarray(world.xpos)
    objects = case.metadata["objects"]
    objects["sample_poses"]["table"] = pose.tolist()
    facts = objects["table"]
    facts["T_world_table"] = pose.tolist()
    facts["center_xy_m"] = pose[:2, 3].tolist()
    facts["size_xy_m"] = (2.0 * table.size[:2]).tolist()
    facts["thickness_m"] = float(2.0 * table.size[2])
    facts["top_z_m"] = float(pose[2, 3] + pose[2, 2] * table.size[2])


def _model_hash(case: PickBowlCase) -> str:
    return hashlib.sha256(Path(case.metadata["xml_path"]).read_bytes()).hexdigest()


def capture_task_state(case: PickBowlCase) -> TaskState:
    """从唯一场景采样 Task 边界，避免从标称 target 猜终点。"""
    model, data = case.model, case.data
    robot = case.metadata["robot"]
    names = tuple(robot["joint_groups"]["right"])
    gripper_names = tuple(robot["grippers"]["right"]["joint_names"])
    mujoco.mj_forward(model, data)
    return TaskState(
        model_sha256=_model_hash(case), joint_names=names,
        joint_positions_rad=tuple(float(data.qpos[model.joint(name).qposadr[0]])
                                  for name in names),
        gripper_width_m=sum(float(data.qpos[model.joint(name).qposadr[0]])
                            for name in gripper_names),
        door_angle_rad=float(data.qpos[model.joint("microwave_door_joint").qposadr[0]]),
        knob_angle_rad=float(data.qpos[model.joint("knob_0_joint").qposadr[0]]),
        object_poses_world={name: _pose(data, name) for name in _free_body_joints(model)},
    )


def apply_task_state(case: PickBowlCase, state: TaskState) -> None:
    """把上一 Skill 实际终态写回同一资产，不允许偷偷换场景或重置对象。"""
    model, data = case.model, case.data
    robot = case.metadata["robot"]
    if state.model_sha256 != _model_hash(case):
        raise ValueError("Task 状态的场景资产与当前 MuJoCo 模型不同")
    if state.joint_names != tuple(robot["joint_groups"]["right"]):
        raise ValueError("Task 状态的机器人关节顺序不同")
    free = _free_body_joints(model)
    if set(free) != set(state.object_poses_world):
        raise ValueError("Task 状态的自由物体集合不同")
    if state.attachments:
        raise ValueError("附着状态不能只通过 MuJoCo qpos 恢复；必须使用适配器碰撞快照")
    for name, value in zip(state.joint_names, state.joint_positions_rad, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    gripper_names = robot["grippers"]["right"]["joint_names"]
    for name in gripper_names:
        data.qpos[model.joint(name).qposadr[0]] = state.gripper_width_m / len(gripper_names)
    data.qpos[model.joint("microwave_door_joint").qposadr[0]] = state.door_angle_rad
    data.qpos[model.joint("knob_0_joint").qposadr[0]] = state.knob_angle_rad
    for name, joint_id in free.items():
        pose = state.object_poses_world[name]
        address = int(model.jnt_qposadr[joint_id])
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        data.qpos[address:address + 7] = [*pose[:3, 3], *quaternion]
    mujoco.mj_forward(model, data)


def door_partition_bounds(phases, part: str) -> tuple[int, int]:
    """按旧 DoorCycleTask 的 intermediate_task 边界拆开门会话。"""
    if part not in {"open", "close"}:
        raise ValueError("门分段只支持 open 或 close")
    names = tuple(phase.skill_id for phase in phases)
    if not names or "push_close" not in names:
        raise ValueError("门阶段缺少关门 Skill")
    boundary = names.index("push_close")
    if not ("handle_pull" in names[:boundary]
            and set(names[:boundary]) <= {"handle_pull", "push_open", "task_transition"}
            and set(names[boundary:]) == {"push_close"}):
        raise ValueError("门阶段顺序不符合旧 Task1")
    return (0, boundary) if part == "open" else (boundary, len(names))


def legacy_skipped_stage_two_phases(phases) -> tuple:
    """旧同步 Viser 配置 85° 已达到目标，故不执行内侧推门。

    旧 Task1 此时从把手的 clearance pose 直接进入物体 Task，也不运行
    常规推门分支中的回默认 TCP 过渡。关门 Knowledge 仍按 85° 起点生成。
    """
    names = tuple(phase.skill_id for phase in phases)
    if not ({"handle_pull", "push_open", "task_transition", "push_close"} <= set(names)):
        raise ValueError("旧阶段二跳过配置缺少完整原始门阶段")
    selected = tuple(phase for phase in phases
                     if phase.skill_id in {"handle_pull", "push_close"})
    door_partition_bounds(selected, "open")
    return selected


def _select_task1_push_phase(session) -> dict | None:
    """在真实上一阶段终态按旧同步流程为当前门角筛选 Knowledge 候选。"""
    from gene_agent.motion.curobo.core import load_realman_motion_config
    from gene_agent.motion.curobo.scene import export_collision_scene

    phase = session.phases[session.phase_index]
    if phase.skill_id not in {"push_open", "push_close"} or phase.name == "contact":
        return None
    if phase.name == "pre_contact":
        index = 0
    elif phase.name.startswith("push_"):
        index = int(phase.name.removeprefix("push_"))
    else:
        return None
    knowledge = session.knowledge[phase.skill_id]
    candidate_set = knowledge.door_candidate_sets[index]
    session._prepare_phase_scene(phase)
    case = session.case
    folder = (session.output_dir / f"revision_{session.revision:03d}"
              / f"phase_{session.phase_index:02d}" / "legacy_candidate_selection")
    # 旧可达性检查允许接触门板；这里沿用当前门 Skill 的动态门板
    # 碰撞表示，其余静态场景与真实夹爪开口照常导出。
    snapshot = export_collision_scene(
        case.model, case.data, case.metadata["robot"], session.group, folder,
        moving_body_names=("microwave_door",),
    )
    settings = load_realman_motion_config(
        session.group, case.metadata["robot"],
        gripper_snapshot=snapshot.get("gripper_snapshot"),
    )
    planner = session.planner_pool.acquire(
        settings, snapshot, **session.planner_options(),
    ).planner
    names = tuple(case.metadata["robot"]["joint_groups"][session.group])
    q = tuple(float(case.data.qpos[case.model.joint(name).qposadr[0]]) for name in names)
    chosen, facts = select_legacy_push_candidate(
        planner, names, q, candidate_set.candidates,
    )
    facts.update(
        phase_id=phase.phase_id, angle_deg=float(np.rad2deg(candidate_set.door_angle_rad)),
        collision_snapshot_ref=str(folder / "collision_snapshot.json"),
        snapshot_sha256=snapshot["snapshot_sha256"],
    )
    if chosen is None:
        return {"success": False, **facts}
    # 与现场门 Skill 共用纯目标替换规则；原候选及阶段顺序不变。
    session.phases = apply_selected_push_candidate(
        session.phases, session.knowledge[phase.skill_id], session.phase_index,
        chosen,
    )
    return {"success": True, **facts}


def run_door_partition(session, step: str, incoming: TaskState, *, part: str,
                       skipped_stage_two_angle_deg: float | None = None,
                       progressive_push_candidates: bool = False,
                       on_door_phase_complete: Callable[[object], None] | None = None) -> TaskStepResult:
    """复用门会话逐阶段 planner，并在其重置后注入上一 Skill 的实终态。

    公开 begin 会回到 home；本编排器只在 begin 后、首个 cuRobo 调用前
    写回完整状态，并同时修正会话用于失败恢复的初态。结束后读取门会话
    自身的真实轨迹末帧，不以目标关节角或标称 TCP 冒充规划终点。
    """
    first, last = door_partition_bounds(session.phases, part)
    first_row = len(session.results)
    session.begin_plan_sequence(None if part == "open" else "push_close")
    if session.phase_index != first:
        raise RuntimeError("门会话实际阶段起点与旧 Task1 边界不同")
    session.stop_index = last
    apply_task_state(session.case, incoming)
    actual_start = capture_task_state(session.case)
    session._sequence_state["initial_qpos"] = session.case.data.qpos.copy()
    session._sequence_state["total_stages"] = last - first
    report = {"done": False}
    while not report["done"]:
        selection = (_select_task1_push_phase(session)
                     if progressive_push_candidates else None)
        if selection is not None and not selection["success"]:
            session.cancel_plan_sequence()
            rows = tuple(session.results[first_row:]) + ({
                "phase_id": selection["phase_id"], "success": False,
                "operation": "knowledge_candidate_selection",
                "candidate_selection": selection,
            },)
            return TaskStepResult(step, actual_start, capture_task_state(session.case),
                                  False, rows, {
                "code": "goal_ik_no_solution",
                "message": "旧 Knowledge 当前门角所有候选 IK 不可达",
                "phase_id": selection["phase_id"],
                "candidate_selection": selection,
            })
        report = session.plan_sequence_step()
        if selection is not None:
            session.results[-1]["knowledge_candidate_selection"] = selection
        if on_door_phase_complete is not None:
            # 只读观察者须在下一轮候选筛选之前取得实际轨迹。候选全部
            # 无解时 cancel 会清空增量序列，不能等 partition 返回再冻结。
            on_door_phase_complete(session)
    if session.timeline is not None:
        session.seek_timeline(session.timeline.duration_s)
    rows = tuple(session.results[first_row:])
    if report["success"] and part == "open" and skipped_stage_two_angle_deg is not None:
        # 旧 Viser 把手实际只拉到 25°；阶段二由外部门状态切到 85°，
        # 不存在从 25° 到 85° 的机器人轨迹，因此另记场景事件。
        joint = session.case.model.joint("microwave_door_joint")
        address = joint.qposadr[0]
        before = float(session.case.data.qpos[address])
        after = float(np.deg2rad(skipped_stage_two_angle_deg))
        session.case.data.qpos[address] = after
        mujoco.mj_forward(session.case.model, session.case.data)
        rows += ({
            "phase_id": "task/stage_two_skipped/external_door_state_update",
            "success": True, "operation": "external_door_state_update",
            "from_angle_deg": float(np.rad2deg(before)),
            "to_angle_deg": float(skipped_stage_two_angle_deg),
        },)
    end = capture_task_state(session.case)
    failure = None
    if not report["success"]:
        original = report.get("first_failure") or {}
        failure = {
            "code": original.get("failure_category") or original.get("status") or "planning_failed",
            "message": original.get("message") or "门阶段规划失败",
            "phase_id": original.get("phase_id"),
        }
    return TaskStepResult(step, actual_start, end, bool(report["success"]), rows, failure)


def state_after_insert_report(start: TaskState, report: dict) -> TaskState:
    """从放入实际撤离轨迹和 release 事实建立关门起点。"""
    if not report.get("success"):
        raise ValueError("失败的放入报告没有可交接终态")
    rows = tuple(report.get("stages") or ())
    endpoint = next((row["result"]["goal_state"] for row in reversed(rows)
                     if (row.get("result") or {}).get("goal_state")), None)
    release = next((row for row in rows if row.get("name") == "release"), None)
    object_pose = report.get("actual_released_object_world")
    if endpoint is None or release is None or object_pose is None:
        raise ValueError("放入报告缺少撤离终点、release 或 Bowl 实际位姿")
    if tuple(endpoint["joint_names"]) != start.joint_names:
        raise ValueError("放入终点机器人关节顺序不同")
    poses = dict(start.object_poses_world)
    if "bowl" not in poses:
        raise ValueError("当前场景缺少 Bowl")
    poses["bowl"] = np.asarray(object_pose, dtype=float)
    return replace(
        start, joint_positions_rad=tuple(float(value) for value in endpoint["positions_rad"]),
        gripper_width_m=float(release["details"]["width_m"]),
        object_poses_world=poses, attachments=(),
    )


def _skill_failure(report: dict, *, default_code: str) -> dict:
    failed = next((row for row in report.get("stages", ())
                   if row.get("status") == "failed"), {})
    result = failed.get("result") or {}
    error = result.get("failure") or {}
    return {
        "code": error.get("kind") or default_code,
        "message": error.get("summary") or report.get("message") or "Skill 规划失败",
        "stage": failed.get("name"),
    }


def run_bowl_insert_partition(case: PickBowlCase, incoming: TaskState,
                              *, seed: int) -> TaskStepResult:
    """在开门终态场景规划 Pick→Insert，并只从 release 后实轨迹交接。"""
    from gene_agent.knowledge import bowl_grasp_candidates
    from gene_agent.motion.curobo.scene import export_collision_scene
    from gene_agent.tasks.policies.microwave_task.planning.grasp import run_pick_bowl_skill_planning
    from gene_agent.tasks.policies.microwave_task.planning.insert import run_insert_bowl_skill_planning

    apply_task_state(case, incoming)
    start = capture_task_state(case)
    snapshot = export_collision_scene(
        case.model, case.data, case.metadata["robot"], "right",
        case.output_dir / "task1_collision" / "before_insert",
    )
    bowl_pose = _pose(case.data, "bowl")
    active = replace(
        case, revision=case.revision + 1, snapshot=snapshot, bowl_pose=bowl_pose,
        candidates=bowl_grasp_candidates(bowl_pose, pose_source="mujoco_gt"),
    )
    pick = run_pick_bowl_skill_planning(active, seed=seed)
    rows = tuple({"skill": "pick_bowl", **row} for row in pick.get("stages", ()))
    if not pick.get("success"):
        return TaskStepResult("insert_bowl", start, start, False, rows,
                              _skill_failure(pick, default_code="pick_failed"))
    insert = run_insert_bowl_skill_planning(active, seed=seed)
    rows += tuple({"skill": "insert_bowl", **row} for row in insert.get("stages", ()))
    if not insert.get("success"):
        return TaskStepResult("insert_bowl", start, start, False, rows,
                              _skill_failure(insert, default_code="insert_failed"))
    end = state_after_insert_report(start, insert)
    apply_task_state(case, end)
    return TaskStepResult("insert_bowl", start, capture_task_state(case), True, rows)


def run_rotate_knob_partition(case: PickBowlCase, incoming: TaskState,
                              *, output_dir: str | Path,
                              task_path: str | Path = ROOT / "data/task_1/skills/knob_0_task.json") -> TaskStepResult:
    """在同一 MuJoCo 场景中接上旧旋钮 Skill，保留实际门角和 Bowl 位姿。"""
    from gene_agent.tasks.policies.microwave_task.planning.knob import run_rotate_knob_0_planning

    if incoming.attachments:
        return TaskStepResult("rotate_knob", incoming, incoming, False, (), {
            "code": "attached_object_unavailable",
            "message": "旋钮 Task 边界仍有附着物；旧流程要求 Bowl 已 release/detach",
        })
    apply_task_state(case, incoming)
    start = capture_task_state(case)
    report = run_rotate_knob_0_planning(
        output_dir, task_path=task_path, case=case,
    )
    rows = tuple(report.get("stages") or ())
    failure = None
    if not report.get("success"):
        failed = next((row for row in rows if row.get("status") == "failed"), {})
        result_failure = ((failed.get("result") or {}).get("failure") or {})
        failure = {
            "code": result_failure.get("kind") or "knob_" + str(
                report.get("failure_stage") or failed.get("name") or "planning_failed"),
            "message": (result_failure.get("summary") or report.get("failure_reason")
                        or failed.get("axis_path_failure") or "旋钮 Skill 规划失败"),
            "stage": report.get("failure_stage") or failed.get("name"),
            "candidate_count": report.get("candidate_count"),
            "grasp_reachable_count": report.get("grasp_reachable_count"),
        }
    return TaskStepResult("rotate_knob", start, capture_task_state(case),
                          bool(report.get("success")), rows, failure)


def run_extract_partition(case: PickBowlCase, incoming: TaskState,
                          *, seed: int, output_dir: Path) -> TaskStepResult:
    """先把上一段实终态写入共享场景，再运行同模型的取出与放桌面。"""
    from gene_agent.tasks.policies.microwave_task.planning.extract import run_extract_bowl_from_current_case

    if incoming.attachments:
        return TaskStepResult("extract_place_bowl", incoming, incoming, False, (), {
            "code": "incoming_attachment_present",
            "message": "取出前 Bowl 必须已解除附着",
        })
    apply_task_state(case, incoming)
    return run_extract_bowl_from_current_case(
        case, incoming, seed=seed, output_dir=output_dir,
    )


@dataclass(frozen=True)
class Task1Catalog:
    """同一 MuJoCo 资产上的阶段适配器和明确未满足的注入边界。"""

    case: PickBowlCase
    door_session: object
    initial_state: TaskState
    adapters: dict[str, Callable[[TaskState], TaskStepResult]]
    output_dir: Path
    configuration: dict


def _unsupported_adapter(step: str, entry: str, reason: str):
    def run(start: TaskState) -> TaskStepResult:
        return TaskStepResult(step, start, start, False, (), {
            "code": "stateful_adapter_unavailable",
            "message": f"{entry} 尚不能从上一 Skill 的完整状态规划：{reason}",
        })
    return run


def build_task1_catalog(output_dir: str | Path, *,
                        variant: str = "task1_visual_explicit",
                        knob_profile_variant: str = "default",
                        pull_angle_deg: float | None = None,
                        push_angle_deg: float | None = None,
                        push_start_angle_deg: float | None = None,
                        microwave_xy_m: tuple[float, float] | None = None,
                        scene_edits: SceneEdits | None = None,
                        scene_revision: int = 0,
                        on_door_phase_complete: Callable[[object], None] | None = None) -> Task1Catalog:
    """构造一份场景；旧资产映射与缺失接口都作为显式边界处理。"""
    from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession
    from gene_agent.tasks.policies.microwave_task.planning.extract import apply_legacy_table_geometry_in_place
    from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene

    if variant not in {"task1_visual_explicit", "legacy_viser"}:
        raise ValueError(f"未知 Task1 门配置：{variant}")
    if knob_profile_variant not in {"default", "tip_x_5_azimuth_240"}:
        raise ValueError(f"未知旋钮 profile 配置：{knob_profile_variant}")
    # 角度属于 Task 调参输入；只覆盖复制出的门配置，不修改原始默认文件。
    # 旧 Viser 模式的 85° 是跳过第二阶段后的外部状态，不能把它冒充
    # 可任意调节的真实 push 轨迹。
    angle_edits = {"pull_angle_deg": pull_angle_deg, "push_angle_deg": push_angle_deg,
                   "push_start_angle_deg": push_start_angle_deg}
    for key, value in angle_edits.items():
        if value is not None and (not np.isfinite(value) or not 0. < float(value) <= 90.):
            raise ValueError(f"{key} 必须位于 (0, 90]°")
    if variant == "legacy_viser" and (push_start_angle_deg is not None or
            (push_angle_deg is not None and float(push_angle_deg) != 85.)):
        raise ValueError("旧 Viser 跳过分支的 85° 不是可编辑的 push 终点；请切换显式推门模式")
    edits = scene_edits or SceneEdits()
    if edits.robot_joint_positions_rad:
        raise ValueError("Task1 固定从机器人 home 开始，不能编辑机器人初始关节")
    if abs(edits.object_joint_positions_rad.get("microwave_door_joint", 0.0)) > 1e-9:
        raise ValueError("Task1 固定从关门初态开始，不能编辑门初角")
    if microwave_xy_m is not None and "microwave" in edits.object_poses_world:
        raise ValueError("microwave_xy_m 与完整 microwave 位姿不能同时指定")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    door_task = json.loads((ROOT / "data/task_1/skills/door_task.json").read_text(encoding="utf-8"))
    knob_task = json.loads((ROOT / "data/task_1/skills/knob_0_task.json").read_text(encoding="utf-8"))
    microwave_pose = np.asarray(door_task["microwave_pose_world"], dtype=float)
    if not np.allclose(microwave_pose, knob_task["microwave_pose_world"], atol=1e-9):
        raise ValueError("门与旋钮配置的 microwave 资产 frame 不同")
    source_microwave_pose = microwave_pose.copy()
    pose_overrides = dict(edits.object_poses_world)
    table_edit_pose = pose_overrides.pop("table", None)
    if "microwave" in pose_overrides:
        microwave_pose = np.asarray(pose_overrides["microwave"], dtype=float).copy()
    if microwave_xy_m is not None:
        xy = np.asarray(microwave_xy_m, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("微波炉 XY 摆位必须是两个有限数")
        microwave_pose[:2, 3] = xy
    pose_overrides["microwave"] = microwave_pose
    seed = int(door_task["seed"])
    if seed != int(knob_task["seed"]):
        raise ValueError("门与旋钮配置的随机种子不同")
    prepared = prepare_pick_bowl_scene(
        output / "scene", revision=scene_revision, seed=seed,
        pose_overrides=pose_overrides,
        joint_overrides=dict(edits.object_joint_positions_rad),
    )
    door_config = json.loads((ROOT / "data/task_1/skills/door_skill_debug.json").read_text(encoding="utf-8"))
    door_config["microwave_pose_world"] = microwave_pose.tolist()
    if pull_angle_deg is not None:
        door_config["pull_angle_deg"] = float(pull_angle_deg)
    if push_angle_deg is not None:
        door_config["push_angle_deg"] = float(push_angle_deg)
    if push_start_angle_deg is not None:
        door_config["push_start_angle_deg"] = float(push_start_angle_deg)
    if variant == "legacy_viser":
        # DoorDebugSession 只允许 push 起始角 < 目标角。旧 Viser 的 85°
        # 起始角恰好等于目标，故用 25° 构造但不执行 push Knowledge；
        # 真正生效的 85° 跳过语义由本 Task 编排层明确记录。
        door_config["push_angle_deg"] = 85.0
        door_config["push_start_angle_deg"] = 25.0
        door_config["legacy_stage_two_start_angle_deg"] = 85.0
    config_path = output / "door_config.json"
    config_path.write_text(json.dumps(door_config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    session = DoorDebugSession(config_path=config_path, run_dir=prepared.output_dir)
    session.output_dir = output / "door"
    case = session.case
    # 取出 Skill 的旧桌面几何必须在首个门阶段前作用于同一模型。
    table_alignment = apply_legacy_table_geometry_in_place(case, output)
    table_edit = None
    if table_edit_pose is not None or edits.table_size_xy_m is not None:
        from gene_agent.motion.curobo.scene import export_collision_scene

        # 旧取出桌面先作用于共享 model；用户再编辑桌面最终 pose/长宽。
        # 源 XML 本来就不包含旧桌面修正，实际碰撞与视觉都取本模型。
        table = case.model.geom("scene_table")
        if table_edit_pose is not None:
            table.pos[:] = table_edit_pose[:3, 3]
            # 编译后的桌面默认标记为 sameframe；只改 quat 时 MuJoCo 会继续
            # 使用 identity xmat，必须解除该优化才会让画面与碰撞同步旋转。
            case.model.geom_sameframe[table.id] = 0
            table.quat[:] = Rotation.from_matrix(table_edit_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if edits.table_size_xy_m is not None:
            table.size[:2] = np.asarray(edits.table_size_xy_m, dtype=float) / 2.0
        case.revision += 1
        mujoco.mj_forward(case.model, case.data)
        case.snapshot = export_collision_scene(
            case.model, case.data, case.metadata["robot"], "right",
            output / "task1_collision/shared_edited_table",
        )
        table_edit = {
            "center_world_m": table.pos.tolist(),
            "size_xy_m": (2.0 * table.size[:2]).tolist(),
            "collision_snapshot_sha256": case.snapshot["snapshot_sha256"],
            "scene_revision": case.revision,
        }
    _sync_runtime_table_metadata(case)
    session.rebuild()
    if variant == "legacy_viser":
        session.phases = legacy_skipped_stage_two_phases(session.phases)
        session.stop_index = len(session.phases)
    initial = capture_task_state(case)
    knob_task_path = ROOT / "data/task_1/skills/knob_0_task.json"
    knob_profile_change = None
    if (knob_profile_variant != "default" or microwave_xy_m is not None
            or "microwave" in edits.object_poses_world):
        configured_knob_task = dict(knob_task)
        configured_knob_task["microwave_pose_world"] = microwave_pose.tolist()
        if knob_profile_variant == "tip_x_5_azimuth_240":
            default_profile = ROOT / knob_task["grasp_profile_path"]
            profiles = json.loads(default_profile.read_text(encoding="utf-8"))
            parameters = profiles["profiles"][knob_task["grasp_profile_name"]]["parameters"]
            before = {
                "azimuth_deg": list(parameters["azimuth_deg"]["values"]),
                "tip_x_deg": list(parameters["tip_x_deg"]["values"]),
            }
            parameters["azimuth_deg"]["values"] = [240]
            parameters["tip_x_deg"]["values"] = [5]
            profile_path = output / "knob_grasp_profile.json"
            profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
            configured_knob_task["grasp_profile_path"] = str(profile_path)
            knob_profile_change = {
                "source_profile_ref": str(default_profile),
                "azimuth_deg": {"before": before["azimuth_deg"], "after": [240]},
                "tip_x_deg": {"before": before["tip_x_deg"], "after": [5]},
            }
        knob_task_path = output / "knob_task_config.json"
        knob_task_path.write_text(
            json.dumps(configured_knob_task, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    adapters = {
        "open_door_first": lambda state: run_door_partition(
            session, "open_door_first", state, part="open",
            skipped_stage_two_angle_deg=85.0 if variant == "legacy_viser" else None,
            progressive_push_candidates=variant != "legacy_viser",
            on_door_phase_complete=on_door_phase_complete),
        "insert_bowl": lambda state: run_bowl_insert_partition(case, state, seed=seed),
        "close_door_first": lambda state: run_door_partition(
            session, "close_door_first", state, part="close",
            progressive_push_candidates=variant != "legacy_viser",
            on_door_phase_complete=on_door_phase_complete),
        "rotate_knob": lambda state: run_rotate_knob_partition(
            case, state, output_dir=output / "knob", task_path=knob_task_path),
        "open_door_second": lambda state: run_door_partition(
            session, "open_door_second", state, part="open",
            skipped_stage_two_angle_deg=85.0 if variant == "legacy_viser" else None,
            progressive_push_candidates=variant != "legacy_viser",
            on_door_phase_complete=on_door_phase_complete),
        "extract_place_bowl": lambda state: run_extract_partition(
            case, state, seed=seed, output_dir=output / "extract"),
        "close_door_second": lambda state: run_door_partition(
            session, "close_door_second", state, part="close",
            progressive_push_candidates=variant != "legacy_viser",
            on_door_phase_complete=on_door_phase_complete),
    }
    configuration = {
        "variant": variant,
        "door_pull_angle_deg": float(door_config["pull_angle_deg"]),
        "door_push_start_angle_deg": (85.0 if variant == "legacy_viser" else
                                      float(door_config.get(
                                          "push_start_angle_deg", door_config["pull_angle_deg"]))),
        "door_push_angle_deg": float(door_config["push_angle_deg"]),
        "microwave_pose_world": microwave_pose.tolist(),
        "source_microwave_pose_world": source_microwave_pose.tolist(),
        "microwave_xy_override_m": None if microwave_xy_m is None else list(
            map(float, microwave_xy_m)),
        "seed": seed,
        "door_push_candidate_selection": (
            "legacy85_skip_open_static_close" if variant == "legacy_viser" else
            "progressive_16_then_remaining"
        ),
        "scene_revision": case.revision,
        "scene_edits": {
            "object_poses_world": {name: pose.tolist()
                                   for name, pose in edits.object_poses_world.items()},
            "object_joint_positions_rad": dict(edits.object_joint_positions_rad),
            "table_size_xy_m": (None if edits.table_size_xy_m is None else
                                list(edits.table_size_xy_m)),
        },
        "shared_table_alignment": table_alignment,
        "shared_table_edit": table_edit,
        "knob_profile_variant": knob_profile_variant,
        "knob_profile_change": knob_profile_change,
        "knob_task_config_ref": str(knob_task_path),
        "door_config_ref": str(config_path),
        "legacy_default_note": (
            "旧 Viser 默认阶段二 85° 起始角并跳过内侧推门；"
            "本目录在旧跳过分支中只保留把手 25° 轨迹和外部门状态更新"
            if variant == "legacy_viser" else
            "旧 Viser 默认阶段二 85° 起始角并跳过内侧推门；"
            "本目录使用显式 25°/90° 门调试配置，不能冒充旧默认成功基线"
        ),
        "asset_frame_note": (
            "门/旋钮用当前 calibrated Concept；Bowl 放入的旧资产映射仍为实验视觉映射"
        ),
    }
    return Task1Catalog(case, session, initial, adapters, output, configuration)


__all__ = ["capture_task_state", "apply_task_state", "door_partition_bounds",
           "legacy_skipped_stage_two_phases", "run_door_partition",
           "state_after_insert_report", "run_bowl_insert_partition",
           "select_legacy_push_candidate",
           "run_rotate_knob_partition", "run_extract_partition",
           "Task1Catalog", "build_task1_catalog"]
