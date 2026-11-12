"""把现有微波炉在线规划结果交给共用纯视觉运行模型。

首批只接门 Skill：这里不生成门的目标，也不替换 DoorDebugSession 的
cuRobo 规划。碰撞资产来自每阶段实际导出的 collision snapshot。
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.visualization.pick_bowl.scene import _collision_group
from gene_agent.visualization.viser_sim.model import VisualRun, VisualStage
from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState


def _collision_facts(snapshot_ref: str | None) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], int,
]:
    if not snapshot_ref:
        return (), (), (), (), 0
    snapshot = json.loads(Path(snapshot_ref).read_text(encoding="utf-8"))
    model = snapshot.get("scene_model") or {}
    active_names = {
        str(name) for objects in model.values() if isinstance(objects, dict)
        for name in objects
    }
    # cuRobo world 使用细粒度 obstacle 名称；面板先按语义 body 汇总，
    # 仍保留原始 snapshot 路径，供需要时逐 geom 检查碰撞来源。
    body_by_obstacle = {
        str(record["obstacle_name"]): str(record["body_name"])
        for record in snapshot.get("records", ())
        if record.get("obstacle_name") and record.get("body_name")
    }
    enabled = tuple(sorted({body_by_obstacle.get(name, name) for name in active_names}))
    frozen_groups = set(snapshot.get("frozen_groups", ()))
    if not frozen_groups and snapshot.get("active_group") in ("left", "right"):
        frozen_groups.add("left" if snapshot["active_group"] == "right" else "right")
    display_groups = tuple(sorted({
        _collision_group(record, frozen_groups)
        for record in snapshot.get("records", ())
        if str(record.get("obstacle_name")) in active_names
    }))
    moving = tuple(sorted(str(name) for name in snapshot.get("moving_body_names", ())))
    disabled = tuple(sorted(str(name) for name in snapshot.get("disabled_body_names", ())))
    return enabled, moving, disabled, display_groups, len(active_names)


def _json_safe(value):
    """只在对外清单边界转换 NumPy 数值，内部 planner 结果不被改写。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def plan_door_visual_run(session, skill_id: str | None = None) -> VisualRun:
    """运行一个门 Skill 或门循环，并冻结本次实际规划出的阶段。

    ``DoorDebugSession`` 本来就是可在线修改场景的同步会话。调用前可通过
    它现有的编辑接口修改微波炉 pose、门角和过渡点；本函数只取本次新增的
    result，防止重复点击规划时混入上一次的阶段记录。
    """
    first_row = len(session.results)
    report = session.plan_sequence(skill_id)
    return door_visual_run_from_session(
        session, skill_id, first_row=first_row, report=report
    )


def door_visual_run_from_session(
    session, skill_id: str | None, *, first_row: int, report: dict,
) -> VisualRun:
    """复用在线界面已经算出的阶段；转换记录绝不能触发第二次规划。"""
    rows = tuple(session.results[first_row:])
    if not rows:
        raise RuntimeError("门会话没有返回本次规划阶段")
    phase_by_id = {phase.phase_id: phase for phase in session.phases}
    segments = (
        {} if session.timeline is None
        else {segment.stage_id: segment for segment in session.timeline.segments}
    )
    stages: list[VisualStage] = []
    for row in rows:
        phase_id = str(row["phase_id"])
        phase = phase_by_id[phase_id]
        success = bool(row.get("success"))
        snapshot_ref = row.get("collision_snapshot_ref")
        (collision_enabled, collision_allowed, collision_disabled,
         collision_display_groups, collision_obstacle_count) = _collision_facts(snapshot_ref)
        if success:
            failure = None
        else:
            failure = row.get("failure_summary")
            if not isinstance(failure, dict) or not failure:
                failure = {
                    "code": row.get("failure_category") or row.get("status") or "unknown",
                    "message": row.get("message") or "规划未成功",
                }
        timing = row.get("timing") or {}
        diagnostics = _json_safe({
            key: row[key]
            for key in (
                "approximate_ik", "target_selection", "target_selection_attempts",
                "linear_path_metrics", "planner_mode", "use_cuda_graph",
                "requested_operation", "fallback",
                "knowledge_candidate_selection",
                "timing", "failure_detail", "ik_diagnostic", "solver_diagnostic",
                "collision_verification", "failure_visualization",
            )
            if key in row and row[key] is not None
        })
        diagnostics["contact_policy"] = phase.contact_policy
        if collision_disabled:
            diagnostics["collision_disabled_body_names"] = list(collision_disabled)
        if snapshot_ref:
            diagnostics["collision_display_groups"] = list(collision_display_groups)
            diagnostics["collision_obstacle_count"] = collision_obstacle_count
        stages.append(VisualStage(
            stage_id=phase_id,
            skill_id=phase.skill_id,
            motion_kind=(row.get("operation") or phase.motion_kind),
            success=success,
            trajectory=(segments.get(phase_id) if success and phase.motion_kind != "gripper" else None),
            target_tcp_world=phase.tcp_pose_world,
            collision_enabled=collision_enabled,
            collision_allowed=collision_allowed,
            collision_snapshot_ref=snapshot_ref,
            gripper_command=(phase.name if phase.motion_kind == "gripper" else None),
            gripper_width_m=(float(row.get("gripper_width_m", phase.gripper_width_m))
                             if phase.motion_kind == "gripper" else None),
            state_duration_s=(segments[phase_id].duration_s
                              if success and phase.motion_kind == "gripper"
                              and phase_id in segments else None),
            door_angle_rad=phase.door_angle_rad,
            planning_duration_s=timing.get("stage_wall_s", 0.0),
            failure=_json_safe(failure),
            diagnostics=diagnostics,
        ))
    run = VisualRun(
        selection_id=skill_id or "microwave_door_cycle",
        scene_revision=int(session.revision),
        stages=tuple(stages),
    )
    if run.success != bool(report.get("success")):
        raise RuntimeError("门会话总结果与逐阶段结果不一致")
    return run


def door_visual_playback(session, run: VisualRun, *,
                         initial_qpos: np.ndarray | None = None) -> VisualPlayback:
    """从本轮真实场景和关节初态构造可任意 seek 的纯视觉时间轴。

    DoorDebugSession 仍负责旧版 Knowledge、碰撞与 cuRobo 规划；这里仅使用
    同一 MuJoCo 模型作 TCP 正运动学，并保存 free-joint 物体的初始世界位姿。
    """
    case = session.case
    model, data = case.model, case.data
    robot = case.metadata["robot"]
    names = tuple(robot["joint_groups"][session.group])
    base_qpos = data.qpos.copy() if initial_qpos is None else np.asarray(initial_qpos, dtype=float).copy()
    if base_qpos.shape != data.qpos.shape:
        raise ValueError("门回放初始 qpos 与当前模型不匹配")
    base_data = mujoco.MjData(model)
    base_data.qpos[:] = base_qpos
    if model.nmocap:
        base_data.mocap_pos[:] = data.mocap_pos
        base_data.mocap_quat[:] = data.mocap_quat
    mujoco.mj_forward(model, base_data)
    joint_positions = np.asarray([
        base_qpos[model.joint(name).qposadr[0]] for name in names
    ], dtype=float)
    door_joint = model.joint("microwave_door_joint")
    door_angle = float(base_qpos[door_joint.qposadr[0]])
    objects = {}
    for body_id in range(1, model.nbody):
        body = model.body(body_id)
        if int(body.jntnum[0]) != 1:
            continue
        joint_id = int(body.jntadr[0])
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        pose = np.eye(4)
        pose[:3, :3] = base_data.xmat[body_id].reshape(3, 3)
        pose[:3, 3] = base_data.xpos[body_id]
        objects[name] = pose
    gripper = robot.get("grippers", {}).get(session.group)
    if gripper is None:
        # 无夹爪的最小机器人也可以回放 TCP/门/物体状态；旧会话若提供
        # 夹爪宽度接口，仍沿用其返回值，避免把已有开口硬写成 0。
        width = float(getattr(session, "_gripper_width", lambda: 0.)())
    else:
        width = (sum(float(base_qpos[model.joint(name).qposadr[0]])
                     for name in gripper["joint_names"])
                 if gripper.get("mode") == "parallel_slide_proxy" else 0.0)
    initial = VisualState(names, joint_positions, width,
                          objects, door_angle)
    scratch = mujoco.MjData(model)
    tcp_site = robot["tcp_sites"][session.group]

    def tcp_pose(joint_names: tuple[str, ...], positions: np.ndarray) -> np.ndarray:
        scratch.qpos[:] = base_qpos
        for name, value in zip(joint_names, positions, strict=True):
            scratch.qpos[model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(model, scratch)
        site = scratch.site(tcp_site)
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return pose

    return VisualPlayback(run, initial, tcp_pose)


class OnlineDoorVisualController:
    """在线界面每个 tick 规划一段；结束后只转换这一次的结果。"""

    def __init__(self, session):
        self.session = session
        self.run: VisualRun | None = None
        self.playback: VisualPlayback | None = None
        self.selection_id: str | None = None
        self._first_row = 0
        self._visible_first_row = 0
        self._close_initial_qpos: np.ndarray | None = None
        self._active = False

    @property
    def planning(self) -> bool:
        return self._active

    def begin(self, selection_id: str | None) -> dict:
        if self._active:
            self.cancel()
        self.run = None
        self.playback = None
        self.selection_id = selection_id
        self._first_row = len(self.session.results)
        self._visible_first_row = self._first_row
        self._close_initial_qpos = None
        report = self.session.begin_plan_sequence(selection_id)
        self._active = not bool(report["done"])
        return report

    def step(self) -> dict:
        if not self._active:
            raise RuntimeError("当前没有正在规划的 Skill/Task")
        if (self.selection_id == "microwave_door_close"
                and self._close_initial_qpos is None
                and self.session.phases[self.session.phase_index].skill_id == "push_close"):
            # 这里的 qpos 来自本轮刚完成的开门及过渡阶段；完成规划后门会话
            # 会恢复初始场景，所以必须在首个关门阶段开始前保存回放起点。
            self._visible_first_row = len(self.session.results)
            self._close_initial_qpos = self.session.case.data.qpos.copy()
        try:
            selection = None
            if self.selection_id in {None, "push_open", "push_close",
                                     "microwave_door_open", "microwave_door_close"}:
                # 单 Skill、门循环及任务分区都复用旧 Task1 的逐角两批 IK。
                # 不改 Knowledge 几何或 Skill 阶段顺序，只从当前实际门角的
                # 候选中选首个可达 TCP，避免显示目标与实际规划目标脱节。
                import gene_agent.visualization.microwave.catalog as visual_catalog
                selection = visual_catalog._select_task1_push_phase(self.session)
            if selection is not None and not selection["success"]:
                row = {
                    "phase_id": selection["phase_id"], "success": False,
                    "operation": "knowledge_candidate_selection",
                    "failure_category": "goal_ik_no_solution",
                    "status": "goal_ik_no_solution",
                    "message": "旧 Knowledge 当前门角所有候选 IK 不可达",
                    "failure_summary": {
                        "code": "goal_ik_no_solution",
                        "message": "旧 Knowledge 当前门角所有候选 IK 不可达",
                    },
                    "knowledge_candidate_selection": selection,
                    "collision_snapshot_ref": selection["collision_snapshot_ref"],
                }
                self.session.results.append(row)
                report = self.session._finish_plan_sequence(
                    success=False, first_failure=row,
                )
            else:
                report = self.session.plan_sequence_step()
                if selection is not None:
                    self.session.results[-1]["knowledge_candidate_selection"] = selection
        except Exception:
            self._active = False
            raise
        if report["done"]:
            self._active = False
            first_row = self._visible_first_row
            if (self.selection_id == "microwave_door_close"
                    and self._close_initial_qpos is None):
                # 前置开门失败时没有关门轨迹；只展示真实首败诊断，
                # 不能把该次运行标成已完成的关门任务。
                first_row = len(self.session.results) - 1
            self.run = door_visual_run_from_session(
                self.session, self.selection_id,
                first_row=first_row, report=report,
            )
            if self._close_initial_qpos is None:
                self.playback = door_visual_playback(self.session, self.run)
            else:
                self.playback = door_visual_playback(
                    self.session, self.run, initial_qpos=self._close_initial_qpos,
                )
        return report

    def cancel(self) -> None:
        if self._active:
            self.session.cancel_plan_sequence()
            self._active = False


__all__ = [
    "plan_door_visual_run", "door_visual_run_from_session", "door_visual_playback",
    "OnlineDoorVisualController",
]
