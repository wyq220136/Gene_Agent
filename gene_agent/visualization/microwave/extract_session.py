"""Bowl 取出与桌面放置的在线单 Skill 会话。

每个 revision 从桌面 Bowl 的同一 MuJoCo 场景实际规划 Pick、Insert；
Insert 的 release 和撤离终态才是 Extract 的起点。Place 仅在同一轮
Extract 建立真实规划 attachment 后运行。前序报告只提供起始状态，
所选 Skill 的轨迹始终由当前场景重新规划。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.visualization.viser_sim.model import SceneEdits, VisualRun, VisualStage
from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState

from gene_agent.tasks.policies.microwave_task.planning.extract import _world_legacy_frame, apply_legacy_table_geometry_in_place, extract_profile_targets, extract_visual_run, run_extract_bowl_skill_planning, table_tcp_after_scene_edit
from gene_agent.tasks.policies.microwave_task.operations.bowl import load_object_operation_profile
from gene_agent.tasks.policies.microwave_task.planning.insert import run_insert_bowl_skill_planning
from gene_agent.visualization.microwave.bowl_session import DEFAULT_CONFIG, ROOT, ObjectVisualSession, object_visual_run
from gene_agent.tasks.policies.microwave_task.planning.bowl import _body_pose, prepare_pick_bowl_scene
from gene_agent.tasks.policies.microwave_task.planning.grasp import run_pick_bowl_skill_planning


EXTRACT_PLACE_SKILLS = {"extract_bowl": "炉内取碗", "place_bowl": "桌面放碗"}
EXTRACT_SUBTASKS = {"microwave_object_extract": "炉内取碗 → 桌面放碗"}


def extract_subtask_visual_run(report: dict, output_dir: Path) -> VisualRun:
    """保留 Extract→Place 真阶段与失败前缀，检查两段关节轨迹衔接。"""
    combined = extract_visual_run(report, output_dir)
    return VisualRun("microwave_object_extract", combined.scene_revision,
                     combined.stages)


def _commit_visual_state(case, state: VisualState) -> None:
    """把前序规划终态一次写入共享 GT，再从该状态导出新的完整 world。

    本函数只在 Pick→Insert 已成功且 VisualPlayback 已验证连续后调用；
    规划本身不会悄悄改变 ``case.data``，后继 Extract 的碰撞必须从这里
    的 qpos、Bowl pose、门和旋钮角重新导出。
    """
    if state.attachments:
        raise ValueError("Insert 后 Bowl 必须已 detach，才允许取出")
    if "bowl" not in state.object_poses_world:
        raise ValueError("Insert 终态缺少实际 Bowl 位姿")
    model, data = case.model, case.data
    for name, value in zip(state.joint_names, state.joint_positions_rad, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = float(value)
    fingers = case.metadata["robot"]["grippers"]["right"]["joint_names"]
    for name in fingers:
        data.qpos[model.joint(name).qposadr[0]] = float(state.gripper_width_m) / len(fingers)
    if state.door_angle_rad is not None:
        data.qpos[model.joint("microwave_door_joint").qposadr[0]] = state.door_angle_rad
    for name, value in state.object_joint_positions_rad.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    pose = state.object_poses_world["bowl"]
    free = model.joint("bowl_free").qposadr[0]
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
    data.qpos[free:free + 3] = pose[:3, 3]
    data.qpos[free + 3:free + 7] = quat[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    case.bowl_pose = _body_pose(data, "bowl")
    case.snapshot = export_collision_scene(
        model, data, case.metadata["robot"], "right",
        case.output_dir / "collision/after_online_insert",
    )


class ExtractPlaceVisualSession(ObjectVisualSession):
    """同一场景编辑、前序 Skill、Extract 和 Place 的逐步在线编排。"""

    def __init__(self, *, output_dir: Path | None = None, config_path: Path | None = None):
        self.output_dir = Path(output_dir or ROOT / "runs" /
            f"extract-place-online-{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = json.loads(Path(config_path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
        self._plan_state = None
        self.run = self.playback = self.prerequisite_report = None
        self.case = self._build(self.config, 0)
        self._refresh_targets()
        self.initial = self._initial_state()

    def _build(self, config: dict, revision: int):
        """先装入旧桌面，再将用户 pose/尺寸编辑同时提交到 visual 和碰撞。"""
        poses = {name: np.asarray(pose, dtype=float)
                 for name, pose in config.get("object_poses_world", {}).items()}
        table_pose = poses.pop("table", None)
        case = prepare_pick_bowl_scene(
            self.output_dir / f"revision_{revision:03d}", revision=revision,
            seed=int(config.get("seed", 123)), pose_overrides=poses,
            joint_overrides=config.get("object_joint_positions_rad", {}),
        )
        apply_legacy_table_geometry_in_place(case, case.output_dir)
        table = case.model.geom("scene_table")
        if table_pose is not None:
            table.pos[:] = table_pose[:3, 3]
            table.quat[:] = Rotation.from_matrix(table_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            # 编译时原桌面姿态恒等，MuJoCo 用 sameframe 快捷路径忽略
            # 运行期写入的 quat；解除它才能同步更新 geom_xmat 与碰撞。
            case.model.geom_sameframe[table.id] = 0
        if config.get("table_size_xy_m") is not None:
            table.size[:2] = np.asarray(config["table_size_xy_m"], dtype=float) / 2.0
        mujoco.mj_forward(case.model, case.data)
        # ``sample_poses`` 是 GUI 数值与拖动控件的源；必须与实际 geom 一致。
        actual_table = np.eye(4)
        actual_table[:3, :3] = case.data.geom("scene_table").xmat.reshape(3, 3)
        actual_table[:3, 3] = case.data.geom("scene_table").xpos
        case.metadata["objects"]["sample_poses"]["table"] = actual_table.tolist()
        case.snapshot = export_collision_scene(
            case.model, case.data, case.metadata["robot"], "right",
            case.output_dir / "collision/current_legacy_table",
        )
        # 这是新 revision 的基线构建，而非一次已执行的 Skill 状态跳变。
        case.revision = revision
        return case

    def _refresh_targets(self) -> None:
        profile = load_object_operation_profile(ROOT / "data/task_1", "Bowl_0")
        world_legacy, _ = _world_legacy_frame(self.case)
        self.extract_targets, table = extract_profile_targets(world_legacy, profile)
        self.table_target = table_tcp_after_scene_edit(self.case, table)
        self.open_width_m = float(profile["grasp_pregrasp_opening_width_m"])

    def _initial_state(self) -> VisualState:
        base = super()._initial_state()
        knobs = {name: float(self.case.data.qpos[self.case.model.joint(name).qposadr[0]])
                 for name in ("knob_0_joint", "knob_1_joint")}
        return VisualState(base.joint_names, base.joint_positions_rad,
                           base.gripper_width_m, base.object_poses_world,
                           base.door_angle_rad, base.attachments, knobs)

    def apply_edits(self, edits: SceneEdits) -> None:
        super().apply_edits(edits)
        self._refresh_targets()

    def begin_plan(self, selection: str) -> dict:
        if selection not in EXTRACT_PLACE_SKILLS and selection not in EXTRACT_SUBTASKS:
            raise ValueError(f"未知 Bowl Skill 或子任务：{selection}")
        if self.run is not None:
            self.apply_edits(SceneEdits())
        self._plan_state = {"selection": selection, "step": "pick",
                            "table_initial": self._initial_state()}
        return {"done": False, "message": "正在当前场景规划 Pick 前序"}

    def _failed_prerequisite(self, name: str, prior_run: VisualRun,
                             initial: VisualState, report: dict) -> dict:
        selection = self._plan_state["selection"]
        source = report.get("extract", report)
        attempts = source.get("attempts") or []
        first = next((row for row in attempts
                      if not (row.get("result") or {}).get("success", False)), None)
        if first is not None:
            raw = (first.get("result") or {}).get("failure") or {}
            original = {"category": raw.get("kind", "planning_failed"),
                        "message": raw.get("summary", "前序规划失败"),
                        "candidate_id": first.get("candidate_id"),
                        "stage": first.get("stage"),
                        "backend_status": raw.get("backend_status")}
        else:
            raw = prior_run.stages[-1].failure or {}
            original = {"category": raw.get("category", "planning_failed"),
                        "message": raw.get("message", "前序规划失败")}
        report_name = {"pick": "skill_pick_bowl_result.json",
                       "insert": "skill_insert_bowl_result.json",
                       "extract": "skill_extract_bowl_result.json"}[name]
        failed = VisualStage(
            f"{selection}/prerequisite_{name}", selection, "attachment", False,
            failure={"category": "prerequisite_failed",
                     "message": f"当前场景 {name} 前序规划失败，未执行 {selection}",
                     f"{name}_failure": original,
                     "attempt_count": len(attempts),
                     "report_ref": str(self.case.output_dir / report_name)},
            diagnostics={"prerequisite": name, "visual_only": True},
        )
        return self._finish_run(VisualRun(selection, self.case.revision, (failed,)), initial)

    def plan_step(self) -> dict:
        """每次 tick 最多规划一个同步 Skill；失败只显示首败，不伪造后继。"""
        if self._plan_state is None:
            raise RuntimeError("请先 begin_plan")
        state = self._plan_state
        selection = state["selection"]
        seed = int(self.config.get("seed", 123))
        if state["step"] == "pick":
            pick = run_pick_bowl_skill_planning(self.case, seed=seed)
            self.prerequisite_report = {"pick": pick}
            run = object_visual_run(pick, "pick_bowl", self.case.output_dir)
            if not run.success:
                return self._failed_prerequisite("pick", run, state["table_initial"], pick)
            state["pick_run"] = run
            state["step"] = "insert"
            return {"done": False, "message": "Pick 成功；正在同 revision 规划 Insert 前序"}
        if state["step"] == "insert":
            insert = run_insert_bowl_skill_planning(self.case, seed=seed)
            self.prerequisite_report["insert"] = insert
            run = object_visual_run(insert, "insert_bowl", self.case.output_dir,
                                    initial_attachments=("bowl",))
            if not run.success:
                return self._failed_prerequisite("insert", run, state["table_initial"], insert)
            combined = VisualRun(selection, self.case.revision,
                                 state["pick_run"].stages + run.stages)
            predecessor = VisualPlayback(combined, state["table_initial"], self.tcp_pose)
            endpoint = predecessor.sample(predecessor.duration_s).state
            _commit_visual_state(self.case, endpoint)
            state["extract_initial"] = self._initial_state()
            state["step"] = "extract"
            return {"done": False, "message": "Insert 实际 release 已提交；正在规划炉内 Extract"}
        report = run_extract_bowl_skill_planning(
            self.case, seed=seed, current_scene=True,
            stop_after_extract=(selection == "extract_bowl"),
            follow_table_edits=True,
        )
        self.prerequisite_report["extract"] = report
        if selection in EXTRACT_SUBTASKS:
            # Extract 和 Place 在同一适配器、同一模型及同一 attachment 中
            # 连续规划；失败时直接返回真实成功前缀与首个失败阶段。
            run = extract_subtask_visual_run(report, self.case.output_dir)
            return self._finish_run(run, state["extract_initial"])
        part = dict(report["extract"], scene_revision=report["scene_revision"],
                    instance_id="bowl")
        extract_run = object_visual_run(part, "extract_bowl", self.case.output_dir)
        if selection == "extract_bowl":
            return self._finish_run(extract_run, state["extract_initial"])
        if not extract_run.success:
            return self._failed_prerequisite("extract", extract_run,
                                             state["extract_initial"], report)
        predecessor = VisualPlayback(extract_run, state["extract_initial"], self.tcp_pose)
        place_initial = predecessor.sample(predecessor.duration_s).state
        if place_initial.attachments != ("bowl",):
            raise RuntimeError("Extract 成功报告未建立 Bowl attachment")
        place = dict(report["place"], scene_revision=report["scene_revision"],
                     instance_id="bowl")
        place_run = object_visual_run(place, "place_bowl", self.case.output_dir,
                                      initial_attachments=("bowl",))
        return self._finish_run(place_run, place_initial)


class ExtractTaskVisualSession(ExtractPlaceVisualSession):
    """供在线子任务界面独立注入的同场景会话入口。"""


__all__ = ["EXTRACT_PLACE_SKILLS", "EXTRACT_SUBTASKS",
           "ExtractPlaceVisualSession", "ExtractTaskVisualSession",
           "extract_subtask_visual_run"]
