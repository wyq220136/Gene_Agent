"""门 Skill 在线纯视觉编辑器；业务状态全部来自 DoorDebugSession。"""
from __future__ import annotations

import json
from threading import RLock
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession, TRANSITION_NAMES
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.visualization.grasp_tuner.gripper_preview import GripperMeshPreview
from gene_agent.visualization.grasp_tuner.pose_ik import MuJoCoPoseIK
from gene_agent.knowledge.reachability import assess_pose_ik
from gene_agent.visualization.grasp_tuner.microwave_scene import _body_pose
from gene_agent.visualization.pick_bowl.scene import PickBowlScene, _pose


def _matrix(position, wxyz):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position, dtype=float)
    transform[:3, :3] = Rotation.from_quat(np.asarray(wxyz, dtype=float)[[1, 2, 3, 0]]).as_matrix()
    return transform


class LiveDoorApp:
    """目标浏览不改变规划状态；点击规划时才调用当前场景的 cuRobo。"""

    CAMERAS = ("全景", "门接触", "夹爪侧面")
    SKILL_LABELS = {
        "handle_pull": "抓把手拉门",
        "push_open": "从内侧推开",
        "push_close": "从外侧推关",
    }

    def __init__(self, server, *, config_path: Path | None = None,
                 mode: str = "skill", skill_id: str = "push_close",
                 planner_mode: str | None = None):
        if mode not in {"skill", "task"}:
            raise ValueError("mode 必须是 skill 或 task")
        if mode == "skill" and skill_id not in self.SKILL_LABELS:
            raise ValueError(f"未知门 Skill：{skill_id}")
        self.server = server
        session_options = {}
        if config_path is not None:
            session_options["config_path"] = config_path
        if planner_mode is not None:
            session_options["planner_mode"] = planner_mode
        self.session = DoorDebugSession(**session_options)
        self.mode = mode
        self.skill_id = skill_id if mode == "skill" else None
        self.session.start_skill(self.skill_id)
        self.scope_indices = self._compute_scope_indices()
        self.lock = RLock()
        self.syncing = False
        self.preview_playing = False
        self.planning_requested = False
        self.sequence_planning_requested = False
        self.running = False
        self.timeline_playing = False
        self.view_kind = "knowledge"
        self.timeline_time_s = 0.0
        self.last_tick = time.monotonic()
        self.preview_elapsed = 0.
        self.ik_cache = {}
        self._shown_collision_ref = None
        self.preview_data = mujoco.MjData(self.session.case.model)
        mujoco.mj_copyData(self.preview_data, self.session.case.model, self.session.case.data)
        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 238, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False,
                                   show_share_button=False, brand_color=(53, 112, 169))
        case = self.session.case
        self.scene = PickBowlScene(server, case.model, case.data, case.metadata, case.snapshot)
        self.scene.set_collision_visible(False)
        self.ghost = GripperMeshPreview(server, case, prefix="/door_skill/target_gripper",
                                        color_rgba=(244, 190, 43, 145))
        if not self.ghost.available:
            raise ValueError("当前场景缺少真实夹爪 visual mesh")
        self.ghost.set_visible(True)
        microwave = _body_pose(case.data, "microwave")
        self.microwave_control = server.scene.add_transform_controls(
            "/door_skill/edit_microwave", scale=.15,
            position=microwave[:3, 3], wxyz=_pose(microwave)[0])
        self.target_control = server.scene.add_transform_controls(
            "/door_skill/edit_transition", scale=.09)
        self.target_frame = server.scene.add_frame(
            "/door_skill/knowledge_tcp", axes_length=.04, axes_radius=.0015)
        self.ik_gap = None
        gui = server.gui
        scope_name = (self.SKILL_LABELS[skill_id] if mode == "skill"
                      else "门循环 Task：拉门 → 推开 → 推关")
        self.scope_title = gui.add_markdown(
            f"### {scope_name} · 在线纯视觉调试\n"
            "黄色夹爪表示 Knowledge 目标；规划后播放真实 cuRobo 连续关节轨迹。"
            "这里检查几何、IK、路径和碰撞，不代表 MuJoCo 物理接触已经通过。"
        )
        self.skill_selector = (
            gui.add_dropdown(
                "选择 Skill",
                options=tuple(self.SKILL_LABELS),
                initial_value=self.skill_id,
            )
            if mode == "skill" else None
        )
        self.result_status = gui.add_markdown("**规划状态：** 尚未规划。")
        self.view_status = gui.add_markdown("正在生成 Knowledge 目标…")
        self.stage = gui.add_slider("当前范围内的阶段", min=0, max=len(self.scope_indices)-1,
                                    step=1, initial_value=0)
        self.phase_info = gui.add_markdown("")
        self.preview_button = gui.add_button("播放／暂停 Knowledge 目标动画")
        self.run_button = gui.add_button(
            "规划并播放整个 Skill" if mode == "skill" else "规划并播放整个门循环 Task",
            color="blue")
        self.timeline = gui.add_slider("规划轨迹时间 (s)", min=0., max=1., step=.01,
                                       initial_value=0.)
        self.timeline_button = gui.add_button("播放／暂停规划轨迹")
        self.timeline_speed = gui.add_slider("播放速度", min=.1, max=2., step=.1,
                                             initial_value=1.)
        with gui.add_folder("高级：逐阶段规划", expand_by_default=False):
            self.plan_button = gui.add_button("规划并播放下一阶段")
        self.stop_button = gui.add_button("停止／恢复编辑初态")
        self.save_button = gui.add_button("保存当前参数到任务目录")
        self.restore_button = gui.add_button("恢复选中点的 Knowledge 默认值")
        self.camera = gui.add_dropdown("视角", options=self.CAMERAS, initial_value=self.CAMERAS[0])
        self.show_visual = gui.add_checkbox("显示 visual", initial_value=True)
        self.preview_ik = gui.add_checkbox("Knowledge 动画中显示纯 IK 机器人姿态", initial_value=True)
        self.show_collision = gui.add_checkbox("显示场景碰撞几何", initial_value=False)
        self.collision_info = gui.add_markdown("碰撞层未显示；规划时会保存各阶段碰撞清单。")
        self.attachment_info = gui.add_markdown("**Attach：** 无。门是场景关节物体，不附着到夹爪。")
        self.timing_info = gui.add_markdown("**耗时：** 尚无规划记录。")
        self.planner_info = gui.add_markdown(
            f"**Planner：** `{self.session.planner_mode.value}` · "
            f"CUDA Graph {'开启' if self.session.planner_mode.value == 'cuda_graph_benchmark' else '关闭'}"
        )
        with gui.add_folder("操作角度（修改后重建目标）", expand_by_default=True):
            self.pull = gui.add_slider("把手拉门终点 (°)", min=1, max=45, step=1,
                                       initial_value=float(self.session.config["pull_angle_deg"]))
            self.push = gui.add_slider("内侧推开终点 (°)", min=30, max=90, step=1,
                                       initial_value=float(self.session.config["push_angle_deg"]))
        with gui.add_folder("选中阶段的有界近似 IK", expand_by_default=False):
            gui.add_markdown(
                "默认非接触过渡点与连续 push 点最多 20 mm / 10°；"
                "首次 contact 仍保持严格。修改后旧轨迹失效。"
            )
            self.approx_enabled = gui.add_checkbox("允许近似目标", initial_value=False)
            self.approx_position = gui.add_slider("位置误差上限 (mm)", min=1, max=100, step=1, initial_value=20)
            self.approx_rotation = gui.add_slider("姿态误差上限 (°)", min=1, max=45, step=1, initial_value=10)
        with gui.add_folder("微波炉位姿（世界坐标）", expand_by_default=False):
            self.microwave_fields = self._pose_fields("微波炉", microwave)
        with gui.add_folder("选中过渡点（世界坐标）", expand_by_default=False):
            self.stage_fields = self._pose_fields(
                "过渡点", self.session.phases[self.scope_indices[0]].tcp_pose_world)
            gui.add_markdown("仅 pre-grasp、pre-contact、撤离点可覆盖；接触目标必须回 Knowledge 调整。")
        if self.skill_selector is not None:
            self.skill_selector.on_update(lambda _: self._change_skill())
        self.stage.on_update(lambda _: self._show_target())
        self.preview_button.on_click(lambda _: self._toggle_preview())
        self.plan_button.on_click(lambda _: self._request_plan(False))
        self.run_button.on_click(lambda _: self._request_plan(True))
        self.timeline_button.on_click(lambda _: self._toggle_timeline())
        self.timeline.on_update(lambda _: self._seek_timeline())
        self.stop_button.on_click(lambda _: self._stop())
        self.save_button.on_click(lambda _: self._save())
        self.restore_button.on_click(lambda _: self._restore_stage())
        self.camera.on_update(lambda _: self._set_camera())
        self.show_visual.on_update(lambda _: self.scene.set_visual_visible(self.show_visual.value))
        self.preview_ik.on_update(lambda _: self._show_target())
        self.show_collision.on_update(lambda _: self._toggle_collision())
        for widget in (self.approx_enabled, self.approx_position, self.approx_rotation):
            widget.on_update(lambda _: self._edit_budget())
        self.pull.on_update(lambda _: self._edit_angles())
        self.push.on_update(lambda _: self._edit_angles())
        for widget in self.microwave_fields.values():
            widget.on_update(lambda _: self._edit_microwave_numbers())
        for widget in self.stage_fields.values():
            widget.on_update(lambda _: self._edit_stage_numbers())
        self.microwave_control.on_update(lambda _: self._edit_microwave_transform())
        self.target_control.on_update(lambda _: self._edit_stage_transform())
        self._set_camera()
        self._show_target()

    def _pose_fields(self, prefix, pose):
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        result = {}
        for i, axis in enumerate("xyz"):
            result[axis] = self.server.gui.add_number(f"{prefix} {axis.upper()} (m)",
                                                       initial_value=float(pose[i, 3]), step=.005)
        for axis, value in zip(("roll", "pitch", "yaw"), rpy):
            result[axis] = self.server.gui.add_number(f"{prefix} {axis} (°)",
                                                       initial_value=float(value), step=1.)
        return result

    def _read_fields(self, fields):
        xyz = [float(fields[axis].value) for axis in "xyz"]
        rpy = [float(fields[axis].value) for axis in ("roll", "pitch", "yaw")]
        pose = np.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        return pose

    def _write_fields(self, fields, pose):
        self.syncing = True
        try:
            for i, axis in enumerate("xyz"):
                fields[axis].value = float(pose[i, 3])
            for axis, value in zip(("roll", "pitch", "yaw"),
                                   Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)):
                fields[axis].value = float(value)
        finally:
            self.syncing = False

    def _compute_scope_indices(self):
        """Skill 查看器只暴露本 Skill；Task 查看器包含显式过渡段。"""
        return tuple(
            index for index, phase in enumerate(self.session.phases)
            if self.skill_id is None or phase.skill_id == self.skill_id
        )

    def _subset(self):
        return list(self.scope_indices)

    def _current_global_index(self) -> int:
        local = int(np.clip(int(self.stage.value), 0, len(self.scope_indices) - 1))
        return self.scope_indices[local]

    def _current_phase(self):
        return self.session.phases[self._current_global_index()]

    def _reset_scope_after_edit(self, preferred_phase_id: str | None = None) -> None:
        """重建后仍停留在同一个检查范围和尽可能相同的阶段。"""
        self.session.start_skill(self.skill_id)
        self.scope_indices = self._compute_scope_indices()
        target_local = 0
        if preferred_phase_id is not None:
            for local, global_index in enumerate(self.scope_indices):
                if self.session.phases[global_index].phase_id == preferred_phase_id:
                    target_local = local
                    break
        self.syncing = True
        try:
            self.stage.max = len(self.scope_indices) - 1
            self.stage.value = target_local
        finally:
            self.syncing = False

    def _change_skill(self) -> None:
        """在同一 Viser 会话内切换 Skill，并清除上一 Skill 的规划状态。"""
        if self.syncing or self.skill_selector is None:
            return
        selected = str(self.skill_selector.value)
        if selected not in self.SKILL_LABELS:
            self.result_status.content = f"**Skill 切换失败：** 未知 Skill `{selected}`。"
            return
        with self.lock:
            self.preview_playing = self.running = self.timeline_playing = False
            self.planning_requested = self.sequence_planning_requested = False
            self.session.cancel_plan_sequence()
            self.skill_id = selected
            self.session.start_skill(selected)
            self.scope_indices = self._compute_scope_indices()
            self.timeline_time_s = 0.0
            self.view_kind = "knowledge"
            self.syncing = True
            try:
                self.stage.max = len(self.scope_indices) - 1
                self.stage.value = 0
                self.timeline.max = 1.0
                self.timeline.value = 0.0
            finally:
                self.syncing = False
            self.scope_title.content = (
                f"### {self.SKILL_LABELS[selected]} · 在线纯视觉调试\n"
                "黄色夹爪表示 Knowledge 目标；规划后播放真实 cuRobo 连续关节轨迹。"
                "这里检查几何、IK、路径和碰撞，不代表 MuJoCo 物理接触已经通过。"
            )
            self._show_target()
            self.result_status.content = (
                f"**已切换 Skill：** `{selected}`。场景与参数保持不变，"
                "上一 Skill 的轨迹已清除。"
            )

    def _set_camera(self):
        case = self.session.case
        center = case.data.body("microwave").xpos.copy()
        handle = case.data.geom("microwave_door_handle_visual").xpos.copy()
        if self.camera.value == "全景":
            base = case.data.body(f"{self.session.group}_base_link").xpos
            look = .42 * handle + .58 * base
            pos = look + np.array([-.85, .65, .62])
        elif self.camera.value == "门接触":
            look = handle
            pos = look + np.array([-.48, .17, .15])
        else:
            phase = self._current_phase()
            self._set_gripper_camera(phase.tcp_pose_world)
            return
        self._apply_camera(pos, look)

    def _set_gripper_camera(self, tcp_pose_world):
        """从夹爪侧面观察两指间隙；连续轨迹播放时随实际 TCP 移动。"""
        look = tcp_pose_world[:3, 3]
        pos = (look + .29 * tcp_pose_world[:3, 1]
               - .08 * tcp_pose_world[:3, 2] + np.array([0., 0., .06]))
        self._apply_camera(pos, look)

    def _apply_camera(self, position, look_at):
        pos = np.asarray(position, dtype=float)
        look = np.asarray(look_at, dtype=float)
        self.server.initial_camera.position = pos.tolist()
        self.server.initial_camera.look_at = look.tolist()
        self.server.initial_camera.up = [0., 0., 1.]
        for client in self.server.get_clients().values():
            client.camera.position = pos.tolist()
            client.camera.look_at = look.tolist()
            client.camera.up_direction = [0., 0., 1.]

    def _show_target(self):
        with self.lock:
            if self.syncing:
                return
            self.view_kind = "knowledge"
            local_index = int(self.stage.value)
            index = self._current_global_index()
            phase = self.session.phases[index]
            self.ghost.set_visible(True)
            mujoco.mj_copyData(self.preview_data, self.session.case.model, self.session.case.data)
            joint = self.session.case.model.joint("microwave_door_joint")
            self.preview_data.qpos[joint.qposadr[0]] = phase.door_angle_rad
            ik_note = "纯 IK 预览关闭"
            if self.preview_ik.value:
                key = (self.session.revision, phase.phase_id)
                if key not in self.ik_cache:
                    robot = self.session.case.metadata["robot"]
                    solver = MuJoCoPoseIK(
                        self.session.case.model, self.preview_data,
                        tuple(robot["joint_groups"][self.session.group]),
                        robot["tcp_sites"][self.session.group])
                    self.ik_cache[key] = solver.solve(phase.tcp_pose_world)
                result = self.ik_cache[key]
                if result.joint_positions is not None:
                    names = self.session.case.metadata["robot"]["joint_groups"][self.session.group]
                    for name, value in zip(names, result.joint_positions, strict=True):
                        self.preview_data.qpos[self.session.case.model.joint(name).qposadr[0]] = value
                budget = self.session.budget_for(phase.phase_id)
                quality = (assess_pose_ik(result,
                    max_position_error_m=budget["max_position_error_m"],
                    max_rotation_error_deg=np.degrees(budget["max_rotation_error_rad"]))
                    if budget else None)
                label = ("严格 IK 可解" if result.reachable else
                         "预算内近似候选" if quality and quality["acceptable"] else
                         "最近尝试姿态（未达目标）")
                ik_note = (f"{label}；位置误差 {result.position_error_m*1000:.1f} mm，"
                           f"姿态误差 {result.rotation_error_deg:.1f}°；未检查路径/场景碰撞")
            mujoco.mj_forward(self.session.case.model, self.preview_data)
            if self.ik_gap is not None:
                self.ik_gap.remove()
                self.ik_gap = None
            if self.preview_ik.value:
                site = self.preview_data.site(
                    self.session.case.metadata["robot"]["tcp_sites"][self.session.group])
                if np.linalg.norm(site.xpos - phase.tcp_pose_world[:3, 3]) > .004:
                    segment = np.asarray([[site.xpos, phase.tcp_pose_world[:3, 3]]], dtype=np.float32)
                    self.ik_gap = self.server.scene.add_line_segments(
                        "/door_skill/ik_gap", segment, (228, 67, 54), thickness=.004)
            self.scene.update_visuals(self.preview_data)
            if self.show_collision.value:
                self._refresh_collision(self.preview_data, phase)
            self.ghost.select(phase.tcp_pose_world, opening_width_m=phase.gripper_width_m)
            wxyz, position = _pose(phase.tcp_pose_world)
            self.target_frame.wxyz, self.target_frame.position = wxyz, position
            self.target_control.wxyz, self.target_control.position = wxyz, position
            self.target_control.visible = phase.name in TRANSITION_NAMES
            self.restore_button.visible = phase.name in TRANSITION_NAMES
            self._write_fields(self.stage_fields, phase.tcp_pose_world)
            budget = self.session.budget_for(phase.phase_id)
            self.syncing = True
            try:
                self.approx_enabled.value = budget is not None
                self.approx_position.value = (budget["max_position_error_m"] * 1000 if budget else 20.)
                self.approx_rotation.value = (np.degrees(budget["max_rotation_error_rad"]) if budget else 10.)
            finally:
                self.syncing = False
            self.phase_info.content = (
                                       f"**{local_index+1}/{len(self.scope_indices)} {phase.phase_id}**  \n"
                                       f"门角 {np.degrees(phase.door_angle_rad):.1f}° · "
                                       f"路径 {phase.motion_kind} · 接触策略 {phase.contact_policy}  \n"
                                       f"目标 TCP {np.round(phase.tcp_pose_world[:3, 3], 3).tolist()} m · "
                                       f"近似 IK {'启用' if budget else '严格'}")
            self.view_status.content = (
                f"**当前视图：Knowledge 目标** · revision {self.session.revision}  \n"
                f"{ik_note}。黄色夹爪始终表示原目标。"
            )
            if self.camera.value == "夹爪侧面":
                self._set_camera()

    def _refresh_collision(self, data, phase):
        """按本帧真实几何重新冻结碰撞代理；绝不沿用编辑前旧位姿。"""
        # 当前即将显示的是 Knowledge 临时快照，之前记住的规划快照已不在
        # 画面里；下次回到轨迹必须重新装载对应阶段的保存快照。
        self._shown_collision_ref = None
        folder = (self.session.output_dir / f"revision_{self.session.revision:03d}" /
                  "visual_collision" / phase.phase_id.replace("/", "_"))
        snapshot = export_collision_scene(
            self.session.case.model, data, self.session.case.metadata["robot"],
            self.session.group, folder,
            moving_body_names=("microwave_door",) if phase.contact_policy != "avoid_door" else ())
        self.scene.refresh(data, snapshot)
        self.scene.set_collision_visible(self.show_collision.value)
        records = snapshot.get("records", [])
        active = [row.get("obstacle_name") for row in records if row.get("obstacle_name")]
        self.collision_info.content = (f"当前目标场景 revision {self.session.revision} · "
                                       f"启用 {len(active)} 个规划障碍；"
                                       f"动态门 {'已排除' if phase.contact_policy != 'avoid_door' else '保留'}。"
                                       f"详情：{folder / 'collision_snapshot.json'}")

    def _toggle_collision(self):
        with self.lock:
            if self.show_collision.value:
                if self.view_kind == "trajectory" and self.session.timeline is not None:
                    sample = self.session.timeline.sample(self.timeline_time_s)
                    segment = self.session.timeline.segments[sample.segment_index]
                    self._show_planned_collision(segment.details, self.session.phase(sample.stage_id))
                else:
                    phase = self._current_phase()
                    self._refresh_collision(self.preview_data, phase)
            else:
                self.scene.set_collision_visible(False)
                self.collision_info.content = "碰撞层未显示；规划时会保存各阶段碰撞清单。"

    def _edit_microwave_transform(self):
        if self.syncing:
            return
        self._apply_edit(lambda: self.session.set_microwave_pose(
            _matrix(self.microwave_control.position, self.microwave_control.wxyz)),
            object_edit=True)

    def _edit_microwave_numbers(self):
        if self.syncing:
            return
        self._apply_edit(lambda: self.session.set_microwave_pose(self._read_fields(self.microwave_fields)),
                         object_edit=True)

    def _edit_stage_transform(self):
        if self.syncing:
            return
        self._apply_edit(lambda: self.session.set_stage_override(
            self._current_phase().phase_id,
            _matrix(self.target_control.position, self.target_control.wxyz)))

    def _edit_stage_numbers(self):
        if self.syncing:
            return
        self._apply_edit(lambda: self.session.set_stage_override(
            self._current_phase().phase_id,
            self._read_fields(self.stage_fields)))

    def _edit_budget(self):
        if self.syncing:
            return
        phase_id = self._current_phase().phase_id
        budget = ({"max_position_error_m": float(self.approx_position.value) / 1000.,
                   "max_rotation_error_rad": float(np.deg2rad(self.approx_rotation.value))}
                  if self.approx_enabled.value else None)
        self._apply_edit(lambda: self.session.set_stage_budget(phase_id, budget))

    def _edit_angles(self):
        if self.syncing:
            return
        self._apply_edit(lambda: self.session.set_angles(self.pull.value, self.push.value))

    def _apply_edit(self, action, *, object_edit=False):
        with self.lock:
            self.preview_playing = self.running = self.timeline_playing = False
            self.planning_requested = self.sequence_planning_requested = False
            self.session.cancel_plan_sequence()
            preferred_phase_id = self._current_phase().phase_id
            try:
                action()
                self.ik_cache.clear()
                pose = _body_pose(self.session.case.data, "microwave")
                self._write_fields(self.microwave_fields, pose)
                self.syncing = True
                try:
                    self.microwave_control.wxyz, self.microwave_control.position = _pose(pose)
                    self.pull.value = float(self.session.config["pull_angle_deg"])
                    self.push.value = float(self.session.config["push_angle_deg"])
                finally:
                    self.syncing = False
                self._reset_scope_after_edit(preferred_phase_id)
                self._show_target()
                self.result_status.content = (
                    f"**规划状态：** 参数已更新为 revision {self.session.revision}；"
                    "旧规划已失效，请重新规划。"
                )
            except Exception as error:
                self.result_status.content = f"**参数未应用：** {type(error).__name__}: {error}"
                self.session.output_dir.mkdir(parents=True, exist_ok=True)
                (self.session.output_dir / "ui_errors.log").write_text(traceback.format_exc(), encoding="utf-8")
                if object_edit:
                    pose = _body_pose(self.session.case.data, "microwave")
                    self.microwave_control.wxyz, self.microwave_control.position = _pose(pose)

    def _restore_stage(self):
        with self.lock:
            self._apply_edit(lambda: self.session.set_stage_override(
                self._current_phase().phase_id, None))

    def _toggle_preview(self):
        self.preview_playing = not self.preview_playing
        self.preview_elapsed = 0.
        self.running = self.timeline_playing = False

    def _selected_skill_id(self):
        return self.skill_id

    def _toggle_timeline(self):
        if self.session.timeline is None:
            self.view_status.content = "**当前视图：** 尚无可播放的规划轨迹。"
            return
        self.preview_playing = self.running = False
        if self.timeline_time_s >= self.session.timeline.duration_s - 1e-9:
            self.timeline_time_s = 0.0
        self.timeline_playing = not self.timeline_playing

    def _seek_timeline(self):
        if self.syncing or self.session.timeline is None:
            return
        self.timeline_playing = False
        self.timeline_time_s = float(self.timeline.value)
        self._show_timeline_frame()

    def _show_timeline_frame(self):
        self.view_kind = "trajectory"
        sample = self.session.seek_timeline(self.timeline_time_s)
        self.scene.update_visuals(self.session.case.data)
        if self.camera.value == "夹爪侧面":
            self._set_gripper_camera(self.session._current_tcp_pose())
        self.ghost.set_visible(False)
        self.target_control.visible = False
        phase = self.session.phase(sample.stage_id)
        details = self.session.timeline.segments[sample.segment_index].details
        approximate = bool((details or {}).get("approximate_ik"))
        actual_motion = (details or {}).get("operation") or phase.motion_kind
        fallback = (details or {}).get("fallback")
        fallback_note = " · 直线失败后按旧 Task1 规则改用普通规划" if fallback else ""
        owner = "Task 过渡" if sample.skill_id == "task_transition" else f"Skill {sample.skill_id}"
        self.phase_info.content = (
            f"**规划轨迹：{sample.stage_id}**  \n"
            f"归属 {owner} · 路径 {actual_motion}{fallback_note} · 接触策略 {phase.contact_policy}  \n"
            f"时间 {sample.time_s:.2f} / {self.session.timeline.duration_s:.2f} s · "
            f"近似 IK {'是' if approximate else '否'}"
        )
        self.view_status.content = (
            f"**当前视图：cuRobo 连续轨迹** · {self.session.timeline_selection} · "
            f"阶段 {sample.segment_index + 1}/{len(self.session.timeline.segments)}。"
            "这是 cuRobo 关节轨迹的运动学播放，尚未做 MuJoCo 物理接触验收。"
        )
        timing = (details or {}).get("timing") or {}
        if timing:
            self.timing_info.content = (
                f"**本阶段耗时：** 总计 {float(timing.get('stage_wall_s') or 0):.3f} s · "
                f"碰撞导出 {float(timing.get('collision_export_s') or 0):.3f} s · "
                f"planner {'复用' if timing.get('planner_cache_hit') else '新建'} "
                f"{float(timing.get('planner_setup_s') or 0):.3f} s · "
                f"候选与规划 {float(timing.get('candidate_and_plan_s') or 0):.3f} s"
            )
        if self.show_collision.value:
            self._show_planned_collision(details, phase)

    def _show_planned_collision(self, details, phase):
        """轨迹播放严格显示该阶段规划时保存的碰撞快照。"""
        reference = (details or {}).get("collision_snapshot_ref")
        if not reference:
            self._shown_collision_ref = None
            self.scene.set_collision_visible(False)
            self.collision_info.content = "本阶段没有保存碰撞快照。"
            return
        if reference == self._shown_collision_ref:
            self.scene.set_collision_visible(True)
            return
        try:
            snapshot = json.loads(Path(reference).read_text(encoding="utf-8"))
            self.scene.refresh(self.session.case.data, snapshot)
            self.scene.set_collision_visible(True)
            self._shown_collision_ref = reference
            bodies = sorted({str(row.get("body_name")) for row in snapshot.get("records", [])
                             if row.get("body_name")})
            preview = "、".join(bodies[:8]) + ("…" if len(bodies) > 8 else "")
            self.collision_info.content = (
                f"**本阶段规划碰撞层：** {len(snapshot.get('records', []))} 个代理；"
                f"对象：{preview or '无'}；门板"
                f"{'参与避障' if phase.contact_policy == 'avoid_door' else '因接触阶段被排除'}。  \n"
                f"快照：`{reference}`"
            )
        except Exception as error:
            self.collision_info.content = f"碰撞快照读取失败：{type(error).__name__}: {error}"

    def _failure_markdown(self, failure: dict) -> str:
        summary = failure.get("failure_summary") or {}
        lines = [
            f"**规划失败：{failure.get('phase_id', '未知阶段')}**",
            str(summary.get("conclusion") or failure.get("message")
                or failure.get("failure_category") or "原因未知"),
        ]
        if summary.get("start_state_valid") is not None:
            lines.append(f"- 起点严格检查：{'通过' if summary['start_state_valid'] else '失败'}")
        if summary.get("ik_seed_count") is not None:
            lines.append(
                f"- 严格 IK：{summary.get('ik_success_count') or 0}/"
                f"{summary['ik_seed_count']} 个 seed 成功"
            )
        if summary.get("local_candidate_count") is not None:
            lines.append(
                f"- 局部旋转候选：{summary.get('local_candidate_success_count') or 0}/"
                f"{summary['local_candidate_count']} 成功"
            )
        if summary.get("best_position_error_m") is not None:
            lines.append(
                f"- 最近候选误差：位置 {summary['best_position_error_m']*1000:.1f} mm，"
                f"姿态 {float(summary.get('best_rotation_error_rad') or 0)*180/np.pi:.1f}°"
            )
        if summary.get("scene_collision") is not None or summary.get("self_collision") is not None:
            lines.append(
                f"- 场景碰撞：{summary.get('scene_collision')}；"
                f"自碰撞：{summary.get('self_collision')}；"
                f"关节限位：{summary.get('joint_limit_violation')}"
            )
        if summary.get("collision_groups"):
            lines.append("- 碰撞对象：" + "、".join(summary["collision_groups"]))
        if summary.get("world_disabled_still_failed") is True:
            lines.append("- 关闭场景碰撞后仍失败：现有证据不支持把失败归因于场景物体")
        lines.append(f"- 完整记录：`{self.session.output_dir}`")
        return "  \n".join(lines)

    def _show_timing(self, row: dict) -> None:
        timing = row.get("timing") or {}
        if not timing:
            return
        self.timing_info.content = (
            f"**阶段 `{row.get('phase_id', '未知')}` 规划计算耗时：** "
            f"总计 {float(timing.get('stage_wall_s') or 0):.3f} s · "
            f"碰撞导出 {float(timing.get('collision_export_s') or 0):.3f} s · "
            f"planner {'复用' if timing.get('planner_cache_hit') else '新建'} "
            f"{float(timing.get('planner_setup_s') or 0):.3f} s · "
            f"候选与规划 {float(timing.get('candidate_and_plan_s') or 0):.3f} s"
        )
        self.planner_info.content = (
            f"**Planner：** `{row.get('planner_mode', self.session.planner_mode.value)}` · "
            f"CUDA Graph {'开启' if row.get('use_cuda_graph') else '关闭'}"
        )

    def _request_plan(self, all_stages):
        with self.lock:
            self.preview_playing = self.timeline_playing = False
            if all_stages:
                self.planning_requested = False
                self.session.cancel_plan_sequence()
                report = self.session.begin_plan_sequence(self._selected_skill_id())
                self.sequence_planning_requested = True
                self.result_status.content = (
                    f"**规划中：0/{report['total_stages']}** · 下一阶段 "
                    f"`{report['next_phase_id']}`。每完成一段会更新一次。"
                )
                return
            if self.session.planned_q is None and self.session.phase_index == 0:
                self.session.start_skill(self._selected_skill_id())
            self.running = False
            self.planning_requested = True
            self.result_status.content = "**规划中：** 正在准备当前阶段…"

    def _stop(self):
        with self.lock:
            self.preview_playing = self.running = self.planning_requested = False
            self.sequence_planning_requested = self.timeline_playing = False
            self.session.cancel_plan_sequence()
            if self.session.timeline is not None:
                self.timeline_time_s = 0.0
                self.syncing = True
                try:
                    self.timeline.value = 0.0
                finally:
                    self.syncing = False
                self._show_timeline_frame()
            else:
                self.session.reset_visual_state()
                self.session.start_skill(self._selected_skill_id())
                self.session.planned_q = None
                self._show_target()
            self.result_status.content = "**已停止。** 已恢复当前检查范围的起点。"

    def _save(self):
        with self.lock:
            try:
                saved = self.session.save_config()
                self.view_status.content = f"**已保存参数：** `{saved}`"
            except Exception as error:
                self.view_status.content = f"**保存失败：** {type(error).__name__}: {error}"

    def tick(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_tick
            self.last_tick = now
            if self.preview_playing:
                self.preview_elapsed += elapsed
            if self.preview_playing and self.preview_elapsed >= .35:
                self.preview_elapsed = 0.
                self.stage.value = (int(self.stage.value) + 1) % len(self.scope_indices)
                self._show_target()
            if self.sequence_planning_requested:
                try:
                    state = self.session._sequence_state or {}
                    next_phase = (self.session.phases[self.session.phase_index].phase_id
                                  if self.session.phase_index < self.session.stop_index else "完成")
                    self.result_status.content = (
                        f"**规划中：{len(state.get('segments', []))}/"
                        f"{state.get('total_stages', '?')}** · 当前阶段 `{next_phase}`…"
                    )
                    report = self.session.plan_sequence_step()
                    self.sequence_planning_requested = not report["done"]
                    if not report["done"]:
                        last = report.get("last_result") or {}
                        timing = last.get("timing") or {}
                        self._show_timing(last)
                        self.result_status.content = (
                            f"**规划中：{report['completed_segments']}/{report['total_stages']}** · "
                            f"刚完成 `{last.get('phase_id')}`，耗时 "
                            f"{float(timing.get('stage_wall_s') or 0):.3f} s；"
                            f"下一阶段 `{report['next_phase_id']}`。"
                        )
                    elif not report["success"]:
                        failure = report.get("first_failure") or {}
                        self._show_timing(failure)
                        if self.session.timeline is not None:
                            self.timeline_time_s = 0.0
                            self.syncing = True
                            try:
                                self.timeline.max = max(self.session.timeline.duration_s, .01)
                                self.timeline.value = 0.0
                            finally:
                                self.syncing = False
                            self._show_timeline_frame()
                            self._show_timing(failure)
                        self.result_status.content = self._failure_markdown(failure)
                    elif report["done"]:
                        self.timeline_time_s = 0.0
                        self.syncing = True
                        try:
                            self.timeline.max = max(float(report["duration_s"]), .01)
                            self.timeline.value = 0.0
                        finally:
                            self.syncing = False
                        self.timeline_playing = True
                        self._show_timeline_frame()
                        # 规划计算时间不是轨迹播放时间。否则上一阶段数秒的
                        # solver 墙钟会让刚开始播放的时间轴直接跳到中后段。
                        self.last_tick = time.monotonic()
                        elapsed = 0.0
                        self.result_status.content = (
                            f"**规划成功：** {report['completed_segments']} 个阶段，"
                            f"墙钟总耗时 {report['elapsed_wall_s']:.3f} s；"
                            "正在播放连续轨迹。"
                        )
                except Exception as error:
                    self.sequence_planning_requested = False
                    self.timeline_playing = False
                    self.result_status.content = f"**完整规划异常：** {type(error).__name__}: {error}"
                    self.session.output_dir.mkdir(parents=True, exist_ok=True)
                    (self.session.output_dir / "ui_errors.log").write_text(
                        traceback.format_exc(), encoding="utf-8")
            if self.planning_requested and self.session.planned_q is None:
                self.planning_requested = False
                try:
                    row = self.session.plan_next()
                    self._show_timing(row)
                    if not row.get("success"):
                        self.running = False
                        self.result_status.content = self._failure_markdown(row)
                    elif row.get("operation") == "gripper":
                        self.scene.update_visuals(self.session.case.data)
                        self.result_status.content = f"**夹爪阶段完成：** `{row['phase_id']}`（运动学）"
                        self.planning_requested = self.running
                    else:
                        timing = row.get("timing") or {}
                        self.result_status.content = (
                            f"**规划成功：** `{row['phase_id']}`；"
                            f"近似 IK：{bool(row.get('approximate_ik'))}；"
                            f"阶段总耗时 {float(timing.get('stage_wall_s') or 0):.3f} s"
                        )
                except Exception as error:
                    self.running = False
                    self.result_status.content = f"**规划异常：** {type(error).__name__}: {error}"
                    self.session.output_dir.mkdir(parents=True, exist_ok=True)
                    (self.session.output_dir / "ui_errors.log").write_text(traceback.format_exc(), encoding="utf-8")
            if self.session.planned_q is not None:
                self.session.advance()
                self.scene.update_visuals(self.session.case.data)
                if self.session.planned_q is None:
                    if self.session.phase_index < self.session.stop_index:
                        self.planning_requested = self.running
                    else:
                        self.running = False
                        self.result_status.content = "**规划轨迹动画已完成。** 尚未做物理接触验收。"
            if self.timeline_playing and self.session.timeline is not None:
                self.timeline_time_s = min(
                    self.timeline_time_s + elapsed * float(self.timeline_speed.value),
                    self.session.timeline.duration_s,
                )
                self.syncing = True
                try:
                    self.timeline.value = self.timeline_time_s
                finally:
                    self.syncing = False
                self._show_timeline_frame()
                if self.timeline_time_s >= self.session.timeline.duration_s - 1e-9:
                    self.timeline_playing = False
