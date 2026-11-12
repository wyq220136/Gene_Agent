"""在线纯视觉门 Skill/Task 检查器；共用 VisualRun 与 Viser 场景。

门的 Knowledge 和 cuRobo 规划仍由 DoorDebugSession 负责。本界面只负责
编辑摆位、逐阶段触发原规划、显示真实轨迹、碰撞快照和结构化失败。
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
import time
import traceback
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession
from gene_agent.visualization.microwave.door_session import OnlineDoorVisualController
from gene_agent.visualization.grasp_tuner.microwave_scene import (
    _body_pose, set_preview_microwave_pose,
)
from gene_agent.visualization.pick_bowl.scene import _pose

from .model import VisualStage
from .scene import ViserVisualScene


DOOR_SKILLS = {
    "handle_pull": "抓把手拉门",
    "push_open": "从内侧推开",
    "push_close": "从外侧推关",
}
DOOR_TASKS = {
    "microwave_door_cycle": "开门并关门",
    "microwave_door_open": "开门",
    "microwave_door_close": "关门",
}


def stage_description(stage: VisualStage) -> str:
    """人和 Codex 读取同一阶段事实，不从画面推测碰撞或 IK 结论。"""
    status = "规划成功" if stage.success else "规划失败；未生成轨迹"
    lines = [f"**{stage.stage_id}** · {status} · 路径 `{stage.motion_kind}`"]
    if stage.planning_duration_s is not None:
        lines.append(f"规划耗时 {stage.planning_duration_s:.3f} s")
    display_groups = stage.diagnostics.get("collision_display_groups")
    enabled = "、".join(display_groups or stage.collision_enabled) or "无"
    allowed = "、".join(stage.collision_allowed) or "无"
    count = stage.diagnostics.get("collision_obstacle_count")
    amount = f"（{int(count)} 个代理）" if count is not None else ""
    lines.append(f"参与规划碰撞：{enabled}{amount}")
    policy = stage.diagnostics.get("contact_policy")
    if policy:
        lines.append(f"阶段接触策略：`{policy}`")
    if stage.collision_allowed:
        lines.append(f"允许接触或动态排除：{allowed}")
    disabled = stage.diagnostics.get("collision_disabled_body_names") or ()
    if disabled:
        lines.append(f"本阶段未加入规划碰撞：{'、'.join(disabled)}")
    lines.append(f"Attach：{'、'.join(stage.attach_after) or '无'}")
    approximate = stage.diagnostics.get("approximate_ik")
    if isinstance(approximate, dict):
        parts = ["近似 IK"]
        if approximate.get("position_error_m") is not None:
            parts.append(f"位置误差 {float(approximate['position_error_m'])*1000:.1f} mm")
        if approximate.get("rotation_error_rad") is not None:
            parts.append(f"姿态误差 {np.degrees(approximate['rotation_error_rad']):.1f}°")
        lines.append("；".join(parts))
    elif approximate:
        lines.append("近似 IK：已使用；数值见运行记录")
    if not stage.success:
        failure = stage.failure or {}
        lines.append(f"失败代码：`{failure.get('code') or failure.get('category') or 'unknown'}`")
        lines.append(str(failure.get("message") or failure.get("conclusion") or failure))
    if stage.collision_snapshot_ref:
        ref = Path(stage.collision_snapshot_ref)
        lines.append(f"碰撞快照：`{ref.parent.name}/{ref.name}`；完整路径见运行清单")
    return "  \n".join(lines)


def _matrix(position: Any, wxyz: Any) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        np.asarray(wxyz, dtype=float)[[1, 2, 3, 0]]
    ).as_matrix()
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def _gripper_side_camera_position(position: np.ndarray,
                                  rotation: np.ndarray) -> np.ndarray:
    """将观察方向约束在水平面附近，腕部翻转也不会变成俯视镜头。"""
    side = np.asarray(rotation[:, 1], dtype=float).copy()
    side[2] = 0.
    length = float(np.linalg.norm(side))
    if length < .25:
        side = np.array([-.8, .6, 0.])
    else:
        side /= length
    return np.asarray(position, dtype=float) + .32 * side + np.array([0., 0., .09])


class DoorVisualApp:
    """Skill 和 Task 分别实例化；共用阶段显示和时间轴实现。"""

    def __init__(self, server: Any, *, mode: str, config_path: Path | None = None,
                 initial_selection: str | None = None):
        if mode not in {"skill", "task"}:
            raise ValueError("mode 必须是 skill 或 task")
        self.server = server
        self.mode = mode
        self.options = DOOR_SKILLS if mode == "skill" else DOOR_TASKS
        self.selection_id = initial_selection or next(iter(self.options))
        if self.selection_id not in self.options:
            raise ValueError(f"未知的 {mode}：{self.selection_id}")
        self.session = DoorDebugSession(**({"config_path": config_path} if config_path else {}))
        self.session.start_skill(self._session_selection())
        self.controller = OnlineDoorVisualController(self.session)
        self.playback = None
        self.playing = False
        self.time_s = 0.
        self.last_tick = time.monotonic()
        self._syncing = False
        self._pose_applied = True
        self._plan_started_s = None
        # Viser GUI 回调与主线程 tick 并发；切换阶段会重建碰撞节点。
        # 串行更新可防止滑块拖动时修改已删除的 BoxHandle。
        self._state_lock = RLock()
        self._shown_stage_id: str | None = None
        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 238, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False,
                                   show_share_button=False, brand_color=(53, 112, 169))
        case = self.session.case
        self.scene = ViserVisualScene(
            server, case.model, case.data, case.metadata,
            gripper_group=self.session.group,
            tcp_site_name=case.metadata["robot"]["tcp_sites"][self.session.group],
            door_joint_name="microwave_door_joint",
        )
        pose = _body_pose(case.data, "microwave")
        self.pending_microwave_pose = pose.copy()
        gui = server.gui
        kind = "Skill" if mode == "skill" else "Task"
        gui.add_markdown(f"### 在线纯视觉 {kind} 检查\n轨迹来自 cuRobo；没有运行物理接触。")
        self.selection = gui.add_dropdown(
            f"选择 {kind}", options=tuple(self.options), initial_value=self.selection_id)
        self.apply_button = gui.add_button("应用新位姿并重建场景", color="blue")
        self.plan_button = gui.add_button(f"规划所选 {kind}", color="blue")
        self.status = gui.add_markdown("当前场景尚未规划。")
        self.stage_slider = gui.add_slider("阶段", min=0, max=1, step=1, initial_value=0)
        self.stage_info = gui.add_markdown("")
        self.time_slider = gui.add_slider("轨迹时间 (s)", min=0., max=1., step=.01,
                                          initial_value=0.)
        self.play_button = gui.add_button("播放／暂停连续轨迹")
        self.visual_toggle = gui.add_checkbox("显示 visual", initial_value=True)
        self.collision_toggle = gui.add_checkbox("显示本阶段规划碰撞", initial_value=False)
        self.collision_group = gui.add_dropdown(
            "只看碰撞物体组", options=("全部",), initial_value="全部")
        self.collision_wireframe = gui.add_checkbox("碰撞线框", initial_value=False)
        self.camera = gui.add_dropdown("相机", options=("前上方", "夹爪侧面"),
                                       initial_value="前上方")
        with gui.add_folder("微波炉位姿（位置 m，旋转 rad）", expand_by_default=False):
            self.pose_fields = {}
            xyz = pose[:3, 3]
            angles = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
            for label, initial in zip(("x", "y", "z", "roll", "pitch", "yaw"),
                                      (*xyz, *angles), strict=True):
                field = gui.add_number(label, initial_value=float(initial), step=.01)
                self.pose_fields[label] = field
                field.on_update(self._locked(lambda _: self._pose_from_numbers()))
        self.microwave_control = server.scene.add_transform_controls(
            "/viser_sim/edit_microwave", scale=.22, line_width=3.5,
            depth_test=False,
            wxyz=_pose(pose)[0], position=_pose(pose)[1])
        self.selection.on_update(self._locked(lambda _: self._change_selection()))
        self.apply_button.on_click(self._locked(lambda _: self._apply_pose()))
        self.plan_button.on_click(self._locked(lambda _: self._start_plan()))
        self.stage_slider.on_update(self._locked(lambda _: self._seek_stage()))
        self.time_slider.on_update(self._locked(lambda _: self._seek_time()))
        self.play_button.on_click(self._locked(lambda _: self._toggle_play()))
        self.visual_toggle.on_update(self._locked(lambda _: self.scene.set_visual_visible(
            self.visual_toggle.value)))
        self.collision_toggle.on_update(self._locked(lambda _: self.scene.set_collision_visible(
            self.collision_toggle.value)))
        self.collision_group.on_update(self._locked(lambda _: self._select_collision_group()))
        self.collision_wireframe.on_update(self._locked(lambda _: self.scene.renderer.set_collision_wireframe(
            self.collision_wireframe.value)))
        self.camera.on_update(self._locked(lambda _: self._set_camera()))
        self.microwave_control.on_update(self._locked(lambda _: self._pose_from_transform()))
        self._show_knowledge_preview()
        self._set_camera()

    def _locked(self, callback):
        """所有 GUI 修改与主循环共用一把可重入锁。"""
        def invoke(event):
            with self._state_lock:
                return callback(event)
        return invoke

    def _session_selection(self) -> str | None:
        return (self.selection_id if self.mode == "skill"
                or self.selection_id != "microwave_door_cycle" else None)

    def _scope(self):
        if self.mode == "skill":
            return tuple(phase for phase in self.session.phases
                         if phase.skill_id == self.selection_id)
        if self.selection_id == "microwave_door_open":
            return tuple(phase for phase in self.session.phases
                         if phase.skill_id in {"handle_pull", "push_open"})
        if self.selection_id == "microwave_door_close":
            return tuple(phase for phase in self.session.phases
                         if phase.skill_id == "push_close")
        return tuple(self.session.phases)

    def _show_knowledge_preview(self) -> None:
        phases = self._scope()
        self._syncing = True
        try:
            self.stage_slider.max = max(1, len(phases) - 1)
            self.stage_slider.value = min(int(self.stage_slider.value), len(phases) - 1)
        finally:
            self._syncing = False
        phase = phases[int(self.stage_slider.value)]
        if self.selection_id == "microwave_door_close":
            # 预览展示名义已开门姿态；点击规划后 begin 会重置为关闭初态，
            # 并重新计算真实开门前序，不沿用此预览状态充当规划起点。
            door = self.session.case.model.joint("microwave_door_joint")
            self.session.case.data.qpos[door.qposadr[0]] = phase.door_angle_rad
            mujoco.mj_forward(self.session.case.model, self.session.case.data)
        preview = VisualStage(
            phase.phase_id, phase.skill_id, "door_state", True,
            target_tcp_world=phase.tcp_pose_world,
            door_angle_rad=phase.door_angle_rad,
        )
        self.scene.set_stage(preview)
        self.scene.renderer.update_visuals(self.session.case.data)
        self.scene.set_collision_visible(False)
        self.stage_info.content = (
            f"**Knowledge 目标：{phase.phase_id}**  \n"
            f"期望路径 `{phase.motion_kind}` · 接触策略 `{phase.contact_policy}` · "
            f"门角 {np.degrees(phase.door_angle_rad):.1f}°。"
        )

    def _invalidate(self) -> None:
        self.controller.cancel()
        self.controller.run = None
        self.controller.playback = None
        self.playback = None
        self.playing = False
        self._shown_stage_id = None
        self.scene.set_stage(VisualStage(
            "unapplied_scene", "scene", "door_state", True))
        self.scene.set_collision_visible(False)
        self._syncing = True
        try:
            self.collision_group.options = ("全部",)
            self.collision_group.value = "全部"
        finally:
            self._syncing = False

    def _select_collision_group(self) -> None:
        if self._syncing:
            return
        group = str(self.collision_group.value)
        self.scene.renderer.select_collision_group(
            None if group == "全部" else group,
            isolate=group != "全部",
        )

    def _preview_pose(self, pose: np.ndarray) -> None:
        self.pending_microwave_pose = pose.copy()
        self._pose_applied = False
        self._invalidate()
        set_preview_microwave_pose(self.session.case, pose)
        self.scene.renderer.update_visuals(self.session.case.data)
        self.status.content = "摆位预览已改变；原规划和碰撞快照已失效。请先应用新位姿，再规划。"
        self._set_camera()

    def _pose_from_transform(self) -> None:
        if self._syncing:
            return
        pose = _matrix(self.microwave_control.position, self.microwave_control.wxyz)
        angles = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
        self._syncing = True
        try:
            for label, value in zip(("x", "y", "z", "roll", "pitch", "yaw"),
                                    (*pose[:3, 3], *angles), strict=True):
                self.pose_fields[label].value = float(value)
        finally:
            self._syncing = False
        self._preview_pose(pose)

    def _pose_from_numbers(self) -> None:
        if self._syncing:
            return
        pose = np.eye(4)
        pose[:3, 3] = [float(self.pose_fields[name].value) for name in ("x", "y", "z")]
        pose[:3, :3] = Rotation.from_euler(
            "xyz", [float(self.pose_fields[name].value)
                    for name in ("roll", "pitch", "yaw")]
        ).as_matrix()
        self._syncing = True
        try:
            self.microwave_control.wxyz, self.microwave_control.position = _pose(pose)
        finally:
            self._syncing = False
        self._preview_pose(pose)

    def _apply_pending(self) -> None:
        configured = self.session.config.get("microwave_pose_world")
        if configured is None or not np.allclose(configured, self.pending_microwave_pose):
            self.session.set_microwave_pose(self.pending_microwave_pose)
        self.scene.state.capture_base()

    def _apply_pose(self) -> None:
        """显式提交预览位姿，重建门 Knowledge；此处不启动规划。"""
        try:
            self._invalidate()
            self._apply_pending()
            self.session.start_skill(self._session_selection())
            self.scene.state.capture_base()
            self._show_knowledge_preview()
            self._pose_applied = True
            self.status.content = (f"新位姿已应用，场景 revision {self.session.revision}；"
                                   "请点击规划所选 Skill/Task。")
        except Exception as error:
            self._show_error(error)

    def _change_selection(self) -> None:
        if self._syncing:
            return
        self._invalidate()
        self._apply_pending()
        self._pose_applied = True
        self.selection_id = str(self.selection.value)
        self.session.start_skill(self._session_selection())
        self.scene.state.capture_base()
        self._show_knowledge_preview()
        self.status.content = f"已选 `{self.selection_id}`；点击按钮规划当前场景。"
        if self.selection_id == "microwave_door_close":
            self.status.content += "关门会先重新规划开门前序，回放仅显示关门阶段。"

    def _start_plan(self) -> None:
        if not self._pose_applied:
            self.status.content = "请先点击“应用新位姿并重建场景”，再规划。"
            return
        try:
            self._plan_started_s = time.perf_counter()
            self._invalidate()
            self._apply_pending()
            report = self.controller.begin(self._session_selection())
            self.scene.state.capture_base()
            self.status.content = (
                f"规划中：0/{report['total_stages']}；当前 `{report['next_phase_id']}`。"
            )
            self.last_tick = time.monotonic()
        except Exception as error:
            self._show_error(error)

    def _show_error(self, error: Exception) -> None:
        self.playing = False
        self.status.content = f"**异常：** {type(error).__name__}: {error}"
        if self._plan_started_s is not None:
            self.status.content += f"；本次规划墙钟耗时 {time.perf_counter() - self._plan_started_s:.3f} s"
            self._plan_started_s = None
        self.session.output_dir.mkdir(parents=True, exist_ok=True)
        (self.session.output_dir / "ui_errors.log").write_text(
            traceback.format_exc(), encoding="utf-8")

    def _seek_stage(self) -> None:
        if self._syncing:
            return
        if self.playback is None:
            self._show_knowledge_preview()
            return
        index = min(int(self.stage_slider.value), len(self.controller.run.stages) - 1)
        stage = self.controller.run.stages[index]
        self.time_s = self.playback.stage_start_s(stage.stage_id)
        self.playing = False
        self._show_frame()

    def _seek_time(self) -> None:
        if self._syncing or self.playback is None:
            return
        self.playing = False
        self.time_s = float(self.time_slider.value)
        self._show_frame()

    def _toggle_play(self) -> None:
        if self.playback is None:
            return
        if self.time_s >= self.playback.duration_s - 1e-9:
            self.time_s = 0.
        self.playing = not self.playing
        self.last_tick = time.monotonic()

    def _show_frame(self) -> None:
        if self.playback is None:
            return
        sample = self.playback.sample(self.time_s)
        stage = next(item for item in self.controller.run.stages
                     if item.stage_id == sample.stage_id)
        if self._shown_stage_id != stage.stage_id:
            self.scene.set_stage(stage)
            self._shown_stage_id = stage.stage_id
            groups = ("全部",) + tuple(self.scene.renderer.collision_groups)
            selected = str(self.collision_group.value)
            if selected not in groups:
                selected = "全部"
            self._syncing = True
            try:
                self.collision_group.options = groups
                self.collision_group.value = selected
            finally:
                self._syncing = False
            self._select_collision_group()
        self.scene.show_sample(sample)
        self.scene.set_collision_visible(self.collision_toggle.value)
        self.stage_info.content = stage_description(stage)
        index = self.controller.run.stages.index(stage)
        self._syncing = True
        try:
            self.stage_slider.value = index
            self.time_slider.value = self.time_s
        finally:
            self._syncing = False
        if self.camera.value == "夹爪侧面":
            self._set_camera()

    def _set_camera(self) -> None:
        case = self.session.case
        if self.camera.value == "夹爪侧面":
            site = case.data.site(case.metadata["robot"]["tcp_sites"][self.session.group])
            look = np.asarray(site.xpos).copy()
            rotation = np.asarray(site.xmat).reshape(3, 3)
            # 相机随 TCP 平移，但不要随夹爪翻腕绕到正上方：那会遮住
            # 指尖与门/把手的接触。
            position = _gripper_side_camera_position(look, rotation)
        else:
            handle = np.asarray(case.data.geom("microwave_door_handle_visual").xpos)
            base = np.asarray(case.data.body(f"{self.session.group}_base_link").xpos)
            look = .42 * handle + .58 * base
            position = look + np.array([-.85, .65, .62])
        self.server.initial_camera.position = position.tolist()
        self.server.initial_camera.look_at = look.tolist()
        self.server.initial_camera.up = [0., 0., 1.]
        for client in self.server.get_clients().values():
            client.camera.position = position.tolist()
            client.camera.look_at = look.tolist()
            client.camera.up_direction = [0., 0., 1.]

    def tick(self) -> None:
        with self._state_lock:
            self._tick_locked()

    def _tick_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_tick
        self.last_tick = now
        if self.controller.planning:
            try:
                report = self.controller.step()
                if not report["done"]:
                    self.status.content = (
                        f"规划中：{report['completed_segments']}/{report['total_stages']}；"
                        f"下一段 `{report['next_phase_id']}`。"
                    )
                    return
                self.playback = self.controller.playback
                self.scene.state.capture_base()
                self.time_s = 0.
                self._shown_stage_id = None
                self._syncing = True
                try:
                    self.stage_slider.max = max(1, len(self.controller.run.stages) - 1)
                    self.stage_slider.value = 0
                    self.time_slider.max = max(.01, self.playback.duration_s)
                    self.time_slider.value = 0.
                finally:
                    self._syncing = False
                self._show_frame()
                self.controller.session.output_dir.mkdir(parents=True, exist_ok=True)
                (self.controller.session.output_dir / "visual_run_manifest.json").write_text(
                    json.dumps(self.controller.run.manifest(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if report["success"]:
                    self.status.content = (
                        f"规划成功：{len(self.controller.run.stages)} 段；"
                        f"计算墙钟 {report['elapsed_wall_s']:.3f} s；"
                        f"动画时长 {self.playback.duration_s:.2f} s。"
                    )
                    self.playing = True
                else:
                    self.status.content = (stage_description(self.controller.run.stages[-1])
                                           + f"  \n本次规划墙钟耗时 {report['elapsed_wall_s']:.3f} s。")
                    self.playing = False
                self._plan_started_s = None
                self.last_tick = time.monotonic()
                return
            except Exception as error:
                self._show_error(error)
                return
        if self.playing and self.playback is not None:
            self.time_s = min(self.time_s + elapsed, self.playback.duration_s)
            self._show_frame()
            if self.time_s >= self.playback.duration_s - 1e-9:
                self.playing = False


__all__ = ["DoorVisualApp", "DOOR_SKILLS", "DOOR_TASKS", "stage_description"]
