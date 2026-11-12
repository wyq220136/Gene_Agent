"""Bowl 在线 Skill 检查器，共用 ViserVisualScene 和 VisualPlayback。"""
from __future__ import annotations

from functools import wraps
from threading import RLock

import mujoco
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.visualization.microwave.bowl_session import ObjectVisualSession, OBJECT_SKILLS
from gene_agent.tasks.policies.microwave_task.planning.bowl import _body_pose
from gene_agent.visualization.grasp_tuner.gripper_preview import GripperMeshPreview
from gene_agent.visualization.pick_bowl.scene import _geom_pose, _pose
from .model import SceneEdits
from .scene import ViserVisualScene


def _serialized(method):
    """GUI 回调与同步 tick 使用同一可重入锁，避免跨线程修改共享场景。"""
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class ObjectVisualApp:
    """编辑草稿显式应用；规划使用新 GT，画面只播放 planner 返回的轨迹。"""

    selections = OBJECT_SKILLS
    selection_label = '单个 Skill'
    title = 'Bowl Skill'
    sample_bowl_candidates = False
    progressive_preview = False

    def __init__(self, server, *, session=None, initial_selection='pick_bowl', config_path=None):
        if initial_selection not in self.selections:
            raise ValueError(f'不支持的在线 Skill：{initial_selection}')
        self._lock = RLock()
        self.server = server
        self.session = session or ObjectVisualSession(config_path=config_path)
        # 与共享录制器保持公开接口一致；不另建录像用场景或轨迹。
        self.controller = self.session
        self.playback = None
        self.playing = False
        self.time_s = 0.
        self._last_tick = time.monotonic()
        self.planning = False
        self._plan_started_s = None
        self._planning_wall_s = 0.
        self._previewing_prefix = False
        self._preview_rate = 1.
        self._pending_edits = False
        self._shown_stage = None
        self._syncing = False
        self._sample_round = 0
        self._sample_indices = ()
        server.scene.set_up_direction('+z')
        server.gui.configure_theme(dark_mode=False, control_width='large', show_logo=False)
        gui = server.gui
        gui.add_markdown(f'### {self.title} · 在线纯视觉检查\n'
                         '黄色夹爪表示抓取目标预览；蓝线表示 cuRobo 规划轨迹。'
                         '画面中的碗跟随夹爪是规划的附着状态，未验证物理夹持。')
        self.selection = gui.add_dropdown(self.selection_label, options=tuple(self.selections), initial_value=initial_selection)
        self.status = gui.add_markdown('尚未规划；可先编辑摆位，再点击规划。')
        self.enable_drag = gui.add_checkbox('显示物体拖动与旋转控件', initial_value=True)
        self.apply_button = gui.add_button('应用摆位并更新抓取目标与碰撞', color='blue')
        self.run_button = gui.add_button(f'规划所选{self.selection_label}', color='blue')
        self.stop_button = gui.add_button('暂停并回到轨迹起点')
        self.play_button = gui.add_button('播放／暂停规划轨迹', disabled=True)
        self.timeline = gui.add_slider('轨迹时间 (s)', min=0., max=1., step=.01, initial_value=0., disabled=True)
        self.speed = gui.add_slider('播放速度', min=.1, max=2., step=.1, initial_value=1.)
        self.preview_partitions = (gui.add_checkbox('逐分区快速预览已规划轨迹', initial_value=True)
                                   if self.progressive_preview else None)
        self.stage = gui.add_dropdown('规划阶段', options=('尚未规划',), initial_value='尚未规划', disabled=True)
        self.phase_info = gui.add_markdown('抓取目标预览只显示姿态；规划后再查看路径与碰撞结果。')
        self.camera = gui.add_dropdown('相机', options=('前上方', '夹爪侧面'), initial_value='前上方')
        with gui.add_folder('场景显示与目标预览', expand_by_default=False):
            self.show_collision = gui.add_checkbox('显示本阶段规划碰撞代理', initial_value=False)
            self.show_visual = gui.add_checkbox('显示模型外观', initial_value=True)
            self.candidate = gui.add_slider('抓取姿态编号', min=0, max=len(self.session.case.candidates)-1,
                                            step=1, initial_value=0)
            self.preview_button = gui.add_button('查看当前抓取姿态')
            if self.sample_bowl_candidates:
                # Task 页面主要检查整套任务摆位；抽样只改变预览，规划仍使用
                # 原有全部抓取候选及其顺序，不能把预览误当成筛选结果。
                self.candidate.visible = self.preview_button.visible = False
                self.sample_count = gui.add_slider('随机预览抓取姿态数', min=1,
                    max=min(8, len(self.session.case.candidates)), step=1, initial_value=5)
                self.sample_button = gui.add_button('换一组抓取姿态')
        self.pose_fields = {}
        with gui.add_folder('场景摆位草稿（应用后重建）', expand_by_default=False):
            gui.add_markdown('单位为 m / °；先应用再规划。放入碗前会在同一场景规划抓碗，后续动作从抓碗终点开始。')
            for name in ('bowl', 'microwave', 'table'):
                pose = self._object_pose(name)
                angles = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
                values = [*pose[:3, 3], *angles]
                labels = ('X (m)', 'Y (m)', 'Z (m)', 'roll (°)', 'pitch (°)', 'yaw (°)')
                self.pose_fields[name] = [gui.add_number(f'{name} {label}', initial_value=float(value),
                                                        step=.005 if i < 3 else 1.)
                                          for i, (label, value) in enumerate(zip(labels, values))]
            self.door_angle = gui.add_slider('门初始角 (°)', min=0., max=90., step=1.,
                                              initial_value=float(np.degrees(self.session.initial.door_angle_rad)))
            size = 2 * self.session.case.model.geom('scene_table').size[:2]
            self.table_sizes = [gui.add_number(f'桌面 {axis} 长度 (m)', initial_value=float(value), step=.01)
                                for axis, value in zip(('X', 'Y'), size)]
            self.save_button = gui.add_button('保存当前场景配置')
        self.drag_controls = {}
        for name in self.pose_fields:
            wxyz, position = _pose(self._object_pose(name))
            handle = server.scene.add_transform_controls(f'/object_skill/edit/{name}',
                wxyz=wxyz, position=position,
                scale=.17 if name == 'bowl' else .22,
                line_width=3.5, depth_test=False, visible=True)
            self.drag_controls[name] = handle
            handle.on_update(lambda _, name=name: self._pose_from_drag(name))
            for field in self.pose_fields[name]:
                field.on_update(lambda _, name=name: self._pose_from_numbers(name))
        for field in self.table_sizes:
            field.on_update(lambda _: self._preview_drafts())
        self.enable_drag.on_update(lambda _: self._show_drag())
        self._bind_scene()
        self.run_button.on_click(lambda _: self._start_plan())
        self.apply_button.on_click(lambda _: self._apply_edits())
        self.save_button.on_click(lambda _: self._save())
        self.play_button.on_click(lambda _: self._toggle_play())
        self.stop_button.on_click(lambda _: self._stop())
        self.timeline.on_update(lambda _: self._seek())
        self.stage.on_update(lambda _: self._seek_stage())
        self.selection.on_update(lambda _: self._change_selection())
        self.preview_button.on_click(lambda _: self._show_knowledge())
        if self.sample_bowl_candidates:
            self.sample_count.on_update(lambda _: self._show_knowledge())
            self.sample_button.on_click(lambda _: self._next_candidate_sample())
        self.camera.on_update(lambda _: self._set_camera())
        self.show_collision.on_update(lambda _: self._set_visibility())
        self.show_visual.on_update(lambda _: self._set_visibility())
        self._set_camera()
        self._show_knowledge()

    @_serialized
    def _object_pose(self, name):
        return np.asarray(self.session.case.metadata['objects']['sample_poses'][name], dtype=float)

    @_serialized
    def _field_pose(self, name):
        values = [float(field.value) for field in self.pose_fields[name]]
        pose = np.eye(4)
        pose[:3, 3] = values[:3]
        pose[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
        return pose

    @_serialized
    def _set_visibility(self):
        self.scene.set_visual_visible(self.show_visual.value)
        self.scene.set_collision_visible(self.show_collision.value and not self._pending_edits)

    @_serialized
    def _show_drag(self):
        for handle in self.drag_controls.values():
            handle.visible = self.enable_drag.value and not self.planning

    @_serialized
    def _pose_from_drag(self, name):
        if self._syncing or self.planning:
            return
        handle = self.drag_controls[name]
        rotation = Rotation.from_quat(np.asarray(handle.wxyz)[[1, 2, 3, 0]])
        values = [*handle.position, *rotation.as_euler('xyz', degrees=True)]
        self._syncing = True
        try:
            for field, value in zip(self.pose_fields[name], values):
                field.value = float(value)
        finally:
            self._syncing = False
        self._preview_drafts()

    @_serialized
    def _pose_from_numbers(self, name):
        if self._syncing or self.planning:
            return
        self._syncing = True
        try:
            handle = self.drag_controls[name]
            handle.wxyz, handle.position = _pose(self._field_pose(name))
        finally:
            self._syncing = False
        self._preview_drafts()

    @_serialized
    def _preview_drafts(self):
        if self._syncing or self.planning:
            return
        self.playing = False
        self._pending_edits = True
        self.play_button.disabled = self.timeline.disabled = self.stage.disabled = True
        self.show_collision.disabled = True
        self.scene.set_collision_visible(False)
        case = self.session.case
        scratch = mujoco.MjData(case.model)
        scratch.qpos[:] = self._edit_qpos
        mujoco.mj_forward(case.model, scratch)
        self.scene.renderer.update_visuals(scratch)
        for name in self.pose_fields:
            delta = self._field_pose(name) @ np.linalg.inv(self._object_pose(name))
            for geom_id, handle in self.scene.renderer.visual_handles.items():
                if name == 'table':
                    owned = case.model.geom(geom_id).name == 'scene_table'
                else:
                    root_id = case.model.body(name).id
                    body_id = int(case.model.geom_bodyid[geom_id])
                    while body_id and body_id != root_id:
                        body_id = int(case.model.body_parentid[body_id])
                    owned = body_id == root_id
                if owned:
                    handle.wxyz, handle.position = _pose(delta @ _geom_pose(scratch, geom_id))
                    if name == 'table':
                        handle.dimensions = (*[float(field.value) for field in self.table_sizes],
                                             float(2 * case.model.geom(geom_id).size[2]))
        self.phase_info.content = '当前显示的是摆位草稿；抓取目标和规划碰撞尚未更新，请点击“应用摆位”。'

    @_serialized
    def _bind_scene(self):
        # 重建模型前删除本界面持有的旧节点；仅替换 Python renderer 会
        # 让上一版碰撞代理和 TCP 路径留在 Viser 的持久场景缓冲区中。
        if hasattr(self, 'scene'):
            for handle in (*self.scene.renderer.visual_handles.values(),
                           *self.scene.renderer.collision_handles.values()):
                handle.remove()
            self.scene._target_handle.remove()
            if self.scene._path_handle is not None:
                self.scene._path_handle.remove()
            for ghost in self.ghosts:
                if ghost.root is not None:
                    ghost.root.remove()
        case = self.session.case
        self._edit_qpos = case.data.qpos.copy()
        robot = case.metadata['robot']
        self.scene = ViserVisualScene(self.server, case.model, case.data, case.metadata,
                                     gripper_group='right', tcp_site_name=robot['tcp_sites']['right'],
                                     door_joint_name='microwave_door_joint')
        self.ghost = GripperMeshPreview(self.server, case, prefix='/object_skill/knowledge_gripper_0',
                                        color_rgba=(244, 190, 43, 145))
        self.ghosts = [self.ghost]
        self._shown_stage = None
        self.scene.set_visual_visible(self.show_visual.value)
        self.scene.set_collision_visible(self.show_collision.value)

    @_serialized
    def _invalidate(self):
        self.playing = False
        self._pending_edits = False
        self.playback = None
        self.session.run = self.session.playback = None
        self.time_s = 0.
        self.play_button.disabled = self.timeline.disabled = self.stage.disabled = True
        self._syncing = True
        try:
            self.timeline.value = 0.
            self.stage.options = ('尚未规划',)
            self.stage.value = '尚未规划'
        finally:
            self._syncing = False

    @_serialized
    def _change_selection(self):
        if self.planning:
            return
        # 原场景可能停在附着/释放终态；重建名义初态，不能把回放帧当新规划输入。
        self.session.apply_edits(SceneEdits())
        self._invalidate()
        self._bind_scene()
        self._show_knowledge()
        self.status.content = f'已切换{self.selection_label}；请从当前场景初态重新规划。'

    @_serialized
    def _apply_edits(self):
        if self.planning:
            return
        try:
            poses = {}
            for name, fields in self.pose_fields.items():
                values = [float(field.value) for field in fields]
                pose = np.eye(4)
                pose[:3, 3] = values[:3]
                pose[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
                poses[name] = pose
            self.session.apply_edits(SceneEdits(object_poses_world=poses,
                object_joint_positions_rad={'microwave_door_joint': float(np.deg2rad(self.door_angle.value))},
                table_size_xy_m=tuple(float(field.value) for field in self.table_sizes)))
            self._invalidate()
            self._bind_scene()
            self._show_knowledge()
            self._set_busy(False)
            self.status.content = f'场景版本 {self.session.case.revision} 已更新；请重新规划轨迹。'
        except Exception as error:
            self.status.content = f'摆位未应用：{type(error).__name__}: {error}'

    @_serialized
    def _save(self):
        if not self.planning:
            self.phase_info.content = f'配置已保存：`{self.session.save_config()}`'

    @_serialized
    def _show_knowledge(self):
        if self.planning:
            return
        self.playing = False
        # Knowledge 是独立的名义初态几何视图，不保留上一个规划阶段的
        # attachment 碰撞快照或蓝色路径，避免把目标候选误读成已规划动作。
        self.scene.renderer._build_collision({})
        self.scene._shown_snapshot_ref = None
        self._shown_stage = None
        if self.scene._path_handle is not None:
            self.scene._path_handle.remove()
            self.scene._path_handle = None
        self.scene._target_handle.visible = False
        scratch = mujoco.MjData(self.session.case.model)
        scratch.qpos[:] = self._edit_qpos
        mujoco.mj_forward(self.session.case.model, scratch)
        self.scene.renderer.update_visuals(scratch)
        if self.sample_bowl_candidates:
            candidates = self.session.case.candidates
            count = min(int(self.sample_count.value), len(candidates))
            seed = (int(self.session.config.get('seed', 123))
                    + 1000003 * self.session.case.revision + 7919 * self._sample_round)
            self._sample_indices = tuple(int(index) for index in
                np.random.default_rng(seed).choice(len(candidates), size=count, replace=False))
            while len(self.ghosts) < count:
                index = len(self.ghosts)
                self.ghosts.append(GripperMeshPreview(
                    self.server, self.session.case,
                    prefix=f'/object_skill/knowledge_gripper_{index}',
                    color_rgba=(244, 190, 43, 115)))
            for index, ghost in enumerate(self.ghosts):
                ghost.set_visible(index < count)
                if index < count:
                    item = candidates[self._sample_indices[index]]
                    ghost.select(np.asarray(item['T_world_tcp']),
                                 opening_width_m=float(item['parameters']['opening_width_m']))
            ids = '、'.join(candidates[index]['candidate_id'] for index in self._sample_indices)
            self.phase_info.content = (
                f'随机预览 {count}/{len(candidates)} 个抓取姿态（seed {seed}）：{ids}。'
                '仅用于观察姿态；规划仍检查全部候选，结果以实际选中候选为准。')
            return
        item = self.session.case.candidates[int(self.candidate.value)]
        self.ghost.set_visible(True)
        self.ghost.select(np.asarray(item['T_world_tcp']),
                          opening_width_m=float(item['parameters']['opening_width_m']))
        self.phase_info.content = (f"抓取目标：{item['candidate_id']} · "
                                   f'场景版本 {self.session.case.revision}；尚未检查路径与碰撞。')

    @_serialized
    def _next_candidate_sample(self):
        if self.planning:
            return
        self._sample_round += 1
        self._show_knowledge()

    @_serialized
    def _set_busy(self, busy):
        self.run_button.disabled = self.apply_button.disabled = self.selection.disabled = busy
        self.preview_button.disabled = busy
        if self.preview_partitions is not None:
            self.preview_partitions.disabled = busy
        if self.sample_bowl_candidates:
            self.sample_count.disabled = self.sample_button.disabled = busy
        for fields in self.pose_fields.values():
            for field in fields:
                field.disabled = busy
        self.door_angle.disabled = busy
        self.enable_drag.disabled = busy
        for field in self.table_sizes:
            field.disabled = busy
        for control in self.drag_controls.values():
            control.visible = not busy and self.enable_drag.value
        self.show_collision.disabled = busy or self._pending_edits
        self.play_button.disabled = busy or self.playback is None
        self.timeline.disabled = busy or self.playback is None
        self.stage.disabled = busy or self.playback is None

    @_serialized
    def _start_plan(self):
        if self.planning:
            return
        if self._pending_edits:
            self.status.content = '请先应用摆位草稿，再规划新场景。'
            return
        self.playing = False
        self._previewing_prefix = False
        started = time.perf_counter()
        report = self.session.begin_plan(self.selection.value)
        self._invalidate()
        self._plan_started_s = started
        self._planning_wall_s = time.perf_counter() - started
        self.planning = True
        self._set_busy(True)
        self.status.content = report['message'] + '；完成一段后更新进度。'

    @_serialized
    def _toggle_play(self):
        if self.playback is None or self.planning:
            return
        if self.time_s >= self.playback.duration_s:
            self.time_s = 0.
        self.playing = not self.playing
        self._last_tick = time.monotonic()

    @_serialized
    def _stop(self):
        if self.planning:
            self.session.cancel_plan()
            self.planning = False
            self._previewing_prefix = False
            self._plan_started_s = None
            self._set_busy(False)
            self.status.content = '已取消尚未开始的后续规划步骤。'
        self.playing = False
        self.time_s = 0.
        if self.playback is not None:
            self._show_frame()

    @_serialized
    def _seek(self):
        if not self._syncing and self.playback is not None and not self.planning:
            self.playing = False
            self.time_s = float(self.timeline.value)
            self._show_frame()

    @_serialized
    def _seek_stage(self):
        if not self._syncing and self.playback is not None and not self.planning:
            self.playing = False
            self.time_s = self.playback.stage_start_s(self.stage.value)
            self._show_frame()

    @_serialized
    def _show_frame(self):
        if self.playback is None:
            return
        sample = self.playback.sample(self.time_s)
        stage = next(stage for stage in self.session.run.stages if stage.stage_id == sample.stage_id)
        if stage.stage_id != self._shown_stage:
            self.scene.set_stage(stage)
            self._shown_stage = stage.stage_id
            self.scene.set_collision_visible(self.show_collision.value)
        self.scene.show_sample(sample)
        for ghost in self.ghosts:
            ghost.set_visible(False)
        self._syncing = True
        try:
            self.timeline.value = self.time_s
            self.stage.value = sample.stage_id
        finally:
            self._syncing = False
        self.phase_info.content = (
            f'**{stage.stage_id}** · 路径 `{stage.motion_kind}` · 轨迹 {self.time_s:.2f}/{self.playback.duration_s:.2f} s  \n'
            f'本阶段轨迹求解耗时：{format(stage.planning_duration_s, ".3f") + " s" if stage.planning_duration_s is not None else "状态事件"}  \n'
            f'Attach：{"、".join(sample.state.attachments) or "无"}（规划状态）  \n'
            f'碰撞范围：{stage.diagnostics.get("collision_scope", "未提供")}  \n'
            f'启用障碍 {len(stage.collision_enabled)} 个；排除：{"、".join(stage.collision_allowed) or "无"}  \n'
            f'快照：`{str(Path(stage.collision_snapshot_ref).relative_to(self.session.case.output_dir)) if stage.collision_snapshot_ref and Path(stage.collision_snapshot_ref).is_relative_to(self.session.case.output_dir) else stage.collision_snapshot_ref or "无"}`'
        )
        if self.camera.value == '夹爪侧面':
            self._set_camera()

    @_serialized
    def _set_camera(self):
        case = self.session.case
        if self.camera.value == '夹爪侧面':
            site = case.data.site(case.metadata['robot']['tcp_sites']['right'])
            if (self.selection.value in {'insert_bowl', 'extract_bowl'}
                    or (self._shown_stage or '').startswith(('insert_bowl/', 'extract_bowl/'))):
                # 放入和取出都从炉口方向看：相机绑定微波炉坐标系，避免
                # 夹爪滚转把镜头带到打开的门板后面，遮住碗与指尖接触。
                look = .5 * (site.xpos + case.data.body('bowl').xpos)
                rotation = case.data.body('microwave').xmat.reshape(3, 3)
                position = look + rotation @ np.array([-.05, -.45, .35])
            else:
                look = site.xpos.copy()
                rotation = site.xmat.reshape(3, 3)
                position = look + .29 * rotation[:, 1] - .08 * rotation[:, 2] + [0., 0., .06]
        else:
            # 右侧控制栏会遮住画面右边；把机器人基座也纳入构图中心，
            # 让机械臂、微波炉、碗和桌面同时留在可见区域。
            look = (.35 * case.data.body('bowl').xpos
                    + .25 * case.data.body('microwave').xpos
                    + .40 * case.data.body('right_base_link').xpos)
            position = look + np.array([-1.05, .72, .82])
        self.server.initial_camera.position = position.tolist()
        self.server.initial_camera.look_at = look.tolist()
        self.server.initial_camera.up = [0., 0., 1.]
        for client in self.server.get_clients().values():
            client.camera.position = position.tolist()
            client.camera.look_at = look.tolist()
            client.camera.up_direction = [0., 0., 1.]

    @_serialized
    def tick(self):
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now
        if self.planning:
            step_started = None
            try:
                if self._previewing_prefix:
                    self.time_s = min(self.time_s + elapsed * self._preview_rate,
                                      self.playback.duration_s)
                    self._show_frame()
                    if self.time_s >= self.playback.duration_s:
                        self._previewing_prefix = False
                        self.status.content = '已播放当前分区；下一步继续规划。'
                    return
                step_started = time.perf_counter()
                report = self.session.plan_step()
                self._planning_wall_s += time.perf_counter() - step_started
                step_started = None
                if not report['done']:
                    self.status.content = report['message']
                    if self.preview_partitions is not None and self.preview_partitions.value:
                        partial = report.get('partial_run')
                        playback = report.get('partial_playback')
                        if partial is not None and playback is not None:
                            previous_count = len(self.session.run.stages) if self.session.run else 0
                            self.session.run = partial
                            self.session.playback = playback
                            self.playback = playback
                            self.timeline.max = max(.01, playback.duration_s)
                            self.stage.options = tuple(stage.stage_id for stage in partial.stages)
                            first_new = partial.stages[previous_count].stage_id
                            self.time_s = playback.stage_start_s(first_new)
                            self._show_frame()
                            remaining = playback.duration_s - self.time_s
                            # 快速预览最长约 2 秒；结束后仍可按原速检查完整轨迹。
                            self._preview_rate = max(1., remaining / 2.)
                            self._previewing_prefix = remaining > 0.
                            self.status.content += '；正在快速预览刚完成的分区。'
                    return
                run = report['run']
                self.planning = False
                elapsed_s = self._planning_wall_s
                self._plan_started_s = None
                self._previewing_prefix = False
                self.playback = self.session.playback
                self._bind_scene()
                self.timeline.max = max(.01, self.playback.duration_s)
                self.stage.options = tuple(stage.stage_id for stage in run.stages)
                self.time_s = 0.
                self._show_frame()
                self.status.content = (f'**规划成功** · 场景版本 {run.scene_revision} · {len(run.stages)} 阶段。'
                    if run.success else '**规划失败**：' + json.dumps(run.stages[-1].failure, ensure_ascii=False))
                self.status.content += f'  \n累计规划计算耗时：{elapsed_s:.3f} s。'
                self.status.content += f'  \n记录：`{self.session.case.output_dir.parent.name}/{self.session.case.output_dir.name}`'
                stage_time = sum(stage.planning_duration_s or 0. for stage in run.stages)
                self.status.content += (f'  \n已记录的阶段求解耗时合计：{stage_time:.3f} s；'
                                        '计算总耗时还包括候选筛选、场景准备和结果记录。')
                self.playing = run.success
            except Exception as error:
                if step_started is not None:
                    self._planning_wall_s += time.perf_counter() - step_started
                self.planning = False
                self._previewing_prefix = False
                elapsed_s = self._planning_wall_s if self._plan_started_s is not None else None
                self._plan_started_s = None
                self.session.cancel_plan()
                self._invalidate()
                self.status.content = f'规划异常：{type(error).__name__}: {error}'
                if elapsed_s is not None:
                    self.status.content += f'；累计规划计算耗时：{elapsed_s:.3f} s。'
                (self.session.output_dir / 'ui_error.log').write_text(traceback.format_exc(), encoding='utf-8')
            finally:
                self._set_busy(self.planning)
                self._last_tick = time.monotonic()
            return
        if self.playing and self.playback is not None:
            self.time_s = min(self.time_s + elapsed * float(self.speed.value), self.playback.duration_s)
            self._show_frame()
            if self.time_s >= self.playback.duration_s:
                self.playing = False
