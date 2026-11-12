"""完整微波炉 Task1 在线界面，复用同一编辑器、场景与真实状态时间轴。"""
from __future__ import annotations

import numpy as np
from gene_agent.tasks.policies.microwave_task.visual_session import Task1VisualSession, TASK1_SELECTION
from gene_agent.tasks.policies.microwave_task.visual import TASK1_STEP_ORDER
from .object_app import ObjectVisualApp, _serialized


class Task1VisualApp(ObjectVisualApp):
    selections = {TASK1_SELECTION: '完整微波炉 Task1'}
    selection_label = '完整 Task'
    title = '微波炉 Task1 · 7 个连续分区'
    sample_bowl_candidates = True
    progressive_preview = True

    def __init__(self, server, *, session=None, config_path=None, initial_selection=TASK1_SELECTION):
        active_session = session or Task1VisualSession(config_path=config_path)
        legacy = active_session.config.get('variant', 'legacy_viser') == 'legacy_viser'
        self.title = ('Task1 · 拉门后直接设置门角' if legacy
                      else 'Task1 · 拉门与推门')
        self.mode_note = ('把手拉到 25° 后，门角直接更新到 85°；中间没有机器人推门轨迹。'
                          if legacy else '把手拉门后由机器人继续推门；下方可以调整目标角。')
        self._mode_run = None
        super().__init__(server, session=active_session, initial_selection=initial_selection)
        self.partition = server.gui.add_dropdown('Task 分区', options=TASK1_STEP_ORDER,
                                                initial_value=TASK1_STEP_ORDER[0], disabled=True)
        self.partition.on_update(lambda _: self._seek_partition())
        with server.gui.add_folder('门操作目标角（应用后重建）', expand_by_default=False):
            server.gui.add_markdown('单位为度；修改后点击“应用摆位并更新抓取目标与碰撞”，'
                                    '再规划完整 Task。当前配置若没有推门轨迹，请先加载推门配置。')
            door = active_session.catalog.door_session.config
            self.pull_target = server.gui.add_number('拉把手目标角 (°)',
                initial_value=float(door['pull_angle_deg']), step=1., disabled=legacy)
            self.push_target = server.gui.add_number('推门目标角 (°)',
                initial_value=float(door['push_angle_deg']), step=1., disabled=legacy)
        server.gui.add_markdown('**当前配置**：' + self.mode_note
            + f'  \n旋钮 profile：`{active_session.config.get("knob_profile_variant", "tip_x_5_azimuth_240")}`。'
            + '编辑后从关门初态重新规划全部7个分区。')
        self._set_busy(False)

    @_serialized
    def tick(self):
        super().tick()
        if not self.planning and self.session.run is not None and self._mode_run is not self.session.run:
            self.status.content += '  \n**动作说明**：' + self.mode_note
            self._mode_run = self.session.run

    @_serialized
    def _object_pose(self, name):
        # Task1 的共享桌面先应用旧几何，再叠加用户编辑；源 XML metadata
        # 不能代替当前 model 的实际位姿。编辑器从同一 MuJoCo 初态读取。
        case = self.session.case
        target = case.data.geom('scene_table') if name == 'table' else case.data.body(name)
        pose = np.eye(4)
        pose[:3, :3] = target.xmat.reshape(3, 3)
        pose[:3, 3] = target.xpos
        return pose

    @_serialized
    def _set_busy(self, busy):
        super()._set_busy(busy)
        # Task1 的首段必须从关门初态规划；catalog 会拒绝非零门初角。
        self.door_angle.disabled = True
        if hasattr(self, 'partition'):
            self.partition.disabled = busy or self.playback is None
        if hasattr(self, 'pull_target'):
            legacy = self.session.config.get('variant', 'legacy_viser') == 'legacy_viser'
            self.pull_target.disabled = self.push_target.disabled = busy or legacy

    @_serialized
    def _apply_edits(self):
        if self.planning:
            return
        if self.session.config.get('variant', 'legacy_viser') == 'legacy_viser':
            return super()._apply_edits()
        pull, push = float(self.pull_target.value), float(self.push_target.value)
        if not 0. < pull < push <= 90.:
            self.status.content = '门目标角未应用：必须满足 0° < 拉门目标角 < 推门目标角 ≤ 90°。'
            return
        previous = dict(self.session.config)
        revision = self.session.case.revision
        self.session.config.update(pull_angle_deg=pull, push_angle_deg=push)
        super()._apply_edits()
        if self.session.case.revision == revision:
            # 场景编辑器已经显示具体错误；恢复未提交的角度配置。
            self.session.config = previous

    @_serialized
    def _seek_stage(self):
        if self._syncing or self.playback is None or self.planning:
            return
        selected = next(stage for stage in self.session.run.stages if stage.stage_id == self.stage.value)
        super()._seek_stage()
        if selected.state_duration_s == 0.:
            # 零时长事件不占用虚构的播放时间，但仍要允许用户选中该事件
            # 检查其语义和快照，不能被后一个同起点阶段抢走下拉框选择。
            self.scene.set_stage(selected)
            self.scene.set_collision_visible(self.show_collision.value)
            self._shown_stage = selected.stage_id
            self._syncing = True
            try:
                self.stage.value = selected.stage_id
            finally:
                self._syncing = False
            self.phase_info.content = (f'**{selected.stage_id}** · 零时长状态事件  \n'
                f'开口：{selected.gripper_width_m} m；Attach：{"、".join(selected.attach_after) or "无"}  \n'
                f'碰撞范围：{selected.diagnostics.get("collision_scope", "状态事件；未调用 planner")}  \n'
                f'快照：`{selected.collision_snapshot_ref or "无"}`')

    @_serialized
    def _seek_partition(self):
        if self._syncing or self.playback is None or self.planning:
            return
        stage = next((stage for stage in self.session.run.stages
                      if stage.diagnostics.get('task_partition') == self.partition.value), None)
        if stage is not None:
            self.playing = False
            self.time_s = self.playback.stage_start_s(stage.stage_id)
            self._show_frame()

    @_serialized
    def _show_frame(self):
        super()._show_frame()
        if self.playback is None:
            return
        sample = self.playback.sample(self.time_s)
        stage = next(stage for stage in self.session.run.stages if stage.stage_id == sample.stage_id)
        if hasattr(self, 'partition'):
            self._syncing = True
            try:
                self.partition.value = stage.diagnostics['task_partition']
            finally:
                self._syncing = False
        self.phase_info.content += (f'  \n门角：{np.degrees(sample.state.door_angle_rad):.1f}°；'
            f'旋钮角：{np.degrees(sample.state.object_joint_positions_rad.get("knob_0_joint", 0.)):.1f}°')
        if stage.diagnostics.get('external_state_update'):
            self.phase_info.content += '  \n**门角由任务状态直接更新；没有机器人推门轨迹。**'
        if stage.diagnostics.get('direct_joint_command'):
            self.phase_info.content += '  \n**末关节直接命令；该段未调用 cuRobo。**'

    @_serialized
    def _set_camera(self):
        case = self.session.case
        if self.camera.value != '夹爪侧面':
            # 完整任务需要同时看见微波炉、机器人与桌面碗；单 Bowl Skill 的
            # 炉口近景会裁掉机器人。这里与门 Task 使用同一个前上方机位。
            handle = np.asarray(case.data.geom('microwave_door_handle_visual').xpos)
            base = np.asarray(case.data.body('right_base_link').xpos)
            look = .42 * handle + .58 * base
            position = look + np.array([-.85, .65, .62])
        else:
            site = case.data.site(case.metadata['robot']['tcp_sites']['right'])
            stage = self._shown_stage or ''
            if '/insert_bowl/' in stage or '/extract_bowl/' in stage or '/place_bowl/' in stage:
                look = .5 * (site.xpos + case.data.body('bowl').xpos)
                position = look + case.data.body('microwave').xmat.reshape(3,3) @ np.array([-.05,-.45,.35])
            else:
                from .app import _gripper_side_camera_position
                look = site.xpos.copy()
                position = _gripper_side_camera_position(look, site.xmat.reshape(3,3))
        self.server.initial_camera.position = position.tolist()
        self.server.initial_camera.look_at = look.tolist()
        self.server.initial_camera.up = [0., 0., 1.]
        for client in self.server.get_clients().values():
            client.camera.position = position.tolist()
            client.camera.look_at = look.tolist()
            client.camera.up_direction = [0., 0., 1.]
