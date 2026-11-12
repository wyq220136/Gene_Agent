"""旋钮 Skill/子任务界面，复用共享编辑器与真实轨迹播放组件。"""
from __future__ import annotations

import mujoco
import numpy as np
from gene_agent.visualization.microwave.knob_session import KnobVisualSession, KNOB_SKILLS, KNOB_TASKS, KNOB_PROFILES
from .object_app import ObjectVisualApp, _serialized


class KnobVisualApp(ObjectVisualApp):
    selections = KNOB_SKILLS
    selection_label = '单个 Skill'
    title = '旋钮 Skill'

    def __init__(self, server, *, session=None, initial_selection='rotate_knob', config_path=None):
        super().__init__(server, session=session or KnobVisualSession(config_path=config_path),
                         initial_selection=initial_selection)
        self.candidate.label = '旋钮 Knowledge 候选'
        self.candidate.max = max(1, len(self.session.knob_candidates)-1)
        self.candidate.disabled = len(self.session.knob_candidates) == 1
        self.profile = server.gui.add_dropdown('旋钮候选配置', options=tuple(KNOB_PROFILES),
                            initial_value=self.session.config['knob_profile_variant'])
        self.profile.on_update(lambda _: self._change_profile())
        server.gui.add_markdown('tip_x_5_azimuth_240：tip X 5° / azimuth 240°；default：原始六候选。'
            '两者都从当前场景重新规划。旋钮角只在 adapter 明确提交时更新，未验证接触动力学。')

    @_serialized
    def _change_profile(self):
        if self.planning or self._syncing:
            return
        if self._pending_edits:
            self._syncing = True
            try:
                self.profile.value = self.session.config['knob_profile_variant']
            finally:
                self._syncing = False
            self.status.content = '请先应用摆位草稿，再切换候选配置。'
            return
        self.session.set_profile(self.profile.value)
        self._invalidate()
        self._bind_scene()
        self.candidate.max = max(1, len(self.session.knob_candidates)-1)
        self.candidate.value = 0
        self.candidate.disabled = len(self.session.knob_candidates) == 1
        self._show_knowledge()
        self.status.content = '候选配置已重建；请重新规划。'

    @_serialized
    def _show_knowledge(self):
        if self.planning:
            return
        self.playing = False
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
        if self.candidate.value >= len(self.session.knob_candidates):
            self.candidate.value = 0
        item = self.session.knob_candidates[int(self.candidate.value)]
        self.ghost.set_visible(True)
        self.ghost.select(item.tcp_pose_world, opening_width_m=.08)
        self.phase_info.content = (f'**旋钮 Knowledge：{item.candidate_id}** · '
            f'revision {self.session.case.revision}；仅目标几何，不代表路径可行。')

    @_serialized
    def _set_busy(self, busy):
        super()._set_busy(busy)
        if hasattr(self, 'profile'):
            self.profile.disabled = busy

    # 旋钮含真实零时长 gripper 命令，采用共享 Task1 检查语义而不插入假等待。
    @_serialized
    def _seek_stage(self):
        if self._syncing or self.playback is None or self.planning:
            return
        selected = next(s for s in self.session.run.stages if s.stage_id == self.stage.value)
        super()._seek_stage()
        if selected.state_duration_s == 0.:
            self.scene.set_stage(selected)
            self.scene.set_collision_visible(self.show_collision.value)
            self._shown_stage = selected.stage_id
            self._syncing = True
            try:
                self.stage.value = selected.stage_id
            finally:
                self._syncing = False
            self.phase_info.content = (f'**{selected.stage_id}** · 零时长状态事件  \n'
                f'开口：{selected.gripper_width_m} m；Attach：无  \n'
                f'快照：`{selected.collision_snapshot_ref or "无"}`')

    @_serialized
    def _show_frame(self):
        super()._show_frame()
        if self.playback is not None:
            sample = self.playback.sample(self.time_s)
            self.phase_info.content += f'  \n旋钮角：{np.degrees(sample.state.object_joint_positions_rad.get("knob_0_joint", 0.)):.2f}°'
            stage = next(s for s in self.session.run.stages if s.stage_id == sample.stage_id)
            if stage.state_update_at_end:
                self.phase_info.content += '  \n**画面在本阶段末尾更新旋钮角；尚未仿真夹爪带动旋钮的物理接触。**'
            if stage.diagnostics.get('direct_joint_command'):
                self.phase_info.content += '  \n**末关节直接命令；该段未调用 cuRobo。**'

    @_serialized
    def _set_camera(self):
        if self.camera.value != '夹爪侧面':
            return super()._set_camera()
        case = self.session.case
        look = case.data.site('knob_0_site').xpos.copy()
        rotation = case.data.body('microwave').xmat.reshape(3,3)
        position = look + rotation @ np.array([.20, -.36, .16])
        self.server.initial_camera.position = position.tolist()
        self.server.initial_camera.look_at = look.tolist()
        self.server.initial_camera.up = [0.,0.,1.]
        for client in self.server.get_clients().values():
            client.camera.position = position.tolist()
            client.camera.look_at = look.tolist()
            client.camera.up_direction = [0.,0.,1.]


class KnobTaskVisualApp(KnobVisualApp):
    selections = KNOB_TASKS
    selection_label = '子任务 Task'
    title = '微波炉旋钮子任务'

    def __init__(self, server, *, initial_selection='microwave_knob_rotate', **kwargs):
        super().__init__(server, initial_selection=initial_selection, **kwargs)
