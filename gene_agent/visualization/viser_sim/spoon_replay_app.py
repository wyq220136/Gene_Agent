"""在统一 Skill/Task Viser 入口检查勺子任务的实际 MuJoCo 回放。"""

from __future__ import annotations

import bisect
from pathlib import Path
import time

import mujoco
import numpy as np

from gene_agent.feedback.tabletop.spoon_run import SPOON_SKILLS, SPOON_TASKS, find_latest_successful_run, load_replay
from gene_agent.visualization.pick_bowl.scene import PickBowlScene


class SpoonReplayApp:
    """时间轴只读取实际 qpos；切换 Skill 不重新运行规划或物理。"""

    def __init__(self, server, *, initial_selection: str, run_dir: Path | None = None,
                 replay_loader=load_replay, latest_run_finder=find_latest_successful_run,
                 skill_labels=SPOON_SKILLS, task_labels=SPOON_TASKS,
                 intro: str | None = None, camera_look=None, camera_position=None,
                 status_formatter=None):
        self.server = server
        self._replay_loader = replay_loader
        self._status_formatter = status_formatter
        self._camera_look = camera_look or [0.40, -0.03, 0.10]
        self._camera_position = camera_position or [-0.28, 0.60, 0.54]
        self.run_dir = (Path(run_dir).resolve() if run_dir is not None else
                        latest_run_finder())
        self.kind = "task" if initial_selection in task_labels else "skill"
        options = task_labels if self.kind == "task" else skill_labels
        self.report, self.frames, scene_xml = replay_loader(self.run_dir, initial_selection)
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        if any(len(row["qpos"]) != self.model.nq for row in self.frames):
            raise ValueError("run 的 qpos 长度与 MuJoCo 场景不一致")
        self.times = [row["sim_time_s"] for row in self.frames]
        self.selection_id = initial_selection
        self.playing = False
        self._last_tick = time.monotonic()
        server.scene.set_up_direction("+z")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False)
        server.gui.add_markdown(intro or (
            "### 勺子盛球 · MuJoCo 实际执行回放\n"
            "时间轴读取执行时记录的 qpos。夹勺在双侧指垫接触后使用显式仿真辅助 weld；"
            "小球由 MuJoCo 接触运动。"))
        self.selection = server.gui.add_dropdown(
            "Skill" if self.kind == "skill" else "Task",
            options=tuple(options), initial_value=initial_selection)
        self.status = server.gui.add_markdown("")
        self.play_button = server.gui.add_button("播放／暂停")
        self.timeline = server.gui.add_slider(
            "实际执行时间 (s)", min=0., max=max(.01, self._duration),
            step=.01, initial_value=0.)
        self.stage = server.gui.add_dropdown(
            "跳转阶段", options=self._stage_names,
            initial_value=self._stage_names[0])
        self.selection.on_update(lambda _: self._switch_selection())
        self.play_button.on_click(lambda _: self._toggle_play())
        self.timeline.on_update(lambda _: self._seek(float(self.timeline.value)))
        self.stage.on_update(lambda _: self._seek_stage())
        self.renderer = PickBowlScene(server, self.model, self.data, {}, {})
        self._set_camera()
        self._seek(0.)

    @property
    def _duration(self) -> float:
        return self.times[-1] - self.times[0]

    @property
    def _stage_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row["stage"] for row in self.frames))

    def _set_camera(self) -> None:
        look = self._camera_look
        position = self._camera_position
        self.server.initial_camera.position = position
        self.server.initial_camera.look_at = look
        self.server.initial_camera.up = [0., 0., 1.]
        for client in self.server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = look
            client.camera.up_direction = [0., 0., 1.]

    def _switch_selection(self) -> None:
        selected = str(self.selection.value)
        if selected == self.selection_id:
            return
        self.report, self.frames, _ = self._replay_loader(self.run_dir, selected)
        self.times = [row["sim_time_s"] for row in self.frames]
        self.selection_id = selected
        self.playing = False
        self.timeline.max = max(.01, self._duration)
        self.stage.options = self._stage_names
        self.stage.value = self._stage_names[0]
        self._seek(0.)

    def _seek_stage(self) -> None:
        selected = str(self.stage.value)
        for row in self.frames:
            if row["stage"] == selected:
                self._seek(row["sim_time_s"] - self.times[0])
                return

    def _seek(self, relative_time_s: float) -> None:
        relative = float(np.clip(relative_time_s, 0., self._duration))
        index = max(0, bisect.bisect_right(self.times, self.times[0] + relative) - 1)
        row = self.frames[index]
        self.data.qpos[:] = row["qpos"]
        self.data.time = row["sim_time_s"]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_visuals(self.data)
        skills = self.report.get("skills", {})
        cargo_count = len(skills.get("scoop", {}).get("cargo_ball_ids", ()))
        transfer_count = len(skills.get("pour", {}).get("transferred_ball_ids", ()))
        self.status.content = (self._status_formatter(self.report, row, relative,
                                                       self._duration, self.run_dir)
                               if self._status_formatter is not None else (
            f"**{row['stage']}** · {relative:.2f}/{self._duration:.2f} s · "
            f"run: `{self.run_dir}`  \n"
            f"Task 成功：`{self.report['success']}`；盛球 `{cargo_count}` 个，"
            f"转移 `{transfer_count}` 个；夹持：`仿真辅助`；"
            "画面：`实际 MuJoCo 状态`。"))
        if abs(float(self.timeline.value) - relative) > .005:
            self.timeline.value = relative

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing and float(self.timeline.value) >= self._duration - .01:
            self._seek(0.)
        self._last_tick = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now
        if self.playing:
            target = float(self.timeline.value) + elapsed
            self._seek(target)
            if target >= self._duration:
                self.playing = False
