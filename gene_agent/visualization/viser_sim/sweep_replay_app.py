"""在通用 Skill/Task 入口查看清扫 run 的 MuJoCo 实际状态。"""

from __future__ import annotations

import bisect
from pathlib import Path
import time

import mujoco
import numpy as np

from gene_agent.feedback.tabletop.sweep_run import find_latest_compatible_run, load_replay
from gene_agent.visualization.pick_bowl.scene import PickBowlScene


SWEEP_SKILLS = {"pick_brush": "抓取刷子 · MuJoCo 实际回放",
                "sweep_cube": "刷子清扫 · MuJoCo 实际回放",
                "pick_brush_long": "从上向下抓取长刷 · MuJoCo 实际回放",
                "sweep_long_brush": "长刷清扫 · MuJoCo 实际回放"}
SWEEP_TASKS = {"sweep_cube_into_dustpan": "刷子扫方块入簸箕 · MuJoCo 完整任务",
               "sweep_long": "长刷扫方块入簸箕 · MuJoCo 完整任务"}
DEFAULT_SCENE = Path("data/realworld/brush_cube_dustpan_0/scene.xml")


class SweepReplayApp:
    """时间轴只播放运行中保存的实际 qpos，不重新规划或执行。"""

    def __init__(self, server, *, initial_selection: str, run_dir: Path | None = None):
        self.server = server
        self.run_dir = (Path(run_dir).resolve() if run_dir is not None else
                        find_latest_compatible_run(DEFAULT_SCENE))
        self.kind = "task" if initial_selection in SWEEP_TASKS else "skill"
        options = SWEEP_TASKS if self.kind == "task" else SWEEP_SKILLS
        self.report, self.frames, scene_xml = load_replay(self.run_dir, initial_selection)
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        if any(len(row["qpos"]) != self.model.nq for row in self.frames):
            raise ValueError("run 的 qpos 长度与当前 MuJoCo 场景不一致")
        self.times = [row["sim_time_s"] for row in self.frames]
        self.selection_id = initial_selection
        self.playing = False
        self._last_tick = time.monotonic()
        server.scene.set_up_direction("+z")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False)
        server.gui.add_markdown(
            "### 刷子清扫 · MuJoCo 实际执行回放\n"
            "时间轴来自执行时保存的 qpos；抓刷柄经双侧接触核验后使用仿真辅助 weld。"
            "方块位移与入盘由 MuJoCo 接触产生。")
        self.selection = server.gui.add_dropdown(
            "清扫 Skill" if self.kind == "skill" else "清扫 Task",
            options=tuple(options), initial_value=initial_selection)
        self.status = server.gui.add_markdown("")
        from gene_agent.visualization.operation_evidence import operation_evidence_markdown
        shared_description = operation_evidence_markdown(self.report)
        if shared_description: server.gui.add_markdown(shared_description)
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
        look = [0.38, 0., 0.10]
        position = [-0.35, 0.65, 0.58]
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
        self.report, self.frames, _ = load_replay(self.run_dir, selected)
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
        self.status.content = (
            f"**{row['stage']}** · {relative:.2f}/{self._duration:.2f} s · "
            f"run: `{self.run_dir}`  \n"
            f"完整任务成功：`{self.report['success']}`；"
            "夹持：`仿真辅助`；当前画面：`实际 MuJoCo 状态`。")
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
