"""只读 MuJoCo Skill 回放：根据 ReplayTimeline 恢复已记录状态并查看。"""

from __future__ import annotations

import json
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from gene_agent.feedback.state import restore_state, state_id
from gene_agent.visualization.replay import ReplayFrame, ReplayTimeline
from gene_agent.visualization.style import camera_presets, configure_camera


def planned_initial_qpos(model: mujoco.MjModel, run: str | Path,
                         robot_metadata: dict | None) -> np.ndarray:
    """恢复场景构建时的机器人 home 和微波炉关节；XML 默认 qpos 不等于规划初态。"""
    initial = mujoco.MjData(model)
    for name, value in (robot_metadata or {}).get("home_joint_positions", {}).items():
        initial.qpos[model.joint(name).qposadr[0]] = float(value)
    metadata_path = Path(run) / "case_meta.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for name, value in metadata.get("joint_overrides", {}).items():
            initial.qpos[model.joint(name).qposadr[0]] = float(value)
    mujoco.mj_forward(model, initial)
    return initial.qpos.copy()


def restore_replay_frame(model: mujoco.MjModel, data: mujoco.MjData,
                         frame: ReplayFrame, run: str | Path, *,
                         robot_metadata: dict | None = None,
                         active_group: str | None = None,
                         baseline_qpos: np.ndarray | None = None) -> str:
    """恢复画面状态；完整采样只含 qpos，故不把它的 state_id 当重算哈希。

    阶段边界旧记录缺 qpos，但 state.npz 含积分状态，恢复后严格核对 state_id。
    旧中间采样只有部分关节位置，不能据此伪造碗和夹爪的同一时刻画面。
    """
    run = Path(run)
    if frame.sample_quality == "boundary_state" and frame.state_ref:
        saved = run / Path(frame.state_ref).parent / "state.npz"
        if saved.is_file():
            restore_state(model, data, saved)
            if state_id(model, data) != frame.state_id:
                raise ValueError(f"阶段边界状态 ID 与保存状态不一致：{frame.stage_id}")
            if frame.qpos is not None and not np.allclose(data.qpos, frame.qpos, atol=1e-8, rtol=0):
                raise ValueError(f"阶段边界 qpos 与保存状态不一致：{frame.stage_id}")
            return "boundary_state_npz"
    if frame.qpos is None:
        if frame.mode == "planned" and frame.joint_positions_rad is not None:
            # 旧 Viser 的纯视觉路径只把规划关节写到场景，不调用 mj_step。
            # 每帧从 XML 初态重建，再应用夹爪与 attach，随机拖动时间轴
            # 也不会沿用上一个阶段的 Bowl 位置或开口。
            mujoco.mj_resetData(model, data)
            baseline = (baseline_qpos if baseline_qpos is not None else
                        planned_initial_qpos(model, run, robot_metadata))
            data.qpos[:] = baseline
            for name, value in zip(frame.joint_names, frame.joint_positions_rad):
                data.qpos[model.joint(name).qposadr[0]] = float(value)
            if (frame.gripper_width_m is not None and robot_metadata is not None
                    and active_group is not None):
                for name in robot_metadata["grippers"][active_group]["joint_names"]:
                    data.qpos[model.joint(name).qposadr[0]] = frame.gripper_width_m / 2.0
            mujoco.mj_forward(model, data)
            attachment = frame.attachment or {}
            if attachment.get("T_tcp_object") is not None:
                if robot_metadata is None or active_group is None:
                    raise ValueError("视觉 attach 帧缺少机器人 TCP 元数据")
                instance = str(attachment["object_instance_id"])
                joint_id = int(model.body(instance).jntadr[0])
                if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    raise ValueError("视觉 attach 对象需要 free joint")
                site = data.site(robot_metadata["tcp_sites"][active_group])
                world_tcp = np.eye(4)
                world_tcp[:3, :3] = site.xmat.reshape(3, 3)
                world_tcp[:3, 3] = site.xpos
                world_object = world_tcp @ np.asarray(attachment["T_tcp_object"], dtype=float)
                adr = int(model.jnt_qposadr[joint_id])
                data.qpos[adr:adr + 3] = world_object[:3, 3]
                data.qpos[adr + 3:adr + 7] = Rotation.from_matrix(
                    world_object[:3, :3],
                ).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)
            elif frame.object_poses_world:
                for instance, raw_pose in frame.object_poses_world.items():
                    pose = np.asarray(raw_pose, dtype=float)
                    if pose.shape != (4, 4) or not np.isfinite(pose).all():
                        raise ValueError(f"{instance} 的规划物体位姿无效")
                    joint_id = int(model.body(instance).jntadr[0])
                    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                        raise ValueError(f"{instance} 的视觉位姿要求 free joint")
                    adr = int(model.jnt_qposadr[joint_id])
                    data.qpos[adr:adr + 3] = pose[:3, 3]
                    data.qpos[adr + 3:adr + 7] = Rotation.from_matrix(
                        pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)
            return "planned_kinematic"
        raise ValueError(
            f"{frame.stage_id} 的 {frame.sample_quality} 缺少完整 qpos；"
            "旧部分采样只能查看保存的阶段边界"
        )
    qpos = np.asarray(frame.qpos, dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError(f"{frame.stage_id} 的完整 qpos 维度或数值无效：期望 {model.nq}")
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    if frame.sim_time_s is not None:
        data.time = float(frame.sim_time_s)
    mujoco.mj_forward(model, data)
    return "replay_qpos"


class SkillMujocoViewer:
    """独立持有 MjData；选择帧只改变该查看器，不调用 Skill 或规划器。"""

    def __init__(self, run: str | Path, *, skill: str | None = None) -> None:
        self.run = Path(run).resolve()
        self.timeline = (ReplayTimeline.load(self.run) if skill is None else
                         ReplayTimeline.load(self.run, skill=skill))
        model_ref = self.timeline.view.model_ref
        if not model_ref:
            raise FileNotFoundError("Skill run 缺少 model/scene.xml，无法进行 MuJoCo 检查")
        self.model = mujoco.MjModel.from_xml_path(str(self.run / model_ref))
        self.data = mujoco.MjData(self.model)
        self.frame: ReplayFrame | None = None
        self.state_source: str | None = None
        self._selected_stage: str | None = None
        self._time_s = 0.0
        metadata = self.run / Path(model_ref).parent / "robot/robot_metadata.json"
        self.robot_metadata = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else None
        self.planned_base_qpos = planned_initial_qpos(self.model, self.run, self.robot_metadata)
        result = self.run / "result.json"
        if not result.is_file():
            result = next((self.run / name for name in (
                "skill_rotate_knob_0_result.json", "skill_open_door_result.json",
                "skill_close_door_result.json") if (self.run / name).is_file()), result)
        report = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        self.active_group = report.get("robot_group") or (
            "right" if getattr(self.timeline.view, "source_format", None) in {"skill_plan", "microwave_actual_record"}
            and self.robot_metadata and "right" in self.robot_metadata["joint_groups"]
            else None
        )

    def select(self, stage_id: str | None = None, mode: str | None = None) -> ReplayFrame:
        self.timeline.select(stage_id, mode=mode or self.timeline.mode)
        self._selected_stage = stage_id
        return self.seek(0.0)

    def seek(self, time_s: float) -> ReplayFrame:
        frame = self.timeline.seek(time_s)
        self.state_source = restore_replay_frame(
            self.model, self.data, frame, self.run,
            robot_metadata=self.robot_metadata, active_group=self.active_group,
            baseline_qpos=self.planned_base_qpos,
        )
        self.frame = frame
        self._time_s = time_s
        return frame

    def sample_times(self) -> tuple[float, ...]:
        """以 ReplayTimeline 已选片段的实际采样为逐帧位置，不做物理插值。"""
        times = set()
        for segment in self.timeline._segments:
            for sample in segment.samples:
                # 旧 run 的部分采样缺碗和夹爪的完整状态，交互播放只跳到可复现的边界。
                boundary = (sample.quality == "boundary_state" and sample.state_ref
                            and (self.run / Path(sample.state_ref).parent / "state.npz").is_file())
                if sample.qpos is not None or boundary or (
                    getattr(self.timeline, "mode", None) == "planned"
                    and getattr(sample, "positions", None) is not None
                ):
                    times.add(segment.start_s + sample.time_s)
        if not times:
            raise ValueError("所选阶段没有完整 qpos 或可复现的阶段边界")
        return tuple(sorted(times))

    def step(self, direction: int) -> ReplayFrame:
        times = self.sample_times()
        if direction > 0:
            selected = next((value for value in times if value > self._time_s + 1e-8), times[-1])
        else:
            selected = next((value for value in reversed(times) if value < self._time_s - 1e-8), times[0])
        return self.seek(selected)

    def _camera(self, name: str):
        presets = camera_presets(
            self.model, self.data, self.timeline.view.instance_id,
            robot_metadata=self.robot_metadata if self.active_group else None,
            active_group=self.active_group if self.robot_metadata else None,
        )
        if name not in presets:
            raise ValueError(f"相机 {name!r} 不可用；可用视角：{sorted(presets)}")
        return configure_camera(mujoco.MjvCamera(), presets[name]), presets[name]

    def save_snapshot(self, output: str | Path, *, camera_name: str = "gripper_side",
                      width: int = 960, height: int = 720) -> dict:
        if self.frame is None:
            self.seek(0.0)
        assert self.frame is not None
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        camera, params = self._camera(camera_name)
        old_width, old_height = self.model.vis.global_.offwidth, self.model.vis.global_.offheight
        try:
            self.model.vis.global_.offwidth = max(width, old_width)
            self.model.vis.global_.offheight = max(height, old_height)
            with mujoco.Renderer(self.model, width=width, height=height) as renderer:
                renderer.update_scene(self.data, camera=camera)
                image = Image.fromarray(renderer.render())
        finally:
            self.model.vis.global_.offwidth = old_width
            self.model.vis.global_.offheight = old_height
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, min(width - 8, 590), 64), fill=(20, 27, 37))
        draw.text((18, 15), f"{self.frame.stage_id} | {self.frame.mode} | {self.frame.sample_quality}", fill="white")
        timing = (f"sim t={self.frame.sim_time_s:.3f} s" if self.frame.sim_time_s is not None
                  else f"planned t={self.frame.sample_time_s:.3f} s")
        draw.text((18, 38), f"{timing} | target={self.timeline.view.instance_id} | gripper={self.active_group}",
                  fill=(225, 232, 240))
        image.save(output)
        manifest = {
            "schema_version": 1, "source_run": str(self.run), "image": str(output),
            "target_instance_id": self.timeline.view.instance_id,
            "stage_id": self.frame.stage_id, "mode": self.frame.mode,
            "global_time_s": self.frame.global_time_s,
            "sample_time_s": self.frame.sample_time_s,
            "sim_time_s": self.frame.sim_time_s,
            "sample_quality": self.frame.sample_quality,
            "state_id": self.frame.state_id,
            "state_source": self.state_source,
            "scene_revision": self.frame.scene_revision,
            "collision_snapshot_ref": self.frame.collision_snapshot_ref,
            "camera_name": camera_name, "camera": params,
            "note": "完整采样只恢复 qpos 与时间；state_id 沿用原运行记录，不代表重新计算的积分状态哈希。",
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest

    def launch(self, *, camera_name: str = "gripper_side") -> None:
        """本地键盘：空格播放/暂停，左右逐样本，方括号切阶段，A 选完整 Skill。"""
        import glfw
        import mujoco.viewer

        stages = [stage.stage_id for stage in self.timeline.view.stages]
        commands: SimpleQueue[int] = SimpleQueue()
        def on_key(key: int) -> None:
            commands.put(key)
        if self.frame is None:
            self.seek(0.0)
        playing = False
        anchor_wall = time.monotonic()
        anchor_time = self._time_s
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=on_key) as viewer:
            while viewer.is_running():
                # 被动 viewer 有自己的渲染线程；所有 MjData 与相机改动都持锁。
                with viewer.lock():
                    while not commands.empty():
                        key = commands.get()
                        if key == glfw.KEY_SPACE:
                            playing = not playing
                            anchor_wall, anchor_time = time.monotonic(), self._time_s
                        elif key in (glfw.KEY_RIGHT, glfw.KEY_LEFT):
                            playing = False
                            self.step(1 if key == glfw.KEY_RIGHT else -1)
                        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
                            playing = False
                            index = stages.index(self._selected_stage) if self._selected_stage in stages else -1
                            index += -1 if key == glfw.KEY_LEFT_BRACKET else 1
                            self.select(stages[index % len(stages)])
                        elif key == glfw.KEY_A:
                            playing = False
                            self.select(None)
                    if playing:
                        current = min(anchor_time + time.monotonic() - anchor_wall,
                                      self.timeline.duration_s)
                        times = self.sample_times()
                        visible = next((value for value in reversed(times) if value <= current + 1e-8), times[0])
                        if visible != self._time_s:
                            self.seek(visible)
                        if current >= self.timeline.duration_s:
                            playing = False
                    camera, _ = self._camera(camera_name)
                    viewer.cam.type = camera.type
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance = camera.distance
                    viewer.cam.azimuth = camera.azimuth
                    viewer.cam.elevation = camera.elevation
                viewer.sync()
                time.sleep(1 / 60)
