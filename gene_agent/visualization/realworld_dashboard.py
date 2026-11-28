"""一个 Viser 内的六视图：初始对齐、规划、点云、夹爪、物体及联合回放。"""
from types import SimpleNamespace
import json
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.visualization.afford_scene import _wxyz
from gene_agent.visualization.pick_bowl.scene import PickBowlScene
from gene_agent.visualization.realworld_scene import RealworldSceneApp
from gene_agent.visualization.realworld_pick_debug import _add_line_segments_compat
from gene_agent.visualization.realworld_timeline import apply_observed_robot, bbox_segments

PANELS = ("1 · 初始标定 / 点云对齐", "2 · 规划轨迹", "3 · 感知点云逐帧",
          "4 · 点云 + 夹爪状态", "5 · 点云 + 物体 axis / bbox", "6 · 点云 + 夹爪 + 物体")


class ScopedScene:
    """复用既有显示器，用独立父节点控制整组可见性。"""
    def __init__(self, scene, prefix):
        self.scene, self.prefix = scene, prefix

    def __getattr__(self, name):
        method = getattr(self.scene, name)
        if name.startswith("add_"):
            return lambda path, *args, **kwargs: method(self.prefix + path, *args, **kwargs)
        return method


def scoped_server(server, prefix):
    return SimpleNamespace(scene=ScopedScene(server.scene, prefix), gui=server.gui,
                           initial_camera=server.initial_camera)


class RealworldDashboard:
    def __init__(self, server, scene, model, data, rgbd, warning, timeline, *, follow=False):
        self.server, self.scene, self.model, self.data = server, scene, model, data
        self.timeline, self.follow = timeline, follow
        self.base_qpos = data.qpos.copy()
        self.panel, self.dirty, self.playing = 0, True, False
        self.last_tick = self.last_refresh = time.monotonic()
        self.dynamic_handles, self.path_handle = [], None
        self.previous_path = None
        server.gui.add_markdown("## 真机统一回放\n\n六个视图共用保存的 world 坐标系，单位米。"
            "动态视图播放运行中已采集的 RGB-D；不重新感知、配准、规划或控制硬件。")
        with server.gui.add_folder("视图选择（六个按钮）"):
            self.panel_buttons = []
            for index, label in enumerate(PANELS):
                button = server.gui.add_button(label)
                button.on_click(lambda _, i=index: self.select_panel(i))
                self.panel_buttons.append(button)
        self.title = server.gui.add_markdown("")
        self.static_root = server.scene.add_frame("/unified/static", show_axes=False)
        with server.gui.add_folder("初始现场与对齐图层") as self.static_folder:
            self.static = RealworldSceneApp(scoped_server(server, "/unified/static"),
                                            scene, model, data, rgbd, warning)
            self.alignment_info = server.gui.add_markdown("")
            self._sdk_alignment()
        self.dynamic_root = server.scene.add_frame("/unified/perception", show_axes=False)
        self.robot_root = server.scene.add_frame("/unified/robot", show_axes=False)
        self.dynamic_data = mujoco.MjData(model)
        self.dynamic_data.qpos[:] = self.base_qpos
        mujoco.mj_forward(model, self.dynamic_data)
        self.robot = PickBowlScene(scoped_server(server, "/unified/robot"), model,
                                   self.dynamic_data, scene.get("robot_metadata", {}), {})
        for geom_id, handle in self.robot.visual_handles.items():
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(model.geom_bodyid[geom_id])) or "").lower()
            # 动态背景只有本帧点云；旧静态物体 mesh 不遮住新的感知结果。
            handle.visible = any(name.startswith(group + "_") or name == group
                                 for group in ("left", "right"))
        with server.gui.add_folder("感知时间轴（视图3～6共用）") as self.dynamic_folder:
            self.frame = server.gui.add_slider("观测序号", min=0,
                max=max(1, len(timeline.frames)-1), step=1, initial_value=0,
                disabled=not timeline.frames)
            self.sample_mode = server.gui.add_checkbox("播放夹爪原始检测样本（需要带时间戳的新日志）", initial_value=False,
                                                       disabled=not timeline.feedback)
            self.feedback_frame = server.gui.add_slider("夹爪检测样本", min=0,
                max=max(1, len(timeline.feedback)-1), step=1, initial_value=0, disabled=not timeline.feedback)
            self.sample_mode.on_update(lambda _: self.invalidate())
            self.feedback_frame.on_update(lambda _: self.invalidate())
            self.previous = server.gui.add_button("上一帧")
            self.next = server.gui.add_button("下一帧")
            self.play = server.gui.add_button("播放 / 暂停感知")
            self.fps = server.gui.add_slider("感知播放帧率", min=1, max=15, step=1, initial_value=2)
            self.baseline = server.gui.add_checkbox("叠加第一帧点云（青色，固定参照）", initial_value=False)
            self.frame_info = server.gui.add_markdown("")
            self.rgb_image = server.gui.add_image(np.zeros((2, 2, 3), dtype=np.uint8), label="所选帧 RGB / 当帧 SAM3 mask")
            self.state_info = server.gui.add_markdown("")
            self.frame.on_update(lambda _: self.invalidate())
            self.baseline.on_update(lambda _: self.invalidate())
            self.previous.on_click(lambda _: self.step(-1))
            self.next.on_click(lambda _: self.step(1))
            self.play.on_click(lambda _: self.toggle_play())
        with server.gui.add_folder("规划时间轴（保存轨迹，非实测执行）") as self.plan_folder:
            options = tuple(f"{i:03d} · {row['label']}" for i, row in enumerate(timeline.trajectories)) or ("无保存轨迹",)
            self.plan = server.gui.add_dropdown("轨迹段", options=options, initial_value=options[0],
                                                disabled=not timeline.trajectories)
            self.plan_frame = server.gui.add_slider("轨迹采样帧", min=0, max=1, step=1, initial_value=0,
                                                   disabled=not timeline.trajectories)
            self.plan_play = server.gui.add_button("播放 / 暂停轨迹", disabled=not timeline.trajectories)
            self.plan_info = server.gui.add_markdown("没有保存轨迹时保持空白，不补造规划。")
            self.plan.on_update(lambda _: self.change_plan())
            self.plan_frame.on_update(lambda _: self.invalidate())
            self.plan_play.on_click(lambda _: self.toggle_play())
        self.notice = server.gui.add_markdown("\n\n".join(timeline.warnings))
        self.select_panel(0)
        self.tick()

    def invalidate(self):
        self.dirty = True

    def select_panel(self, index):
        self.panel, self.playing, self.dirty = index, False, True

    def step(self, delta):
        self.playing = False
        if self.sample_mode.value and self.panel in (3, 5) and self.timeline.feedback:
            self.feedback_frame.value = min(max(0, int(self.feedback_frame.value)+delta), len(self.timeline.feedback)-1)
        elif self.timeline.frames:
            self.frame.value = min(max(0, int(self.frame.value)+delta), len(self.timeline.frames)-1)
        self.dirty = True

    def toggle_play(self):
        self.playing = not self.playing
        self.last_tick = time.monotonic()

    def change_plan(self):
        self.plan_frame.value = 0
        self.previous_path = None
        self.playing, self.dirty = False, True

    def _sdk_alignment(self):
        source = self.scene["provenance"]
        details = []
        for group, pose in source.get("tcp_poses_world", {}).items():
            sdk = np.asarray(pose)
            self.server.scene.add_frame(f"/unified/static/sdk_tcp/{group}", position=sdk[:3, 3],
                                        wxyz=_wxyz(sdk), axes_length=.13, axes_radius=.003)
            site = self.scene.get("robot_metadata", {}).get("tcp_sites", {}).get(group)
            if site:
                urdf = np.eye(4)
                urdf[:3, :3] = np.asarray(self.data.site(site).xmat).reshape(3, 3)
                urdf[:3, 3] = self.data.site(site).xpos
                self.server.scene.add_frame(f"/unified/static/urdf_tcp/{group}", position=urdf[:3, 3],
                    wxyz=_wxyz(urdf), axes_length=.08, axes_radius=.002)
                error = np.linalg.inv(sdk) @ urdf
                details.append(f"{group} SDK FK TCP / 模型 FK TCP："
                    f"{np.linalg.norm(error[:3, 3])*1000:.3f} mm / "
                    f"{Rotation.from_matrix(error[:3, :3]).magnitude()*180/np.pi:.3f}°")
        self.alignment_info.content = ("\n\n".join(details) or
            "本日志未保存 SDK FK TCP；只能核对已保存模型、SDK基座与相机坐标链，不能补算成历史 SDK 实测。")

    def _clear_dynamic(self):
        for handle in self.dynamic_handles:
            handle.remove()
        self.dynamic_handles = []

    def _dynamic(self):
        self._clear_dynamic()
        if not self.timeline.frames:
            self.frame_info.content = "本运行只有初始场景包，没有多帧 capture；动态视图缺数据。"
            return
        index = min(int(self.frame.value), len(self.timeline.frames)-1)
        feedback = None
        if self.sample_mode.value and self.panel in (3, 5) and self.timeline.feedback:
            feedback = self.timeline.feedback[min(int(self.feedback_frame.value), len(self.timeline.feedback)-1)]
            preceding = [i for i, r in enumerate(self.timeline.frames) if r["timestamp_ms"] <= feedback["timestamp_ms"]]
            index = preceding[-1] if preceding else 0
        row = self.timeline.frames[index]
        try:
            cloud = self.timeline.cloud(index)
            self.dynamic_handles.append(self.server.scene.add_point_cloud("/unified/perception/cloud",
                points=cloud["points"], colors=cloud["colors"], point_size=.0025))
            drift = ""
            if self.baseline.value:
                first = self.timeline.cloud(0)
                self.dynamic_handles.append(self.server.scene.add_point_cloud("/unified/perception/reference",
                    points=first["points"], colors=(30, 220, 245), point_size=.0015))
                # 同内外参及分辨率才比较同像素深度；不做 ICP。
                ids, a, b = np.intersect1d(first["ids"], cloud["ids"], return_indices=True)
                same_camera = (row["camera"]["serial"] == self.timeline.frames[0]["camera"]["serial"]
                    and row["camera"]["intrinsic_matrix"] == self.timeline.frames[0]["camera"]["intrinsic_matrix"]
                    and cloud["rgb"].shape == first["rgb"].shape
                    and np.allclose(row["T_world_camera"], self.timeline.frames[0]["T_world_camera"], atol=1e-9, rtol=0))
                if len(ids) and same_camera:
                    delta = cloud["depth_m"][b]-first["depth_m"][a]
                    drift = f"同采样像素深度差中位数 {np.median(delta)*1000:.2f} mm，|差|P95 {np.percentile(abs(delta),95)*1000:.2f} mm。含物体运动/遮挡，不是纯传感器漂移估计。"
            show_objects = self.panel in (4, 5)
            self.rgb_image.image = cloud["overlay"] if show_objects else cloud["rgb"]
            provenance = row["provenance"]
            sample_note = ""
            if feedback is not None:
                provenance = {**provenance,
                    "gripper_widths_m": {**provenance.get("gripper_widths_m", {}), feedback["group"]: feedback["width_m"]},
                    "live_state": {**provenance.get("live_state", {}), "gripper_width_sources": {
                        **provenance.get("live_state", {}).get("gripper_width_sources", {}), feedback["group"]: "SDK原始检测样本"}}}
                if feedback["joint_state"]:
                    provenance["joint_positions_by_group_rad"] = {
                        **provenance.get("joint_positions_by_group_rad", {}),
                        feedback["group"]: feedback["joint_state"]["positions_rad"]}
                lag = feedback["timestamp_ms"]-row["timestamp_ms"]
                sample_note = (f"\n\n夹爪检测时刻 {feedback['timestamp_ms']:.3f}；背景帧与检测相差 {lag:.1f} ms。"
                    "点云为最近已采帧；初始化检测早于第一帧时使用首个后续帧，不冒称同步拍摄。"
                    f"\n\n原始反馈：`{json.dumps(feedback['sdk_reading'], ensure_ascii=False)}`")
            objects = self.timeline.objects(index) if show_objects else []
            for i, obj in enumerate(objects):
                pose = obj["pose"]
                self.dynamic_handles.append(self.server.scene.add_frame(f"/unified/perception/pose_{i}",
                    position=pose[:3, 3], wxyz=_wxyz(pose), axes_length=.09, axes_radius=.002))
                self.dynamic_handles.append(self.server.scene.add_label(f"/unified/perception/name_{i}",
                    text=obj["object_id"] + " · FoundationPose", position=pose[:3, 3]))
                if obj["bounds"] is not None:
                    self.dynamic_handles.append(_add_line_segments_compat(self.server.scene,
                        f"/unified/perception/bbox_{i}", points=bbox_segments(obj["bounds"], pose),
                        colors=(255, 180, 35), thickness=.002, thickness_units="world"))
            show_robot = self.panel in (3, 5)
            has_robot = bool(provenance.get("joint_positions_by_group_rad"))
            self.robot_root.visible = show_robot and has_robot
            if show_robot and has_robot:
                apply_observed_robot(self.model, self.dynamic_data, self.base_qpos, provenance)
                self.robot.update_visuals(self.dynamic_data)
                for group, pose in ({} if feedback is not None else provenance.get("tcp_poses_world", {})).items():
                    pose = np.asarray(pose)
                    self.dynamic_handles.append(self.server.scene.add_frame(f"/unified/perception/gripper_tcp_{group}",
                        position=pose[:3, 3], wxyz=_wxyz(pose), axes_length=.1, axes_radius=.003))
            widths = provenance.get("gripper_widths_m", {})
            sources = provenance.get("live_state", {}).get("gripper_width_sources", {})
            states = [f"{group} 开口 {width*1000:.3f} mm · `{sources.get(group, '旧日志未标注来源')}`"
                      for group, width in widths.items()] if show_robot else []
            self.frame_info.content = (f"**{index+1}/{len(self.timeline.frames)} · frame={row['frame_number']}**\n\n"
                f"采集 Unix ms `{row['timestamp_ms']:.3f}`；目录 `{row['label']}`。\n\n"
                "坐标链：固定 T_world_sdk × 本帧已记录 T_sdk_camera × 相机深度反投影。无逐帧自动配准。\n\n" + drift)
            retained = provenance.get("perception_refresh", {}).get("object_observations", {})
            self.state_info.content = ("\n\n".join(states) + ("\n\n缺少本帧关节状态，不显示机器人。" if show_robot and not has_robot else "")
                + sample_note + "\n\n夹爪按程序消费状态离散播放；复用值不是当帧重新检测，未测量帧不插值。"
                + (f"\n\n当帧原始 FoundationPose：{', '.join(o['object_id'] for o in objects) or '无'}；"
                   f"当帧 SAM3 mask：{len(cloud['masks'])}。tracking 帧没有重新分割时不贴旧 mask。"
                   "\n\n物体沿用来源（未画成当帧 FP）：\n```json\n"
                   + json.dumps(retained, ensure_ascii=False, indent=2) + "\n```" if show_objects else ""))
        except (ValueError, KeyError, OSError) as error:
            self._clear_dynamic()
            self.robot_root.visible = False
            self.frame_info.content = f"本帧无法显示：{error}；没有替换成其他帧。"

    def _planning(self):
        if not self.timeline.trajectories:
            return
        index = int(self.plan.value.split(" · ")[0])
        row = self.timeline.trajectories[index]
        self.plan_frame.max = max(1, len(row["positions_rad"])-1)
        frame = min(int(self.plan_frame.value), len(row["positions_rad"])-1)
        if self.previous_path != index:
            if self.path_handle is not None:
                self.path_handle.remove()
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.base_qpos
            group = row["joint_names"][0].split("_joint_")[0]
            site = self.scene["robot_metadata"]["tcp_sites"][group]
            points = []
            for q in row["positions_rad"]:
                for name, value in zip(row["joint_names"], q):
                    scratch.qpos[self.model.joint(name).qposadr[0]] = value
                mujoco.mj_forward(self.model, scratch)
                points.append(scratch.site(site).xpos.copy())
            points = np.asarray(points)
            self.path_handle = (_add_line_segments_compat(self.server.scene, "/unified/planned_tcp",
                points=np.stack((points[:-1], points[1:]), axis=1), colors=(30, 150, 255),
                thickness=.004, thickness_units="world") if len(points) > 1 else None)
            self.previous_path = index
        self.data.qpos[:] = self.base_qpos
        for name, value in zip(row["joint_names"], row["positions_rad"][frame]):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        self.static.geometry.update_visuals(self.data)
        self.static.geometry.set_robot_spheres_visible(False)
        self.plan_info.content = (f"**保存的规划 / FK 预览** · {frame+1}/{len(row['positions_rad'])}；"
            f"轨迹时间 {row['times_s'][frame]:.3f}s\n\n来源 `{row['source']}`。背景固定为初始点云；"
            "物体和规划障碍属于初始快照，不表示逐帧碰撞验证或真实连续执行。")

    def tick(self):
        now = time.monotonic()
        if self.follow and now-self.last_refresh >= 2.:
            self.last_refresh = now
            previous = [(r["label"], r["revision"]) for r in self.timeline.frames]
            self.timeline.refresh()
            if self.timeline.refresh_reports():
                self.plan.options = tuple(f"{i:03d} · {r['label']}" for i, r in enumerate(self.timeline.trajectories)) or ("无保存轨迹",)
                self.plan.disabled = self.plan_frame.disabled = self.plan_play.disabled = not self.timeline.trajectories
                self.sample_mode.disabled = self.feedback_frame.disabled = not self.timeline.feedback
                self.feedback_frame.max = max(1, len(self.timeline.feedback)-1)
                self.dirty = True
            if [(r["label"], r["revision"]) for r in self.timeline.frames] != previous:
                self.frame.max = max(1, len(self.timeline.frames)-1)
                self.frame.disabled = not self.timeline.frames
                self.dirty = True
            self.notice.content = "\n\n".join(self.timeline.warnings)
        if self.playing:
            period = .08 if self.panel == 1 else 1 / self.fps.value
            if now-self.last_tick >= period:
                self.last_tick = now
                if self.panel == 1 and self.timeline.trajectories:
                    row = self.timeline.trajectories[int(self.plan.value.split(" · ")[0])]
                    self.plan_frame.value = (int(self.plan_frame.value)+1) % len(row["positions_rad"])
                elif self.panel in (3, 5) and self.sample_mode.value and self.timeline.feedback:
                    self.feedback_frame.value = (int(self.feedback_frame.value)+1) % len(self.timeline.feedback)
                elif self.panel >= 2 and self.timeline.frames:
                    self.frame.value = (int(self.frame.value)+1) % len(self.timeline.frames)
                self.dirty = True
        if not self.dirty:
            return
        self.dirty = False
        self.title.content = "### " + PANELS[self.panel]
        self.static_root.visible = self.panel < 2
        self.dynamic_root.visible = self.panel >= 2
        self.robot_root.visible = False
        self.static_folder.visible = self.panel == 0
        self.plan_folder.visible = self.panel == 1
        self.dynamic_folder.visible = self.panel >= 2
        self.sample_mode.visible = self.feedback_frame.visible = self.panel in (3, 5)
        if self.path_handle is not None:
            self.path_handle.visible = self.panel == 1
        if self.panel == 0:
            self.data.qpos[:] = self.base_qpos
            mujoco.mj_forward(self.model, self.data)
            self.static.geometry.update_visuals(self.data)
        elif self.panel == 1:
            self._planning()
        else:
            self._dynamic()
