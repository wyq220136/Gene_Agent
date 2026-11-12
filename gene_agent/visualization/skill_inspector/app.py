"""已保存 Skill run 的 Viser 检查界面；显式提交编辑时新建规划 revision。"""

from __future__ import annotations

import html
import json
import traceback
from functools import wraps
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.visualization.replay import ReplayTimeline
from gene_agent.simulation.skill_viewer import planned_initial_qpos
from .scene import SkillInspectorScene, stage_facts


def _details(items: list[str], title: str) -> str:
    rows = "<br>".join(html.escape(item) for item in items)
    return f"<details><summary>{title}（{len(items)}）</summary>{rows}</details>"


_MODE_LABELS = {"planned": "规划轨迹", "actual": "仿真实际执行"}
_PLANNING_LABELS = {"not_applicable": "无需路径规划", "planned": "规划成功",
                    "unsupported": "后端不支持", "failed": "规划失败", "not_executed": "未规划"}
_EXECUTION_LABELS = {"completed": "已执行", "failed": "执行失败",
                     "not_executed": "未执行"}
_STAGE_LABELS = {"open": "开爪", "pregrasp": "移动到预抓取位姿",
                 "approach": "直线接近", "final": "直线接近", "close": "闭爪",
                 "lift": "直线抬升", "hold": "保持", "preplace": "移动到炉前",
                 "insert": "直线放入", "release": "松爪", "detach": "解除附着",
                 "retreat_1": "直线撤离一", "retreat_2": "直线撤离二"}


def _seconds(value: Any) -> str:
    return "未记录" if value is None else f"{float(value):.3f} s"


def _stage_status(stage: Any) -> str:
    planning = _PLANNING_LABELS.get(stage.planning_status, stage.planning_status)
    execution = _EXECUTION_LABELS.get(stage.execution_status, stage.execution_status)
    return f"{_STAGE_LABELS.get(stage.name, stage.name)}：{planning}，{execution}"


def _scene_update(function):
    """串行化 GUI 回调和播放循环对 MuJoCo data、Viser 节点的修改。

    Viser 的普通 GUI 回调运行在线程池；主线程同时调用 tick。阶段切换会销毁并
    重建碰撞节点，因此不能让另一线程仍拿着旧 handle 更新 visible。RLock 允许
    选择阶段、逐样本与播放内部再次调用 seek，而不产生自锁。
    """
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            return function(self, *args, **kwargs)
    return wrapped


class SkillInspectorApp:
    """回放使用同一份阶段采样；提交编辑后用同一套控件打开新的 Skill run。"""

    def __init__(self, server: Any, run_dir: str | Path, *, stage_id: str | None = None,
                 mode: str | None = None, show_collision: bool = False,
                 show_visual: bool = True, initial_time_s: float = 0.0,
                 skill: str | None = None,
                 read_only: bool = False,
                 scene_edit_root: Path | None = None,
                 prepare_scene: Callable | None = None,
                 run_skill_planning: Callable | None = None) -> None:
        self.server = server
        self.root = Path(run_dir).resolve()
        self.timeline = ReplayTimeline.load(self.root, skill=skill)
        self.view = self.timeline.view
        self.read_only = read_only or self.view.source_format == "microwave_actual_record"
        self.requested_skill = skill
        model_ref = self.view.model_ref or ("source_scene/scene.xml"
                                            if (self.root / "source_scene/scene.xml").is_file() else None)
        if model_ref is None:
            raise FileNotFoundError("运行记录缺少 MuJoCo 场景模型")
        self.model = mujoco.MjModel.from_xml_path(str(self.root / model_ref))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        metadata_ref = Path(model_ref).parent / "robot/robot_metadata.json"
        self.robot_metadata = json.loads((self.root / metadata_ref).read_text(encoding="utf-8"))
        self.planned_base_qpos = planned_initial_qpos(self.model, self.root, self.robot_metadata)
        self.data.qpos[:] = self.planned_base_qpos
        mujoco.mj_forward(self.model, self.data)
        case_meta = self.root / "case_meta.json"
        self._initial_joint_overrides = (json.loads(case_meta.read_text(encoding="utf-8"))
                                         .get("joint_overrides", {}) if case_meta.is_file() else {})
        result_path = next((self.root / name for name in (
            "result.json", "skill_rotate_knob_0_result.json",
            "skill_open_door_result.json", "skill_close_door_result.json")
            if (self.root / name).is_file()), self.root / "result.json")
        self.robot_group = (json.loads(result_path.read_text(encoding="utf-8")).get("robot_group", "right")
                            if result_path.is_file() else "right")
        self.tcp_site = self.robot_metadata["tcp_sites"][self.robot_group]
        self.show_collision = show_collision
        self.show_visual = show_visual
        self.scene_edit_root = Path(scene_edit_root or self.root / "scene_edits")
        self.prepare_scene = prepare_scene
        self.run_skill_planning = run_skill_planning
        self._planning = False
        self.selected_stage = stage_id
        self.mode = mode or self.timeline.mode
        if self.mode == "actual" and not self._actual_available(stage_id):
            raise ValueError("所选范围存在未执行阶段，无法显示 actual")
        self.timeline.select(stage_id, mode=self.mode)
        self._playing = False
        self._state_lock = getattr(self, "_state_lock", RLock())
        self._updating = False
        self._last_tick = time.monotonic()
        self._time_s = 0.0
        self._shown_stage: str | None = None

        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 232, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False,
                                   show_share_button=False, brand_color=(53, 112, 169))
        self.scene = SkillInspectorScene(server, self.root, self.model, self.data,
                                         planned_base_qpos=self.planned_base_qpos)
        self.scene.renderer.set_visual_visible(self.show_visual)
        self.editable_names = tuple(
            name for name in ("bowl", "microwave", "table")
            if not self.read_only and self.view.skill in {"pick", "insert"} and
            (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
                or name == "table" and
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_table") >= 0)
        )
        self._edit_base_poses = {name: self.scene.object_pose(name) for name in self.editable_names}
        self._initial_scene_poses = self._capture_initial_scene_poses()
        self._configure_camera()
        self._build_gui()
        self._update_stage_and_path()
        self.seek(initial_time_s)
        if self.show_collision:
            self._on_collision(None)

        if not getattr(self, "_client_hook_registered", False):
            @server.on_client_connect
            def _connected(client: Any) -> None:
                client.camera.position = (-1.26, 1.69, 1.00)
                client.camera.look_at = (.16, .20, .10)
                client.camera.up_direction = (0., 0., 1.)
                client.camera.fov = .95
                client.flush()
                client.get_render(height=8, width=8, transport_format="png", timeout=8.0)
                print(f"[skill-inspector] scene-ready stage={self.selected_stage or 'all'} mode={self.mode}", flush=True)
            self._client_hook_registered = True

    def _configure_camera(self) -> None:
        self.server.initial_camera.position = (-1.26, 1.69, 1.00)
        self.server.initial_camera.look_at = (.16, .20, .10)
        self.server.initial_camera.up = (0., 0., 1.)
        self.server.initial_camera.fov = .95

    def _capture_initial_scene_poses(self) -> dict[str, np.ndarray]:
        """从完整 Skill 的首帧取基线，避免把抬升中途的 pose 当场景初态。"""
        baseline = ReplayTimeline.load(self.root, skill=self.requested_skill)
        try:
            baseline.select(None, mode="actual")
        except ValueError:
            baseline.select(None, mode="planned")
        frame = baseline.seek(0.)
        saved = self.data.qpos.copy()
        try:
            if frame.qpos is not None and len(frame.qpos) == self.model.nq:
                self.data.qpos[:] = frame.qpos
                mujoco.mj_forward(self.model, self.data)
            return {name: self.scene.object_pose(name) for name in self.editable_names}
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_forward(self.model, self.data)

    def _actual_available(self, stage_id: str | None) -> bool:
        stages = (self.view.stages if stage_id is None else
                  [stage for stage in self.view.stages if stage.stage_id == stage_id])
        return bool(stages) and (self.view.source_format == "microwave_actual_record" or
                                 all(stage.execution_status == "completed" for stage in stages))

    def _build_gui(self) -> None:
        gui = self.server.gui
        self.summary = gui.add_markdown("")
        options = ("完整 Skill",) + tuple(s.stage_id for s in self.view.stages)
        self.stage_control = gui.add_dropdown("检查范围", options=options,
                                              initial_value=self.selected_stage or "完整 Skill")
        self.stage_control.on_update(self._on_selection)
        mode_options = ((_MODE_LABELS['actual'],) if self.view.source_format == 'microwave_actual_record'
                        else tuple(_MODE_LABELS.values()))
        self.mode_control = gui.add_dropdown("查看数据", options=mode_options,
                                             initial_value=_MODE_LABELS[self.mode])
        self.mode_control.on_update(self._on_selection)
        self.mode_help = gui.add_markdown("规划轨迹＝cuRobo 保存的目标运动；仿真实际执行＝MuJoCo 运行时采样。")
        self.collision_control = gui.add_checkbox("显示规划碰撞层", initial_value=self.show_collision)
        self.collision_control.on_update(self._on_collision)
        self.visual_control = gui.add_checkbox("显示 visual 外观", initial_value=self.show_visual)
        self.visual_control.on_update(self._on_visual)
        self.collision_group_control = gui.add_dropdown("碰撞对象组", options=("全部",), initial_value="全部")
        self.collision_group_control.on_update(self._on_collision_group)
        self.collision_isolate_control = gui.add_checkbox("只看选中碰撞组", initial_value=False)
        self.collision_isolate_control.on_update(self._on_collision_group)
        self.collision_info = gui.add_markdown("")
        self.scene_status = gui.add_markdown("")
        self.play_control = gui.add_button("播放 / 暂停")
        self.play_control.on_click(self._on_play)
        self.back_control = gui.add_button("上一采样")
        self.back_control.on_click(lambda event: self.step(-1))
        self.next_control = gui.add_button("下一采样")
        self.next_control.on_click(lambda event: self.step(1))
        self.time_control = gui.add_slider("时间 (s)", min=0.,
                                           max=max(self.timeline.duration_s, .001), step=.001,
                                           initial_value=0.)
        self.time_control.on_update(self._on_time)
        self.frame_info = gui.add_markdown("")
        self.facts_info = gui.add_markdown("")
        if not self.read_only and self.view.skill in {"pick", "insert"}:
            self.edit_info = gui.add_markdown(
                "**场景位姿预览**：拖动会一起移动画面中的 visual 和碰撞代理；机器人与轨迹仍来自原 Skill。点击应用后才用新位姿重新规划。"
            )
            self._build_object_editors()
        else:
            self._edit_numbers = {}
            self._edit_transforms = {}
            gui.add_markdown("已保存记录只读回放；任务结果与仿真辅助范围以原报告为准。"
                             if self.read_only else
                             "当前旋钮 run 用任务配置重建场景；界面只读检查规划，调整配置后重新运行规划命令。")

    def _build_object_editors(self) -> None:
        gui = self.server.gui
        labels = {"bowl": "碗", "microwave": "微波炉", "table": "桌面"}
        self._edit_numbers: dict[str, dict[str, Any]] = {}
        self._edit_transforms: dict[str, Any] = {}
        for name in self.editable_names:
            pose = self._edit_base_poses[name]
            self._edit_numbers[name] = {}
            with gui.add_folder(f"预览移动{labels.get(name, name)}", expand_by_default=False):
                for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                    control = gui.add_number(f"{axis} (m)", initial_value=float(pose[index, 3]), step=.005)
                    control.on_update(lambda event, object_name=name: self._on_edit_numbers(object_name))
                    self._edit_numbers[name][axis] = control
                for axis in ("roll", "pitch", "yaw"):
                    control = gui.add_number(f"{axis} (rad)", initial_value=0., step=.02)
                    control.on_update(lambda event, object_name=name: self._on_edit_numbers(object_name))
                    self._edit_numbers[name][axis] = control
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            transform = self.server.scene.add_transform_controls(
                f"/skill_inspector/edit/{name}", scale=.10,
                position=tuple(pose[:3, 3]), wxyz=tuple(q[[3, 0, 1, 2]]))
            transform.visible = False
            transform.on_update(lambda event, object_name=name: self._on_edit_transform(object_name))
            self._edit_transforms[name] = transform
        self.edit_object_control = gui.add_dropdown("拖动预览物体", options=("关闭",) + self.editable_names,
                                                    initial_value="关闭")
        self.edit_object_control.on_update(self._on_edit_object)
        self.reset_preview_button = gui.add_button("清除位姿预览")
        self.reset_preview_button.on_click(self._on_reset_preview)
        self.replan_button = gui.add_button("应用位姿并重新规划 Skill", color="blue")
        self.replan_button.on_click(self._on_apply_and_plan)
        self.replan_info = gui.add_markdown(
            "应用会重建场景和碰撞快照，规划新的 Skill 阶段，随后用当前页面重新打开新 run。"
            "旧 run 保留，物理执行仍需单独运行。"
        )

    @_scene_update
    def _on_selection(self, event: Any) -> None:
        if self._updating:
            return
        stage_id = None if self.stage_control.value == "完整 Skill" else str(self.stage_control.value)
        mode = next(key for key, label in _MODE_LABELS.items()
                    if label == str(self.mode_control.value))
        if mode == "actual" and not self._actual_available(stage_id):
            mode = "planned"
            self._updating = True
            self.mode_control.value = _MODE_LABELS[mode]
            self._updating = False
        self.selected_stage, self.mode = stage_id, mode
        self._playing = False
        self.timeline.select(stage_id, mode=mode)
        self._shown_stage = None
        self._updating = True
        self.time_control.max = max(self.timeline.duration_s, .001)
        self.time_control.value = 0.0
        self._updating = False
        self._update_stage_and_path()
        self.seek(0.0)

    @_scene_update
    def _on_collision(self, event: Any) -> None:
        self.show_collision = bool(self.collision_control.value)
        if self.show_collision:
            current = next(s for s in self.view.stages if s.stage_id == self._shown_stage)
            if current.collision_snapshot_ref is None:
                available = next((s for s in self.view.stages if s.collision_snapshot_ref), None)
                if available is not None:
                    # open/close 等阶段没有独立的规划快照。主动切到有快照的阶段，
                    # 让用户点击开关后立即看到真实参与规划的代理，并标明其来源。
                    self._updating = True
                    self.stage_control.value = available.stage_id
                    self._updating = False
                    self._on_selection(None)
        self.scene.renderer.set_collision_visible(self.show_collision)
        self._update_collision_info()

    @_scene_update
    def _on_visual(self, event: Any) -> None:
        self.scene.renderer.set_visual_visible(bool(self.visual_control.value))

    @_scene_update
    def _on_collision_group(self, event: Any) -> None:
        if self._updating:
            return
        selected = str(self.collision_group_control.value)
        self.scene.renderer.select_collision_group(None if selected == "全部" else selected,
                                                   isolate=bool(self.collision_isolate_control.value))

    def _update_collision_info(self) -> None:
        stage = next(s for s in self.view.stages if s.stage_id == self._shown_stage)
        groups = tuple(self.scene.renderer.collision_groups)
        self._updating = True
        try:
            self.collision_group_control.options = ("全部",) + groups
            if self.collision_group_control.value not in ("全部",) + groups:
                self.collision_group_control.value = "全部"
        finally:
            self._updating = False
        self.collision_info.content = (
            f"碰撞层来源：**{stage.stage_id}** 的规划快照；{len(self.scene.renderer.collision_handles)} 个代理，"
            f"{len(groups)} 组。碰撞代理只用于规划检查，不能替代 visual 外观。"
            if stage.collision_snapshot_ref else
            "本阶段没有独立规划碰撞快照；打开碰撞层会跳到有快照的阶段。"
        )

    def _update_scene_status(self, frame: Any) -> None:
        """把当前快照的碰撞范围和 attachment 放在时间轴上方常显位置。"""
        if frame.collision_snapshot_ref:
            groups = self.scene.renderer.collision_groups
            obstacles = [f"{label} {len(names)}" for label, names in groups.items()
                         if label != "活动机械臂碰撞球"]
            spheres = len(groups.get("活动机械臂碰撞球", ()))
            sphere_label = (f"活动臂碰撞球 {spheres} 个" if
                            "active_robot_spheres" in self.scene._collision_snapshot else
                            "活动臂碰撞球未记录")
            collision = (f"参与碰撞避免：{', '.join(obstacles) or '无场景障碍'}；"
                         f"{sphere_label}。")
        else:
            collision = "参与碰撞避免：本阶段没有独立规划请求或碰撞快照。"
        moving = self.scene._collision_snapshot.get("moving_body_names") or []
        if moving:
            collision += (f"运动关节 {', '.join(moving)} 从静态规划中排除；"
                          "其接触由保存的逐帧 MuJoCo 复核记录说明。")
        attachment = (frame.attachment or {}).get("object_instance_id") or "无"
        preview = ("**位姿预览未提交**：物体 visual 与碰撞代理已移动；"
                   "机械臂、轨迹和实际规划仍是原 Skill。 "
                   if self.scene._preview_poses else "")
        self.scene_status.content = (
            f"{preview}**当前 run 场景约束** · {collision} 附着代理：{attachment}。"
            "附着代理表示规划器把物体随工具移动，不等于物理夹持成功。"
        )

    def _sync_edit_controls(self) -> None:
        """未编辑的物体随回放帧更新拖动控件，避免控件停留在旧时刻。"""
        self._updating = True
        try:
            for name in self.editable_names:
                if name in self.scene._preview_poses:
                    continue
                pose = self.scene.object_pose(name)
                self._edit_base_poses[name] = pose
                for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                    self._edit_numbers[name][axis].value = float(pose[index, 3])
                for axis in ("roll", "pitch", "yaw"):
                    self._edit_numbers[name][axis].value = 0.
                q = Rotation.from_matrix(pose[:3, :3]).as_quat()
                self._edit_transforms[name].position = tuple(pose[:3, 3])
                self._edit_transforms[name].wxyz = tuple(q[[3, 0, 1, 2]])
        finally:
            self._updating = False

    @_scene_update
    def _on_edit_object(self, event: Any) -> None:
        selected = str(self.edit_object_control.value)
        for name, control in self._edit_transforms.items():
            control.visible = name == selected

    @_scene_update
    def _on_edit_numbers(self, name: str) -> None:
        if self._updating:
            return
        values = [float(self._edit_numbers[name][axis].value) for axis in ("x", "y", "z")]
        angles = [float(self._edit_numbers[name][axis].value) for axis in ("roll", "pitch", "yaw")]
        if not np.isfinite([*values, *angles]).all():
            self.edit_info.content = "位姿预览失败：位置和角度必须是有限数值。"
            return
        pose = self._edit_base_poses[name].copy()
        pose[:3, 3] = values
        pose[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix() @ pose[:3, :3]
        self._playing = False
        self.scene.preview_object_pose(name, pose)
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        self._updating = True
        try:
            self._edit_transforms[name].position = tuple(values)
            self._edit_transforms[name].wxyz = tuple(q[[3, 0, 1, 2]])
        finally:
            self._updating = False
        self.edit_info.content = (f"正在预览移动 **{name}** 的 visual 和碰撞代理；"
                                  "实际运行、轨迹与原碰撞快照未改变。")
        self._update_scene_status(self.timeline.seek(self._time_s))

    @_scene_update
    def _on_edit_transform(self, name: str) -> None:
        if self._updating:
            return
        control = self._edit_transforms[name]
        p, q = np.asarray(control.position, dtype=float), np.asarray(control.wxyz, dtype=float)
        if not np.isfinite([*p, *q]).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
            self.edit_info.content = "位姿预览失败：拖动控件给出了无效 pose。"
            return
        pose = np.eye(4)
        pose[:3, 3] = p
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        self._playing = False
        self.scene.preview_object_pose(name, pose)
        delta = pose[:3, :3] @ self._edit_base_poses[name][:3, :3].T
        angles = Rotation.from_matrix(delta).as_euler("xyz")
        self._updating = True
        try:
            for axis, value in zip(("x", "y", "z"), p):
                self._edit_numbers[name][axis].value = float(value)
            for axis, value in zip(("roll", "pitch", "yaw"), angles):
                self._edit_numbers[name][axis].value = float(value)
        finally:
            self._updating = False
        self.edit_info.content = (f"正在预览移动 **{name}** 的 visual 和碰撞代理；"
                                  "实际运行、轨迹与原碰撞快照未改变。")
        self._update_scene_status(self.timeline.seek(self._time_s))

    @_scene_update
    def _on_reset_preview(self, event: Any) -> None:
        self.scene.clear_object_preview()
        self._sync_edit_controls()
        self.edit_info.content = "位姿预览已清除；画面已恢复为当前回放记录。"
        self._update_scene_status(self.timeline.seek(self._time_s))

    @_scene_update
    def _on_apply_and_plan(self, event: Any) -> None:
        """提交新场景、规划 PickSkill，并用现有页面重新打开新 run。"""
        if self._planning:
            return
        edited = {name: pose.copy() for name, pose in self.scene._preview_poses.items()}
        if not edited:
            self.replan_info.content = "请先拖动或输入至少一个物体的新位姿，再点击重新规划。"
            return
        # 未编辑物体仍取原 Skill 的首帧；从抬升中途进入界面也不会把悬空碗
        # 或已移动的机械臂状态错误地当成新的场景初态。
        overrides = {name: pose.copy() for name, pose in self._initial_scene_poses.items()}
        overrides.update(edited)
        self._planning = True
        self._playing = False
        self.replan_button.disabled = True
        target = None
        try:
            self.scene_edit_root.mkdir(parents=True, exist_ok=True)
            revision = 1
            target = self.scene_edit_root / f"revision_{revision:03d}"
            while target.exists():
                revision += 1
                target = self.scene_edit_root / f"revision_{revision:03d}"
            self.replan_info.content = f"正在重建场景并规划 Skill：{target}。"
            from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene
            from gene_agent.tasks.policies.microwave_task.planning.grasp import run_pick_bowl_skill_planning
            prepare = self.prepare_scene or prepare_pick_bowl_scene
            plan = self.run_skill_planning or run_pick_bowl_skill_planning
            case = prepare(target, revision=revision, seed=123, pose_overrides=overrides,
                           joint_overrides=self._initial_joint_overrides)
            plan(case, seed=123)
            if self.view.skill == "insert":
                from gene_agent.tasks.policies.microwave_task.planning.insert import run_insert_bowl_skill_planning
                run_insert_bowl_skill_planning(case, seed=123)
            # 规划失败的阶段也有报告。新页面仍需打开它，才能沿用相同的阶段、
            # 轨迹和失败诊断控件，而不是在旧回放上叠加一套特殊控件。
            self._reload_run(case.output_dir)
        except Exception as error:
            message = f"重新规划或打开新 Skill run 失败：{html.escape(str(error))}。"
            if target is not None:
                target.mkdir(parents=True, exist_ok=True)
                (target / "ui_errors.log").write_text(traceback.format_exc(), encoding="utf-8")
                message += f" 详细错误：`{target / 'ui_errors.log'}`"
            self.replan_info.content = message
        finally:
            self._planning = False
            self.replan_button.disabled = False

    def _reload_run(self, run_dir: Path) -> None:
        """重建原有 UI 和场景；播放器对象保持不变，主循环无需切换引用。"""
        old_root = self.root
        options = dict(show_collision=self.show_collision, show_visual=self.show_visual,
                       skill=self.requested_skill,
                       scene_edit_root=self.scene_edit_root, prepare_scene=self.prepare_scene,
                       run_skill_planning=self.run_skill_planning)
        # 先验证报告结构，再移除旧控件。若新模型加载失败，重新打开旧 run。
        ReplayTimeline.load(run_dir, skill=self.requested_skill)
        try:
            self.server.gui.reset()
            self.server.scene.reset()
            self.__init__(self.server, run_dir, **options)
            print(f"[skill-inspector] 已打开新 Skill run：{run_dir}")
        except Exception:
            self.server.gui.reset()
            self.server.scene.reset()
            self.__init__(self.server, old_root, **options)
            raise

    @_scene_update
    def _on_play(self, event: Any) -> None:
        self._playing = not self._playing
        self._last_tick = time.monotonic()

    @_scene_update
    def _on_time(self, event: Any) -> None:
        if not self._updating:
            self._playing = False
            self.seek(float(self.time_control.value))

    def _sample_times(self) -> list[float]:
        return sorted({round(segment.start_s + sample.time_s, 9)
                       for segment in self.timeline._segments for sample in segment.samples})

    @_scene_update
    def step(self, direction: int) -> None:
        times = self._sample_times()
        if direction > 0:
            candidates = [t for t in times if t > self._time_s + 1e-8]
            next_time = candidates[0] if candidates else self.timeline.duration_s
        else:
            candidates = [t for t in times if t < self._time_s - 1e-8]
            next_time = candidates[-1] if candidates else 0.0
        self._playing = False
        self.seek(next_time)

    def _update_stage_and_path(self) -> None:
        frames = []
        for t in self._sample_times():
            frames.append(self.timeline.seek(min(t, self.timeline.duration_s)))
        self.scene.set_path(frames, self.tcp_site)
        statuses = _details([_stage_status(stage) for stage in self.view.stages], "查看各阶段状态")
        completed = sum(stage.execution_status == "completed" for stage in self.view.stages)
        planned = sum(stage.planning_status == "planned" for stage in self.view.stages)
        physical = ("是" if self.view.physical_success is True else
                    "否" if self.view.physical_success is False else "未判定")
        physical_label = (f"物理抓起：{physical}" if self.view.skill == "pick" else
                          "关节动作：运动学预览，未验证接触物理")
        if self.view.source_format == 'microwave_actual_record':
            physical_label = 'MuJoCo 保存采样；可能包含显式辅助或运动学提交，见原报告'
        progress = (f"已完成 {completed}/{len(self.view.stages)} 阶段" if
                    self.view.skill == "pick" else
                    f"已规划 {planned}/{len(self.view.stages)}、"
                    f"已执行 {completed}/{len(self.view.stages)} 阶段")
        self.summary.content = (f"**{self.view.skill} · {self.view.instance_id}**  "
                                f"回放总时长 {_seconds(self.timeline.duration_s)} · "
                                f"{physical_label} · "
                                f"{progress}\n\n{statuses}")

    def _show_facts(self, stage_id: str) -> None:
        stage = next(s for s in self.view.stages if s.stage_id == stage_id)
        self.scene.set_stage(stage, show_collision=self.show_collision, tcp_site=self.tcp_site)
        self._update_collision_info()
        facts = stage_facts(self.root, stage)
        included = [str(x) for x in facts["collision_objects"]]
        excluded = [f"{x['geom']}: {x['reason']}" for x in facts["excluded"]]
        policy = facts["mounting_contact_policy"] or {}
        attachment = facts["attachment"] or {}
        target = facts["target_pose"]
        if target is None and self.scene.current_target_pose is not None:
            target = self.scene.current_target_pose.tolist()
        target_xyz = (", ".join(f"{v:.4f}" for v in np.asarray(target)[:3, 3])
                      if target is not None else "未记录")
        articulation = (f"关节角：{facts['articulation_start_angle_rad']:.3f} → "
                        f"{facts['articulation_end_angle_rad']:.3f} rad；"
                        "机器人为规划轨迹，物体关节为运动学跟随，未验证接触。\n\n"
                        if facts["kinematic_preview"] else "")
        moving = facts["moving_body_names"]
        contact_check = facts["dynamic_door_contact_check"] or {}
        dynamic_line = (
            f"运动门：{', '.join(moving)} 不在本阶段静态规划障碍中；"
            f"MuJoCo 逐帧复核 {contact_check.get('sample_count', 0)} 帧，"
            f"非法接触 {len(contact_check.get('forbidden_contacts', []))} 类。"
            "连续扫掠和实际抓取力未验证。\n\n" if moving else ""
        )
        self.facts_info.content = (
            f"### {stage.stage_id}\n"
            f"路径：**{facts['path_kind'] or '无'}** · "
            f"{_PLANNING_LABELS.get(facts['planning_status'], facts['planning_status'])} · "
            f"{_EXECUTION_LABELS.get(facts['execution_status'], facts['execution_status'])}\n\n"
            f"目标 TCP：({target_xyz}) m"
            f"{' · 旧记录由规划终点 FK 计算' if facts['target_pose'] is None and target is not None else ''}\n\n"
            f"规划墙钟耗时：{_seconds(facts['planning_duration_s'])} · "
            f"MuJoCo 仿真轨迹时长：{_seconds(facts['execution_duration_s'])}\n\n"
            f"失败：{html.escape(str(facts['failure'] or '无'))}\n\n"
            f"{articulation}"
            f"{dynamic_line}"
            f"碰撞快照：`{facts['collision_sha256'] or '无'}`\n\n"
            f"允许的安装接触：{', '.join(policy.get('robot_links', [])) or '未记录'}；"
            f"对象 {len(policy.get('obstacle_names', []))} 个\n\n"
            f"附着物体：{attachment.get('object_instance_id') or '无'}\n\n"
            f"实际采样：{facts['sample_format'] or '未记录'}\n\n"
            + _details([json.dumps(target, ensure_ascii=False)] if target is not None else [],
                       "目标 pose 矩阵") + "\n\n"
            + _details([json.dumps(policy, ensure_ascii=False)] if policy else [], "安装接触规则") + "\n\n"
            + _details([json.dumps(attachment, ensure_ascii=False)] if attachment else [], "attachment 详情") + "\n\n"
            + _details([json.dumps(facts["axis_path_validation"], ensure_ascii=False)]
                       if facts["axis_path_validation"] else [], "绕轴路径复核") + "\n\n"
            + _details([json.dumps(contact_check, ensure_ascii=False)]
                       if contact_check else [], "运动门逐帧接触复核") + "\n\n"
            + _details(included, "参与规划的碰撞对象") + "\n\n"
            + _details(excluded, "排除的几何与原因") + "\n\n"
            + _details([str(x) for x in facts["limitations"]], "碰撞快照限制")
        )

    @_scene_update
    def seek(self, time_s: float) -> None:
        self._time_s = min(max(time_s, 0.0), self.timeline.duration_s)
        frame = self.timeline.seek(self._time_s)
        stage = next(s for s in self.view.stages if s.stage_id == frame.stage_id)
        if frame.stage_id != self._shown_stage:
            self._shown_stage = frame.stage_id
            self._show_facts(frame.stage_id)
        self.scene.show_frame(frame, base_state_ref=stage.start_state_ref,
                              gripper_group=self.robot_group, tcp_site=self.tcp_site)
        self._update_scene_status(frame)
        self._sync_edit_controls()
        self._updating = True
        self.time_control.value = self._time_s
        self._updating = False
        quality = {"complete_actual_sample": "完整实测状态",
                   "saved_actual_boundary": "保存的实际状态边界",
                   "boundary_state": "阶段边界实测状态",
                   "partial_actual_sample": "部分实测：物体状态未知",
                   "planned_trajectory": "规划轨迹",
                   "failed_stage_marker": "失败阶段，无轨迹"}.get(frame.sample_quality, frame.sample_quality)
        sim_time = (f"{frame.sim_time_s:.3f} s" if frame.sim_time_s is not None else "未知")
        width = (f"{frame.gripper_width_m * 1000:.1f} mm"
                 if frame.gripper_width_m is not None else "未知")
        self.frame_info.content = (
            f"时间 {self._time_s:.3f}/{self.timeline.duration_s:.3f}s · "
            f"阶段 {frame.stage_id} · 阶段内 {frame.local_time_s:.3f}s · "
            f"采样 {frame.sample_time_s:.3f}s · {quality}\n\n"
            f"state ID：`{frame.state_id or '该采样未记录'}` · "
            f"仿真时刻：{sim_time}（场景重置后的累计时间） · 夹爪开口：{width}")

    @_scene_update
    def tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now
        if self._playing:
            self.seek(min(self._time_s + elapsed, self.timeline.duration_s))
            if self._time_s >= self.timeline.duration_s:
                self._playing = False
