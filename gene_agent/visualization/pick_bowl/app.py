"""真实抓碗规划结果的 Viser 检查与显式重新规划界面。"""

from __future__ import annotations

from pathlib import Path
import json
import traceback
from typing import Any, Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case, prepare_pick_bowl_scene, run_pick_bowl_planning, plan_pick_bowl_candidate
from .scene import PickBowlScene, _compiled_mesh, _geom_pose, _pose


def _wxyz(pose: np.ndarray) -> tuple[float, float, float, float]:
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return tuple(float(x) for x in q[[3, 0, 1, 2]])


def _body_pose(data: Any, name: str) -> np.ndarray:
    body = data.geom("scene_table") if name == "table" else data.body(name)
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose


def camera_presets(case: Any, *, candidate_id: str | None = None) -> dict[str, dict[str, Any]]:
    """全景/碗近景固定；夹爪视角从所选候选的掌端沿 TCP +Z 看向指尖。"""
    bowl = _body_pose(case.data, "bowl")[:3, 3]
    focus = tuple(float(x) for x in bowl)
    candidates = case.candidates
    candidate = next((item for item in candidates if item["candidate_id"] == candidate_id), None)
    if candidate is None and candidates:
        candidate = candidates[0]
    grasp = np.asarray(candidate["T_world_tcp"], dtype=float) if candidate is not None else np.eye(4)
    origin, rotation = grasp[:3, 3], grasp[:3, :3]
    # 正对掌背会被夹爪底板挡住视线；从掌端侧前方斜看，保留“沿指尖方向”透视。
    palm = origin + rotation @ np.array([0., -.25, -.25])
    fingers = origin + rotation @ np.array([0., 0., .04])
    upwards = rotation @ np.array([0., 1., 0.])
    return {
        "overview": {"position": (-1.26, 1.69, 1.00), "look_at": (.16, .20, .10),
                     "up": (0., 0., 1.), "fov": .95},
        "bowl": {"position": (focus[0] + .42, focus[1] - .65, focus[2] + .38),
                 "look_at": focus, "up": (0., 0., 1.), "fov": .8},
        "gripper": {"position": tuple(float(x) for x in palm),
                    "look_at": tuple(float(x) for x in fingers),
                    "up": tuple(float(x) for x in upwards), "fov": .85},
    }


class PickBowlPlanningApp:
    """查看已保存事实；只有按钮回调才创建新 revision 并调用 cuRobo。"""

    def __init__(
        self, server: Any, run_dir: Path, seed: int = 123, *, view: str = "overview",
        show_collision: bool = False,
        initial_candidate_id: str | None = None,
        trajectory_source: str = "抓取目标",
        load_case: Callable = load_pick_bowl_case,
        prepare_scene: Callable = prepare_pick_bowl_scene,
        run_planning: Callable = run_pick_bowl_planning,
        plan_candidate: Callable = plan_pick_bowl_candidate,
        loaded_case: Any | None = None,
    ) -> None:
        if view not in ("overview", "bowl", "gripper"):
            raise ValueError(f"未知视角: {view}")
        if trajectory_source not in ("抓取目标", "Skill pregrasp", "Skill approach", "TCP 直线探针"):
            raise ValueError(f"未知轨迹阶段: {trajectory_source}")
        self.server = server
        opened_dir = Path(run_dir).resolve()
        self.run_root = opened_dir.parent if opened_dir.name.startswith("revision_") else opened_dir
        self.seed = seed
        self.view = view
        self.show_collision = bool(show_collision)
        self.initial_candidate_id = initial_candidate_id
        self.initial_trajectory_source = trajectory_source
        self.prepare_scene = prepare_scene
        self.load_case = load_case
        self.run_planning = run_planning
        self.plan_candidate = plan_candidate
        self.case = loaded_case if loaded_case is not None else load_case(opened_dir)
        skill_path = self.case.output_dir / "skill_pick_bowl_result.json"
        self.skill_report = json.loads(skill_path.read_text(encoding="utf-8")) if skill_path.exists() else None
        linear_path = self.case.output_dir / "linear_probe_result.json"
        self.linear_probe = json.loads(linear_path.read_text(encoding="utf-8")) if linear_path.exists() else None
        if (self.skill_report is not None and
                self.skill_report.get("snapshot_sha256") != self.case.snapshot["snapshot_sha256"]):
            raise ValueError("Skill 报告与当前碰撞快照哈希不一致")
        editable = ["bowl", "microwave"]
        if mujoco.mj_name2id(self.case.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_table") >= 0:
            editable.append("table")
        self.editable_names = tuple(editable)
        self.pending = {name: _body_pose(self.case.data, name) for name in self.editable_names}
        self.pending_table_size_xy_m = (
            tuple(float(2 * value) for value in self.case.model.geom("scene_table").size[:2])
            if "table" in self.editable_names else None
        )
        self.editable_joint_names = tuple(name for name in (
            "microwave_door_joint", "knob_0_joint", "knob_1_joint"
        ) if mujoco.mj_name2id(self.case.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0)
        self.pending_joints = {
            name: float(self.case.data.qpos[self.case.model.joint(name).qposadr[0]])
            for name in self.editable_joint_names
        }
        self._joint_preview_data = None
        self.pending_dirty = False
        self._syncing = False
        self._planning = False
        self._preview_data = None
        self._candidate_handles: list[Any] = []
        self._candidate_frames: dict[str, Any] = {}
        self._controls: dict[str, Any] = {}

        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 232, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False,
                                   show_share_button=False, brand_color=(53, 112, 169))
        self.scene = PickBowlScene(server, self.case.model, self.case.data, self.case.metadata, self.case.snapshot)
        self.scene.set_collision_visible(self.show_collision)
        self._capture_preview_geoms()
        initial_id = self.initial_candidate_id or (self.case.report or {}).get("selected_candidate_id")
        self.camera_presets = camera_presets(self.case, candidate_id=initial_id)
        self._set_initial_camera(view)
        self._build_controls()
        self._render_results()
        if trajectory_source in ("TCP 直线探针", "Skill approach"):
            # 离线截图默认显示轨迹中段，避免只看到home与目标ghost。
            trajectory = ((self._selected_plan().get("result") or {}).get("trajectory") or {})
            rows = trajectory.get("positions_rad") or []
            if rows:
                self.trajectory_slider.value = len(rows) // 2
                self._on_trajectory(None)

        if not getattr(self, "_client_hook_registered", False):
            @server.on_client_connect
            def _on_client(client: Any) -> None:
                self.configure_client_camera(client, self.camera_presets[self.view])
                client.flush()
                client.get_render(height=8, width=8, transport_format="png", timeout=8.0)
                print(f"[pick-bowl] scene-ready: view={self.view}; revision={self.case.revision}", flush=True)
            self._client_hook_registered = True

    def _set_initial_camera(self, view: str) -> None:
        preset = self.camera_presets[view]
        self.server.initial_camera.position = preset["position"]
        self.server.initial_camera.look_at = preset["look_at"]
        self.server.initial_camera.up = preset["up"]
        self.server.initial_camera.fov = preset["fov"]

    @staticmethod
    def configure_client_camera(client: Any, preset: dict) -> None:
        client.camera.position = preset["position"]
        client.camera.look_at = preset["look_at"]
        client.camera.up_direction = preset["up"]
        client.camera.fov = preset["fov"]
        client.camera.near = .01
        client.camera.far = 10.

    def _build_controls(self) -> None:
        gui = self.server.gui
        self.status = gui.add_markdown("")
        self.view_control = gui.add_dropdown("视角", options=("overview", "bowl", "gripper"), initial_value=self.view)
        self.view_control.on_update(self._on_view)
        self.collision_toggle = gui.add_checkbox("显示规划碰撞层", initial_value=self.show_collision)
        self.collision_toggle.on_update(lambda event: self.scene.set_collision_visible(bool(self.collision_toggle.value)))
        self.visual_toggle = gui.add_checkbox("显示 visual 资产", initial_value=True)
        self.visual_toggle.on_update(lambda event: self.scene.set_visual_visible(bool(self.visual_toggle.value)))
        self.xyz_controls: dict[str, dict[str, Any]] = {}
        self.rotation_controls: dict[str, dict[str, Any]] = {}
        self.table_size_controls: dict[str, Any] = {}
        self.transform_controls: dict[str, Any] = {}
        labels = {"bowl": "碗", "microwave": "微波炉", "table": "桌面"}
        for name in self.editable_names:
            with gui.add_folder(f"编辑 {labels[name]}", expand_by_default=False):
                self.xyz_controls[name] = {}
                self.rotation_controls[name] = {}
                for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                    control = gui.add_number(f"{axis} (m)", initial_value=float(self.pending[name][index, 3]), step=.005)
                    control.on_update(lambda event, object_name=name: self._on_numbers(object_name))
                    self.xyz_controls[name][axis] = control
                for axis in ("roll", "pitch", "yaw"):
                    control = gui.add_number(f"{axis} (rad)", initial_value=0., step=.02)
                    control.on_update(lambda event, object_name=name: self._on_numbers(object_name))
                    self.rotation_controls[name][axis] = control
                if name == "table" and self.pending_table_size_xy_m is not None:
                    for label, key, index in (("长 x (m)", "length", 0), ("宽 y (m)", "width", 1)):
                        control = gui.add_number(
                            label, initial_value=self.pending_table_size_xy_m[index],
                            min=0.001, step=.01,
                        )
                        control.on_update(lambda event: self._on_table_size())
                        self.table_size_controls[key] = control
            self._set_transform(name)
        self.joint_controls: dict[str, Any] = {}
        if self.editable_joint_names:
            with gui.add_folder("编辑微波炉关节初态", expand_by_default=False):
                labels = {"microwave_door_joint": "门角度", "knob_0_joint": "旋钮 0 角度",
                          "knob_1_joint": "旋钮 1 角度"}
                for name in self.editable_joint_names:
                    joint = self.case.model.joint(name)
                    low, high = (float(x) for x in joint.range)
                    control = gui.add_slider(labels[name], min=low, max=high, step=.01,
                                             initial_value=self.pending_joints[name])
                    control.on_update(lambda event, joint_name=name: self._on_joint(joint_name))
                    self.joint_controls[name] = control
        self.ik_mode_control = gui.add_dropdown(
            "IK 筛选", options=("serial", "batch"),
            initial_value=(self.case.report or {}).get("ik_screening_mode", "serial"),
        )
        self.replan_button = gui.add_button("重新规划", color="blue")
        self.replan_button.on_click(self._on_replan)
        self.candidate_control = gui.add_dropdown("候选 TCP", options=self._candidate_options(),
                                                  initial_value=self._candidate_options()[0])
        self.candidate_control.on_update(self._on_candidate)
        self.trajectory_source_control = gui.add_dropdown(
            "轨迹阶段", options=("抓取目标", "Skill pregrasp", "Skill approach", "TCP 直线探针"),
            initial_value=self.initial_trajectory_source,
        )
        self.trajectory_source_control.on_update(self._on_trajectory_source)
        self.plan_candidate_button = gui.add_button("规划当前候选轨迹", color="blue")
        self.plan_candidate_button.on_click(self._on_plan_candidate)
        self.all_candidates_toggle = gui.add_checkbox("显示全部候选", initial_value=False)
        self.all_candidates_toggle.on_update(self._on_all_candidates)
        self.obstacle_control = gui.add_dropdown("碰撞对象", options=self._obstacle_options(),
                                                 initial_value=self._obstacle_options()[0])
        self.obstacle_control.on_update(self._on_obstacle)
        self.isolate_collision_toggle = gui.add_checkbox("只看选中碰撞对象", initial_value=False)
        self.isolate_collision_toggle.on_update(self._on_obstacle_filter)
        self.trajectory_slider = gui.add_slider("轨迹帧", min=0, max=1, step=1, initial_value=0)
        self.trajectory_slider.on_update(self._on_trajectory)
        self.details = gui.add_markdown("")

    def _candidate_options(self) -> tuple[str, ...]:
        return tuple(item["candidate_id"] for item in self.case.candidates) or ("(无候选)",)

    def _obstacle_options(self) -> tuple[str, ...]:
        return tuple(self.scene.collision_groups) or ("(无障碍)",)

    def _capture_preview_geoms(self) -> None:
        """保存已提交场景的世界几何位姿，待应用预览只改 Viser handle。"""
        self._preview_geoms: dict[str, list[tuple[Any, int]]] = {
            name: [] for name in self.editable_names
        }
        model, data = self.case.model, self.case.data
        for geom_id, handle in self.scene.visual_handles.items():
            body_id = int(model.geom_bodyid[geom_id])
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            owner = "table" if geom_name == "scene_table" and "table" in self.pending else None
            while body_id > 0:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name == "bowl":
                    owner = "bowl"
                    break
                if body_name in ("microwave", "microwave_support"):
                    owner = "microwave"
                    break
                body_id = int(model.body_parentid[body_id])
            if owner is None:
                continue
            self._preview_geoms[owner].append((handle, geom_id))

    def _preview_pending_visuals(self, name: str) -> None:
        delta = self.pending[name] @ np.linalg.inv(_body_pose(self.case.data, name))
        source = self._joint_preview_data if self._joint_preview_data is not None else self.case.data
        for handle, geom_id in self._preview_geoms[name]:
            pose = delta @ _geom_pose(source, geom_id)
            handle.position = tuple(float(x) for x in pose[:3, 3])
            handle.wxyz = _wxyz(pose)
            if name == "table" and self.pending_table_size_xy_m is not None:
                baseline = 2 * np.asarray(self.case.model.geom_size[geom_id, :2], dtype=float)
                handle.scale = (
                    float(self.pending_table_size_xy_m[0] / baseline[0]),
                    float(self.pending_table_size_xy_m[1] / baseline[1]), 1.,
                )

    def _on_table_size(self) -> None:
        if self._syncing or not self.table_size_controls:
            return
        values = (float(self.table_size_controls["length"].value),
                  float(self.table_size_controls["width"].value))
        if not np.isfinite(values).all() or min(values) <= 0:
            self._show_error("桌面长宽必须是正的有限米数值")
            return
        self.pending_table_size_xy_m = values
        self.pending_dirty = True
        self._refresh_pending_visuals()
        self._update_status()

    def _refresh_pending_visuals(self, source_data: Any | None = None) -> None:
        """把关节初态与物体位姿叠加在副本上，绝不改已冻结的规划状态。"""
        model = self.case.model
        self._joint_preview_data = mujoco.MjData(model)
        mujoco.mj_copyData(self._joint_preview_data, model, source_data or self.case.data)
        for name, value in self.pending_joints.items():
            self._joint_preview_data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, self._joint_preview_data)
        self.scene.update_visuals(self._joint_preview_data)
        for name in self.pending:
            self._preview_pending_visuals(name)

    def _on_joint(self, name: str) -> None:
        if self._syncing:
            return
        value = float(self.joint_controls[name].value)
        low, high = (float(x) for x in self.case.model.joint(name).range)
        if not np.isfinite(value) or not low <= value <= high:
            self._show_error(f"{name} 必须在 [{low:.3f}, {high:.3f}] rad 内")
            return
        self.pending_joints[name] = value
        self.pending_dirty = True
        self._refresh_pending_visuals()
        self._update_status()

    def _set_transform(self, name: str) -> None:
        old = self.transform_controls.get(name)
        if old is not None:
            old.remove()
        pose = self.pending[name]
        control = self.server.scene.add_transform_controls(
            f"/pick_bowl/edit/{name}", scale=.11, position=tuple(pose[:3, 3]), wxyz=_wxyz(pose)
        )
        control.on_update(lambda event, object_name=name: self._on_transform(object_name))
        self.transform_controls[name] = control

    def _on_numbers(self, name: str) -> None:
        if self._syncing:
            return
        values = [float(self.xyz_controls[name][axis].value) for axis in ("x", "y", "z")]
        angles = [float(self.rotation_controls[name][axis].value) for axis in ("roll", "pitch", "yaw")]
        if not np.isfinite([*values, *angles]).all():
            self._show_error("编辑值必须是有限数值")
            return
        pose = self.pending[name].copy()
        pose[:3, 3] = values
        base = _body_pose(self.case.data, name)
        # 三轴角是相对已提交位姿、绕 world 轴的增量；点击重规划前不写 MuJoCo GT。
        pose[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix() @ base[:3, :3]
        self.pending[name] = pose
        self.pending_dirty = True
        self._syncing = True
        try:
            self.transform_controls[name].position = tuple(values)
            self.transform_controls[name].wxyz = _wxyz(pose)
        finally:
            self._syncing = False
        self._refresh_pending_visuals()
        self._update_status()

    def _on_transform(self, name: str) -> None:
        if self._syncing:
            return
        control = self.transform_controls[name]
        p = np.asarray(control.position, dtype=float)
        q = np.asarray(control.wxyz, dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
            self._show_error("拖动位姿无效")
            return
        pose = np.eye(4)
        pose[:3, 3] = p
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        self.pending[name] = pose
        self.pending_dirty = True
        base = _body_pose(self.case.data, name)
        delta = pose[:3, :3] @ base[:3, :3].T
        angles = Rotation.from_matrix(delta).as_euler("xyz")
        self._syncing = True
        try:
            for axis, value in zip(("x", "y", "z"), p):
                self.xyz_controls[name][axis].value = float(value)
            for axis, angle in zip(("roll", "pitch", "yaw"), angles):
                self.rotation_controls[name][axis].value = float(angle)
        finally:
            self._syncing = False
        self._refresh_pending_visuals()
        self._update_status()

    def _on_view(self, event: Any) -> None:
        self.view = str(self.view_control.value)
        for client in self.server.get_clients().values():
            self.configure_client_camera(client, self.camera_presets[self.view])

    def _on_candidate(self, event: Any) -> None:
        self.camera_presets["gripper"] = camera_presets(
            self.case, candidate_id=str(self.candidate_control.value)
        )["gripper"]
        if self.view == "gripper":
            for client in self.server.get_clients().values():
                self.configure_client_camera(client, self.camera_presets["gripper"])
        self._update_trajectory_sources()
        self._update_trajectory_handle()
        self._update_candidate_visibility()
        self.trajectory_slider.value = 0
        self._refresh_pending_visuals()
        self.scene.set_robot_spheres_visible(True)
        self._render_details()

    def _skill_stage(self, name: str) -> dict[str, Any] | None:
        report = self.skill_report or {}
        if report.get("selected_candidate_id") != str(self.candidate_control.value):
            return None
        return next((stage for stage in report.get("stages", [])
                     if stage.get("name") == name and stage.get("status") == "planned"), None)

    def _skill_pregrasp(self) -> dict[str, Any] | None:
        return self._skill_stage("pregrasp")

    def _update_trajectory_sources(self) -> None:
        options = ["抓取目标"]
        if self._skill_pregrasp():
            options.append("Skill pregrasp")
        if self._skill_stage("approach"):
            options.append("Skill approach")
        if (self.linear_probe is not None and self.linear_probe.get("selected_candidate_id")
                == str(self.candidate_control.value)):
            options.append("TCP 直线探针")
        current = str(self.trajectory_source_control.value)
        self.trajectory_source_control.options = tuple(options)
        self.trajectory_source_control.value = current if current in options else "抓取目标"

    def _on_trajectory_source(self, event: Any) -> None:
        self._update_trajectory_handle()
        self._update_candidate_visibility()
        self._update_status()
        self._render_details()

    def _selected_plan(self) -> dict:
        """仅返回当前候选自己的规划，绝不把默认候选轨迹挂到其它候选。"""
        if str(self.trajectory_source_control.value) == "TCP 直线探针":
            probe = self.linear_probe
            if probe is None or probe.get("selected_candidate_id") != str(self.candidate_control.value):
                return {}
            return {"candidate_id": probe["selected_candidate_id"], "stage": "linear_tcp",
                    "result": probe["result"], "tcp_path_world_m": probe["tcp_path_world_m"]}
        if str(self.trajectory_source_control.value) == "Skill pregrasp":
            stage = self._skill_pregrasp()
            return ({"candidate_id": str(self.candidate_control.value), "stage": "pregrasp",
                     "result": stage.get("result"),
                     "tcp_path_world_m": stage.get("tcp_path_world_m", [])}
                    if stage is not None else {})
        if str(self.trajectory_source_control.value) == "Skill approach":
            stage = self._skill_stage("approach")
            return ({"candidate_id": str(self.candidate_control.value), "stage": "approach",
                     "result": stage.get("result"),
                     "tcp_path_world_m": stage.get("tcp_path_world_m", [])}
                    if stage is not None else {})
        report = self.case.report or {}
        selected = str(self.candidate_control.value)
        extra = (report.get("candidate_plans") or {}).get(selected)
        if extra is not None:
            return extra
        initial = report.get("plan_attempt") or {}
        return initial if initial.get("candidate_id") == selected else {}

    def _on_plan_candidate(self, event: Any) -> None:
        if self._planning:
            return
        if self.pending_dirty:
            self._show_error("场景有未应用编辑；请先重新规划，再规划当前候选轨迹")
            return
        selected = str(self.candidate_control.value)
        if selected not in self._candidate_options():
            self._show_error("没有可规划的候选")
            return
        if self._selected_plan():
            self._update_trajectory_handle()
            self._update_candidate_visibility()
            self._render_details()
            return
        self._planning = True
        self.plan_candidate_button.disabled = True
        try:
            plan = self.plan_candidate(self.case, selected, seed=self.seed)
            # 正式后端负责持久化；测试替身与未来后端只需返回同一结构即可刷新 UI。
            if self.case.report is None:
                self.case.report = {}
            self.case.report.setdefault("candidate_plans", {})[selected] = plan
            self._update_trajectory_handle()
            self._update_candidate_visibility()
            self._update_status()
            self._render_details()
        except Exception as error:
            self._show_error(f"候选轨迹规划失败：{error}")
            self.run_root.mkdir(parents=True, exist_ok=True)
            with (self.run_root / "ui_errors.log").open("a", encoding="utf-8") as stream:
                stream.write(traceback.format_exc() + "\n")
        finally:
            self._planning = False
            self.plan_candidate_button.disabled = False

    def _on_obstacle(self, event: Any) -> None:
        # 主动选择资产时展示该代理；总开关和筛选框仍可随时恢复完整场景。
        self.collision_toggle.value = True
        self.isolate_collision_toggle.value = True
        self.scene.set_collision_visible(True)
        self._on_obstacle_filter(event)

    def _on_obstacle_filter(self, event: Any) -> None:
        self.scene.select_collision_group(str(self.obstacle_control.value),
                                          isolate=bool(self.isolate_collision_toggle.value))
        self._render_details()

    def _on_all_candidates(self, event: Any) -> None:
        self._update_candidate_visibility()

    def _update_candidate_visibility(self) -> None:
        selected = str(self.candidate_control.value)
        show_all = bool(self.all_candidates_toggle.value)
        for name, handle in self._candidate_frames.items():
            handle.visible = show_all or name == selected
        if getattr(self, "_skill_pregrasp_handle", None) is not None:
            self._skill_pregrasp_handle.visible = (
                selected == (self.skill_report or {}).get("selected_candidate_id")
                and str(self.trajectory_source_control.value) == "Skill pregrasp"
            )
        plan = self._selected_plan()
        trajectory = (plan.get("result") or {}).get("trajectory") or {}
        self.trajectory_slider.disabled = not bool(trajectory.get("positions_rad"))
        if getattr(self, "_trajectory_handle", None) is not None:
            self._trajectory_handle.visible = True
        candidate = next((item for item in self.case.candidates if item["candidate_id"] == selected), None)
        if candidate is not None:
            pose = np.asarray(candidate["T_world_tcp"], dtype=float)
            self._grasp_proxy.wxyz = _wxyz(pose)
            self._grasp_proxy.position = tuple(float(x) for x in pose[:3, 3])
            self._update_grasp_preview(candidate)

    def _build_grasp_preview(self) -> None:
        """复用真实夹爪的 visual mesh；子节点位姿始终相对于目标 TCP。"""
        self._preview_gripper_geoms: dict[int, Any] = {}
        self._preview_gripper_data = None
        robot = self.case.metadata.get("robot", {})
        site_name = robot.get("tcp_sites", {}).get("right")
        if not site_name or mujoco.mj_name2id(self.case.model, mujoco.mjtObj.mjOBJ_SITE, site_name) < 0:
            return
        self._preview_gripper_data = mujoco.MjData(self.case.model)
        for geom_id in range(self.case.model.ngeom):
            name = mujoco.mj_id2name(self.case.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if (not name.startswith("right_") or not any(x in name for x in ("gripper", "finger", "jaw"))
                    or int(self.case.model.geom_group[geom_id]) == 3
                    or self.case.model.geom_rgba[geom_id, 3] <= 0
                    or int(self.case.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)):
                continue
            mesh = _compiled_mesh(self.case.model, geom_id)
            mesh.visual.face_colors = np.array([245, 190, 55, 255], dtype=np.uint8)
            handle = self.server.scene.add_mesh_trimesh(
                f"/pick_bowl/selected_grasp/{geom_id}_{name}", mesh,
            )
            self._preview_gripper_geoms[geom_id] = handle
            self._candidate_handles.append(handle)

    def _update_grasp_preview(self, candidate: dict) -> None:
        if self._preview_gripper_data is None:
            return
        model, data = self.case.model, self._preview_gripper_data
        mujoco.mj_copyData(data, model, self.case.data)
        gripper = self.case.metadata["robot"].get("grippers", {}).get("right", {})
        parameters = candidate.get("parameters", {})
        # 候选 opening_width_m 是闭合目标；当前规划碰撞球使用冻结快照的开口。
        # 视觉预览必须与求解器一致，否则闭合夹爪会伪装成更大的净空。
        width = float((self.case.snapshot.get("gripper_snapshot") or {}).get(
            "width_m", parameters.get("pregrasp_opening_width_m", gripper.get("home_width_m", .08))
        ))
        if gripper.get("mode") == "parallel_slide_proxy":
            low, high = gripper["width_range_m"]
            width = float(np.clip(width, low, high))
            for joint_name in gripper["joint_names"]:
                data.qpos[model.joint(joint_name).qposadr[0]] = width / 2
        mujoco.mj_forward(model, data)
        site = data.site(self.case.metadata["robot"]["tcp_sites"]["right"])
        world_tcp = np.eye(4)
        world_tcp[:3, :3] = site.xmat.reshape(3, 3)
        world_tcp[:3, 3] = site.xpos
        tcp_world = np.linalg.inv(world_tcp)
        for geom_id, handle in self._preview_gripper_geoms.items():
            handle.wxyz, handle.position = _pose(tcp_world @ _geom_pose(data, geom_id))

    def _on_trajectory(self, event: Any) -> None:
        plan = self._selected_plan()
        trajectory = (plan.get("result") or {}).get("trajectory") or {}
        rows = trajectory.get("positions_rad") or []
        if not rows:
            return
        if self._preview_data is None:
            self._preview_data = mujoco.MjData(self.case.model)
        mujoco.mj_copyData(self._preview_data, self.case.model, self.case.data)
        index = min(int(self.trajectory_slider.value), len(rows) - 1)
        for name, value in zip(trajectory["joint_names"], rows[index]):
            self._preview_data.qpos[self.case.model.joint(name).qposadr[0]] = float(value)
        mujoco.mj_forward(self.case.model, self._preview_data)
        # 轨迹只改变机器人；待应用的关节与物体编辑仍应保持在画面上。
        self._refresh_pending_visuals(self._preview_data)
        self.scene.set_robot_spheres_visible(index == 0)

    def _on_replan(self, event: Any) -> None:
        if self._planning:
            return
        self._planning = True
        self.replan_button.disabled = True
        try:
            next_revision = self.case.revision + 1
            target = self.run_root / f"revision_{next_revision:03d}"
            # 失败的尝试保留现场；重开旧 revision 时也始终创建同级新版本。
            while target.exists():
                next_revision += 1
                target = self.run_root / f"revision_{next_revision:03d}"
            options = dict(revision=next_revision, seed=self.seed,
                           pose_overrides={name: pose.copy() for name, pose in self.pending.items()})
            if self.pending_joints:
                options["joint_overrides"] = self.pending_joints.copy()
            if (self.pending_table_size_xy_m is not None and not np.allclose(
                self.pending_table_size_xy_m,
                2 * np.asarray(self.case.model.geom("scene_table").size[:2], dtype=float),
            )):
                options["table_size_xy_m"] = self.pending_table_size_xy_m
            new_case = self.prepare_scene(target, **options)
            planning_options = {"seed": self.seed}
            if str(self.ik_mode_control.value) == "batch":
                planning_options["ik_mode"] = "batch"
            report = self.run_planning(new_case, **planning_options)
            new_case.report = report
            # 等同于用户重新打开这个 run：旧节点和控件整体清除，所有选择、
            # 轨迹帧、候选与碰撞信息从新报告初始化，避免混用两个 revision。
            self._reload_case(new_case)
        except Exception as error:
            self._show_error(f"重新规划失败：{error}")
            self.run_root.mkdir(parents=True, exist_ok=True)
            with (self.run_root / "ui_errors.log").open("a", encoding="utf-8") as stream:
                stream.write(traceback.format_exc() + "\n")
        finally:
            self._planning = False
            self.replan_button.disabled = False

    def _reload_case(self, new_case: Any) -> None:
        """在当前连接里按新 run 的默认状态重建原页面。"""
        old_case = self.case
        options = dict(seed=self.seed, view=self.view,
                       show_collision=bool(self.collision_toggle.value),
                       load_case=self.load_case, prepare_scene=self.prepare_scene,
                       run_planning=self.run_planning, plan_candidate=self.plan_candidate)
        show_visual = bool(self.visual_toggle.value)
        self.server.gui.reset()
        self.server.scene.reset()
        try:
            self.__init__(self.server, new_case.output_dir, loaded_case=new_case, **options)
        except Exception:
            self.server.gui.reset()
            self.server.scene.reset()
            self.__init__(self.server, old_case.output_dir, loaded_case=old_case, **options)
            raise
        if not show_visual:
            self.visual_toggle.value = False
            self.scene.set_visual_visible(False)

    def _update_trajectory_handle(self) -> None:
        """当前候选无轨迹时清空线和滑条；已有轨迹只显示对应 candidate。"""
        for handle in getattr(self, "_linear_endpoint_handles", []):
            handle.remove()
            if handle in self._candidate_handles:
                self._candidate_handles.remove(handle)
        self._linear_endpoint_handles = []
        if self._trajectory_handle is not None:
            self._trajectory_handle.remove()
            if self._trajectory_handle in self._candidate_handles:
                self._candidate_handles.remove(self._trajectory_handle)
            self._trajectory_handle = None
        plan = self._selected_plan()
        path = plan.get("tcp_path_world_m") or []
        if len(path) >= 2:
            points = np.asarray(path, dtype=float)
            self._trajectory_handle = self.server.scene.add_line_segments(
                "/pick_bowl/trajectory/tcp", points=np.stack((points[:-1], points[1:]), axis=1),
                colors=(29, 137, 198), thickness=.003, thickness_units="world",
            )
            self._candidate_handles.append(self._trajectory_handle)
            if str(self.trajectory_source_control.value) == "TCP 直线探针":
                for name, point, color in (("start", points[0], (25, 155, 88)),
                                           ("goal", points[-1], (198, 53, 85))):
                    handle = self.server.scene.add_icosphere(
                        f"/pick_bowl/trajectory/linear_{name}", radius=.008,
                        color=color, position=tuple(float(v) for v in point),
                    )
                    self._linear_endpoint_handles.append(handle)
                    self._candidate_handles.append(handle)
        rows = ((plan.get("result") or {}).get("trajectory") or {}).get("positions_rad") or []
        self.trajectory_slider.max = max(1, len(rows) - 1)
        self.trajectory_slider.value = 0
        self.trajectory_slider.disabled = not bool(rows)

    def _render_results(self) -> None:
        for handle in self._candidate_handles:
            handle.remove()
        self._candidate_handles.clear()
        self._candidate_frames.clear()
        self._trajectory_handle = None
        attempts = {item["candidate_id"]: item["result"] for item in (self.case.report or {}).get("ik_attempts", [])}
        for index, candidate in enumerate(self.case.candidates):
            pose = np.asarray(candidate["T_world_tcp"], dtype=float)
            result = attempts.get(candidate["candidate_id"], {})
            handle = self.server.scene.add_frame(
                f"/pick_bowl/candidates/{index}_{candidate['candidate_id']}",
                axes_length=.035, axes_radius=.001, wxyz=_wxyz(pose), position=tuple(pose[:3, 3]),
            )
            self._candidate_handles.append(handle)
            self._candidate_frames[candidate["candidate_id"]] = handle
        self._grasp_proxy = self.server.scene.add_frame(
            "/pick_bowl/selected_grasp", show_axes=True, axes_length=.035, axes_radius=.001,
        )
        self._candidate_handles.append(self._grasp_proxy)
        self._build_grasp_preview()
        skill_stage = next((stage for stage in (self.skill_report or {}).get("stages", [])
                            if stage.get("name") == "pregrasp" and stage.get("target")), None)
        if skill_stage is not None:
            pose = np.asarray(skill_stage["target"]["matrix"], dtype=float)
            self._skill_pregrasp_handle = self.server.scene.add_frame(
                "/pick_bowl/skill/pregrasp_target", axes_length=.055, axes_radius=.002,
                wxyz=_wxyz(pose), position=tuple(float(x) for x in pose[:3, 3]),
            )
            self._candidate_handles.append(self._skill_pregrasp_handle)
        options = self._candidate_options()
        self.candidate_control.options = options
        selected = self.initial_candidate_id or (self.case.report or {}).get("selected_candidate_id")
        self.candidate_control.value = selected if selected in options else options[0]
        self.initial_candidate_id = None
        self._update_trajectory_sources()
        self._update_trajectory_handle()
        self._update_candidate_visibility()
        obstacle_options = self._obstacle_options()
        self.obstacle_control.options = obstacle_options
        self.obstacle_control.value = obstacle_options[0]
        self.scene.select_collision_group(None, isolate=False)
        self.isolate_collision_toggle.value = False
        self._update_status()
        self._render_details()

    def _update_status(self) -> None:
        report = self.case.report or {}
        linear = (self.linear_probe or {}).get("result") or {}
        linear_metrics = (linear.get("metrics") or {}).get("linear_path_metrics") or {}
        line = "轨迹仅表示规划结果，未验证闭爪或抬碗。"
        if str(self.trajectory_source_control.value) == "TCP 直线探针":
            line = (f"TCP直线探针：{'成功' if linear.get('success') else '失败'} · "
                    f"横向偏差 {linear_metrics.get('max_lateral_deviation_m', 0) * 1000:.2f} mm · "
                    "净空段，尚非抓取接触执行。")
        elif str(self.trajectory_source_control.value) == "Skill approach":
            stage = self._skill_stage("approach") or {}
            motion = stage.get("result") or {}
            metrics = (motion.get("metrics") or {}).get("linear_path_metrics") or {}
            line = (f"Skill approach：{stage.get('status', '未规划')} · "
                    f"TCP横向偏差 {metrics.get('max_lateral_deviation_m', 0) * 1000:.2f} mm · "
                    "规划轨迹，尚无物理执行。")
        self.status.content = (
            f"### 抓碗规划检查 · revision {self.case.revision}\n"
            f"{'● 有未应用编辑' if self.pending_dirty else '✓ 当前场景已应用'} · "
            f"规划耗时 {report.get('total_planning_duration_s', 0):.3f} s · "
            f"快照 `{self.case.snapshot.get('snapshot_sha256', '未知')[:12]}`\n\n"
            + line
        )

    def _render_details(self) -> None:
        report = self.case.report or {}
        selected = str(self.candidate_control.value)
        attempt = next((x for x in report.get("ik_attempts", []) if x["candidate_id"] == selected), None)
        result = attempt["result"] if attempt else {}
        collision = self.case.snapshot
        records = collision.get("records", [])
        excluded = collision.get("excluded", [])
        obstacle_group = str(self.obstacle_control.value)
        grouped_names = set(self.scene.collision_groups.get(obstacle_group, ()))
        group_records = [item for item in records if item["obstacle_name"] in grouped_names]
        objects = self.case.metadata.get("objects", {}).get("sample_poses", {})
        failure = result.get("failure") or {}
        solver_diagnostic = (failure.get("details") or {}).get("solver_diagnostic") or {}
        plan = self._selected_plan()
        plan_result = plan.get("result") or {}
        planned_id = plan.get("candidate_id")
        if plan:
            trajectory_line = (
                f"- 轨迹属于 `{planned_id}`："
                f"{'成功' if plan_result.get('success') else '失败'}；"
                f"耗时 {plan_result.get('duration_s', 0):.3f} s"
            )
        else:
            trajectory_line = ("- 轨迹未尝试：没有通过目标 IK 的候选。"
                               if report.get("ik_success_count") == 0 else
                               "- 当前候选尚无轨迹；IK 成功也不会自动生成轨迹，可点击“规划当前候选轨迹”。")
        source_assets = report.get("source_assets") or {}
        mounting = collision.get("mounting_contact_policy") or {}
        diagnosis = report.get("ik_diagnostics") or {}
        start = diagnosis.get("start_state") or {}
        lines = ["### 当前候选", f"- `{selected}`：IK {'成功' if result.get('success') else '失败或未运行'}；"
                 f"耗时 {result.get('duration_s', 0):.3f} s",
                 f"- 失败：`{failure.get('kind', '无')}`；后端：`{failure.get('backend_status', '无')}`",
                 "- 黄色夹爪复用当前机器人的 visual mesh，并按目标 TCP 与规划碰撞快照开口摆放；它是**位姿示意**，不代表该候选已有 IK 解或实际轨迹。"]
        snapshot_width = (collision.get("gripper_snapshot") or {}).get("width_m")
        closing_width = next((item.get("parameters", {}).get("opening_width_m")
                              for item in self.case.candidates if item["candidate_id"] == selected), None)
        if snapshot_width is not None:
            lines.append(f"- 规划夹爪开口：{float(snapshot_width) * 1000:.0f} mm（黄色预览及碰撞快照）")
        if closing_width is not None:
            lines.append(f"- 候选闭合目标开口：{float(closing_width) * 1000:.0f} mm（尚未执行闭爪）")
        if solver_diagnostic:
            lines.append(f"- 当前失败原因：`{solver_diagnostic.get('reason', 'unknown')}`；"
                   f"约束可行：`{solver_diagnostic.get('constraint_feasible', '未知')}`")
            if solver_diagnostic.get("position_error_m") is not None:
                lines.append("- 目标误差：位置 "
                    f"{solver_diagnostic['position_error_m']:.4f} / "
                    f"{solver_diagnostic.get('position_tolerance_m', 0):.4f} m；姿态 "
                    f"{solver_diagnostic.get('rotation_error_rad', 0):.4f} / "
                    f"{solver_diagnostic.get('rotation_tolerance_rad', 0):.4f} rad")
            collision_check = solver_diagnostic.get("collision_check") or {}
            if collision_check:
                lines.append(f"- 碰撞复核：自身 `{collision_check.get('self_collision')}`；"
                      f"场景 `{collision_check.get('scene_collision')}`；"
                      f"障碍 `{collision_check.get('colliding_obstacles', [])}`")
        lines.extend(["### 本场景规划",
                      f"- IK 通过 {report.get('ik_success_count', 0)}/{len(self.case.candidates)}；"
                      f"默认规划候选 `{report.get('selected_candidate_id') or '无'}`；当前候选 `{selected}`",
                      f"- IK 筛选方式：`{report.get('ik_screening_mode', 'serial')}`；"
                      f"整批耗时 {report.get('ik_screening_duration_s', 0):.3f} s",
                      trajectory_line,
                      f"- 起点碰撞预检：`{start.get('valid', '未知')}`；{start.get('evidence', '无记录')}",
                      f"- IK 失败分布：`{diagnosis.get('failure_counts', {})}`；归因不明 {diagnosis.get('ambiguous_solver_failures', 0)}"])
        if str(self.trajectory_source_control.value) == "抓取目标":
            lines.append("- 当前轨迹是独立诊断：从 home 直接规划到 grasp；它不是接在 Skill pregrasp 后的连续执行。")
        elif str(self.trajectory_source_control.value) == "Skill pregrasp":
            lines.append("- 当前轨迹为 Skill 的 home→pregrasp；可切换到 Skill approach 查看下一段。")
        elif str(self.trajectory_source_control.value) == "Skill approach":
            stage = self._skill_stage("approach") or {}
            metrics = ((stage.get("result") or {}).get("metrics") or {})
            linear = metrics.get("linear_path_metrics") or {}
            lines.append("- 当前轨迹从pregrasp规划终态出发，按TCP直线约束到grasp；尚未执行闭爪与抬升。")
            lines.append(f"- 实际碰撞策略：{metrics.get('effective_collision_policy', '未知')}；"
                         f"TCP横向偏差：{linear.get('max_lateral_deviation_m', 0) * 1000:.2f} mm")
        else:
            metrics = ((self.linear_probe or {}).get("result") or {}).get("metrics", {}).get("linear_path_metrics", {})
            lines.append("- 当前轨迹从已规划pregrasp关节状态出发，沿TCP局部-Z离开碗30 mm；"
                         "使用严格碰撞，仅验收cuRobo直线后端，不代表抓碗接触段已执行。")
            if metrics:
                lines.append(f"- TCP直线偏差：{metrics['max_lateral_deviation_m']:.5f} m；"
                             f"后退：{metrics['max_backward_step_m']:.5f} m；"
                             f"终点误差：{metrics['goal_error_m']:.5f} m")
        if self.skill_report is not None:
            stage_states = ", ".join(
                f"{stage.get('name')}: {stage.get('status')}"
                for stage in self.skill_report.get("stages", [])
            )
            lines.append(f"- PickSkill 阶段：{stage_states}；报告只说明规划，不代表物理抓取成功")
        if diagnosis.get("reason_counts"):
            lines.append(f"- 可验证原因：`{diagnosis['reason_counts']}`")
        if plan and not plan_result.get("success"):
            plan_failure = plan_result.get("failure") or {}
            plan_details = plan_failure.get("details") or {}
            lines.append(f"- 轨迹失败：`{plan_failure.get('kind', 'unknown')}`；"
                         f"{plan_failure.get('summary', '后端未给出摘要')}")
            internal_ik = plan_details.get("ik_diagnostic") or {}
            if internal_ik:
                lines.append(f"- 规划内部目标 IK：`{internal_ik.get('selected_seed_reason', 'unknown')}`；"
                             f"成功种子 {internal_ik.get('successful_seed_count', '未知')}")
                probe = internal_ik.get("world_obstruction_probe") or {}
                if probe:
                    lines.append(f"- 无环境障碍的诊断试算：`{probe.get('reason')}`；"
                                 f"复核障碍 `{probe.get('blocking_obstacles', [])}`；不作为可执行解")
            path_diagnostic = plan_details.get("solver_diagnostic") or {}
            if path_diagnostic:
                path_collision = path_diagnostic.get("collision_check") or {}
                lines.append(f"- 轨迹复核：`{path_diagnostic.get('reason')}`；"
                             f"首次无效帧 `{path_diagnostic.get('first_invalid_waypoint', '未定位')}`；"
                             f"障碍 `{path_collision.get('colliding_obstacles', [])}`")
        lines.append("### 物体清单")
        lines.extend(f"- `{name}`：{np.asarray(pose)[:3, 3].round(3).tolist()} m" for name, pose in objects.items())
        table = self.case.metadata.get("objects", {}).get("table") or {}
        if table:
            lines.append(f"- 桌面尺寸：{table.get('size_xy_m')} m；厚度 {table.get('thickness_m')} m")
        lines.extend(["### 规划碰撞资产",
                      f"- 启用 {len(records)} 项；排除 {len(excluded)} 项；活动臂球 {len(collision.get('active_robot_spheres', []))} 个",
                      f"- 机器人球配置：`{source_assets.get('robot_collision_config', '见 robot config')}`",
                      f"- 安装接触规则：{mounting.get('robot_links', [])} ↔ {mounting.get('obstacle_names', [])}"])
        if grouped_names:
            kinds = {kind: sum(item["export_kind"] == kind for item in group_records)
                     for kind in ("cuboid", "mesh")}
            approximations = sorted({str(item.get("approximation", "未知")) for item in group_records})
            lines.extend([f"- 当前碰撞对象：`{obstacle_group}`；{len(group_records)} 个环境代理，"
                          f"{len(grouped_names) - len(group_records)} 个机器人球",
                          f"- 类型：{kinds}；近似：{approximations}；示例："
                          f"{[(item['obstacle_name'], item['geom_name']) for item in group_records[:4]]}"])
        self.details.content = "\n".join(lines)

    def _show_error(self, message: str) -> None:
        self.status.content = f"### 抓碗规划检查 · revision {self.case.revision}\n⚠ {message}"


__all__ = ["PickBowlPlanningApp", "camera_presets"]
