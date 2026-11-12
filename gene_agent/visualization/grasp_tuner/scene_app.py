"""完整 MuJoCo GT 场景里的单物体 Knowledge 抓取调参界面。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case
from gene_agent.visualization.pick_bowl.app import camera_presets
from gene_agent.visualization.pick_bowl.scene import PickBowlScene, _geom_pose
from gene_agent.visualization.pick_bowl.app import _wxyz
from .controls import GraspParameterControls
from .overlay import CandidateOverlay
from .session import EditableGraspSession
from .selection import closest_candidate_index, representative_candidate_indices
from .feedback import canonical_sha256, write_live_state
from .gripper_preview import GripperMeshPreview
from .pose_ik import MuJoCoPoseIK


class SceneGraspTunerApp:
    """复用现有 Viser 场景渲染；Knowledge 候选只叠加，不触发规划。"""

    def __init__(self, server: Any, root: str | Path, run_dir: str | Path, *, target: str = "bowl", view: str = "bowl", output_path: str | Path | None = None, state_path: str | Path | None = None, restored: dict | None = None):
        if target != "bowl":
            raise ValueError("首版完整场景调参目标只支持 bowl")
        if view not in {"bowl", "overview", "gripper"}:
            raise ValueError("view 只支持 bowl、overview 或 gripper")
        self.server = server
        self.state_path = Path(state_path).resolve() if state_path is not None else None
        self.run_dir = Path(run_dir).resolve()
        self.input_error: str | None = None
        self.case = load_pick_bowl_case(Path(run_dir))
        self.target = target
        if restored and (restored.get("target") != target or restored.get("scene_snapshot_sha256") != self.case.snapshot.get("snapshot_sha256")):
            raise ValueError("恢复状态与当前目标或场景快照不一致；请使用截图原始 run")
        self.object_pose_world = np.asarray(restored.get("object_pose_world", self.case.bowl_pose) if restored else self.case.bowl_pose, dtype=float)
        if self.object_pose_world.shape != (4, 4):
            raise ValueError("恢复状态中的物体位姿必须为 4×4 矩阵")
        self.editor = EditableGraspSession(root, "task_1_bowl_0", output_path=output_path, object_pose_world=self.object_pose_world)
        if restored:
            if restored.get("profile_name") != self.editor.profile_name:
                raise ValueError("恢复状态的 profile_name 与当前任务不一致")
            self.editor.replace_profile(restored["profile"])
        self.state = self.editor.state
        self.overlay = CandidateOverlay(server, prefix="/scene_knowledge")
        self.ik_results = None
        self._syncing = True

        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 232, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False, show_share_button=False, brand_color=(53, 112, 169))
        self.scene = PickBowlScene(server, self.case.model, self.case.data, self.case.metadata, self.case.snapshot)
        self._bowl_geoms = []
        for geom_id, handle in self.scene.visual_handles.items():
            body_id = int(self.case.model.geom_bodyid[geom_id])
            while body_id:
                if body_id == self.case.model.body("bowl").id:
                    self._bowl_geoms.append((geom_id, handle))
                    break
                body_id = int(self.case.model.body_parentid[body_id])
        self._move_bowl_visual()
        self.scene.set_collision_visible(False)
        self.gripper_preview = GripperMeshPreview(server, self.case)
        self.overlay.set_proxy_visible(not self.gripper_preview.available)
        # 代表夹爪按需创建。默认只显示一个，但用户可以按当前候选总数
        # 提高上限；已创建的 mesh 会复用，不随每次参数变化重新编译。
        self.ghost_previews: list[GripperMeshPreview] = []
        self.representative_indices: tuple[int, ...] = ()
        self._robot_visual_ids = tuple(
            geom_id for geom_id in self.scene.visual_handles
            if any(word in (mujoco.mj_id2name(self.case.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
                   for word in ("right_", "left_", "robot_body", "arm_"))
        )
        bowl = self.case.bowl_pose[:3, 3]
        self.cameras = camera_presets(self.case)
        self.cameras["gripper"] = self._gripper_camera()
        if restored and restored.get("view") == view and isinstance(restored.get("camera"), dict):
            self.cameras[view] = restored["camera"]
        self.camera = self.cameras[view]
        server.initial_camera.position = self.camera["position"]
        server.initial_camera.look_at = self.camera["look_at"]
        server.initial_camera.up = self.camera["up"]
        server.initial_camera.fov = self.camera["fov"]

        @server.on_client_connect
        def _camera(client: Any) -> None:
            client.camera.position = self.camera["position"]
            client.camera.look_at = self.camera["look_at"]
            client.camera.up_direction = self.camera["up"]
            client.camera.fov = self.camera["fov"]
            client.camera.near = .01
            client.camera.far = 10.
            # 浏览器手动旋转视角时同步记录实际相机，截图清单才可准确复现。
            client.camera.on_update(self._on_browser_camera_update)
            # 浏览器真正完成一次 WebGL 绘制后才发 ready，避免自动截图捕获半成品。
            client.flush()
            client.get_render(height=8, width=8, transport_format="png", timeout=8.0)
            print(f"[scene-grasp-tuner] scene-ready: {self.target}; {len(self.state.candidates)} candidates", flush=True)

        gui = server.gui
        gui.add_markdown("### 场景抓取调参\n拖动碗或调节参数后，全部候选会同步更新；纯 IK 需点击按钮，不检查碰撞与路径。")
        self.status = gui.add_markdown("")
        self.view_control = gui.add_dropdown("视角", options=("bowl", "gripper", "overview"), initial_value=view)
        self.view_control.on_update(lambda event: self._change_view())
        self.focus_button = gui.add_button("对焦当前候选")
        self.focus_button.on_click(lambda event: self._focus_candidate())
        self.robot_toggle = gui.add_checkbox("显示当前机器人 visual", initial_value=bool(restored.get("robot_visual_visible", view != "gripper")) if restored else view != "gripper")
        self.robot_toggle.on_update(lambda event: self._set_robot_visible())
        self._set_robot_visible()
        self.real_gripper_toggle = gui.add_checkbox("显示夹爪 visual mesh", initial_value=bool(restored.get("gripper_visual") == "mujoco_mesh") if restored else self.gripper_preview.available)
        self.real_gripper_toggle.disabled = not self.gripper_preview.available
        self.real_gripper_toggle.on_update(lambda event: self._toggle_gripper())
        self.palm_toggle = gui.add_checkbox("显示夹爪掌部", initial_value=bool(restored.get("gripper_palm_visible", view != "gripper")) if restored else view != "gripper")
        self.palm_toggle.disabled = not self.gripper_preview.available
        self.palm_toggle.on_update(lambda event: self._toggle_gripper())
        self.gripper_preview.set_palm_visible(bool(self.palm_toggle.value))
        self.gripper_preview.set_visible(bool(self.real_gripper_toggle.value))
        self.overlay.set_proxy_visible(not bool(self.real_gripper_toggle.value))
        self.collision_toggle = gui.add_checkbox("显示 cuRobo 场景碰撞代理", initial_value=bool(restored.get("collision_visible", False)) if restored else False)
        self.scene.set_collision_visible(bool(self.collision_toggle.value))
        self.collision_toggle.on_update(lambda event: self.scene.set_collision_visible(bool(self.collision_toggle.value)))
        self.show_all = gui.add_checkbox("显示全部候选位置／IK", initial_value=bool(restored.get("show_all_candidates", False)) if restored else False)
        self.show_all.on_update(lambda event: self._show_all())
        max_available = max(0, len(self.state.candidates) - 1) if self.gripper_preview.available else 0
        initial_limit = min(max_available, max(0, int(restored.get("representative_limit", 12)) if restored else 12))
        initial_count = min(initial_limit, max(0, int(restored.get("representative_count", 0)) if restored else 1))
        self.representative_limit = gui.add_number("代表夹爪上限", initial_value=initial_limit,
                                                   min=0, max=max_available, step=1)
        self.representative_limit.disabled = not self.gripper_preview.available
        self.representative_limit.on_update(lambda event: self._on_representative_limit())
        self.representative_count = gui.add_slider("代表夹爪数量", min=0, max=initial_limit, step=1,
                                                  initial_value=initial_count)
        self.representative_count.disabled = not self.gripper_preview.available
        self.representative_count.on_update(lambda event: self._update_representatives())
        self.representative_summary = gui.add_markdown("")
        selected_index = int(restored.get("selected_index", 0)) if restored else 0
        if not 0 <= selected_index < len(self.state.candidates):
            raise ValueError("恢复状态的候选序号超出当前 profile 范围")
        self.candidate = gui.add_slider("查看候选", min=0, max=max(0, len(self.state.candidates)-1), step=1, initial_value=selected_index)
        self.candidate.on_update(lambda event: self._show_selected())
        self.quick_summary = gui.add_markdown("")
        with gui.add_folder("移动目标碗（预览位姿）", expand_by_default=True):
            gui.add_markdown("拖动场景中的坐标手柄，或输入世界坐标。编辑仅用于本次调参，不改原始 run。")
            self.drag_toggle = gui.add_checkbox("显示碗拖拽手柄", initial_value=bool(restored.get("bowl_drag_visible", False)) if restored else False)
            self.drag_toggle.on_update(lambda event: self._toggle_bowl_drag())
            self.object_xyz = [gui.add_number(f"world {axis} (m)", initial_value=float(self.object_pose_world[index, 3]), step=.005)
                               for index, axis in enumerate("XYZ")]
            self.object_yaw = gui.add_number("相对原始 yaw (°)", initial_value=self._object_yaw_deg(), step=1.)
            for control in (*self.object_xyz, self.object_yaw):
                control.on_update(lambda event: self._on_object_numbers())
            self.object_gizmo = server.scene.add_transform_controls("/scene_knowledge/edit_bowl", scale=.10,
                            position=tuple(self.object_pose_world[:3, 3]), wxyz=_wxyz(self.object_pose_world))
            self.object_gizmo.visible = bool(self.drag_toggle.value)
            self.object_gizmo.on_update(lambda event: self._on_object_gizmo())
        self.ik_button = gui.add_button("检查全部候选纯 IK", color="blue")
        self.ik_button.on_click(lambda event: self._check_ik())
        self.ik_summary = gui.add_markdown("IK 尚未检查；青色表示未知。")
        self.parameters = GraspParameterControls(gui, self.editor.profile_json["parameters"], self._regenerate, self._parameter_error)
        profile = self.editor.profile_json
        post = profile.get("postprocess", [])
        first = next((item for item in post if item.get("type") == "translate"), None)
        shift = first["xyz_m"] if first else [0., 0., 0.]
        frame = first["frame"] if first else "object"
        object_y_world = np.round(self.case.bowl_pose[:3, 1], 3).tolist()
        with gui.add_folder("TCP 后处理偏移", expand_by_default=False):
            gui.add_markdown(f"当前 Bowl 的 object +Y 在 world 中的方向：`{object_y_world}`。偏移沿所选坐标系的轴应用。")
            self.shift_frame = gui.add_dropdown("偏移坐标系", options=("object", "tcp"), initial_value=frame)
            self.shift_xyz = [gui.add_slider(f"{axis} (m)", min=-.08, max=.08, step=.001, initial_value=float(value)) for axis, value in zip("XYZ", shift)]
            self.shift_frame.on_update(lambda event: self._regenerate())
            for control in self.shift_xyz:
                control.on_update(lambda event: self._regenerate())
            self.advanced = gui.add_text("高级：完整 postprocess JSON", initial_value=json.dumps(post, ensure_ascii=False), multiline=True)
            self.apply_advanced = gui.add_button("应用高级后处理")
            self.apply_advanced.on_click(lambda event: self._regenerate(use_advanced=True))
        self.save_button = gui.add_button("保存调参结果", color="blue")
        self.save_button.on_click(self._save)
        with gui.add_folder("候选详细信息", expand_by_default=False):
            self.details = gui.add_markdown("")
        self._syncing = False
        self._draw_candidates()
        if restored:
            selected = self.state.candidates[selected_index]
            expected = restored["selected_candidate"]
            if selected.candidate_id != expected["candidate_id"] or not np.allclose(selected.tcp_pose_world, expected["tcp_pose_world"], atol=1e-9):
                raise ValueError("恢复结果与截图候选不一致；请核对 Knowledge 代码与资产版本")
            if "representative_indices" in restored and list(self.representative_indices) != restored["representative_indices"]:
                raise ValueError("恢复后代表夹爪与截图不一致；请核对候选生成结果")
        self.status.content = f"**目标：** bowl；**候选：** {len(self.state.candidates)}；**预览 pose：** `{np.round(self.object_pose_world[:3, 3], 4).tolist()}`"
        self._write_state()

    def _postprocess(self, use_advanced: bool = False) -> list:
        if use_advanced:
            return json.loads(self.advanced.value)
        # 常用的平移由滑块接管，其余 step 按 task profile 原顺序保留。
        steps = self.editor.profile_json.get("postprocess", [])
        item = {"type": "translate", "frame": str(self.shift_frame.value), "xyz_m": [float(control.value) for control in self.shift_xyz]}
        index = next((i for i, step in enumerate(steps) if step.get("type") == "translate"), None)
        if index is None:
            steps.insert(0, item)
        else:
            steps[index] = item
        return steps

    def _regenerate(self, use_advanced: bool = False) -> None:
        if self._syncing:
            return
        try:
            previous = self.state.candidates[int(self.candidate.value)]
            profile = self.editor.profile_json
            postprocess = self._postprocess(use_advanced)
            state = self.editor.regenerate(parameters=self.parameters.read(), postprocess=postprocess, closing_width=profile["closing_width"])
            self.state = state
            if use_advanced:
                self._sync_postprocess_controls(postprocess)
            else:
                self.advanced.value = json.dumps(postprocess, ensure_ascii=False)
            self.candidate.max = max(0, len(state.candidates)-1)
            self.candidate.value = closest_candidate_index(state.candidates, previous)
            self._reconcile_representative_limit()
            self.ik_results = None
            self._draw_candidates()
            self.overlay.set_reachability(None)
            self.ik_summary.content = "参数已变化；请重新检查纯 IK。"
            self.input_error = None
            self.status.content = f"✅ 已重生成 {len(state.candidates)} 个候选；当前结果尚未保存。"
            print(f"[scene-grasp-tuner] regenerated: {len(state.candidates)} candidates", flush=True)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self.input_error = str(error)
            self.status.content = f"❌ 当前输入无效：{error}。画面仍是上一次有效结果，未保存当前输入。"
            print(f"[scene-grasp-tuner] invalid: {error}", flush=True)
            self._write_state()

    def _sync_postprocess_controls(self, steps: list) -> None:
        """高级 JSON 生效后回填常用滑块，避免下一次拖动覆盖 JSON 改动。"""
        first = next((step for step in steps if step.get("type") == "translate"), None)
        frame = first["frame"] if first else "object"
        xyz = first["xyz_m"] if first else [0., 0., 0.]
        self._syncing = True
        try:
            self.shift_frame.value = frame
            for control, value in zip(self.shift_xyz, xyz):
                control.value = float(value)
        finally:
            self._syncing = False

    def _save(self, event: Any) -> None:
        if self.input_error is not None:
            self.status.content = "❌ 当前输入无效；请先修正参数，避免把上一次有效结果误认为当前调参结果。"
            return
        try:
            path = self.editor.save()
            self.status.content = f"✅ 已保存：`{path}`"
            print(f"[scene-grasp-tuner] saved: {path}", flush=True)
        except (OSError, ValueError) as error:
            self.status.content = f"❌ 保存失败：{error}"

    def _draw_candidates(self) -> None:
        self.overlay.update(self.state.candidates, show_all=bool(self.show_all.value), selected=int(self.candidate.value))
        self._show_selected()

    def _show_all(self) -> None:
        self.overlay.toggle_all(bool(self.show_all.value))
        self._write_state()

    def _change_view(self) -> None:
        self.robot_toggle.value = str(self.view_control.value) != "gripper"
        self._set_robot_visible()
        if str(self.view_control.value) == "gripper":
            # 侧视重点是两指与碗沿；掌部仍可由用户手动重新打开。
            self.palm_toggle.value = False
            self._toggle_gripper()
            self.cameras["gripper"] = self._gripper_camera()
        self.camera = self.cameras[str(self.view_control.value)]
        for client in self.server.get_clients().values():
            client.camera.position = self.camera["position"]
            client.camera.look_at = self.camera["look_at"]
            client.camera.up_direction = self.camera["up"]
            client.camera.fov = self.camera["fov"]
        self._write_state()

    def _focus_candidate(self) -> None:
        """只有用户显式对焦才移动近景相机，保证参数修改前后可以直接比较。"""
        if str(self.view_control.value) == "gripper":
            self.cameras["gripper"] = self._gripper_camera()
        self.camera = self.cameras[str(self.view_control.value)]
        for client in self.server.get_clients().values():
            client.camera.position = self.camera["position"]
            client.camera.look_at = self.camera["look_at"]
            client.camera.up_direction = self.camera["up"]
            client.camera.fov = self.camera["fov"]
        self._write_state()

    def _set_robot_visible(self) -> None:
        for geom_id in self._robot_visual_ids:
            self.scene.visual_handles[geom_id].visible = bool(self.robot_toggle.value)
        self._write_state()

    def _show_selected(self) -> None:
        self.overlay.select(int(self.candidate.value))
        item = self.state.candidates[min(int(self.candidate.value), len(self.state.candidates)-1)]
        self.gripper_preview.select(item.tcp_pose_world, opening_width_m=item.closing_width_m)
        self.overlay.selected_frame.visible = not bool(self.real_gripper_toggle.value)
        self._update_representatives()
        ik_note = ""
        if self.ik_results is not None and int(self.candidate.value) < len(self.ik_results):
            result = self.ik_results[int(self.candidate.value)]
            ik_note = f" · IK {'可达' if result.reachable else '未求得'}（位置误差 {result.position_error_m*1000:.1f} mm，角度误差 {result.rotation_error_deg:.1f}°）"
        self.quick_summary.content = (
            f"**候选 {int(self.candidate.value)+1}/{len(self.state.candidates)}** · "
            f"TCP `{np.round(item.tcp_pose_world[:3,3], 3).tolist()}` m{ik_note}"
        )
        self.details.content = (
            f"**候选 ID：** `{item.candidate_id}`  \n"
            f"**TCP world xyz (m)：** `{np.round(item.tcp_pose_world[:3,3], 5).tolist()}`  \n"
            f"**开口宽度 (m)：** `{item.closing_width_m:.5f}`  \n"
            f"**生成参数：** `{json.dumps(item.to_dict()['parameters'], ensure_ascii=False)}`"
        )
        self._write_state()

    def _update_representatives(self) -> None:
        """围绕当前候选挑选少量分散姿态，使用真实手指展示参数变化。"""
        if not hasattr(self, "candidate") or not hasattr(self, "representative_count"):
            return
        selected = int(self.candidate.value)
        count = min(int(self.representative_count.value), max(0, len(self.state.candidates) - 1))
        if not self.gripper_preview.available:
            count = 0
        self._ensure_representative_previews(count)
        self.representative_indices = representative_candidate_indices(self.state.candidates, selected, count)
        visible = bool(self.real_gripper_toggle.value)
        for slot, preview in enumerate(self.ghost_previews):
            active = visible and slot < len(self.representative_indices)
            preview.set_visible(active)
            if active:
                item = self.state.candidates[self.representative_indices[slot]]
                preview.select(item.tcp_pose_world, opening_width_m=item.closing_width_m)
        numbers = "、".join(str(index + 1) for index in self.representative_indices[:12])
        if len(self.representative_indices) > 12:
            numbers += f"…等 {len(self.representative_indices)} 个"
        self.representative_summary.content = (
            f"淡紫色真实手指：候选 {numbers}；点击手指可查看该姿态。"
            if numbers and visible else "仅显示当前候选的真实夹爪。"
        )
        self._write_state()

    def _ensure_representative_previews(self, count: int) -> None:
        """懒加载真实夹爪 mesh；增大上限本身不会立即占用渲染资源。"""
        while len(self.ghost_previews) < count:
            slot = len(self.ghost_previews)
            preview = GripperMeshPreview(
                self.server, self.case, prefix=f"/scene_knowledge/representative_{slot}",
                color_rgba=(191, 169, 185, 95),
            )
            preview.set_visible(False)
            preview.on_click(lambda event, index=slot: self._select_representative(index))
            self.ghost_previews.append(preview)

    def _on_representative_limit(self) -> None:
        if self._syncing:
            return
        self._reconcile_representative_limit()
        self._update_representatives()

    def _reconcile_representative_limit(self) -> None:
        """候选数变动时同步数值框与数量滑块，避免保留越界的旧上限。"""
        available = max(0, len(self.state.candidates) - 1) if self.gripper_preview.available else 0
        previous_syncing = self._syncing
        self._syncing = True
        try:
            self.representative_limit.max = available
            limit = min(available, max(0, int(self.representative_limit.value)))
            self.representative_limit.value = limit
            self.representative_count.max = limit
            if int(self.representative_count.value) > limit:
                self.representative_count.value = limit
        finally:
            self._syncing = previous_syncing

    def _select_representative(self, slot: int) -> None:
        if slot < len(self.representative_indices):
            self.candidate.value = self.representative_indices[slot]

    def _toggle_bowl_drag(self) -> None:
        self.object_gizmo.visible = bool(self.drag_toggle.value)
        self._write_state()

    def _move_bowl_visual(self) -> None:
        """只平移/旋转渲染节点，不改冻结的 MuJoCo 场景和碰撞快照。"""
        delta = self.object_pose_world @ np.linalg.inv(self.case.bowl_pose)
        for geom_id, handle in self._bowl_geoms:
            pose = delta @ _geom_pose(self.case.data, geom_id)
            handle.position = tuple(float(x) for x in pose[:3, 3])
            handle.wxyz = _wxyz(pose)

    def _object_yaw_deg(self) -> float:
        delta = self.object_pose_world[:3, :3] @ self.case.bowl_pose[:3, :3].T
        return float(Rotation.from_matrix(delta).as_euler("xyz", degrees=True)[2])

    def _set_object_pose(self, pose: np.ndarray) -> None:
        if not np.isfinite(pose).all():
            self.status.content = "❌ 物体位姿包含非有限值。"
            return
        previous_pose = self.editor.object_pose_world.copy()
        self.editor.object_pose_world = pose.copy()
        try:
            profile = self.editor.profile_json
            self.state = self.editor.regenerate(parameters=profile["parameters"],
                                                postprocess=profile.get("postprocess", []),
                                                closing_width=profile["closing_width"])
        except (ValueError, TypeError) as error:
            self.editor.object_pose_world = previous_pose
            self.status.content = f"❌ 物体位姿无效：{error}"
            return
        self.object_pose_world = pose.copy()
        self._move_bowl_visual()
        self.ik_results = None
        self._draw_candidates()
        self.overlay.set_reachability(None)
        self.ik_summary.content = "碗位姿已变化；请重新检查纯 IK。"
        self.status.content = f"✅ 碗预览位置：`{np.round(pose[:3, 3], 4).tolist()}`；原始场景未修改。"
        self._write_state()

    def _on_object_numbers(self) -> None:
        if self._syncing:
            return
        pose = self.case.bowl_pose.copy()
        pose[:3, 3] = [float(control.value) for control in self.object_xyz]
        pose[:3, :3] = Rotation.from_euler("z", float(self.object_yaw.value), degrees=True).as_matrix() @ pose[:3, :3]
        self._set_object_pose(pose)
        self._syncing = True
        try:
            self.object_gizmo.position = tuple(pose[:3, 3])
            self.object_gizmo.wxyz = _wxyz(pose)
        finally:
            self._syncing = False

    def _on_object_gizmo(self) -> None:
        if self._syncing:
            return
        pose = np.eye(4)
        pose[:3, 3] = self.object_gizmo.position
        quaternion = np.asarray(self.object_gizmo.wxyz)
        pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        self._set_object_pose(pose)
        self._syncing = True
        try:
            for control, value in zip(self.object_xyz, pose[:3, 3]):
                control.value = float(value)
            self.object_yaw.value = self._object_yaw_deg()
        finally:
            self._syncing = False

    def _check_ik(self) -> None:
        """按需检查全部当前候选；结果只表明运动学位姿收敛。"""
        try:
            self.ik_summary.content = "正在检查纯 IK；不使用碰撞和路径规划…"
            robot = self.case.metadata["robot"]
            group = self.case.snapshot.get("active_group", "right")
            solver = MuJoCoPoseIK(self.case.model, self.case.data,
                                  tuple(robot["joint_groups"][group]), robot["tcp_sites"][group])
            self.ik_results = tuple(solver.solve(item.tcp_pose_world) for item in self.state.candidates)
            passed = sum(result.reachable for result in self.ik_results)
            self.overlay.set_reachability([result.reachable for result in self.ik_results])
            self.show_all.value = True
            self._show_all()
            self.ik_summary.content = f"纯 IK：**{passed}/{len(self.ik_results)}** 可达。绿色可达，红色未求得；不代表无碰撞或路径可行。"
            self._show_selected()
        except Exception as error:
            self.ik_results = None
            self.overlay.set_reachability(None)
            self.ik_summary.content = f"❌ IK 检查失败：{type(error).__name__}: {error}"
            print(f"[scene-grasp-tuner] ik error: {type(error).__name__}: {error}", flush=True)

    def _toggle_gripper(self) -> None:
        visible = bool(self.real_gripper_toggle.value) and self.gripper_preview.available
        self.gripper_preview.set_visible(visible)
        self.gripper_preview.set_palm_visible(bool(self.palm_toggle.value))
        self.overlay.set_proxy_visible(not visible)
        self.overlay.selected_frame.visible = not visible
        self._update_representatives()
        self._write_state()

    def _gripper_camera(self) -> dict:
        pose = self.state.candidates[min(int(self.candidate.value), len(self.state.candidates)-1)].tcp_pose_world if hasattr(self, "candidate") else self.state.candidates[0].tcp_pose_world
        origin, rotation = pose[:3, 3], pose[:3, :3]
        robot = self.case.metadata["robot"]
        group = self.case.snapshot["active_group"]
        gripper = robot["grippers"][group]
        fingers = gripper["fingers"]
        data = self.gripper_preview.data if self.gripper_preview.available else self.case.data
        tcp = data.site(robot["tcp_sites"][group])
        source_rotation = tcp.xmat.reshape(3, 3)
        positive = data.site(fingers["positive"]["inner_site_name"]).xpos
        negative = data.site(fingers["negative"]["inner_site_name"]).xpos
        opening = rotation @ (source_rotation.T @ (positive - negative))
        local_midpoint = source_rotation.T @ ((positive + negative) * .5 - tcp.xpos)
        target = origin + rotation @ local_midpoint
        # 视线沿水平侧向、垂直于两指开合轴：两根手指会在画面中分开。
        side = np.cross(opening, [0., 0., 1.])
        if np.linalg.norm(side) < 1e-6:
            side = rotation[:, 1].copy()
            side[2] = 0.
        side /= np.linalg.norm(side)
        root_name = robot.get("group_root_bodies", {}).get(group)
        if root_name and mujoco.mj_name2id(self.case.model, mujoco.mjtObj.mjOBJ_BODY, root_name) >= 0:
            root = data.body(root_name).xpos
            if np.dot(side, target - root) < 0:
                side = -side
        table = self.case.model.geom("scene_table")
        table_data = self.case.data.geom("scene_table")
        table_top = float(table_data.xpos[2] + np.abs(table_data.xmat.reshape(3, 3)[2]) @ table.size)
        height = max(float(target[2] + .16), table_top + .05)
        position = target + .40 * side
        position[2] = height
        return {
            "position": tuple(float(x) for x in position),
            "look_at": tuple(float(x) for x in target),
            "up": (0., 0., 1.),
            "fov": .85,
        }

    def _on_browser_camera_update(self, camera: Any) -> None:
        """记录浏览器真正看到的相机，而非最后一次选择的预设视角。"""
        self.camera = {
            "position": np.asarray(camera.position, dtype=float).tolist(),
            "look_at": np.asarray(camera.look_at, dtype=float).tolist(),
            "up": np.asarray(camera.up_direction, dtype=float).tolist(),
            "fov": float(camera.fov),
        }
        self._write_state()

    def _parameter_error(self, message: str) -> None:
        self.input_error = message
        self.status.content = f"❌ 无法切换采样方式：{message}。画面仍是上一次有效结果。"
        print(f"[scene-grasp-tuner] mode switch rejected: {message}", flush=True)
        self._write_state()

    def _write_state(self) -> None:
        if self.state_path is None or not hasattr(self, "candidate"):
            return
        profile = self.editor.profile_json
        selected = self.state.candidates[min(int(self.candidate.value), len(self.state.candidates) - 1)]
        payload = {
            "schema_version": 1, "tool": "scene", "run": str(self.run_dir),
            "scene_revision": self.case.revision,
            "scene_snapshot_sha256": self.case.snapshot.get("snapshot_sha256"),
            "target": self.target, "object_pose_world": self.object_pose_world.tolist(),
            "object_pose_source": "edited_preview" if not np.allclose(self.object_pose_world, self.case.bowl_pose) else "mujoco_gt",
            "profile_name": self.editor.profile_name, "profile": profile,
            "profile_sha256": canonical_sha256(profile),
            "selected_index": int(self.candidate.value),
            "selected_candidate": selected.to_dict(), "candidate_count": len(self.state.candidates),
            "ik_scope": "pose_only_no_collision",
            "ik_reachable": None if self.ik_results is None else [item.reachable for item in self.ik_results],
            "ik_position_error_m": None if self.ik_results is None else [item.position_error_m for item in self.ik_results],
            "ik_rotation_error_deg": None if self.ik_results is None else [item.rotation_error_deg for item in self.ik_results],
            "collision_visible": bool(self.collision_toggle.value),
            "show_all_candidates": bool(self.show_all.value),
            "bowl_drag_visible": bool(self.drag_toggle.value),
            "representative_count": int(self.representative_count.value),
            "representative_limit": int(self.representative_limit.value),
            "representative_indices": list(self.representative_indices),
            "view": str(self.view_control.value), "camera": self.camera,
            "gripper_visual": "mujoco_mesh" if bool(self.real_gripper_toggle.value) and self.gripper_preview.available else "box_proxy",
            "gripper_visual_opening_width_m": self.gripper_preview.opening_width_m,
            "gripper_palm_visible": bool(self.palm_toggle.value),
            "robot_visual_visible": bool(self.robot_toggle.value),
            "input_valid": self.input_error is None, "input_error": self.input_error,
        }
        write_live_state(self.state_path, payload)


__all__ = ["SceneGraspTunerApp"]
