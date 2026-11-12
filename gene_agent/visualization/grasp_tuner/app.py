"""Viser 渲染与 GUI 绑定；抓取几何全部来自 Knowledge 的 session 状态。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .session import EditableGraspSession, GraspTunerState, SUPPORTED_PROFILES
from gene_agent.knowledge.geometry import TrifoldConcept


# 相机与显示颜色只属于 Viser 适配层，不参与 Knowledge 候选或其坐标变换。
CAMERA_OFFSET_WORLD_M = np.array((0.28, 0.32, 0.36), dtype=float)
CAMERA_UP_WORLD = np.array((0.0, 1.0, 0.0), dtype=float)
CAMERA_FOV_RAD = math.radians(42.0)
SCENE_BACKGROUND_RGB = (24, 32, 45)
BOWL_COLOR_RGBA = (57, 177, 192, 255)
KNOB_COLOR_RGBA = (239, 163, 65, 255)
PALM_COLOR_RGBA = (255, 209, 91, 255)
FINGER_COLOR_RGBA = (255, 112, 66, 255)


@dataclass(frozen=True, slots=True)
class ProxyGeometry:
    """Viser 示意夹爪在 TCP 局部 frame 中的盒体尺寸和中心。"""

    palm_position: tuple[float, float, float]
    palm_extents: tuple[float, float, float]
    left_finger_position: tuple[float, float, float]
    right_finger_position: tuple[float, float, float]
    finger_extents: tuple[float, float, float]


def build_proxy_geometry(closing_width_m: float) -> ProxyGeometry:
    """计算仅用于可视检查的平行夹爪盒体，所有长度均以米表示。

    TCP 原点位于指尖与掌部之间的接近轴参考点：掌部沿 TCP `-Z` 放置，
    两根手指由掌部朝 TCP `+Z` 延伸。两指中心位于 `±X`，内侧面的间距
    等于候选的 `closing_width_m`；此几何不代表 URDF、碰撞体或接触模型。
    """
    width = float(closing_width_m)
    if not math.isfinite(width) or width < 0.0:
        raise ValueError(f"closing_width_m 必须是有限的非负米制数值，收到 {closing_width_m!r}")

    finger_thickness_m = 0.008
    finger_center_x_m = width / 2.0 + finger_thickness_m / 2.0
    return ProxyGeometry(
        palm_position=(0.0, 0.0, -0.025),
        palm_extents=(0.075, 0.018, 0.014),
        left_finger_position=(finger_center_x_m, 0.0, 0.012),
        right_finger_position=(-finger_center_x_m, 0.0, 0.012),
        finger_extents=(finger_thickness_m, 0.014, 0.055),
    )


def _viser_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 ``T_world_node`` 转为 Viser 使用的 wxyz 四元数与米制位置。"""
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return quaternion_wxyz, matrix[:3, 3]


def _box(extents: tuple[float, float, float]) -> Any:
    """延迟导入 trimesh，使纯 session 测试完全不需要渲染依赖。"""
    import trimesh

    return trimesh.creation.box(extents=extents)


def _colored_box(
    extents: tuple[float, float, float], color: tuple[int, int, int, int]
) -> Any:
    """给示意夹爪盒体指定纯色，增强它与物体及背景的对比。"""
    mesh = _box(extents)
    mesh.visual.face_colors = color
    return mesh


class GraspTunerApp:
    """把 task-owned Knowledge profile 显示成可切换、可重生成的 Viser 场景。"""

    def __init__(self, server: Any, repository_root: str | Path, profile_name: str, *, output_path: str | Path | None = None) -> None:
        self.server = server
        self.repository_root = Path(repository_root).resolve()
        # 各 profile 独立保留草稿；试算失败不会污染最后一次有效候选。
        self.editors = {name: EditableGraspSession(self.repository_root, name, output_path=output_path) for name in SUPPORTED_PROFILES}
        self.states = {name: editor.state for name, editor in self.editors.items()}
        self.state = self.states[profile_name]
        self._scene_handles: list[Any] = []
        self._selected_frame: Any | None = None
        self._palm: Any | None = None
        self._left_finger: Any | None = None
        self._right_finger: Any | None = None
        self._selected_label: Any | None = None

        # 原始资产与 Concept 保持 object frame；碗 mesh 和概念开口轴均以 +Y
        # 为上方，因此场景视角使用 +Y-up，桌面网格也放在水平 XZ 平面。
        server.scene.set_up_direction("+y")
        server.scene.set_background_image(
            np.full((8, 8, 3), SCENE_BACKGROUND_RGB, dtype=np.uint8), format="png"
        )
        server.gui.configure_theme(
            dark_mode=True,
            control_width="medium",
            show_logo=False,
            show_share_button=False,
            brand_color=(57, 177, 192),
        )

        @server.on_client_connect
        def _set_initial_camera(client: Any) -> None:
            self._configure_client_camera(client, self.state)
            print(
                f"[grasp-tuner] scene-ready: client camera fixed for {self.state.profile_name}",
                flush=True,
            )

        self.profile_control = server.gui.add_dropdown(
            "Profile",
            options=SUPPORTED_PROFILES,
            initial_value=profile_name,
            hint="切换 Bowl、knob_0 与三段式把手的 Knowledge profile",
        )
        self.candidate_control = server.gui.add_slider(
            "Selected candidate",
            min=0,
            max=len(self.state.candidates) - 1,
            step=1,
            initial_value=0,
        )
        self.details = server.gui.add_markdown("")
        server.gui.add_markdown("### 参数化调节\n编辑 JSON 参数范围和后处理，点击重新生成；校验错误会显示在下面。")
        self.parameter_control = server.gui.add_text("Parameters (JSON)", initial_value="", multiline=True)
        self.postprocess_control = server.gui.add_text("Postprocess (JSON)", initial_value="", multiline=True)
        self.width_control = server.gui.add_text("Closing width (JSON)", initial_value="", multiline=True)
        self.apply_control = server.gui.add_button("重新生成 / Validate")
        self.save_control = server.gui.add_button("保存调参结果")
        self.edit_status = server.gui.add_markdown("")
        self._load_editor_fields()
        self._render_state()

        @self.profile_control.on_update
        def _on_profile(event: Any) -> None:
            self.state = self.states[str(event.target.value)]
            self._load_editor_fields()
            self.candidate_control.max = len(self.state.candidates) - 1
            self.candidate_control.value = 0
            self._render_state()
            print(f"[grasp-tuner] {self.state.profile_name}: {len(self.state.candidates)} candidates")

        @self.candidate_control.on_update
        def _on_candidate(event: Any) -> None:
            index = min(int(event.target.value), len(self.state.candidates) - 1)
            self._update_selection(index)

        @self.apply_control.on_click
        def _on_apply(event: Any) -> None:
            try:
                editor = self.editors[self.state.profile_name]
                state = editor.regenerate(
                    parameters=json.loads(self.parameter_control.value),
                    postprocess=json.loads(self.postprocess_control.value),
                    closing_width=json.loads(self.width_control.value),
                )
                self.states[self.state.profile_name] = state
                self.state = state
                self.candidate_control.max = len(state.candidates) - 1
                self.candidate_control.value = 0
                self._render_state()
                self.edit_status.content = f"✅ 已重生成 {len(state.candidates)} 个候选；尚未保存。"
                print(f"[grasp-tuner] regenerated {state.profile_name}: {len(state.candidates)} candidates", flush=True)
            except (ValueError, TypeError, KeyError) as error:
                self.edit_status.content = f"❌ 校验失败：`{error}`"
                print(f"[grasp-tuner] validation-error: {error}", flush=True)

        @self.save_control.on_click
        def _on_save(event: Any) -> None:
            try:
                path = self.editors[self.state.profile_name].save()
                self.edit_status.content = f"✅ 已保存到 `{path}`"
                print(f"[grasp-tuner] saved: {path}", flush=True)
            except OSError as error:
                self.edit_status.content = f"❌ 保存失败：`{error}`"
                print(f"[grasp-tuner] save-error: {error}", flush=True)

    def _load_editor_fields(self) -> None:
        """切换 profile 时显示当前已验证的原始单位参数，而非内部弧度。"""
        profile = self.editors[self.state.profile_name].profile_json
        self.parameter_control.value = json.dumps(profile["parameters"], ensure_ascii=False, indent=2)
        self.postprocess_control.value = json.dumps(profile.get("postprocess", []), ensure_ascii=False, indent=2)
        self.width_control.value = json.dumps(profile["closing_width"], ensure_ascii=False, indent=2)
        self.edit_status.content = "修改后点击重新生成；默认保存到开发目录，不覆盖运行配置。"

    @property
    def candidate_count(self) -> int:
        return len(self.state.candidates)

    def _track(self, handle: Any) -> Any:
        self._scene_handles.append(handle)
        return handle

    def _clear_scene(self) -> None:
        for handle in reversed(self._scene_handles):
            handle.remove()
        self._scene_handles.clear()

    def _render_state(self) -> None:
        self._clear_scene()
        scene = self.server.scene
        self._track(
            scene.add_grid(
                "/ground",
                width=0.5,
                height=0.5,
                cell_size=0.025,
                section_size=0.1,
                cell_color=(83, 102, 121),
                section_color=(127, 148, 167),
                plane_color=SCENE_BACKGROUND_RGB,
                plane_opacity=0.28,
                plane="xz",
                position=(0.0, -0.045, 0.0),
            )
        )
        self._track(scene.add_frame("/world", axes_length=0.08, axes_radius=0.002))
        self._render_object()

        # 每个候选都显示小 TCP 坐标轴；红 X 是 opening axis，蓝 Z 是 approach axis。
        for index, candidate in enumerate(self.state.candidates):
            wxyz, position = _viser_pose(candidate.tcp_pose_world)
            self._track(
                scene.add_frame(
                    f"/candidates/tcp_{index:03d}",
                    axes_length=0.018,
                    axes_radius=0.0007,
                    wxyz=wxyz,
                    position=position,
                )
            )

        self._selected_frame = self._track(
            scene.add_frame(
                "/selected_visual_proxy",
                show_axes=True,
                axes_length=0.035,
                axes_radius=0.0015,
            )
        )
        # 代理夹爪只表达 TCP 朝向和 closing width，不代表真实 URDF 或碰撞几何。
        self._palm = self._track(
            scene.add_mesh_trimesh(
                "/selected_visual_proxy/palm",
                _colored_box((0.075, 0.018, 0.014), PALM_COLOR_RGBA),
            )
        )
        finger_mesh = _colored_box((0.008, 0.014, 0.055), FINGER_COLOR_RGBA)
        self._left_finger = self._track(
            scene.add_mesh_trimesh("/selected_visual_proxy/left_finger", finger_mesh)
        )
        self._right_finger = self._track(
            scene.add_mesh_trimesh("/selected_visual_proxy/right_finger", finger_mesh.copy())
        )
        self._selected_label = self._track(
            scene.add_label(
                "/selected_visual_proxy/label",
                text="VISUAL PROXY — parallel gripper",
                position=(0.0, 0.0, 0.046),
                anchor="bottom-center",
            )
        )
        self._update_selection(0)

    def _render_object(self) -> None:
        import trimesh

        visual = self.state.object_visual
        if visual.kind == "trifold":
            concept = self.state.concept
            assert isinstance(concept, TrifoldConcept)
            for name, segment in (("grip", concept.grip_segment), ("connector_positive", concept.positive_connector), ("connector_negative", concept.negative_connector)):
                pose = self.state.object_pose_world @ concept.pose_in_object @ segment.pose_in_trifold
                wxyz, position = _viser_pose(pose)
                mesh = _colored_box(tuple(float(v) for v in segment.size_xyz_m), KNOB_COLOR_RGBA)
                self._track(self.server.scene.add_mesh_trimesh(f"/object/{name}", mesh, wxyz=wxyz, position=position))
            return
        if visual.kind == "mesh":
            assert visual.mesh_path is not None
            mesh = trimesh.load_mesh(visual.mesh_path, process=False)
            mesh.visual.face_colors = BOWL_COLOR_RGBA
            self._track(self.server.scene.add_mesh_trimesh("/object/bowl", mesh))
            return

        assert visual.radius_xy_m is not None
        assert visual.height_m is not None
        assert visual.pose_in_object is not None
        # Viser 没有独立的椭圆柱 primitive；先建单位圆柱，再分别缩放局部 X/Y。
        radius_x, radius_y = visual.radius_xy_m
        mesh = trimesh.creation.cylinder(radius=1.0, height=visual.height_m, sections=64)
        mesh.apply_scale((radius_x, radius_y, 1.0))
        mesh.visual.face_colors = KNOB_COLOR_RGBA
        wxyz, position = _viser_pose(self.state.object_pose_world @ visual.pose_in_object)
        self._track(
            self.server.scene.add_mesh_trimesh(
                "/object/knob",
                mesh,
                wxyz=wxyz,
                position=position,
            )
        )

    def _update_selection(self, index: int) -> None:
        candidate = self.state.candidates[index]
        wxyz, position = _viser_pose(candidate.tcp_pose_world)
        self._selected_frame.wxyz = wxyz
        self._selected_frame.position = position

        # 与无 server 数值 helper 共用方向和开口公式，避免显示几何再度翻转。
        geometry = build_proxy_geometry(candidate.closing_width_m)
        self._palm.position = geometry.palm_position
        self._left_finger.position = geometry.left_finger_position
        self._right_finger.position = geometry.right_finger_position
        self._selected_label.text = f"VISUAL PROXY — candidate {index}"

        parameters = candidate.to_dict()["parameters"]
        parameter_text = json.dumps(parameters, ensure_ascii=False, indent=2)
        self.details.content = (
            "### Selected candidate\n"
            f"- **index:** `{index}` / `{len(self.state.candidates) - 1}`\n"
            f"- **candidate ID:** `{candidate.candidate_id}`\n"
            f"- **closing width:** `{candidate.closing_width_m:.4f} m`\n"
            "- **axes:** TCP `+X`（红）是 opening/closing axis；"
            "TCP `+Z`（蓝）是 approach axis。\n"
            "- **gripper:** `VISUAL PROXY`，仅用于检查位姿与开口，不是真实 URDF。\n\n"
            f"**parameters**\n```json\n{parameter_text}\n```"
        )

    @staticmethod
    def _configure_client_camera(client: Any, state: GraspTunerState) -> None:
        """固定新浏览器连接的取景方向、尺度，并把镜头对准当前物体。"""
        target = state.object_pose_world[:3, 3].copy()
        if state.object_visual.kind == "cylinder":
            assert state.object_visual.pose_in_object is not None
            target_pose = state.object_pose_world @ state.object_visual.pose_in_object
            target = target_pose[:3, 3].copy()
        elif state.object_visual.kind == "trifold":
            concept = state.concept
            assert isinstance(concept, TrifoldConcept)
            target = (state.object_pose_world @ concept.pose_in_object @ concept.grip_segment.pose_in_trifold)[:3, 3]

        camera = client.camera
        camera.position = target + CAMERA_OFFSET_WORLD_M
        camera.look_at = target
        camera.up_direction = CAMERA_UP_WORLD
        camera.fov = CAMERA_FOV_RAD
        camera.near = 0.01
        camera.far = 5.0
