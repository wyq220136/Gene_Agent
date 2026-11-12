"""把 Motion DebugBundle 显示成只读 Viser 场景。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.feedback.motion_debug import MotionDebugBundle
from gene_agent.motion.model import MotionFailureKind

from .cameras import CameraPreset, camera_presets


BACKGROUND_RGB = (22, 28, 36)
ROLE_COLORS = {
    "environment": (115, 126, 140),
    "robot": (238, 121, 43),
    "object": (80, 181, 118),
    "fixture": (69, 143, 224),
}
SUCCESS_COLOR = (54, 211, 121)
IK_FAILURE_COLOR = (245, 166, 35)
COLLISION_COLOR = (245, 76, 76)
OTHER_FAILURE_COLOR = (168, 108, 230)


def _wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return float(w), float(x), float(y), float(z)


def _attempt_color(attempt) -> tuple[int, int, int]:
    if attempt.result.success:
        return SUCCESS_COLOR
    kind = attempt.result.failure.kind
    if kind is MotionFailureKind.GOAL_IK_NO_SOLUTION:
        return IK_FAILURE_COLOR
    if kind in {
        MotionFailureKind.START_COLLISION,
        MotionFailureKind.GOAL_COLLISION,
        MotionFailureKind.PATH_COLLISION,
    }:
        return COLLISION_COLOR
    return OTHER_FAILURE_COLOR


def _segments(path) -> np.ndarray:
    points = np.asarray(path, dtype=float)
    return np.stack((points[:-1], points[1:]), axis=1)


def confirm_client_rendered(client: Any) -> None:
    """等待浏览器实际返回一帧，再允许截图端把页面视为 ready。

    当前场景只使用同步 primitives，没有异步纹理；8×8 小帧足以确认此前广播的
    场景与相机已经被 WebGL client 处理，同时避免为 ready 检查传输大图。
    """
    client.flush()
    client.get_render(height=8, width=8, transport_format="png", timeout=8.0)


class MotionDebugApp:
    """只提供查看与筛选，不修改 approach、候选或 task 配置。"""

    def __init__(
        self,
        server: Any,
        bundle: MotionDebugBundle,
        *,
        initial_view: str = "overview",
    ) -> None:
        self.server = server
        self.bundle = bundle
        self.presets = camera_presets(bundle)
        if initial_view not in self.presets:
            raise ValueError(f"未知 motion debug view: {initial_view}")
        self.initial_view = initial_view
        self._initial_attempt_index = self._choose_initial_attempt(initial_view)
        self._node_names: list[str] = []
        self._layer_handles: dict[str, list[Any]] = defaultdict(list)

        server.scene.set_up_direction("+z")
        server.scene.set_background_image(
            np.full((8, 8, 3), BACKGROUND_RGB, dtype=np.uint8), format="png"
        )
        server.gui.configure_theme(
            dark_mode=True,
            control_width="medium",
            show_logo=False,
            show_share_button=False,
            brand_color=(70, 164, 190),
        )
        self._set_initial_camera(self.presets[initial_view])
        self._build_scene()
        self._build_controls()

        @server.on_client_connect
        def _set_client_camera(client: Any) -> None:
            self.configure_client_camera(client, self.presets[self.initial_view])
            confirm_client_rendered(client)
            print(
                f"[motion-debug] scene-ready: view={self.initial_view}; run={bundle.run_id}",
                flush=True,
            )

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(self._node_names)

    def _add(self, layer: str, name: str, handle: Any) -> Any:
        self._node_names.append(name)
        self._layer_handles[layer].append(handle)
        return handle

    def _build_scene(self) -> None:
        scene = self.server.scene
        name = "/motion_debug/world"
        self._add("scene", name, scene.add_frame(name, axes_length=0.12, axes_radius=0.002))

        for box in self.bundle.scene_boxes:
            name = f"/motion_debug/scene/{box.geometry_id}"
            self._add(
                "scene",
                name,
                scene.add_box(
                    name,
                    dimensions=box.size_xyz_m,
                    color=ROLE_COLORS.get(box.role, ROLE_COLORS["environment"]),
                    opacity=0.42,
                    wxyz=_wxyz(box.pose_world),
                    position=box.pose_world[:3, 3],
                ),
            )

        for attempt in self.bundle.attempts:
            base = f"/motion_debug/attempts/{attempt.attempt_id}"
            pose = attempt.request.target.matrix
            target_name = f"{base}/target"
            self._add(
                "targets",
                target_name,
                scene.add_frame(
                    target_name,
                    axes_length=0.045,
                    axes_radius=0.0015,
                    wxyz=_wxyz(pose),
                    position=pose[:3, 3],
                ),
            )
            status_name = f"{base}/status"
            self._add(
                "targets",
                status_name,
                scene.add_icosphere(
                    status_name,
                    radius=0.012,
                    color=_attempt_color(attempt),
                    opacity=0.88,
                    position=pose[:3, 3],
                ),
            )
            if len(attempt.tcp_path_world_m) >= 2:
                path_name = f"{base}/path"
                self._add(
                    "paths",
                    path_name,
                    scene.add_line_segments(
                        path_name,
                        points=_segments(attempt.tcp_path_world_m),
                        colors=_attempt_color(attempt),
                        thickness=0.004,
                        thickness_units="world",
                    ),
                )

        for marker in self.bundle.collision_markers:
            if marker.position_world_m is None:
                continue
            name = f"/motion_debug/collisions/{marker.marker_id}"
            self._add(
                "collisions",
                name,
                scene.add_icosphere(
                    name,
                    radius=marker.radius_m,
                    color=COLLISION_COLOR,
                    opacity=0.78,
                    wireframe=True,
                    position=marker.position_world_m,
                ),
            )
            label_name = f"{name}/label"
            self._add(
                "collisions",
                label_name,
                scene.add_label(
                    label_name,
                    text=f"{marker.first} ↔ {marker.second}",
                    position=marker.position_world_m,
                    anchor="bottom-center",
                ),
            )

    def _build_controls(self) -> None:
        labels = tuple(
            f"{attempt.attempt_id}: {attempt.label}" for attempt in self.bundle.attempts
        ) or ("(no attempts)",)
        self.attempt_control = self.server.gui.add_dropdown(
            "Attempt", options=labels, initial_value=labels[self._initial_attempt_index]
        )
        self.view_control = self.server.gui.add_dropdown(
            "Camera",
            options=tuple(self.presets),
            initial_value=self.initial_view,
        )
        self.layer_controls = {
            name: self.server.gui.add_checkbox(f"Show {name}", initial_value=True)
            for name in ("scene", "targets", "paths", "collisions")
        }
        self.details = self.server.gui.add_markdown("")
        self._update_details(self._initial_attempt_index)

        @self.attempt_control.on_update
        def _select_attempt(event: Any) -> None:
            if self.bundle.attempts:
                self._update_details(tuple(self.attempt_control.options).index(event.target.value))

        @self.view_control.on_update
        def _select_view(event: Any) -> None:
            preset = self.presets[str(event.target.value)]
            for client in self.server.get_clients().values():
                self.configure_client_camera(client, preset)

        for layer, control in self.layer_controls.items():
            @control.on_update
            def _toggle(event: Any, selected_layer: str = layer) -> None:
                for handle in self._layer_handles[selected_layer]:
                    handle.visible = bool(event.target.value)

    def _choose_initial_attempt(self, view: str) -> int:
        if view == "collision":
            collision_kinds = {
                MotionFailureKind.START_COLLISION,
                MotionFailureKind.GOAL_COLLISION,
                MotionFailureKind.PATH_COLLISION,
            }
            for index, attempt in enumerate(self.bundle.attempts):
                if (
                    attempt.result.failure is not None
                    and attempt.result.failure.kind in collision_kinds
                ):
                    return index
        return 0

    def _update_details(self, index: int) -> None:
        if not self.bundle.attempts:
            self.details.content = "### Motion Debug\n只读视图；当前 bundle 没有规划 attempt。"
            return
        attempt = self.bundle.attempts[index]
        failure = None if attempt.result.failure is None else attempt.result.failure.kind.value
        self.details.content = (
            "### Motion Debug（只读）\n"
            f"- stage: `{attempt.stage_id}`\n"
            f"- candidate: `{attempt.candidate_id}`\n"
            f"- operation: `{attempt.request.operation.value}`\n"
            f"- success: `{attempt.result.success}`\n"
            f"- failure: `{failure}`\n"
            f"- collision policy: `{attempt.request.collision_policy.policy_id}`\n\n"
            "参数修改由 Codex 在 Skill/task 配置中完成；本界面只显示结果。"
        )

    def _set_initial_camera(self, preset: CameraPreset) -> None:
        camera = self.server.initial_camera
        camera.position = preset.position
        camera.look_at = preset.look_at
        camera.up = preset.up_direction

    @staticmethod
    def configure_client_camera(client: Any, preset: CameraPreset) -> None:
        camera = client.camera
        camera.position = preset.position
        camera.look_at = preset.look_at
        camera.up_direction = preset.up_direction
        camera.fov = preset.fov_rad
        camera.near = preset.near_m
        camera.far = preset.far_m


__all__ = ["MotionDebugApp"]
