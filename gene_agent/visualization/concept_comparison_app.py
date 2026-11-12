"""通用 Concept ↔ 参考 mesh 的只读 Viser 界面。"""

from __future__ import annotations

from threading import RLock
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .concept_comparison import ComparisonData


PALETTE = ((239, 91, 61), (35, 172, 102), (28, 130, 225),
           (176, 92, 207), (235, 161, 48))


class ConceptComparisonApp:
    """对任意 part/layer 数据渲染相同的检查界面。"""

    def __init__(self, server: Any, data: ComparisonData, *,
                 initial_part: str | None = None, title: str = "Concept ↔ 参考 mesh",
                 layer_labels: dict[str, str] | None = None) -> None:
        if not data.parts or not data.layers:
            raise ValueError("检查器需要至少一个 part 和一个 Concept 图层")
        initial_part = initial_part or data.parts[0]
        if initial_part not in data.parts:
            raise ValueError(f"未知部件：{initial_part}")
        self.server, self.data, self.title = server, data, title
        self.labels = {layer: (layer_labels or {}).get(layer, layer) for layer in data.layers}
        self.colors = {layer: PALETTE[index % len(PALETTE)]
                       for index, layer in enumerate(data.layers)}
        self._lock = RLock()
        self._handles: list[Any] = []
        self.camera: dict = {}
        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 248, dtype=np.uint8), format="png")
        server.gui.configure_theme(dark_mode=False, control_width="large", show_logo=False,
                                   show_share_button=False, brand_color=(42, 117, 173))
        server.scene.add_frame("/comparison/object_frame", axes_length=.045, axes_radius=.0012)
        server.scene.add_grid("/comparison/grid", width=.7, height=.7, cell_size=.05,
                              plane="xy", position=(0., 0., -.025))

        @server.on_client_connect
        def _client(client: Any) -> None:
            self._set_client_camera(client)
            client.flush()
            client.get_render(height=8, width=8, transport_format="png", timeout=8.)
            print(f"[concept-comparison] scene-ready: {self.part.value}", flush=True)

        gui = server.gui
        gui.add_markdown(f"### {title}\n灰色为参考 mesh，彩色线框为 Concept；"
                         "所有图层须位于同一 object frame。")
        self.part = gui.add_dropdown("检查部件", options=data.parts, initial_value=initial_part)
        self.part.on_update(lambda event: self._refresh(focus=True))
        self.visible = gui.add_checkbox("显示参考 mesh", initial_value=True)
        self.context = gui.add_checkbox("显示整体背景", initial_value=False)
        self.layers = {layer: gui.add_checkbox(f"显示{self.labels[layer]}", initial_value=True)
                       for layer in data.layers}
        for control in (self.visible, self.context, *self.layers.values()):
            control.on_update(lambda event: self._refresh())
        with gui.add_folder("视角", expand_by_default=True):
            for choice, label in (("all_front", "正面看全部"), ("mesh_front", "正面看实物"),
                                  ("all_side", "侧面看全部"), ("all_top", "俯视看全部")):
                button = gui.add_button(label)
                button.on_click(lambda event, mode=choice: self._focus(mode))
        self.status = gui.add_markdown("")
        self._refresh(focus=True)

    def _camera_spec(self, part: str, choice: str) -> dict:
        boxes = [np.asarray(self.data.visual_meshes[part].bounds)]
        if choice.startswith("all"):
            boxes.extend(np.asarray(mesh.bounds) for mesh in self.data.layer_meshes[part].values())
        minimum = np.min([box[0] for box in boxes], axis=0)
        maximum = np.max([box[1] for box in boxes], axis=0)
        center = (minimum + maximum) / 2.
        distance = max(.34, 2.0 * max(float(np.max(maximum - minimum)), .13))
        if choice.endswith("side"):
            offset = [distance, -.55 * distance, .35 * distance]
        elif choice.endswith("top"):
            offset = [.28 * distance, -.36 * distance, distance]
        else:
            offset = [.22 * distance, -distance, .45 * distance]
        return {"position": (center + offset).tolist(), "look_at": center.tolist(),
                "up": [0., 0., 1.], "fov": .8}

    def _set_client_camera(self, client: Any) -> None:
        client.camera.position = self.camera["position"]
        client.camera.look_at = self.camera["look_at"]
        client.camera.up_direction = self.camera["up"]
        client.camera.fov = self.camera["fov"]

    def _focus(self, choice: str = "all_front") -> None:
        with self._lock:
            self.camera = self._camera_spec(str(self.part.value), choice)
            self.server.initial_camera.position = self.camera["position"]
            self.server.initial_camera.look_at = self.camera["look_at"]
            self.server.initial_camera.up = self.camera["up"]
            self.server.initial_camera.fov = self.camera["fov"]
            for client in self.server.get_clients().values():
                self._set_client_camera(client)

    def _add_mesh(self, name: str, mesh: Any, *, color: tuple[int, int, int],
                  opacity: float, wireframe: bool = False) -> None:
        self._handles.append(self.server.scene.add_mesh_simple(
            name, vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32), color=color,
            opacity=opacity, wireframe=wireframe, side="double",
        ))

    def _refresh(self, *, focus: bool = False) -> None:
        with self._lock:
            part = str(self.part.value)
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            if bool(self.context.value) and self.data.context_mesh is not None:
                self._add_mesh("/comparison/context", self.data.context_mesh,
                               color=(185, 193, 201), opacity=.13)
            if bool(self.visible.value):
                self._add_mesh("/comparison/reference", self.data.visual_meshes[part],
                               color=(107, 117, 132), opacity=.58)
            reference_center = np.asarray(self.data.visual_meshes[part].bounds).mean(axis=0)
            metrics = self.data.report["parts"][part]
            for index, layer in enumerate(self.data.layers):
                mesh = self.data.layer_meshes[part].get(layer)
                if mesh is None or not bool(self.layers[layer].value):
                    continue
                name = f"/comparison/layer_{index}"
                self._add_mesh(name, mesh, color=self.colors[layer], opacity=.78, wireframe=True)
                pose = self.data.layer_poses[part][layer]
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
                self._handles.append(self.server.scene.add_frame(
                    name + "_frame", axes_length=.026, axes_radius=.0009,
                    origin_radius=.0018, wxyz=tuple(quat[[3, 0, 1, 2]]),
                    position=tuple(pose[:3, 3]),
                ))
                center = np.asarray(mesh.bounds).mean(axis=0)
                label_position = center + np.array([0., 0., .02 * index])
                self._handles.append(self.server.scene.add_label(
                    name + "_label", text=self.labels[layer],
                    position=tuple(label_position), anchor="bottom-center",
                ))
                layer_metrics = metrics.get(layer) or {}
                if layer_metrics.get("center_distance_mm") is not None:
                    self._handles.append(self.server.scene.add_line_segments(
                        name + "_delta", np.asarray([[reference_center, center]], dtype=np.float32),
                        self.colors[layer], thickness=.002,
                    ))
            self._update_status(part)
            if focus or not self.camera:
                self._focus()

    def _update_status(self, part: str) -> None:
        entry = self.data.report["parts"][part]
        rows = []
        for layer in self.data.layers:
            value = entry.get(layer)
            if value is None:
                result = "未提供"
            elif "center_distance_mm" not in value:
                result = "仅作视觉叠加"
            else:
                delta = np.round(value["aabb_size_delta_mm"], 1).tolist()
                result = f"AABB 中心差 **{value['center_distance_mm']:.1f} mm**；尺寸差 {delta} mm"
            rows.append(f"- {self.labels[layer]}：{result}")
        note = entry.get("comparison_note") or ""
        self.status.content = (f"**{part}**  \n" + "  \n".join(rows) + "  \n" + note
                               + "  \n**数值口径：** 同一 object frame 的 AABB；不检查表面精确配准。")
