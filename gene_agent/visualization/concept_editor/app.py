"""只面向 mesh 的 Viser Concept 标注界面。"""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import CuboidConcept, CylinderConcept, RingConcept, validate_pose

from .model import ConceptEditorModel, ReferenceMesh, concept_mesh


def pose_to_wxyz(pose: ArrayLike) -> tuple[float, float, float, float]:
    """把 ``T_object_part`` 旋转转换为 Viser 的 wxyz 四元数顺序。"""
    checked = validate_pose(pose, field_name="T_object_part")
    x, y, z, w = Rotation.from_matrix(checked[:3, :3]).as_quat()
    return float(w), float(x), float(y), float(z)


def pose_from_wxyz_position(
    wxyz: ArrayLike, position_m: ArrayLike
) -> NDArray[np.float64]:
    """由 Viser gizmo 状态构造 ``T_object_part``。"""
    quaternion = np.asarray(wxyz, dtype=float)
    position = np.asarray(position_m, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("wxyz 必须是四个有限数")
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position_m 必须是三个有限数")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("wxyz 不能是零四元数")
    quaternion = quaternion / norm
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    ).as_matrix()
    pose[:3, 3] = position
    return validate_pose(pose, field_name="T_object_part")


class ViserConceptEditor:
    """Viser 控件的薄封装；所有持久状态都在 ``ConceptEditorModel``。"""

    def __init__(
        self,
        server: Any,
        model: ConceptEditorModel,
        reference_mesh: ReferenceMesh,
        *,
        output_path: str | Path,
    ) -> None:
        self.server = server
        self.model = model
        self.reference_mesh = reference_mesh
        self.output_path = Path(output_path).expanduser().resolve()
        self._updating = False
        # Viser 会在线程池并发调用普通 GUI 回调。所有模型修改和 UI 局部刷新必须
        # 串行执行，否则快速输入时旧事件可能覆盖新值。
        self._action_lock = threading.RLock()
        self._concept_handles: list[Any] = []
        self.transform_control: Any | None = None
        self.geometry_folder: Any | None = None
        self.geometry_controls: dict[str, Any] = {}

        self._build_scene()
        self._build_controls()
        self._sync_from_model()

    @staticmethod
    def _labels(model: ConceptEditorModel) -> tuple[str, ...]:
        return tuple(
            f"{part.index}: {part.name} ({part.template})" for part in model.parts
        ) or ("(no parts)",)

    def _build_scene(self) -> None:
        self.server.scene.add_frame("/object", axes_length=0.06, axes_radius=0.0015)
        self.server.scene.add_mesh_simple(
            "/object/reference_mesh",
            vertices=np.asarray(self.reference_mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.reference_mesh.faces, dtype=np.uint32),
            color=(105, 110, 120),
            opacity=0.58,
            side="double",
        )
        self._set_initial_camera()

    def _set_initial_camera(self) -> None:
        minimum = self.reference_mesh.bounds_min_m
        maximum = self.reference_mesh.bounds_max_m
        center = (minimum + maximum) / 2.0
        extent = max(float(np.max(maximum - minimum)), 0.1)
        self.server.initial_camera.look_at = tuple(center)
        self.server.initial_camera.position = tuple(
            center + np.array([1.4, -1.4, 1.0]) * extent
        )
        self.server.initial_camera.up = (0.0, 0.0, 1.0)

    def _build_controls(self) -> None:
        with self.server.gui.add_folder("Parts", expand_by_default=True):
            labels = self._labels(self.model)
            selected_label = (
                labels[self.model.selected_index]
                if self.model.selected_index is not None
                else labels[0]
            )
            self.part_dropdown = self.server.gui.add_dropdown(
                "当前 part", options=labels, initial_value=selected_label
            )
            self.part_name = self.server.gui.add_text("part name", initial_value="")
            self.rename_button = self.server.gui.add_button("重命名")
            self.new_kind = self.server.gui.add_dropdown(
                "新增类型", options=("cuboid", "cylinder", "ring"), initial_value="cuboid"
            )
            self.new_name = self.server.gui.add_text("新增 part name", initial_value="part")
            self.add_button = self.server.gui.add_button("新增 part", color="blue")
            self.confirm_delete = self.server.gui.add_checkbox("确认删除当前 part", False)
            self.delete_button = self.server.gui.add_button("删除当前 part", color="red")

        with self.server.gui.add_folder("Pose T_object_part", expand_by_default=True):
            self.pose_controls = {
                name: self.server.gui.add_number(
                    label,
                    initial_value=0.0,
                    step=0.001 if name.endswith("_m") else 0.1,
                )
                for name, label in (
                    ("x_m", "X (m)"),
                    ("y_m", "Y (m)"),
                    ("z_m", "Z (m)"),
                    ("roll_deg", "Roll X (deg)"),
                    ("pitch_deg", "Pitch Y (deg)"),
                    ("yaw_deg", "Yaw Z (deg)"),
                )
            }

        with self.server.gui.add_folder("操作", expand_by_default=True):
            self.initialize_button = self.server.gui.add_button("按 mesh 包围盒初始化")
            self.undo_button = self.server.gui.add_button("撤销")
            self.redo_button = self.server.gui.add_button("重做")
            self.focus_mesh_button = self.server.gui.add_button("聚焦 mesh")
            self.focus_part_button = self.server.gui.add_button("聚焦当前 part")
            self.allow_overwrite = self.server.gui.add_checkbox("允许覆盖已有输出", False)
            self.save_button = self.server.gui.add_button("保存 JSON", color="green")
        self.status = self.server.gui.add_markdown("就绪")

        self.part_dropdown.on_update(self._on_part_selected)
        self.rename_button.on_click(self._on_rename)
        self.add_button.on_click(self._on_add)
        self.delete_button.on_click(self._on_delete)
        for control in self.pose_controls.values():
            control.on_update(self._on_pose_numbers)
        self.initialize_button.on_click(self._on_initialize)
        self.undo_button.on_click(self._on_undo)
        self.redo_button.on_click(self._on_redo)
        self.focus_mesh_button.on_click(self._on_focus_mesh)
        self.focus_part_button.on_click(self._on_focus_part)
        self.save_button.on_click(self._on_save)

    def _set_status(self, message: str, *, error: bool = False) -> None:
        prefix = "错误" if error else "状态"
        self.status.content = f"**{prefix}:** {message}"
        print(f"[concept-editor] {prefix}: {message}", flush=True)

    def _run_action(
        self,
        action,
        *,
        resync: bool = True,
        success_message: str | Callable[[], str] | None = None,
    ) -> bool:
        with self._action_lock:
            try:
                action()
                if resync:
                    self._sync_from_model()
                if success_message is not None:
                    message = success_message() if callable(success_message) else success_message
                    self._set_status(message)
                return True
            except Exception as error:
                self._set_status(f"{type(error).__name__}: {error}", error=True)
                # 页面保留简短错误，终端保留完整调用栈，方便定位具体回调和输入值。
                traceback.print_exc()
                return False

    def _part_log_label(self) -> str:
        index = self.model.selected_index
        if index is None:
            return "none"
        part = self.model.parts[index]
        return f"{part.index}:{part.name}"

    @staticmethod
    def _format_numbers(values: list[float] | tuple[float, ...]) -> str:
        return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"

    def _sync_from_model(self) -> None:
        self._updating = True
        try:
            labels = self._labels(self.model)
            self.part_dropdown.options = labels
            if self.model.selected_index is None:
                self.part_dropdown.value = labels[0]
                self.part_name.value = ""
                self._remove_transform()
                self._rebuild_geometry_controls(None)
                self._refresh_concepts()
                self._set_status("当前没有 part；请新增 cuboid、cylinder 或 ring。")
                return

            summary = self.model.parts[self.model.selected_index]
            self.part_dropdown.value = labels[self.model.selected_index]
            self.part_name.value = summary.name
            if not summary.editable:
                self._remove_transform()
                self._rebuild_geometry_controls(None)
                self._refresh_concepts()
                self._set_status(f"{summary.template} 当前只读；保存时会原样保留。")
                return

            concept = self.model.current_concept()
            pose = concept.pose_in_object
            rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
            values = (*pose[:3, 3], *rpy)
            for control, value in zip(self.pose_controls.values(), values):
                control.value = float(value)
                control.disabled = False
            self._rebuild_geometry_controls(concept)
            self._rebuild_transform(pose)
            self._refresh_concepts()
            self._set_status(
                f"正在编辑 `{summary.name}`；输出 `{self.output_path}`。"
            )
        finally:
            self._updating = False

    def _remove_transform(self) -> None:
        if self.transform_control is not None:
            self.transform_control.remove()
            self.transform_control = None

    def _rebuild_transform(self, pose: NDArray[np.float64]) -> None:
        self._remove_transform()
        self.transform_control = self.server.scene.add_transform_controls(
            "/object/selected_part_transform",
            position=pose[:3, 3],
            wxyz=pose_to_wxyz(pose),
            scale=0.12,
        )
        self.transform_control.on_update(self._on_transform)

    def _add_geometry_number(
        self, key: str, label: str, value: float, *, max_value: float = 10.0
    ) -> None:
        control = self.server.gui.add_number(
            label,
            initial_value=float(value),
            min=1e-6,
            max=max_value,
            step=0.001,
        )
        control.on_update(self._on_geometry_numbers)
        self.geometry_controls[key] = control

    def _rebuild_geometry_controls(self, concept: Any | None) -> None:
        if self.geometry_folder is not None:
            self.geometry_folder.remove()
        self.geometry_controls = {}
        self.geometry_folder = self.server.gui.add_folder(
            "Geometry (m)", expand_by_default=True
        )
        if concept is None:
            with self.geometry_folder:
                self.server.gui.add_markdown("当前 part 不可编辑。")
            return
        with self.geometry_folder:
            if isinstance(concept, CuboidConcept):
                for index, axis in enumerate("XYZ"):
                    self._add_geometry_number(
                        f"size_{axis.lower()}_m", f"size {axis} (m)", concept.size_xyz_m[index]
                    )
            elif isinstance(concept, CylinderConcept):
                self._add_geometry_number("radius_x_m", "radius X (m)", concept.radius_xy_m[0])
                self._add_geometry_number("radius_y_m", "radius Y (m)", concept.radius_xy_m[1])
                self._add_geometry_number("height_m", "height Z (m)", concept.height_m)
            elif isinstance(concept, RingConcept):
                self._add_geometry_number("outer_radius_m", "outer radius (m)", concept.outer_radius_m)
                self._add_geometry_number("inner_radius_m", "inner radius (m)", concept.inner_radius_m)
                self._add_geometry_number("height_m", "height Z (m)", concept.height_m)
                self._add_geometry_number(
                    "exist_angle_deg",
                    "exist angle (deg)",
                    float(np.rad2deg(concept.angle_span_rad)),
                    max_value=360.0,
                )

    def _refresh_concepts(self) -> None:
        for handle in self._concept_handles:
            handle.remove()
        self._concept_handles = []
        for part in self.model.parts:
            if not part.editable:
                continue
            preview = concept_mesh(self.model.concept_at(part.index))
            selected = part.index == self.model.selected_index
            self._concept_handles.append(
                self.server.scene.add_mesh_simple(
                    f"/object/concepts/part_{part.index}",
                    vertices=np.asarray(preview.vertices, dtype=np.float32),
                    faces=np.asarray(preview.faces, dtype=np.uint32),
                    color=(0, 105, 235) if selected else (235, 125, 20),
                    opacity=0.92 if selected else 0.42,
                    wireframe=selected,
                    side="double",
                )
            )

    def _on_part_selected(self, _event: Any) -> None:
        if self._updating or not self.model.parts:
            return
        label = str(self.part_dropdown.value)
        labels = self._labels(self.model)
        self._run_action(
            lambda: self.model.select_part(labels.index(label)),
            success_message=f"已选择 {label}",
        )

    def _on_add(self, _event: Any) -> None:
        self._run_action(
            lambda: self.model.add_part(str(self.new_kind.value), str(self.new_name.value)),
            success_message=f"已新增 part `{self.new_name.value}`",
        )

    def _on_rename(self, _event: Any) -> None:
        self._run_action(
            lambda: self.model.rename_current_part(str(self.part_name.value)),
            success_message=f"已重命名为 `{self.part_name.value}`",
        )

    def _on_delete(self, _event: Any) -> None:
        if not self.confirm_delete.value:
            self._set_status("请先勾选“确认删除当前 part”。", error=True)
            return
        self._run_action(
            self.model.delete_current_part,
            success_message="已删除当前 part",
        )
        self.confirm_delete.value = False

    def _pose_from_numbers(self) -> NDArray[np.float64]:
        pose = np.eye(4)
        pose[:3, 3] = [self.pose_controls[key].value for key in ("x_m", "y_m", "z_m")]
        pose[:3, :3] = Rotation.from_euler(
            "xyz",
            [
                self.pose_controls[key].value
                for key in ("roll_deg", "pitch_deg", "yaw_deg")
            ],
            degrees=True,
        ).as_matrix()
        return pose

    def _on_pose_numbers(self, _event: Any) -> None:
        if self._updating:
            return
        with self._action_lock:
            target = getattr(_event, "target", None)
            changed = next(
                (key for key, control in self.pose_controls.items() if control is target),
                "unknown",
            )
            pose_holder: dict[str, NDArray[np.float64]] = {}

            def update() -> None:
                pose = self._pose_from_numbers()
                pose_holder["pose"] = pose
                self.model.set_pose(pose)
                if self.transform_control is not None:
                    self.transform_control.position = tuple(pose[:3, 3])
                    self.transform_control.wxyz = pose_to_wxyz(pose)
                self._refresh_concepts()

            self._run_action(
                update,
                resync=False,
                success_message=lambda: (
                    f"pose updated part={self._part_log_label()} source=numbers "
                    f"changed={changed} "
                    f"position_m={self._format_numbers(list(pose_holder['pose'][:3, 3]))} "
                    f"rpy_deg={self._format_numbers(list(Rotation.from_matrix(pose_holder['pose'][:3, :3]).as_euler('xyz', degrees=True)))}"
                ),
            )

    def _on_transform(self, _event: Any) -> None:
        if self._updating or self.transform_control is None:
            return
        with self._action_lock:
            if getattr(_event, "target", self.transform_control) is not self.transform_control:
                self._set_status("已忽略来自旧 transform gizmo 的事件。")
                return
            pose_holder: dict[str, NDArray[np.float64]] = {}

            def update() -> None:
                assert self.transform_control is not None
                pose = pose_from_wxyz_position(
                    self.transform_control.wxyz, self.transform_control.position
                )
                pose_holder["pose"] = pose
                self.model.set_pose(pose)
                rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
                self._updating = True
                try:
                    for control, value in zip(
                        self.pose_controls.values(), (*pose[:3, 3], *rpy)
                    ):
                        control.value = float(value)
                finally:
                    self._updating = False
                self._refresh_concepts()

            self._run_action(
                update,
                resync=False,
                success_message=lambda: (
                    f"pose updated part={self._part_log_label()} source=gizmo "
                    f"position_m={self._format_numbers(list(pose_holder['pose'][:3, 3]))} "
                    f"wxyz={self._format_numbers(list(pose_to_wxyz(pose_holder['pose'])))}"
                ),
            )

    def _on_geometry_numbers(self, _event: Any) -> None:
        if self._updating:
            return
        with self._action_lock:
            target = getattr(_event, "target", None)
            changed = next(
                (key for key, control in self.geometry_controls.items() if control is target),
                None,
            )
            if changed is None:
                self._set_status("已忽略来自旧 Geometry 控件的事件。")
                return
            values: dict[str, Any] = {}

            def update() -> None:
                concept = self.model.current_concept()
                if isinstance(concept, CuboidConcept):
                    values["size_xyz_m"] = [
                        self.geometry_controls[f"size_{axis}_m"].value for axis in "xyz"
                    ]
                elif isinstance(concept, CylinderConcept):
                    values.update(
                        {
                            key: self.geometry_controls[key].value
                            for key in ("radius_x_m", "radius_y_m", "height_m")
                        }
                    )
                else:
                    values.update(
                        {
                            key: self.geometry_controls[key].value
                            for key in (
                                "outer_radius_m",
                                "inner_radius_m",
                                "height_m",
                                "exist_angle_deg",
                            )
                        }
                    )
                self.model.set_geometry(**values)
                self._refresh_concepts()

            def success() -> str:
                rendered = " ".join(
                    f"{key}="
                    + (
                        self._format_numbers(value)
                        if isinstance(value, (list, tuple))
                        else f"{float(value):.9g}"
                    )
                    for key, value in values.items()
                )
                return (
                    f"geometry updated part={self._part_log_label()} "
                    f"changed={changed} {rendered}"
                )

            self._run_action(
                update,
                resync=False,
                success_message=success,
            )

    def _on_initialize(self, _event: Any) -> None:
        self._run_action(
            lambda: self.model.auto_initialize(self.reference_mesh),
            success_message="已按 mesh 包围盒初始化当前 part",
        )

    def _on_undo(self, _event: Any) -> None:
        self._run_action(self.model.undo, success_message="已撤销")

    def _on_redo(self, _event: Any) -> None:
        self._run_action(self.model.redo, success_message="已重做")

    def _focus(self, minimum: NDArray[np.float64], maximum: NDArray[np.float64]) -> None:
        center = (minimum + maximum) / 2.0
        extent = max(float(np.max(maximum - minimum)), 0.05)
        for client in self.server.get_clients().values():
            client.camera.look_at = center
            client.camera.position = center + np.array([1.4, -1.4, 1.0]) * extent
            client.camera.up = np.array([0.0, 0.0, 1.0])

    def _on_focus_mesh(self, _event: Any) -> None:
        self._run_action(
            lambda: self._focus(
                self.reference_mesh.bounds_min_m, self.reference_mesh.bounds_max_m
            ),
            resync=False,
            success_message="已聚焦参考 mesh",
        )

    def _on_focus_part(self, _event: Any) -> None:
        self._run_action(
            lambda: self._focus(
                concept_mesh(self.model.current_concept()).vertices.min(axis=0),
                concept_mesh(self.model.current_concept()).vertices.max(axis=0),
            ),
            resync=False,
            success_message="已聚焦当前 part",
        )

    def _on_save(self, _event: Any) -> None:
        try:
            saved = self.model.save(
                self.output_path, overwrite=bool(self.allow_overwrite.value)
            )
            self._set_status(f"已保存 `{saved}`。")
        except Exception as error:
            self._set_status(f"{type(error).__name__}: {error}", error=True)


def run_editor(
    *,
    mesh_path: str | Path,
    concept_path: str | Path | None = None,
    output_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8125,
) -> None:
    """加载输入并持续运行 Viser 服务。"""
    import viser

    mesh_source = Path(mesh_path).expanduser().resolve()
    reference = ReferenceMesh.from_path(mesh_source)
    if concept_path is None:
        model = ConceptEditorModel.empty()
        model.add_part("cuboid", "part_0")
        default_output = mesh_source.with_name(f"{mesh_source.stem}_concept.tuned.json")
    else:
        concept_source = Path(concept_path).expanduser().resolve()
        model = ConceptEditorModel.from_json(concept_source)
        default_output = concept_source.with_name(f"{concept_source.stem}.tuned.json")
    output = Path(output_path).expanduser().resolve() if output_path else default_output
    server = viser.ViserServer(host=host, port=int(port))
    ViserConceptEditor(server, model, reference, output_path=output)
    bounds = reference.bounds_max_m - reference.bounds_min_m
    print(f"Concept mesh 标注器: http://{host}:{server.get_port()}")
    print(f"参考 mesh: {mesh_source}；AABB(m): {bounds.tolist()}")
    print(f"输出: {output}")
    if float(np.max(bounds)) > 10.0 or float(np.max(bounds)) < 1e-4:
        print("警告: mesh 尺度异常；工具不会自动缩放，请确认输入单位为米。")
    print("按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对照 object-frame mesh 编辑 Cuboid/Cylinder/Ring Concept"
    )
    parser.add_argument("--mesh", type=Path, required=True, help="参考 mesh，单位米")
    parser.add_argument("--concept", type=Path, help="可选的初始 Concept JSON")
    parser.add_argument("--output", type=Path, help="输出 JSON；默认另存为 .tuned.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    args = parser.parse_args()
    run_editor(
        mesh_path=args.mesh,
        concept_path=args.concept,
        output_path=args.output,
        host=args.host,
        port=args.port,
    )


__all__ = [
    "ViserConceptEditor",
    "main",
    "pose_from_wxyz_position",
    "pose_to_wxyz",
    "run_editor",
]


if __name__ == "__main__":
    main()
