"""Concept 标注器的无 UI 核心。

编辑器保留完整 JSON，只修改当前受支持 part 的已知字段。mesh 顶点、Concept
尺寸和 ``T_object_part`` 平移均使用米；输入 mesh 已位于 object frame。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import (
    Concept,
    CuboidConcept,
    CylinderConcept,
    RingConcept,
    validate_pose,
)


_TEMPLATE_KIND = {
    "Cuboid": "cuboid",
    "Cuboid_Block": "cuboid",
    "Cuboidal_Door": "cuboid",
    "Cylinder": "cylinder",
    "Ring": "ring",
}
_CANONICAL_TEMPLATE = {
    "cuboid": "Cuboid",
    "cylinder": "Cylinder",
    "ring": "Ring",
}


@dataclass(frozen=True, slots=True)
class PartSummary:
    """供 UI/Agent 选择 part 的稳定摘要。"""

    index: int
    name: str
    template: str
    kind: str | None
    editable: bool


def _readonly_array(value: ArrayLike, *, shape_tail: tuple[int, ...], name: str):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数值数组") from error
    if result.ndim != len(shape_tail) or result.shape[1:] != shape_tail[1:]:
        raise ValueError(f"{name} shape 必须为 (N, {', '.join(map(str, shape_tail[1:]))})")
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是非空有限数值数组")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ReferenceMesh:
    """object frame 下的参考三角 mesh。"""

    vertices: ArrayLike
    faces: ArrayLike

    def __post_init__(self) -> None:
        vertices = _readonly_array(self.vertices, shape_tail=(0, 3), name="vertices")
        try:
            faces = np.asarray(self.faces, dtype=np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError("faces 必须是三角面索引") from error
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("faces 必须是非空 (N, 3) 三角面索引")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces 包含越界顶点索引")
        faces = np.array(faces, dtype=np.uint32, copy=True)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @classmethod
    def from_path(cls, path: str | Path) -> "ReferenceMesh":
        """通过 trimesh 读取常见 mesh 格式；至少支持旧工具使用的 OBJ。"""
        import trimesh

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到参考 mesh: {source}")
        loaded = trimesh.load_mesh(source, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = tuple(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"参考 mesh 场景为空: {source}")
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
            raise ValueError(f"无法读取三角 mesh: {source}")
        return cls(loaded.vertices, loaded.faces)

    @property
    def bounds_min_m(self) -> NDArray[np.float64]:
        return np.min(self.vertices, axis=0)

    @property
    def bounds_max_m(self) -> NDArray[np.float64]:
        return np.max(self.vertices, axis=0)


@dataclass(frozen=True, slots=True)
class MeshData:
    """Viser 可直接显示的三角 mesh 数组。"""

    vertices: NDArray[np.float64]
    faces: NDArray[np.uint32]


class ConceptEditorModel:
    """保留未知 JSON 字段的 Concept 编辑状态和撤销历史。"""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
        selected_index: int | None = None,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError("Concept JSON 根节点必须是对象")
        copied = deepcopy(dict(payload))
        items = copied.get("conceptualization")
        if not isinstance(items, list):
            raise ValueError("conceptualization 必须是列表")
        if not all(isinstance(item, dict) for item in items):
            raise ValueError("conceptualization 的每一项必须是对象")
        self._payload = copied
        self.source_path = (
            None if source_path is None else Path(source_path).expanduser().resolve()
        )
        self.selected_index = self._initial_selection(selected_index)
        self._undo: list[tuple[dict[str, Any], int | None]] = []
        self._redo: list[tuple[dict[str, Any], int | None]] = []
        self._validate_unique_names()
        for summary in self.parts:
            if summary.editable:
                self._concept_from_item(self._items[summary.index])

    @classmethod
    def empty(cls, *, category: str = "TunedObject") -> "ConceptEditorModel":
        return cls({"category": category, "conceptualization": []})

    @classmethod
    def from_json(cls, path: str | Path) -> "ConceptEditorModel":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到 Concept JSON: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Concept JSON 无效: {source}: {error}") from error
        return cls(payload, source_path=source)

    @property
    def _items(self) -> list[dict[str, Any]]:
        return self._payload["conceptualization"]

    def _initial_selection(self, requested: int | None) -> int | None:
        if requested is not None:
            if not 0 <= requested < len(self._items):
                raise IndexError("selected_index 超出范围")
            return requested
        for index, item in enumerate(self._items):
            if item.get("template") in _TEMPLATE_KIND:
                return index
        return 0 if self._items else None

    @property
    def parts(self) -> tuple[PartSummary, ...]:
        summaries: list[PartSummary] = []
        for index, item in enumerate(self._items):
            template = str(item.get("template", ""))
            name = item.get("component") or item.get("semantic") or f"{template.lower()}_{index}"
            kind = _TEMPLATE_KIND.get(template)
            summaries.append(
                PartSummary(index, str(name), template, kind, kind is not None)
            )
        return tuple(summaries)

    def _validate_unique_names(self) -> None:
        names = [part.name for part in self.parts]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"part 名称重复: {duplicates[0]}")

    def _snapshot(self) -> tuple[dict[str, Any], int | None]:
        return deepcopy(self._payload), self.selected_index

    def _restore(self, snapshot: tuple[dict[str, Any], int | None]) -> None:
        self._payload, self.selected_index = deepcopy(snapshot[0]), snapshot[1]

    def _commit(self, before: tuple[dict[str, Any], int | None]) -> None:
        self._validate_unique_names()
        if self.selected_index is not None:
            item = self._items[self.selected_index]
            if item.get("template") in _TEMPLATE_KIND:
                self._concept_from_item(item)
        self._undo.append(before)
        self._redo.clear()

    def select_part(self, index: int) -> None:
        if not 0 <= int(index) < len(self._items):
            raise IndexError("part index 超出范围")
        self.selected_index = int(index)

    def _current_item(self) -> dict[str, Any]:
        if self.selected_index is None:
            raise ValueError("当前没有选中的 part")
        return self._items[self.selected_index]

    def _current_editable_item(self) -> dict[str, Any]:
        item = self._current_item()
        if item.get("template") not in _TEMPLATE_KIND:
            raise ValueError(f"当前 part 类型 {item.get('template')!r} 只读")
        return item

    def add_part(self, kind: str, name: str) -> None:
        if kind not in _CANONICAL_TEMPLATE:
            raise ValueError(f"只支持 cuboid、cylinder、ring，实际为 {kind!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("part name 必须是非空字符串")
        if name in {part.name for part in self.parts}:
            raise ValueError(f"part 名称重复: {name}")
        before = self._snapshot()
        pose = np.eye(4).tolist()
        if kind == "cuboid":
            parameters = {"size": [0.03, 0.03, 0.03], "pose_matrix": pose}
        elif kind == "cylinder":
            parameters = {"size": [0.03, 0.03, 0.10], "pose_matrix": pose}
        else:
            parameters = {
                "size": [0.06, 0.05, 0.02],
                "exist_angle": [360.0],
                "pose_matrix": pose,
            }
        self._items.append(
            {
                "template": _CANONICAL_TEMPLATE[kind],
                "component": name,
                "parameters": parameters,
            }
        )
        self.selected_index = len(self._items) - 1
        self._commit(before)

    def delete_current_part(self) -> None:
        self._current_editable_item()
        before = self._snapshot()
        del self._items[self.selected_index]  # type: ignore[index]
        if not self._items:
            self.selected_index = None
        else:
            self.selected_index = min(self.selected_index or 0, len(self._items) - 1)
        self._commit(before)

    def rename_current_part(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("part name 必须是非空字符串")
        current = self._current_editable_item()
        if any(
            part.name == name and part.index != self.selected_index for part in self.parts
        ):
            raise ValueError(f"part 名称重复: {name}")
        before = self._snapshot()
        current["component"] = name
        self._commit(before)

    def set_pose(self, pose_in_object: ArrayLike) -> None:
        pose = validate_pose(pose_in_object, field_name="pose_in_object")
        before = self._snapshot()
        self._current_editable_item().setdefault("parameters", {})[
            "pose_matrix"
        ] = pose.tolist()
        self._commit(before)

    def set_geometry(self, **values: Any) -> None:
        item = self._current_editable_item()
        kind = _TEMPLATE_KIND[item["template"]]
        parameters = item.setdefault("parameters", {})
        before = self._snapshot()
        if kind == "cuboid":
            allowed = {"size_xyz_m"}
            if set(values) != allowed:
                raise ValueError("cuboid 必须且只能设置 size_xyz_m")
            parameters["size"] = list(values["size_xyz_m"])
        elif kind == "cylinder":
            allowed = {"radius_x_m", "radius_y_m", "height_m"}
            if set(values) != allowed:
                raise ValueError("cylinder 必须设置 radius_x_m、radius_y_m、height_m")
            parameters["size"] = [
                values["radius_x_m"],
                values["radius_y_m"],
                values["height_m"],
            ]
        else:
            allowed = {"outer_radius_m", "inner_radius_m", "height_m", "exist_angle_deg"}
            if set(values) != allowed:
                raise ValueError(
                    "ring 必须设置 outer_radius_m、inner_radius_m、height_m、exist_angle_deg"
                )
            parameters["size"] = [
                values["outer_radius_m"],
                values["inner_radius_m"],
                values["height_m"],
            ]
            parameters["exist_angle"] = [values["exist_angle_deg"]]
        try:
            self._commit(before)
        except Exception:
            self._restore(before)
            raise

    def current_concept(self) -> Concept:
        return self._concept_from_item(self._current_editable_item())

    def concept_at(self, index: int) -> Concept:
        """构造指定可编辑 part，不改变当前选择。"""
        if not 0 <= int(index) < len(self._items):
            raise IndexError("part index 超出范围")
        return self._concept_from_item(self._items[int(index)])

    @staticmethod
    def _concept_from_item(item: Mapping[str, Any]) -> Concept:
        template = item.get("template")
        kind = _TEMPLATE_KIND.get(str(template))
        if kind is None:
            raise ValueError(f"不支持的 Concept 类型: {template!r}")
        parameters = item.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("parameters 必须是对象")
        if "size" not in parameters:
            raise ValueError("parameters.size 是必填字段")
        size = parameters["size"]
        pose = parameters.get("pose_matrix", np.eye(4))
        if kind == "cuboid":
            return CuboidConcept(size, pose)
        if kind == "cylinder":
            array = np.asarray(size, dtype=float)
            if array.shape != (3,):
                raise ValueError("Cylinder size 必须为 [radius_x, radius_y, height]")
            return CylinderConcept(array[:2], array[2], pose)
        array = np.asarray(size, dtype=float)
        if array.shape != (3,):
            raise ValueError("Ring size 必须为 [outer_radius, inner_radius, height]")
        angle = parameters.get("exist_angle", [360.0])
        angle_array = np.asarray(angle, dtype=float)
        if angle_array.shape != (1,):
            raise ValueError("Ring exist_angle 必须是单元素数组")
        return RingConcept(
            array[1],
            array[0],
            array[2],
            [0.0, 0.0, 1.0],
            pose,
            float(np.deg2rad(angle_array[0])),
        )

    def auto_initialize(self, mesh: ReferenceMesh) -> None:
        """用 object-frame AABB 给当前 part 一个简单、确定性的初值。"""
        item = self._current_editable_item()
        kind = _TEMPLATE_KIND[item["template"]]
        minimum = mesh.bounds_min_m
        maximum = mesh.bounds_max_m
        extent = maximum - minimum
        if np.any(extent <= 0.0):
            raise ValueError("mesh 包围盒退化，无法初始化 Concept")
        pose = np.eye(4)
        pose[:3, 3] = (minimum + maximum) / 2.0
        before = self._snapshot()
        parameters = item.setdefault("parameters", {})
        parameters["pose_matrix"] = pose.tolist()
        if kind == "cuboid":
            parameters["size"] = extent.tolist()
        elif kind == "cylinder":
            parameters["size"] = [extent[0] / 2.0, extent[1] / 2.0, extent[2]]
        else:
            outer = max(extent[0], extent[1]) / 2.0
            parameters["size"] = [outer, outer * 0.8, extent[2]]
            parameters.setdefault("exist_angle", [360.0])
        self._commit(before)

    def undo(self) -> None:
        if not self._undo:
            raise ValueError("没有可撤销的操作")
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())

    def redo(self) -> None:
        if not self._redo:
            raise ValueError("没有可重做的操作")
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())

    def to_payload(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        output = Path(path).expanduser().resolve()
        if output.exists() and not overwrite:
            raise FileExistsError(f"输出已存在，拒绝覆盖: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return output

    def apply(self, edit: Mapping[str, Any]) -> None:
        """供 UI 和未来 Agent 共用的最小结构化编辑入口。"""
        action = edit.get("action")
        if action == "select_part":
            self.select_part(int(edit["index"]))
        elif action == "add_part":
            self.add_part(str(edit["kind"]), str(edit["name"]))
        elif action == "delete_part":
            self.delete_current_part()
        elif action == "rename_part":
            self.rename_current_part(str(edit["name"]))
        elif action == "set_pose":
            self.set_pose(edit["pose_in_object"])
        elif action == "set_geometry":
            values = edit.get("values")
            if not isinstance(values, Mapping):
                raise ValueError("set_geometry.values 必须是对象")
            self.set_geometry(**dict(values))
        else:
            raise ValueError(f"未知编辑动作: {action!r}")


def _apply_pose(vertices: NDArray[np.float64], pose: NDArray[np.float64]) -> NDArray[np.float64]:
    return (pose[:3, :3] @ vertices.T).T + pose[:3, 3]


def _cuboid_mesh(concept: CuboidConcept) -> MeshData:
    half = concept.size_xyz_m / 2.0
    vertices = np.array(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]
    )
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.uint32,
    )
    return MeshData(_apply_pose(vertices, concept.pose_in_object), faces)


def _cylinder_mesh(concept: CylinderConcept, segments: int = 64) -> MeshData:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    x = concept.radius_xy_m[0] * np.cos(angles)
    y = concept.radius_xy_m[1] * np.sin(angles)
    half = concept.height_m / 2.0
    bottom = np.column_stack([x, y, np.full(segments, -half)])
    top = np.column_stack([x, y, np.full(segments, half)])
    vertices = np.vstack([bottom, top, [0.0, 0.0, -half], [0.0, 0.0, half]])
    faces: list[list[int]] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.extend(
            [
                [index, next_index, segments + next_index],
                [index, segments + next_index, segments + index],
                [2 * segments, next_index, index],
                [2 * segments + 1, segments + index, segments + next_index],
            ]
        )
    return MeshData(
        _apply_pose(vertices, concept.pose_in_object),
        np.asarray(faces, dtype=np.uint32),
    )


def _ring_mesh(concept: RingConcept, segments: int = 64) -> MeshData:
    full = np.isclose(concept.angle_span_rad, 2.0 * np.pi)
    fraction = concept.angle_span_rad / (2.0 * np.pi)
    count = segments if full else max(2, int(np.ceil(segments * fraction)) + 1)
    angles = np.linspace(
        0.0,
        concept.angle_span_rad,
        count,
        endpoint=not full,
    )
    half = concept.height_m / 2.0
    groups = []
    for radius, z in (
        (concept.outer_radius_m, -half),
        (concept.outer_radius_m, half),
        (concept.inner_radius_m, -half),
        (concept.inner_radius_m, half),
    ):
        groups.append(
            np.column_stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.full(count, z)]
            )
        )
    vertices = np.vstack(groups)
    faces: list[list[int]] = []
    pairs = [(index, (index + 1) % count) for index in range(count if full else count - 1)]
    for index, next_index in pairs:
        ob, ot, ib, it = index, count + index, 2 * count + index, 3 * count + index
        nob, not_, nib, nit = (
            next_index,
            count + next_index,
            2 * count + next_index,
            3 * count + next_index,
        )
        faces.extend(
            [
                [ob, nob, not_], [ob, not_, ot],
                [ib, it, nit], [ib, nit, nib],
                [ot, not_, nit], [ot, nit, it],
                [ob, ib, nib], [ob, nib, nob],
            ]
        )
    if not full:
        for index, reverse in ((0, False), (count - 1, True)):
            ob, ot, ib, it = index, count + index, 2 * count + index, 3 * count + index
            caps = [[ob, ot, it], [ob, it, ib]]
            faces.extend([face[::-1] for face in caps] if reverse else caps)
    axis_rotation, _ = Rotation.align_vectors(
        np.asarray([concept.axis]), np.asarray([[0.0, 0.0, 1.0]])
    )
    vertices = axis_rotation.apply(vertices)
    return MeshData(
        _apply_pose(vertices, concept.pose_in_object),
        np.asarray(faces, dtype=np.uint32),
    )


def concept_mesh(concept: Concept) -> MeshData:
    """把纯几何 Concept 转成 object frame 下的预览 mesh。"""
    if isinstance(concept, CuboidConcept):
        return _cuboid_mesh(concept)
    if isinstance(concept, CylinderConcept):
        return _cylinder_mesh(concept)
    if isinstance(concept, RingConcept):
        return _ring_mesh(concept)
    raise TypeError(f"编辑器不支持预览 {type(concept).__name__}")


__all__ = [
    "ConceptEditorModel",
    "MeshData",
    "PartSummary",
    "ReferenceMesh",
    "concept_mesh",
]
