"""把仓库现有结构化 JSON 显式映射为 typed Concept。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .basic import CuboidConcept, CylinderConcept, RingConcept
from .container import ContainerConcept, ContainerWall
from .transforms import validate_pose
from .trifold import TrifoldConcept


Concept: TypeAlias = (
    CuboidConcept | CylinderConcept | RingConcept | TrifoldConcept | ContainerConcept
)


class ConceptLoadError(ValueError):
    """携带来源文件与 JSON 字段路径的 Concept 加载错误。"""


@dataclass(frozen=True, slots=True)
class ConceptPart:
    """一个具名物体 part；``concept.pose_in_object`` 是 ``T_object_part``。"""

    part_name: str
    semantic: str | None
    concept: Concept
    source_path: Path


def load_concepts(path: str | Path) -> tuple[ConceptPart, ...]:
    """加载当前仓库实际使用的五种 legacy Concept 模板。"""
    source, payload, items = _read_concept_file(path)

    parts: list[ConceptPart] = []
    used_names: set[str] = set()
    for index, item in enumerate(items):
        part = _load_part(source, payload, items, index, item)
        part_name = part.part_name
        if part_name in used_names:
            raise _error(
                source,
                _part_label_path(item, index),
                f"part 名称重复：{part_name}",
            )
        parts.append(part)
        used_names.add(part_name)
    return tuple(parts)


def load_concept(
    path: str | Path,
    *,
    part_name: str | None = None,
    semantic: str | None = None,
) -> ConceptPart:
    """按唯一 part 名或 semantic 定位后，只解析该目标 item。

    该入口用于一个文件同时包含可表达与暂不支持几何的 legacy 数据；未选中的
    ``parameters`` 不会进入 typed Concept 映射。
    """
    source, payload, items = _read_concept_file(path)
    if (part_name is None) == (semantic is None):
        raise _error(source, "$.conceptualization", "必须且只能提供 part_name 或 semantic")
    selector = part_name if part_name is not None else semantic
    selector_kind = "part_name" if part_name is not None else "semantic"
    if not isinstance(selector, str) or not selector:
        raise _error(source, "$.conceptualization", f"{selector_kind} 必须是非空字符串")

    matches: list[tuple[int, dict]] = []
    for index, item in enumerate(items):
        item_path = f"$.conceptualization[{index}]"
        template, item_semantic, component = _item_identity(item, source, item_path)
        resolved_name = _resolved_part_name(
            source,
            payload,
            items,
            index,
            template,
            item_semantic,
            component,
        )
        matched = item_semantic == selector if semantic is not None else resolved_name == selector
        if matched:
            matches.append((index, item))
            if len(matches) > 1:
                field = "semantic" if semantic is not None else (
                    "component" if component is not None else "semantic"
                )
                raise _error(source, f"{item_path}.{field}", f"目标 {selector_kind} 重复：{selector}")

    if not matches:
        raise _error(source, "$.conceptualization", f"找不到 {selector_kind}={selector} 的目标")
    index, item = matches[0]
    return _load_part(source, payload, items, index, item)


def _read_concept_file(path: str | Path) -> tuple[Path, dict, list]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 Concept 文件: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _error(source, "$", f"必须是有效 UTF-8 JSON：{error}") from error
    if not isinstance(payload, dict):
        raise _error(source, "$", "根节点必须是对象")
    items = payload.get("conceptualization")
    if not isinstance(items, list):
        raise _error(source, "$.conceptualization", "必须是列表")
    return source, payload, items


def _load_part(source: Path, payload: dict, items: list, index: int, item) -> ConceptPart:
    item_path = f"$.conceptualization[{index}]"
    template, semantic, component = _item_identity(item, source, item_path)
    parameters = _required(item, "parameters", source, item_path)
    if not isinstance(parameters, dict):
        raise _error(source, f"{item_path}.parameters", "必须是对象")
    _validate_unit(parameters, source, f"{item_path}.parameters")
    part_name = _resolved_part_name(
        source, payload, items, index, template, semantic, component
    )
    concept = _load_item(template, parameters, source, f"{item_path}.parameters")
    return ConceptPart(part_name, semantic, concept, source)


def _item_identity(item, source: Path, item_path: str) -> tuple[str, str | None, str | None]:
    if not isinstance(item, dict):
        raise _error(source, item_path, "必须是对象")
    template = _required(item, "template", source, item_path)
    if not isinstance(template, str):
        raise _error(source, f"{item_path}.template", "必须是字符串")
    semantic = _optional_label(item, "semantic", source, item_path)
    component = _optional_label(item, "component", source, item_path)
    return template, semantic, component


def _resolved_part_name(
    source: Path,
    payload: dict,
    items: list,
    index: int,
    template: str,
    semantic: str | None,
    component: str | None,
) -> str:
    if component is not None:
        return component
    if semantic is not None:
        return semantic
    # 这是 Bowl legacy schema 与现有 task profile 之间唯一需要的显式主键映射。
    # 不把所有无标签 Ring 都称为 bowl_rim，避免误标 Plate 等同形模板。
    is_bowl_source = (
        payload.get("category") == "Bowl"
        or source.name == "bowl_concept.json"
        or any(part.lower() == "bowl" for part in source.parts)
    )
    if len(items) == 1 and index == 0 and template == "Ring" and is_bowl_source:
        return "bowl_rim"
    return f"{template.lower()}_{index}"


def _part_label_path(item, index: int) -> str:
    field = "component" if isinstance(item, dict) and item.get("component") is not None else "semantic"
    return f"$.conceptualization[{index}].{field}"


def _load_item(template: str, values: dict, source: Path, path: str) -> Concept:
    if template == "Ring":
        size = _positive_vector3(_required(values, "size", source, path), source, f"{path}.size")
        if size[0] <= size[1]:
            raise _error(source, f"{path}.size", "外半径必须大于内半径")
        angle = _vector(_required(values, "exist_angle", source, path), 1, source, f"{path}.exist_angle")
        if not 0.0 <= angle[0] <= 360.0:
            raise _error(source, f"{path}.exist_angle", "弧长角必须位于 0 到 360 度之间")
        pose = _pose(values, source, path)
        return _construct(
            RingConcept,
            source,
            f"{path}.size",
            size[1],
            size[0],
            size[2],
            [0.0, 0.0, 1.0],
            pose,
            float(np.deg2rad(angle[0])),
        )

    if template == "Cylinder":
        size = _positive_vector3(_required(values, "size", source, path), source, f"{path}.size")
        pose = _pose(values, source, path)
        return _construct(
            CylinderConcept,
            source,
            f"{path}.size",
            size[:2],
            size[2],
            pose,
        )

    if template in {"Cuboid", "Cuboid_Block", "Cuboidal_Door"}:
        size = _positive_vector3(_required(values, "size", source, path), source, f"{path}.size")
        return _construct(
            CuboidConcept, source, path, size, _pose(values, source, path)
        )

    if template == "Trifold_Handle":
        connector_size = _positive_vector3(
            _required(values, "mounting_size", source, path), source, f"{path}.mounting_size"
        )
        grip_size = _positive_vector3(
            _required(values, "grip_size", source, path), source, f"{path}.grip_size"
        )
        separation = _vector(
            _required(values, "mounting_seperation", source, path),
            1,
            source,
            f"{path}.mounting_seperation",
        )[0]
        if separation <= 0.0:
            raise _error(source, f"{path}.mounting_seperation", "必须是有限正数，单位为米")
        return _construct(
            TrifoldConcept,
            source,
            path,
            grip_size,
            connector_size,
            separation,
            _pose(values, source, path),
        )

    if template == "Cuboid_Container":
        return _load_container(values, source, path)

    raise _error(source, path.removesuffix(".parameters") + ".template", f"未知 concept type：{template}")


def _load_container(values: dict, source: Path, path: str) -> ContainerConcept:
    outer = _positive_vector3(
        _required(values, "outer_size", source, path), source, f"{path}.outer_size"
    )
    thickness = _positive_scalar(
        _required(values, "wall_thickness", source, path), source, f"{path}.wall_thickness"
    )
    opening = _required(values, "opening_axis", source, path)
    if opening != "+Y":
        raise _error(source, f"{path}.opening_axis", "当前 Cuboid_Container 只支持 +Y")
    inner = np.array([outer[0] - 2.0 * thickness, outer[2] - 2.0 * thickness])
    if np.any(inner <= 0.0):
        raise _error(source, f"{path}.wall_thickness", "导致容器内轮廓退化")

    wall_values = _required(values, "walls", source, path)
    if not isinstance(wall_values, list):
        raise _error(source, f"{path}.walls", "必须是列表")
    expected_names = {"back_wall", "bottom_wall", "top_wall", "right_wall", "left_wall"}
    walls: list[ContainerWall] = []
    names: set[str] = set()
    for index, spec in enumerate(wall_values):
        wall_path = f"{path}.walls[{index}]"
        if not isinstance(spec, dict):
            raise _error(source, wall_path, "必须是对象")
        name = _required(spec, "name", source, wall_path)
        if name not in expected_names:
            raise _error(source, f"{wall_path}.name", f"未知 container component：{name}")
        if name in names:
            raise _error(source, f"{wall_path}.name", f"container component 重复：{name}")
        size = _positive_vector3(
            _required(spec, "size", source, wall_path), source, f"{wall_path}.size"
        )
        pose_value = _required(spec, "pose_matrix", source, wall_path)
        pose = _validated_pose(pose_value, source, f"{wall_path}.pose_matrix")
        walls.append(_construct(ContainerWall, source, wall_path, name, size, pose))
        names.add(name)
    if names != expected_names:
        missing = ", ".join(sorted(expected_names - names))
        raise _error(source, f"{path}.walls", f"缺少 container component：{missing}")

    bounds = _finite_vector3(
        _required(values, "bounds_min", source, path), source, f"{path}.bounds_min"
    )
    return _construct(
        ContainerConcept,
        source,
        path,
        [outer[0], outer[2]],
        inner,
        outer[1],
        [0.0, 1.0, 0.0],
        _pose(values, source, path),
        bounds,
        tuple(walls),
    )


def _required(mapping: dict, key: str, source: Path, path: str):
    if key not in mapping:
        raise _error(source, f"{path}.{key}", "缺少必填字段")
    return mapping[key]


def _optional_label(mapping: dict, key: str, source: Path, path: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise _error(source, f"{path}.{key}", "必须是非空字符串")
    return value


def _validate_unit(values: dict, source: Path, path: str) -> None:
    # 现有 legacy schema 不带单位字段，其尺寸约定为米；若显式声明则只接受米。
    if "unit" in values and values["unit"] != "m":
        raise _error(source, f"{path}.unit", "只支持单位 m")


def _pose(values: dict, source: Path, path: str):
    value = _required(values, "pose_matrix", source, path)
    return _validated_pose(value, source, f"{path}.pose_matrix")


def _validated_pose(value, source: Path, path: str):
    try:
        return validate_pose(value, field_name=path)
    except ValueError as error:
        raise ConceptLoadError(f"{source}: {error}") from error


def _vector(value, length: int, source: Path, path: str):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise _error(source, path, f"必须是长度为 {length} 的有限数值数组") from error
    if result.shape != (length,) or not np.isfinite(result).all():
        raise _error(source, path, f"必须是长度为 {length} 的有限数值数组")
    return result


def _positive_vector3(value, source: Path, path: str):
    result = _vector(value, 3, source, path)
    if np.any(result <= 0.0):
        raise _error(source, path, "必须是三个有限正数，单位为米")
    return result


def _finite_vector3(value, source: Path, path: str):
    return _vector(value, 3, source, path)


def _positive_scalar(value, source: Path, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _error(source, path, "必须是有限正数，单位为米") from error
    if not np.isfinite(result) or result <= 0.0:
        raise _error(source, path, "必须是有限正数，单位为米")
    return result


def _construct(constructor, source: Path, path: str, *args):
    """在 JSON 映射边界统一补齐 typed constructor 的来源与字段路径。"""
    try:
        return constructor(*args)
    except ConceptLoadError:
        raise
    except ValueError as error:
        raise _error(source, path, str(error)) from error


def _error(source: Path, path: str, detail: str) -> ConceptLoadError:
    return ConceptLoadError(f"{source}: {path}: {detail}")


__all__ = [
    "Concept",
    "ConceptLoadError",
    "ConceptPart",
    "load_concept",
    "load_concepts",
]
