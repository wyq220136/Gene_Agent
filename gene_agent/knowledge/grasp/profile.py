"""Task-owned grasp profile 的严格 JSON loader。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .model import ClosingWidthSpec, GraspProfile, PosePostprocess, SortRule
from .sampling import expand_sample_spec


SUPPORTED_GENERATORS = frozenset(
    {"ring", "cylinder_axial", "cylinder", "cuboidal_block", "trifold_handle"}
)
_PROFILE_FIELDS = frozenset(
    {
        "generator",
        "part",
        "parameters",
        "closing_width",
        "postprocess",
        "sorting",
        "status",
        "acceptance_baseline",
        "status_reason",
    }
)
_METADATA_FIELDS = ("status", "acceptance_baseline", "status_reason")


class _ParameterRule:
    """一个 generator 参数的值类型与可选枚举；只供本 loader 内部使用。"""

    def __init__(self, kind: str, enum_values: frozenset[str] = frozenset()) -> None:
        self.kind = kind
        self.enum_values = enum_values


_NUMBER = _ParameterRule("number")
_BOOLEAN = _ParameterRule("boolean")
_SURFACE = _ParameterRule("enum", frozenset({"top", "bottom"}))
_FACE = _ParameterRule(
    "enum", frozenset({"front", "back", "left", "right", "top", "bottom"})
)

# 这是五个首批 generator 的显式配置契约。字段保持小而具体：
# 必填项来自当前 task profile；可选项只覆盖相应几何公式确实消费的参数。
_GENERATOR_PARAMETER_SCHEMAS: dict[
    str, tuple[dict[str, _ParameterRule], frozenset[str]]
] = {
    "ring": (
        {
            "surface": _SURFACE,
            "azimuth_deg": _NUMBER,
            "tip_x_deg": _NUMBER,
            "tip_y_deg": _NUMBER,
            "depth_m": _NUMBER,
            "vertical_offset_m": _NUMBER,
        },
        frozenset({"surface", "azimuth_deg", "tip_y_deg"}),
    ),
    "cylinder_axial": (
        {
            "surface": _SURFACE,
            "depth_m": _NUMBER,
            "azimuth_deg": _NUMBER,
            "tip_x_deg": _NUMBER,
        },
        frozenset({"surface", "depth_m", "azimuth_deg", "tip_x_deg"}),
    ),
    "cylinder": (
        {
            "cylinder_rotation_deg": _NUMBER,
            "tip_x_deg": _NUMBER,
            "vertical_offset_m": _NUMBER,
            "depth_m": _NUMBER,
        },
        frozenset({"cylinder_rotation_deg", "tip_x_deg"}),
    ),
    "cuboidal_block": (
        {
            "face": _FACE,
            "rotate_z_90": _BOOLEAN,
            "surface_parallel_offset_m": _NUMBER,
            "depth_m": _NUMBER,
            "tip_x_deg": _NUMBER,
        },
        frozenset(),
    ),
    "trifold_handle": (
        {
            "vertical_offset_m": _NUMBER,
            "tip_x_deg": _NUMBER,
            "tip_y_deg": _NUMBER,
            "depth_m": _NUMBER,
        },
        frozenset({"vertical_offset_m", "tip_x_deg", "tip_y_deg", "depth_m"}),
    ),
}
_FORBIDDEN_FRAGMENTS = frozenset(
    {
        "pregrasp",
        "approach",
        "lift",
        "speed",
        "velocity",
        "acceleration",
        "force",
        "collision",
        "planner",
        "robot",
        "sdk",
        "retry",
        "wait",
        "motion",
        "skill",
    }
)


def _unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], *, path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        key = sorted(unknown)[0]
        raise ValueError(f"{path}.{key} 是未知字段")


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} 必须是 JSON 对象")
    return value


def _require_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} 必须是非空字符串")
    return value


def _finite_float(value: Any, *, path: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "有限非负数" if nonnegative else "有限数值"
        raise ValueError(f"{path} 必须是{qualifier}")
    return result


def _scan_forbidden_fields(value: Any, *, path: str) -> None:
    """递归阻止 Skill、motion 与 collision 配置伪装成采样参数。"""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} 的字段名必须是字符串")
            child_path = f"{path}.{key}" if path else key
            fragments = set(key.lower().replace("-", "_").split("_"))
            if fragments & _FORBIDDEN_FRAGMENTS:
                raise ValueError(f"{child_path} 不属于 Knowledge grasp profile")
            _scan_forbidden_fields(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_fields(child, path=f"{path}[{index}]")


def _resolved_parameter_name(name: str, *, path: str) -> str:
    if name.endswith("_deg"):
        return f"{name[:-4]}_rad"
    if name.endswith("_rad"):
        raise ValueError(f"{path} 在 JSON 中必须使用 _deg，内部才转换为 _rad")
    return name


def _sample_value_path(sample_spec: Any, source_path: str, index: int) -> str:
    if isinstance(sample_spec, Mapping) and "values" in sample_spec:
        return f"{source_path}.values[{index}]"
    return f"{source_path}[{index}]"


def _expand_boolean_values(sample_spec: Any, *, path: str) -> tuple[bool, ...]:
    spec = _require_mapping(sample_spec, path=path)
    _unknown_fields(spec, frozenset({"values"}), path=path)
    values = spec.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}.values 必须是非空列表")
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            raise ValueError(f"{path}.values[{index}] 必须是布尔值")
    return tuple(values)


def _parse_parameters(
    value: Any,
    *,
    generator: str,
    path: str,
) -> dict[str, tuple[float | str | bool, ...]]:
    parameters = _require_mapping(value, path=path)
    rules, required = _GENERATOR_PARAMETER_SCHEMAS[generator]
    unknown = set(parameters) - set(rules)
    if unknown:
        key = sorted(unknown)[0]
        raise ValueError(f"{path}.{key} 不是 {generator} 支持的参数")
    missing = required - set(parameters)
    if missing:
        key = sorted(missing)[0]
        raise ValueError(f"{path}.{key} 是 {generator} 的必填参数")
    if generator == "ring":
        offsets = {"depth_m", "vertical_offset_m"} & set(parameters)
        if len(offsets) != 1:
            raise ValueError(f"{path} 的 ring 必须且只能指定 vertical_offset_m（旧 depth_m 可读取）")

    resolved: dict[str, tuple[float | str | bool, ...]] = {}
    for source_name, sample_spec in parameters.items():
        source_path = f"{path}.{source_name}"
        if not isinstance(source_name, str) or not source_name:
            raise ValueError(f"{path} 的参数名必须是非空字符串")
        rule = rules[source_name]
        if rule.kind == "boolean":
            values: tuple[float | str | bool, ...] = _expand_boolean_values(
                sample_spec, path=source_path
            )
        elif isinstance(sample_spec, list):
            # 当前 task 数据用裸字符串列表表达 surface；
            # face 也使用同一枚举形式。
            if not sample_spec:
                raise ValueError(f"{source_path} 不能为空")
            if not all(isinstance(item, str) and item for item in sample_spec):
                raise ValueError(f"{source_path} 的列表简写只支持非空字符串")
            values = tuple(sample_spec)
        else:
            values = expand_sample_spec(sample_spec, path=source_path)

        if rule.kind == "number":
            for index, item in enumerate(values):
                _finite_float(
                    item,
                    path=_sample_value_path(sample_spec, source_path, index),
                )
        elif rule.kind == "enum":
            for index, item in enumerate(values):
                item_path = _sample_value_path(sample_spec, source_path, index)
                if not isinstance(item, str) or item not in rule.enum_values:
                    choices = ", ".join(sorted(rule.enum_values))
                    raise ValueError(f"{item_path} 只支持 {choices}")

        target_name = _resolved_parameter_name(source_name, path=source_path)
        if target_name in resolved:
            raise ValueError(f"{source_path} 解析后与已有参数 {target_name} 重名")
        if source_name.endswith("_deg"):
            converted: list[float] = []
            for index, item in enumerate(values):
                degrees = _finite_float(item, path=f"{source_path}[{index}]")
                converted.append(math.radians(degrees))
            resolved[target_name] = tuple(converted)
        else:
            resolved[target_name] = values
    return resolved


def _parse_closing_width(value: Any, *, path: str) -> ClosingWidthSpec:
    spec = _require_mapping(value, path=path)
    mode = spec.get("mode")
    if mode == "geometry":
        _unknown_fields(spec, frozenset({"mode", "clearance_m"}), path=path)
        if "clearance_m" not in spec:
            raise ValueError(f"{path}.clearance_m 是必填字段")
        return ClosingWidthSpec(
            mode="geometry",
            clearance_m=_finite_float(
                spec["clearance_m"], path=f"{path}.clearance_m", nonnegative=True
            ),
        )
    if mode == "override":
        _unknown_fields(spec, frozenset({"mode", "width_m"}), path=path)
        if "width_m" not in spec:
            raise ValueError(f"{path}.width_m 是必填字段")
        return ClosingWidthSpec(
            mode="override",
            width_m=_finite_float(spec["width_m"], path=f"{path}.width_m", nonnegative=True),
        )
    raise ValueError(f"{path}.mode 只支持 geometry 或 override")


def _vector3(value: Any, *, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{path} 必须是三个有限数值")
    values = tuple(
        _finite_float(item, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    return values  # type: ignore[return-value]


def _parse_postprocess(value: Any, *, path: str) -> tuple[PosePostprocess, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} 必须是列表")
    result: list[PosePostprocess] = []
    for index, raw_step in enumerate(value):
        step_path = f"{path}[{index}]"
        step = _require_mapping(raw_step, path=step_path)
        step_type = step.get("type")
        if step_type == "translate":
            allowed = frozenset({"type", "frame", "xyz_m"})
            required_value = "xyz_m"
        elif step_type == "rotate":
            allowed = frozenset({"type", "frame", "rpy_deg"})
            required_value = "rpy_deg"
        else:
            raise ValueError(f"{step_path}.type 只支持 translate 或 rotate")
        _unknown_fields(step, allowed, path=step_path)
        if "frame" not in step:
            raise ValueError(f"{step_path}.frame 是必填字段")
        frame = step["frame"]
        if frame not in {"object", "tcp"}:
            raise ValueError(f"{step_path}.frame 只支持 object 或 tcp")
        if required_value not in step:
            raise ValueError(f"{step_path}.{required_value} 是必填字段")
        vector = _vector3(step[required_value], path=f"{step_path}.{required_value}")
        if step_type == "translate":
            result.append(PosePostprocess(type="translate", frame=frame, xyz_m=vector))
        else:
            result.append(
                PosePostprocess(
                    type="rotate",
                    frame=frame,
                    rpy_rad=tuple(math.radians(item) for item in vector),
                )
            )
    return tuple(result)


def _parse_sorting(
    value: Any,
    *,
    numeric_parameter_names: frozenset[str],
    path: str,
) -> tuple[SortRule, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} 必须是列表")
    rules: list[SortRule] = []
    for index, raw_rule in enumerate(value):
        rule_path = f"{path}[{index}]"
        rule = _require_mapping(raw_rule, path=rule_path)
        order = rule.get("order")
        if order not in {"ascending", "descending"}:
            raise ValueError(f"{rule_path}.order 只支持 ascending 或 descending")
        if "field" in rule:
            _unknown_fields(
                rule, frozenset({"field", "transform", "order"}), path=rule_path
            )
            field_name = _require_string(rule["field"], path=f"{rule_path}.field")
            transform = rule.get("transform")
            if transform != "absolute":
                raise ValueError(f"{rule_path}.transform 当前只支持 absolute")
            resolved_field = _resolved_parameter_name(
                field_name, path=f"{rule_path}.field"
            )
            if resolved_field not in numeric_parameter_names:
                raise ValueError(
                    f"{rule_path}.field 必须引用 parameters 中已解析的数值字段"
                )
            rules.append(
                SortRule(
                    order=order,
                    field=resolved_field,
                    transform=transform,
                )
            )
        elif "metric" in rule:
            _unknown_fields(
                rule, frozenset({"metric", "frame", "order"}), path=rule_path
            )
            if rule.get("metric") != "top_down":
                raise ValueError(f"{rule_path}.metric 当前只支持 top_down")
            if rule.get("frame") != "world":
                raise ValueError(f"{rule_path}.frame 当前只支持 world")
            rules.append(
                SortRule(order=order, metric="top_down", frame="world")
            )
        else:
            raise ValueError(f"{rule_path} 必须显式包含 field 或 metric")
    return tuple(rules)


def _parse_profile(name: str, value: Any) -> GraspProfile:
    path = f"profiles.{name}"
    profile = _require_mapping(value, path=path)
    _unknown_fields(profile, _PROFILE_FIELDS, path=path)
    if "generator" not in profile:
        raise ValueError(f"{path}.generator 是必填字段")
    generator = _require_string(profile["generator"], path=f"{path}.generator")
    if generator not in SUPPORTED_GENERATORS:
        raise ValueError(f"{path}.generator 不支持 {generator!r}")
    part = profile.get("part")
    if part is not None:
        part = _require_string(part, path=f"{path}.part")
    if "closing_width" not in profile:
        raise ValueError(f"{path}.closing_width 是必填字段")

    metadata: dict[str, Any] = {}
    for key in _METADATA_FIELDS:
        if key in profile:
            metadata[key] = profile[key]
    if "status" in metadata:
        _require_string(metadata["status"], path=f"{path}.status")
    if "status_reason" in metadata:
        _require_string(metadata["status_reason"], path=f"{path}.status_reason")
    if "acceptance_baseline" in metadata and not isinstance(
        metadata["acceptance_baseline"], bool
    ):
        raise ValueError(f"{path}.acceptance_baseline 必须是布尔值")

    parameters = _parse_parameters(
        profile.get("parameters", {}),
        generator=generator,
        path=f"{path}.parameters",
    )
    parameter_rules, _ = _GENERATOR_PARAMETER_SCHEMAS[generator]
    numeric_parameter_names = frozenset(
        _resolved_parameter_name(name, path=f"{path}.parameters.{name}")
        for name in profile.get("parameters", {})
        if parameter_rules[name].kind == "number"
    )
    sorting = _parse_sorting(
        profile.get("sorting", []),
        numeric_parameter_names=numeric_parameter_names,
        path=f"{path}.sorting",
    )

    return GraspProfile(
        name=name,
        generator=generator,
        part_name=part,
        parameters=parameters,
        closing_width=_parse_closing_width(
            profile["closing_width"], path=f"{path}.closing_width"
        ),
        postprocess=_parse_postprocess(
            profile.get("postprocess", []), path=f"{path}.postprocess"
        ),
        sorting=sorting,
        metadata=metadata,
    )


def parse_grasp_profiles(payload: Any) -> Mapping[str, GraspProfile]:
    """把已读取的 task JSON 严格解析为不可变 resolved profile 映射。"""
    root = _require_mapping(payload, path="root")
    _scan_forbidden_fields(root, path="")
    _unknown_fields(root, frozenset({"profiles"}), path="root")
    if "profiles" not in root:
        raise ValueError("root.profiles 是必填字段")
    profiles = _require_mapping(root["profiles"], path="profiles")
    result: dict[str, GraspProfile] = {}
    for name, profile in profiles.items():
        _require_string(name, path="profiles 的 profile 名")
        result[name] = _parse_profile(name, profile)
    return MappingProxyType(result)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 字段 {key!r} 重复")
        result[key] = value
    return result


def load_grasp_profiles(path: str | Path) -> Mapping[str, GraspProfile]:
    """从 UTF-8 JSON 文件加载全部 profile，并拒绝重复键。"""
    profile_path = Path(path)
    try:
        payload = json.loads(
            profile_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except OSError as error:
        raise ValueError(f"无法读取 grasp profile 文件 {profile_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"grasp profile JSON 无效：line {error.lineno}, column {error.colno}"
        ) from error
    return parse_grasp_profiles(payload)
