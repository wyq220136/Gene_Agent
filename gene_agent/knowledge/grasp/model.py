"""抓取候选、诊断与已解析 profile 的小型不可变数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gene_agent.knowledge.geometry import validate_pose


_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _freeze_json(value: Any, *, field_path: str) -> Any:
    """校验 JSON 值并保存独立只读副本，拒绝 NaN/Infinity 和非字符串键。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_path} 必须是有限 JSON 数值")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_path} 的 JSON 对象键必须是字符串")
            frozen[key] = _freeze_json(child, field_path=f"{field_path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(child, field_path=f"{field_path}[{index}]")
            for index, child in enumerate(value)
        )
    raise ValueError(f"{field_path} 必须可写为 JSON")


def _thaw_json(value: Any) -> Any:
    """把内部只读 JSON 树显式转换为标准 dict/list，供产物直接序列化。"""
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _require_nonempty_string(value: Any, *, field_path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_path} 必须是非空字符串")
    return value


@dataclass(frozen=True)
class ClosingWidthSpec:
    """闭合宽度规则；所有长度均以米表示。"""

    mode: str
    clearance_m: float | None = None
    width_m: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "geometry":
            if self.width_m is not None:
                raise ValueError("closing_width.width_m 不适用于 geometry 模式")
            clearance = 0.0 if self.clearance_m is None else self.clearance_m
            if isinstance(clearance, bool) or not isinstance(clearance, (int, float)):
                raise ValueError("closing_width.clearance_m 必须是有限非负数")
            clearance = float(clearance)
            if not math.isfinite(clearance) or clearance < 0.0:
                raise ValueError("closing_width.clearance_m 必须是有限非负数")
            object.__setattr__(self, "clearance_m", clearance)
            return
        if self.mode == "override":
            if self.clearance_m is not None:
                raise ValueError("closing_width.clearance_m 不适用于 override 模式")
            if isinstance(self.width_m, bool) or not isinstance(self.width_m, (int, float)):
                raise ValueError("closing_width.width_m 必须是有限非负数")
            width = float(self.width_m)
            if not math.isfinite(width) or width < 0.0:
                raise ValueError("closing_width.width_m 必须是有限非负数")
            object.__setattr__(self, "width_m", width)
            return
        raise ValueError("closing_width.mode 只支持 geometry 或 override")

    def to_dict(self) -> dict[str, Any]:
        if self.mode == "geometry":
            return {"mode": self.mode, "clearance_m": self.clearance_m}
        return {"mode": self.mode, "width_m": self.width_m}


@dataclass(frozen=True)
class PosePostprocess:
    """已解析的 TCP pose 后处理；平移为米，旋转为弧度。"""

    type: str
    frame: str
    xyz_m: tuple[float, float, float] | None = None
    rpy_rad: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.type not in {"translate", "rotate"}:
            raise ValueError("postprocess.type 只支持 translate 或 rotate")
        if self.frame not in {"object", "tcp"}:
            raise ValueError("postprocess.frame 只支持 object 或 tcp")
        expected = self.xyz_m if self.type == "translate" else self.rpy_rad
        unexpected = self.rpy_rad if self.type == "translate" else self.xyz_m
        expected_name = "xyz_m" if self.type == "translate" else "rpy_rad"
        if unexpected is not None:
            raise ValueError(f"postprocess.{expected_name} 与 type 不匹配")
        if not isinstance(expected, (list, tuple)) or len(expected) != 3:
            raise ValueError(f"postprocess.{expected_name} 必须是三个有限数值")
        vector: list[float] = []
        for value in expected:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"postprocess.{expected_name} 必须是三个有限数值")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"postprocess.{expected_name} 必须是三个有限数值")
            vector.append(number)
        object.__setattr__(self, expected_name, tuple(vector))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type, "frame": self.frame}
        if self.xyz_m is not None:
            result["xyz_m"] = list(self.xyz_m)
        if self.rpy_rad is not None:
            result["rpy_rad"] = list(self.rpy_rad)
        return result


@dataclass(frozen=True)
class SortRule:
    """候选排序规则；field 与 metric 恰有一个。"""

    order: str
    field: str | None = None
    metric: str | None = None
    transform: str | None = None
    frame: str | None = None

    def __post_init__(self) -> None:
        if self.order not in {"ascending", "descending"}:
            raise ValueError("sorting.order 只支持 ascending 或 descending")
        if (self.field is None) == (self.metric is None):
            raise ValueError("sorting 必须且只能设置 field 或 metric")
        if self.field is not None:
            _require_nonempty_string(self.field, field_path="sorting.field")
            if self.transform != "absolute" or self.frame is not None:
                raise ValueError("field sorting 只支持 absolute transform")
        else:
            if self.metric != "top_down" or self.frame != "world" or self.transform is not None:
                raise ValueError("metric sorting 只支持 world frame 的 top_down")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"order": self.order}
        if self.field is not None:
            result["field"] = self.field
        if self.metric is not None:
            result["metric"] = self.metric
        if self.transform is not None:
            result["transform"] = self.transform
        if self.frame is not None:
            result["frame"] = self.frame
        return result


@dataclass(frozen=True)
class GraspProfile:
    """生成器直接消费的 resolved profile。

    参数已展开为确定顺序；角度字段已经由 ``*_deg`` 改名并转换为
    ``*_rad``。
    """

    name: str
    generator: str
    part_name: str | None
    parameters: Mapping[str, Sequence[float | str]]
    closing_width: ClosingWidthSpec
    postprocess: tuple[PosePostprocess, ...] = ()
    sorting: tuple[SortRule, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, field_path="profile.name")
        _require_nonempty_string(self.generator, field_path="profile.generator")
        if self.part_name is not None:
            _require_nonempty_string(self.part_name, field_path="profile.part_name")
        if not isinstance(self.closing_width, ClosingWidthSpec):
            raise ValueError("profile.closing_width 必须是 ClosingWidthSpec")
        object.__setattr__(
            self,
            "parameters",
            _freeze_json(dict(self.parameters), field_path="profile.parameters"),
        )
        postprocess = tuple(self.postprocess)
        sorting = tuple(self.sorting)
        if not all(isinstance(item, PosePostprocess) for item in postprocess):
            raise ValueError("profile.postprocess 必须只包含 PosePostprocess")
        if not all(isinstance(item, SortRule) for item in sorting):
            raise ValueError("profile.sorting 必须只包含 SortRule")
        object.__setattr__(self, "postprocess", postprocess)
        object.__setattr__(self, "sorting", sorting)
        object.__setattr__(
            self, "metadata", _freeze_json(dict(self.metadata), field_path="profile.metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "generator": self.generator,
            "part": self.part_name,
            "parameters": _thaw_json(self.parameters),
            "closing_width": self.closing_width.to_dict(),
            "postprocess": [item.to_dict() for item in self.postprocess],
            "sorting": [item.to_dict() for item in self.sorting],
            "metadata": _thaw_json(self.metadata),
        }
        return result


@dataclass(frozen=True)
class GraspProposal:
    """单个世界坐标系 TCP 抓取候选，不包含 IK、碰撞或路径状态。"""

    candidate_id: str
    generator: str
    part_name: str
    tcp_pose_world: NDArray[np.float64] | ArrayLike
    closing_width_m: float
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, field_path="candidate_id")
        _require_nonempty_string(self.generator, field_path="generator")
        _require_nonempty_string(self.part_name, field_path="part_name")
        object.__setattr__(
            self,
            "tcp_pose_world",
            validate_pose(self.tcp_pose_world, field_name="tcp_pose_world"),
        )
        if isinstance(self.closing_width_m, bool) or not isinstance(
            self.closing_width_m, (int, float)
        ):
            raise ValueError("closing_width_m 必须是有限非负数")
        width = float(self.closing_width_m)
        if not math.isfinite(width) or width < 0.0:
            raise ValueError("closing_width_m 必须是有限非负数")
        object.__setattr__(self, "closing_width_m", width)
        object.__setattr__(
            self,
            "parameters",
            _freeze_json(dict(self.parameters), field_path="parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generator": self.generator,
            "part_name": self.part_name,
            "tcp_pose_world": self.tcp_pose_world.tolist(),
            "closing_width_m": self.closing_width_m,
            "parameters": _thaw_json(self.parameters),
        }


@dataclass(frozen=True)
class CandidateDiagnostic:
    """候选级结构化诊断；message 面向中文人工检查。"""

    candidate_id: str | None
    code: str
    message: str
    context: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.candidate_id is not None:
            _require_nonempty_string(self.candidate_id, field_path="candidate_id")
        if not isinstance(self.code, str) or _CODE_PATTERN.fullmatch(self.code) is None:
            raise ValueError("code 必须是稳定的大写字母、数字或下划线标识")
        _require_nonempty_string(self.message, field_path="message")
        object.__setattr__(
            self, "context", _freeze_json(dict(self.context), field_path="context")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "code": self.code,
            "message": self.message,
            "context": _thaw_json(self.context),
        }


@dataclass(frozen=True)
class GraspGenerationResult:
    """一次生成的有序候选、诊断和实际 resolved profile。"""

    proposals: tuple[GraspProposal, ...] | Sequence[GraspProposal]
    diagnostics: tuple[CandidateDiagnostic, ...] | Sequence[CandidateDiagnostic]
    resolved_profile: GraspProfile

    def __post_init__(self) -> None:
        proposals = tuple(self.proposals)
        diagnostics = tuple(self.diagnostics)
        if not all(isinstance(item, GraspProposal) for item in proposals):
            raise ValueError("proposals 必须只包含 GraspProposal")
        if not all(isinstance(item, CandidateDiagnostic) for item in diagnostics):
            raise ValueError("diagnostics 必须只包含 CandidateDiagnostic")
        if not isinstance(self.resolved_profile, GraspProfile):
            raise ValueError("resolved_profile 必须是 GraspProfile")
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [item.to_dict() for item in self.proposals],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "resolved_profile": self.resolved_profile.to_dict(),
        }


def build_candidate_id(
    profile_name: str,
    generator: str,
    parameter_indices: Mapping[str, int],
) -> str:
    """由 profile、generator 和离散索引生成跨进程稳定的规范 ID。"""
    _require_nonempty_string(profile_name, field_path="profile_name")
    _require_nonempty_string(generator, field_path="generator")
    normalized: list[tuple[str, int]] = []
    for name, index in parameter_indices.items():
        _require_nonempty_string(name, field_path="parameter_indices key")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"parameter_indices.{name} 必须是非负整数")
        normalized.append((name, index))
    normalized.sort(key=lambda item: item[0])
    indices = ",".join(
        f"{quote(name, safe='')}={index}" for name, index in normalized
    )
    return (
        f"profile={quote(profile_name, safe='')}|"
        f"generator={quote(generator, safe='')}|indices={indices}"
    )
