"""保存一次 motion 调试所需的结构化事实。

DebugBundle 可以在无浏览器、无 Viser、无 MuJoCo 和无 GPU 的后台生成。显示层只
读取该文件；任何颜色、相机或 GUI 状态都不能成为规划事实来源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from gene_agent.motion.model import (
    MotionRequest,
    MotionResult,
    PoseTarget,
    motion_request_from_dict,
    motion_result_from_dict,
)


SCHEMA_VERSION = 1


def _text(value: str, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} 不能为空")
    return result


def _xyz(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是有限三维坐标")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class DebugBox:
    geometry_id: str
    label: str
    pose_world: np.ndarray
    size_xyz_m: tuple[float, float, float]
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_id", _text(self.geometry_id, name="geometry ID"))
        object.__setattr__(self, "label", _text(self.label, name="geometry label"))
        object.__setattr__(self, "role", _text(self.role, name="geometry role"))
        # 复用 Motion 的刚体变换验证，避免 feedback 产生另一套 pose 定义。
        object.__setattr__(self, "pose_world", PoseTarget("world", self.pose_world).matrix)
        size = _xyz(self.size_xyz_m, name="box size")
        if any(value <= 0.0 for value in size):
            raise ValueError("box size 必须全部大于 0")
        object.__setattr__(self, "size_xyz_m", size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id,
            "label": self.label,
            "pose_world": self.pose_world.tolist(),
            "size_xyz_m": list(self.size_xyz_m),
            "role": self.role,
        }


@dataclass(frozen=True)
class CollisionMarker:
    marker_id: str
    first: str
    second: str
    position_world_m: tuple[float, float, float] | None
    radius_m: float
    note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "marker_id", _text(self.marker_id, name="marker ID"))
        object.__setattr__(self, "first", _text(self.first, name="collision first"))
        object.__setattr__(self, "second", _text(self.second, name="collision second"))
        object.__setattr__(self, "note", _text(self.note, name="collision note"))
        if self.position_world_m is not None:
            object.__setattr__(
                self,
                "position_world_m",
                _xyz(self.position_world_m, name="collision marker position"),
            )
        radius = float(self.radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("collision marker radius 必须是正有限数")
        object.__setattr__(self, "radius_m", radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "first": self.first,
            "second": self.second,
            "position_world_m": (
                None if self.position_world_m is None else list(self.position_world_m)
            ),
            "radius_m": self.radius_m,
            "note": self.note,
        }


@dataclass(frozen=True)
class MotionDebugAttempt:
    attempt_id: str
    stage_id: str
    candidate_id: str | None
    label: str
    request: MotionRequest
    result: MotionResult
    tcp_path_world_m: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _text(self.attempt_id, name="attempt ID"))
        object.__setattr__(self, "stage_id", _text(self.stage_id, name="stage ID"))
        object.__setattr__(self, "label", _text(self.label, name="attempt label"))
        if self.candidate_id is not None:
            object.__setattr__(
                self, "candidate_id", _text(self.candidate_id, name="candidate ID")
            )
        if not isinstance(self.request, MotionRequest) or not isinstance(
            self.result, MotionResult
        ):
            raise TypeError("attempt request/result 必须使用 Motion 公共契约")
        if self.request.request_id != self.result.request_id:
            raise ValueError("attempt request/result 的 request_id 不一致")
        if self.request.operation != self.result.operation:
            raise ValueError("attempt request/result 的 operation 不一致")
        if self.request.planner_mode != self.result.planner_mode:
            raise ValueError("attempt request/result 的 planner_mode 不一致")
        path = tuple(_xyz(point, name="TCP path point") for point in self.tcp_path_world_m)
        object.__setattr__(self, "tcp_path_world_m", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "stage_id": self.stage_id,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
            "tcp_path_world_m": [list(point) for point in self.tcp_path_world_m],
        }


@dataclass(frozen=True)
class MotionDebugBundle:
    run_id: str
    attempts: tuple[MotionDebugAttempt, ...]
    scene_boxes: tuple[DebugBox, ...] = ()
    collision_markers: tuple[CollisionMarker, ...] = ()
    notes: tuple[str, ...] = ()
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, name="run ID"))
        attempts = tuple(self.attempts)
        boxes = tuple(self.scene_boxes)
        markers = tuple(self.collision_markers)
        if any(not isinstance(value, MotionDebugAttempt) for value in attempts):
            raise TypeError("attempts 必须由 MotionDebugAttempt 组成")
        if any(not isinstance(value, DebugBox) for value in boxes):
            raise TypeError("scene_boxes 必须由 DebugBox 组成")
        if any(not isinstance(value, CollisionMarker) for value in markers):
            raise TypeError("collision_markers 必须由 CollisionMarker 组成")
        self._require_unique((value.attempt_id for value in attempts), "attempt_id")
        self._require_unique((value.geometry_id for value in boxes), "geometry_id")
        self._require_unique((value.marker_id for value in markers), "marker_id")
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "scene_boxes", boxes)
        object.__setattr__(self, "collision_markers", markers)
        object.__setattr__(
            self, "notes", tuple(_text(value, name="debug note") for value in self.notes)
        )

    @staticmethod
    def _require_unique(values: Sequence[str] | Any, name: str) -> None:
        items = tuple(values)
        if len(items) != len(set(items)):
            raise ValueError(f"{name} 必须唯一")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempts": [value.to_dict() for value in self.attempts],
            "scene_boxes": [value.to_dict() for value in self.scene_boxes],
            "collision_markers": [value.to_dict() for value in self.collision_markers],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotionDebugBundle":
        version = value.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"不支持的 motion DebugBundle schema_version={version!r}；"
                f"当前只支持 {SCHEMA_VERSION}"
            )
        return cls(
            run_id=value["run_id"],
            attempts=tuple(
                MotionDebugAttempt(
                    attempt_id=item["attempt_id"],
                    stage_id=item["stage_id"],
                    candidate_id=item.get("candidate_id"),
                    label=item["label"],
                    request=motion_request_from_dict(item["request"]),
                    result=motion_result_from_dict(item["result"]),
                    tcp_path_world_m=tuple(
                        tuple(point) for point in item.get("tcp_path_world_m", ())
                    ),
                )
                for item in value.get("attempts", ())
            ),
            scene_boxes=tuple(
                DebugBox(
                    item["geometry_id"],
                    item["label"],
                    item["pose_world"],
                    tuple(item["size_xyz_m"]),
                    item["role"],
                )
                for item in value.get("scene_boxes", ())
            ),
            collision_markers=tuple(
                CollisionMarker(
                    item["marker_id"],
                    item["first"],
                    item["second"],
                    (
                        None
                        if item.get("position_world_m") is None
                        else tuple(item["position_world_m"])
                    ),
                    item["radius_m"],
                    item["note"],
                )
                for item in value.get("collision_markers", ())
            ),
            notes=tuple(value.get("notes", ())),
        )


def save_motion_debug_bundle(path: str | Path, bundle: MotionDebugBundle) -> Path:
    """在同目录原子写入 bundle，避免异常留下看似完整的半个 JSON。"""
    if not isinstance(bundle, MotionDebugBundle):
        raise TypeError("bundle 必须是 MotionDebugBundle")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        bundle.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def load_motion_debug_bundle(path: str | Path) -> MotionDebugBundle:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("motion DebugBundle 根节点必须是 JSON object")
    return MotionDebugBundle.from_dict(value)


__all__ = [
    "SCHEMA_VERSION",
    "DebugBox",
    "CollisionMarker",
    "MotionDebugAttempt",
    "MotionDebugBundle",
    "save_motion_debug_bundle",
    "load_motion_debug_bundle",
]
