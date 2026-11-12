"""通用规划轨迹时间轴；Skill 和 Task 查看器共享，不依赖具体操作对象。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlannedSegment:
    stage_id: str
    skill_id: str
    joint_names: tuple[str, ...]
    positions_rad: np.ndarray
    sample_dt_s: float
    channels: dict[str, np.ndarray]
    details: dict | None = None

    def __post_init__(self):
        positions = np.asarray(self.positions_rad, dtype=float)
        if (not self.stage_id or not self.skill_id or positions.ndim != 2
                or positions.shape[0] < 2 or positions.shape[1] != len(self.joint_names)
                or not np.isfinite(positions).all()):
            raise ValueError("规划阶段必须包含至少两个有限的具名关节采样")
        if not math.isfinite(self.sample_dt_s) or self.sample_dt_s <= 0:
            raise ValueError("轨迹采样周期必须是正的有限数")
        normalized = {}
        for name, values in self.channels.items():
            array = np.asarray(values, dtype=float)
            if array.shape != (len(positions),) or not np.isfinite(array).all():
                raise ValueError(f"状态通道 {name} 必须与关节采样数量一致")
            normalized[str(name)] = array.copy()
        object.__setattr__(self, "positions_rad", positions.copy())
        object.__setattr__(self, "channels", normalized)
        object.__setattr__(self, "details", dict(self.details or {}))

    @property
    def duration_s(self) -> float:
        return (len(self.positions_rad) - 1) * self.sample_dt_s


@dataclass(frozen=True)
class TimelineSample:
    time_s: float
    stage_id: str
    skill_id: str
    joint_names: tuple[str, ...]
    joint_positions_rad: np.ndarray
    channels: dict[str, float]
    segment_index: int
    segment_time_s: float


class PlannedTimeline:
    def __init__(self, segments, *, boundary_tolerance_rad: float = 1e-3):
        self.segments = tuple(segments)
        if not self.segments:
            raise ValueError("规划时间轴不能为空")
        if not math.isfinite(boundary_tolerance_rad) or boundary_tolerance_rad < 0:
            raise ValueError("阶段边界容差必须是有限非负数")
        names = self.segments[0].joint_names
        for previous, current in zip(self.segments, self.segments[1:]):
            if current.joint_names != names:
                raise ValueError("同一时间轴中的关节名称和顺序必须一致")
            error = float(np.max(np.abs(
                previous.positions_rad[-1] - current.positions_rad[0]
            )))
            if error > boundary_tolerance_rad:
                raise ValueError(
                    f"阶段边界关节状态不连续：{previous.stage_id} → {current.stage_id}，"
                    f"最大跳变 {error:.6f} rad"
                )
        self.joint_names = names
        self.starts_s = np.cumsum([0.] + [item.duration_s for item in self.segments[:-1]])
        self.duration_s = float(sum(item.duration_s for item in self.segments))

    def sample(self, time_s: float) -> TimelineSample:
        time_s = float(np.clip(time_s, 0., self.duration_s))
        index = int(np.searchsorted(self.starts_s, time_s, side="right") - 1)
        index = min(max(index, 0), len(self.segments) - 1)
        segment = self.segments[index]
        local = float(np.clip(time_s - self.starts_s[index], 0., segment.duration_s))
        coordinate = local / segment.sample_dt_s
        left = min(int(np.floor(coordinate)), len(segment.positions_rad) - 1)
        right = min(left + 1, len(segment.positions_rad) - 1)
        fraction = coordinate - left if right > left else 0.
        joints = ((1. - fraction) * segment.positions_rad[left]
                  + fraction * segment.positions_rad[right])
        channels = {
            name: float((1. - fraction) * values[left] + fraction * values[right])
            for name, values in segment.channels.items()
        }
        return TimelineSample(
            time_s, segment.stage_id, segment.skill_id, segment.joint_names,
            joints, channels, index, local,
        )


__all__ = ["PlannedSegment", "PlannedTimeline", "TimelineSample"]
