"""桌面物体的 seed 可复现 pose 采样；不依赖具体任务或机器人。"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Callable

import numpy as np


def sample_tabletop_layout(
    nominal_xyyaw_rad: Mapping[str, tuple[float, float, float]],
    ranges_xyyaw_rad: Mapping[str, tuple[float, float, float]],
    footprint_xy_m: Mapping[str, tuple[float, float]],
    *, table_bounds_xy: tuple[float, float, float, float], seed: int,
    min_clearance_m: float = .015, max_attempts: int = 200,
    candidate_validator: Callable[[dict[str, np.ndarray]], bool] | None = None,
) -> dict:
    """均匀采样 XY/yaw；用旋转后 AABB 保守检查桌边和物体间距。"""
    names = tuple(nominal_xyyaw_rad)
    if not names or set(names) != set(ranges_xyyaw_rad) or set(names) != set(footprint_xy_m):
        raise ValueError("名义 pose、采样范围与 footprint 的物体集合必须一致且非空")
    nominal = {name: np.asarray(nominal_xyyaw_rad[name], dtype=float) for name in names}
    ranges = {name: np.asarray(ranges_xyyaw_rad[name], dtype=float) for name in names}
    footprint = {name: np.asarray(footprint_xy_m[name], dtype=float) for name in names}
    bounds = np.asarray(table_bounds_xy, dtype=float)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
        raise ValueError("桌面边界必须是有限 (xmin,xmax,ymin,ymax)")
    if max_attempts <= 0 or not np.isfinite(min_clearance_m) or min_clearance_m < 0:
        raise ValueError("采样次数须为正，间距须为非负有限值")
    for name in names:
        if (nominal[name].shape != (3,) or ranges[name].shape != (3,)
                or footprint[name].shape != (2,)
                or not all(np.isfinite(value).all()
                           for value in (nominal[name], ranges[name], footprint[name]))
                or np.any(ranges[name] < 0) or np.any(footprint[name] <= 0)):
            raise ValueError(f"{name} 的 pose、范围或 footprint 无效")

    rng = np.random.default_rng(int(seed))
    rejected = {"table_boundary": 0, "object_clearance": 0, "grasp_reachability": 0}
    for attempt in range(1, max_attempts + 1):
        sampled = {name: nominal[name] + rng.uniform(-ranges[name], ranges[name])
                   for name in names}
        boxes = {}
        invalid_reason = None
        for name in names:
            x, y, yaw = sampled[name]
            dx, dy = footprint[name] / 2
            half_x = abs(np.cos(yaw)) * dx + abs(np.sin(yaw)) * dy
            half_y = abs(np.sin(yaw)) * dx + abs(np.cos(yaw)) * dy
            box = (x - half_x, x + half_x, y - half_y, y + half_y)
            if (box[0] < bounds[0] or box[1] > bounds[1]
                    or box[2] < bounds[2] or box[3] > bounds[3]):
                invalid_reason = "table_boundary"
                break
            for other in boxes.values():
                gap_x = max(0., box[0] - other[1], other[0] - box[1])
                gap_y = max(0., box[2] - other[3], other[2] - box[3])
                if np.hypot(gap_x, gap_y) < min_clearance_m:
                    invalid_reason = "object_clearance"
                    break
            if invalid_reason:
                break
            boxes[name] = box
        if invalid_reason:
            rejected[invalid_reason] += 1
            continue
        if candidate_validator is not None and not candidate_validator(sampled):
            rejected["grasp_reachability"] += 1
            continue
        return {"valid": True, "seed": int(seed), "attempts": attempt,
                "distribution": "independent_uniform_xy_yaw_about_nominal",
                "frame": "world", "length_unit": "m", "angle_unit": "rad",
                "nominal_xyyaw_rad": {name: nominal[name].tolist() for name in names},
                "ranges_xyyaw_rad": {name: ranges[name].tolist() for name in names},
                "poses_xyyaw_rad": {name: sampled[name].tolist() for name in names},
                "footprint_xy_m": {name: footprint[name].tolist() for name in names},
                "table_bounds_xy_m": bounds.tolist(),
                "min_clearance_m": min_clearance_m,
                "rejected_attempts": rejected}
    raise ValueError(f"{max_attempts} 次采样未找到有效初态；拒绝原因：{rejected}")
