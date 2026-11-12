"""只用实际仿真采样判断阶段后置条件；不读取规划器的成功标志。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def _door_angle(spec: Mapping, samples: Sequence[Mapping]) -> dict:
    target = float(spec["target_rad"])
    tolerance = float(spec["tolerance_rad"])
    window = float(spec["stable_window_s"])
    contact_key = str(spec["required_contact_key"])
    if (not all(math.isfinite(value) for value in (target, tolerance, window))
            or tolerance <= 0 or window <= 0 or not contact_key):
        raise ValueError("门角判定需要有限目标、正的容差与稳定时长、接触字段")
    result = {
        "kind": "door_angle", "status": "unknown", "code": "insufficient_observation",
        "source": "mujoco_actual_samples", "target_rad": target,
        "tolerance_rad": tolerance, "stable_window_s": window,
        "required_contact_key": contact_key,
        "actual_rad": None, "contact_sample_count": 0,
    }
    if not samples:
        return result
    if any("sim_time_s" not in item or "door_angle_rad" not in item
           or contact_key not in item for item in samples):
        result["code"] = "missing_observation"
        return result
    times = [float(item["sim_time_s"]) for item in samples]
    angles = [float(item["door_angle_rad"]) for item in samples]
    contacts = [int(item[contact_key]) for item in samples]
    if (not all(math.isfinite(value) for value in (*times, *angles))
            or any(next_time <= time for time, next_time in zip(times, times[1:]))
            or any(count < 0 for count in contacts)):
        raise ValueError("门角采样时间必须递增、数值有限且接触数非负")
    result["actual_rad"] = angles[-1]
    result["contact_sample_count"] = sum(count > 0 for count in contacts)
    if times[-1] - times[0] < window:
        return result
    window_samples = [angle for time, angle in zip(times, angles)
                      if time >= times[-1] - window - 1e-9]
    if len(window_samples) < 2:
        return result
    result["max_window_error_rad"] = max(abs(angle - target) for angle in window_samples)
    if not result["contact_sample_count"]:
        result.update(status="fail", code="no_contact")
    elif result["max_window_error_rad"] > tolerance:
        result.update(status="fail", code="angle_not_stable")
    else:
        result.update(status="pass", code="goal_reached")
    return result


def evaluate_checks(specs: list[dict], samples: list[dict]) -> list[dict]:
    """返回逐项判定；缺失或过短的观测为 unknown，不算物理通过。"""
    results = []
    for spec in specs:
        if spec.get("kind") == "door_angle":
            results.append(_door_angle(spec, samples))
        else:
            raise ValueError(f"未知阶段判定：{spec.get('kind')}")
    return results
