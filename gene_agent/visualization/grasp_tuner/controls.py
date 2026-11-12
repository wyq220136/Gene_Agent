"""两个 Knowledge 可视化工具共用的参数范围滑块。

这里仅在原始 task profile 的单位中编辑采样规格。角度以度表示，米制参数
以米表示；解析和合法性检查仍由 Knowledge profile loader 统一完成。
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

from gene_agent.knowledge.grasp.sampling import expand_sample_spec


PARAMETER_LABELS = {
    "azimuth_deg": "环上方位角 azimuth (°)",
    "cylinder_rotation_deg": "柱体周向角 (°)",
    "tip_x_deg": "TCP 局部 X 倾角 (°)",
    "tip_y_deg": "TCP 局部 Y 倾角 (°)",
    "depth_m": "抓取深度 (m)",
    "vertical_offset_m": "沿物体轴线偏移 (m)",
    "surface_parallel_offset_m": "沿抓取面偏移 (m)",
}


def _limits(name: str, values: list[float]) -> tuple[float, float, float]:
    if name == "tip_y_deg":
        # 倾角主要用于微调；已有 profile 的端点仍然必须落在滑块范围内。
        return min(-30.0, min(values)), max(30.0, max(values)), 0.5
    if name == "tip_x_deg":
        return min(-90.0, min(values)), max(90.0, max(values)), 0.5
    if name in {"azimuth_deg", "cylinder_rotation_deg"}:
        return min(0.0, min(values)), max(360.0, max(values)), 1.0
    if name.endswith("_deg"):
        return min(-180.0, min(values)), max(180.0, max(values)), 1.0
    if "offset" in name:
        return -0.2, 0.2, 0.001
    if "depth" in name:
        return 0.0, 0.2, 0.001
    low, high = min(values), max(values)
    return min(low, -0.1), max(high, 0.2), 0.001


def numeric_spec(spec: Any) -> tuple[str, list[float]]:
    """只识别数值规格；枚举和布尔值交给原 JSON 编辑器。"""
    if isinstance(spec, dict) and {"start", "stop", "step"} <= set(spec):
        return "范围", [float(spec["start"]), float(spec["stop"]), float(spec["step"])]
    values = spec.get("values") if isinstance(spec, dict) else spec
    if isinstance(values, list) and values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return ("固定" if len(values) == 1 else "列表"), [float(v) for v in values]
    raise TypeError("该参数不是数值采样规格")


def _range_from_values(values: list[float], *, name: str, tick: float) -> tuple[float, float, float]:
    """只在列表本身是等差序列时转换，避免悄悄增加或删除抓取候选。"""
    if len(values) == 1:
        return values[0], values[0], tick
    step = values[1] - values[0]
    if step == 0 or any(abs(value - (values[0] + index * step)) > 1e-8 for index, value in enumerate(values)):
        raise ValueError(f"{name}: 只有等差列表可以转换为范围；请保留列表模式")
    return values[0], values[-1], step


class GraspParameterControls:
    """为 profile 的数值参数生成固定值/范围/原始列表三种编辑方式。"""

    def __init__(self, gui: Any, parameters: dict, on_change: Callable[[], None], on_error: Callable[[str], None] | None = None):
        self.original = copy.deepcopy(parameters)
        self.on_change = on_change
        self.on_error = on_error
        self.rows: dict[str, dict[str, Any]] = {}
        self.folder = gui.add_folder("抓取参数滑块", expand_by_default=True)
        with self.folder:
            for name, spec in parameters.items():
                try:
                    mode, values = numeric_spec(spec)
                except TypeError:
                    continue
                low, high, tick = _limits(name, values)
                start = values[0]
                stop = values[1] if mode == "范围" else values[-1]
                interval = abs(values[2]) if mode == "范围" else tick
                with gui.add_folder(PARAMETER_LABELS.get(name, name), expand_by_default=name in {"azimuth_deg", "cylinder_rotation_deg"}):
                    kind = gui.add_dropdown("采样方式", options=("固定", "范围", "列表"), initial_value=mode)
                    fixed = gui.add_slider("固定值", min=low, max=high, step=tick, initial_value=start)
                    first = gui.add_slider("起点", min=low, max=high, step=tick, initial_value=start)
                    last = gui.add_slider("终点", min=low, max=high, step=tick, initial_value=stop)
                    step = gui.add_slider("步长（绝对值）", min=tick, max=90.0 if name.endswith("_deg") else 0.05, step=tick, initial_value=max(tick, interval))
                    listing = gui.add_text("列表 JSON", initial_value=json.dumps(values, ensure_ascii=False))
                row = {"mode": kind, "fixed": fixed, "start": first, "stop": last, "step": step, "list": listing}
                row["previous_mode"] = mode
                row["order"] = spec.get("order") if mode == "范围" else None
                self.rows[name] = row
                for key in ("mode", "fixed", "start", "stop", "step", "list"):
                    row[key].on_update(lambda event, item=name: self._changed(item))
                self._visibility(name)
        self._syncing = False

    def _visibility(self, name: str) -> None:
        row = self.rows[name]
        mode = row["mode"].value
        row["fixed"].visible = mode == "固定"
        for key in ("start", "stop", "step"):
            row[key].visible = mode == "范围"
        row["list"].visible = mode == "列表"

    def _changed(self, name: str) -> None:
        if getattr(self, "_syncing", False):
            return
        row = self.rows[name]
        current, previous = str(row["mode"].value), row["previous_mode"]
        if current != previous:
            try:
                if current == "列表":
                    if previous == "范围":
                        spec = {key: float(row[key].value) for key in ("start", "stop", "step")}
                        if spec["stop"] < spec["start"]:
                            spec["step"] = -spec["step"]
                        if row["order"] is not None:
                            spec["order"] = row["order"]
                        values = expand_sample_spec(spec, path=name)
                    else:
                        values = [float(row["fixed"].value)]
                    self._syncing = True
                    row["list"].value = json.dumps(values, ensure_ascii=False)
                    self._syncing = False
                elif current == "范围":
                    values = json.loads(row["list"].value) if previous == "列表" else [float(row["fixed"].value)]
                    start, stop, signed_step = _range_from_values(values, name=name, tick=_limits(name, values)[2])
                    self._syncing = True
                    for key, value in (("start", start), ("stop", stop), ("step", abs(signed_step))):
                        row[key].value = value
                    self._syncing = False
                    # 从显式列表进入范围时，顺序应由列表决定，不能继承历史 range 的排序。
                    if previous == "列表":
                        row["order"] = None
                elif previous == "范围":
                    row["fixed"].value = float(row["start"].value)
                else:
                    values = json.loads(row["list"].value)
                    row["fixed"].value = float(values[0])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                self._syncing = True
                row["mode"].value = previous
                self._syncing = False
                self._visibility(name)
                if self.on_error is not None:
                    self.on_error(str(error))
                return
            row["previous_mode"] = current
        self._visibility(name)
        self.on_change()

    def read(self) -> dict:
        result = copy.deepcopy(self.original)
        for name, row in self.rows.items():
            mode = row["mode"].value
            if mode == "固定":
                result[name] = {"values": [float(row["fixed"].value)]}
            elif mode == "范围":
                start, stop, step = (float(row[key].value) for key in ("start", "stop", "step"))
                if stop < start:
                    step = -step
                result[name] = {"start": start, "stop": stop, "step": step}
                if row["order"] is not None:
                    result[name]["order"] = row["order"]
            else:
                values = json.loads(row["list"].value)
                if not isinstance(values, list) or not values:
                    raise ValueError(f"{name}: 列表必须是非空 JSON 数组")
                result[name] = {"values": values}
        return result

    def remove(self) -> None:
        self.folder.remove()


__all__ = ["GraspParameterControls", "numeric_spec"]
