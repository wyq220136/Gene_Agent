"""单一单调墙钟、层级区间与互斥账本；不参与操作决策。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import heapq
import inspect
import json
import math
from pathlib import Path
import time

_CURRENT = ContextVar("realworld_runtime_timing", default=None)
_STACK = ContextVar("realworld_timing_spans", default=())
_NS = 1_000_000_000


def _timestamp(wall_ns=None):
    value = datetime.now(timezone.utc) if wall_ns is None else datetime.fromtimestamp(wall_ns / _NS, timezone.utc)
    return value.astimezone().isoformat(timespec="milliseconds")


def _union_ns(intervals):
    total, right = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


class RuntimeTimingRecorder:
    """总时间来自起止时钟；组件/阶段独占时间各自与总时间守恒。"""
    def __init__(self, output_dir=None, *, started_ns=None, started_wall_ns=None, clock_ns=None):
        self.clock_ns = clock_ns or time.perf_counter_ns
        self._started_ns = self.clock_ns() if started_ns is None else int(started_ns)
        self._started = self._started_ns / _NS  # 兼容已有 perf_counter 秒偏移调用方。
        self.output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
        self._write_allowed = True
        self.started_at = _timestamp(started_wall_ns)
        self.finished_at = None
        self.total_duration_s = None
        self._total_ns = None
        self.status = "running"
        self.error = None
        self.events = []
        self._sequence = 0
        self._open = {}

    def bind_output(self, output_dir):
        if output_dir is not None and self.output_dir is None:
            self.output_dir = Path(output_dir).expanduser().resolve()
            # 延迟绑定的 CLI 不得因拒绝复用旧目录而覆盖旧的计时报告。
            self._write_allowed = not self.output_dir.exists() or not any(self.output_dir.iterdir())

    def record(self, component, operation, duration_s, *, started_offset_s=None,
               ended_offset_s=None, status="success", metadata=None, event_id=None,
               parent_id=None, depth=0, stage_id=None):
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0:
            return
        if event_id is None:
            self._sequence += 1
            event_id = self._sequence
        event = dict(event_id=event_id, parent_id=parent_id, depth=depth,
                     component=str(component), operation=str(operation), duration_s=duration,
                     status=str(status), timing_kind="reported_duration", stage_id=stage_id)
        if started_offset_s is not None:
            start = float(started_offset_s)
            end = start + duration if ended_offset_s is None else float(ended_offset_s)
            if not all(math.isfinite(value) for value in (start, end)) or end < start:
                return
            event.update(started_offset_s=start, ended_offset_s=end, timing_kind="wall_interval")
        if metadata:
            event["metadata"] = dict(metadata)
        self.events.append(event)

    def finish(self, *, status, error=None):
        if self._total_ns is None:
            self._total_ns = max(0, self.clock_ns() - self._started_ns)
            self.finished_at = _timestamp()
            self.total_duration_s = self._total_ns / _NS
            self.status = str(status)
            if error is not None:
                self.error = f"{type(error).__name__}: {error}"
        payload = self.to_dict()
        self._write(payload)
        return payload

    def to_dict(self):
        total_ns = self._total_ns if self._total_ns is not None else max(0, self.clock_ns() - self._started_ns)
        snapshot_at = self.finished_at or _timestamp()
        events = list(self.events)
        for event in self._open.values():
            events.append({**event, "ended_offset_s": total_ns / _NS,
                           "duration_s": max(0., total_ns / _NS - event["started_offset_s"])})
        components, intervals, starts, points = {}, [], {}, {0, total_ns}
        for event in events:
            row = components.setdefault(event["component"], dict(call_count=0, total_duration_s=0.,
                wall_duration_s=0., reported_duration_s=0., inclusive_sum_s=0.,
                exclusive_duration_s=0., max_duration_s=0., slowest_operation=None))
            row["call_count"] += 1
            if event["duration_s"] >= row["max_duration_s"]:
                row.update(max_duration_s=event["duration_s"], slowest_operation=event["operation"])
            if event["timing_kind"] != "wall_interval":
                row["reported_duration_s"] += event["duration_s"]
                continue
            start = min(total_ns, max(0, round(event["started_offset_s"] * _NS)))
            end = min(total_ns, max(start, round(event["ended_offset_s"] * _NS)))
            row["inclusive_sum_s"] += (end - start) / _NS
            intervals.append((start, end, event))
            starts.setdefault(start, []).append((start, end, event))
            points.update((start, end))
        for name, row in components.items():
            row["wall_duration_s"] = _union_ns((start, end) for start, end, event in intervals
                                              if event["component"] == name) / _NS
            row["total_duration_s"] = row["wall_duration_s"]
        # 最深层区间优先；同深度重叠取后开始者，保证每纳秒只进入一个桶。
        exclusive, stages, active = {}, {}, []
        ordered = sorted(points)
        for left, right in zip(ordered, ordered[1:]):
            for start, end, event in starts.get(left, ()):
                heapq.heappush(active, (-event["depth"], -start, -event["event_id"], end, event))
            while active and active[0][3] <= left:
                heapq.heappop(active)
            event = active[0][4] if active else None
            component = event["component"] if event else "unattributed"
            stage = str(event.get("stage_id") or "outside_stages") if event else "outside_stages"
            exclusive[component] = exclusive.get(component, 0) + right - left
            stages[stage] = stages.get(stage, 0) + right - left
        for name, value in exclusive.items():
            if name in components:
                components[name]["exclusive_duration_s"] = value / _NS
        stage_events = [dict(event_id=e["event_id"], parent_id=e["parent_id"],
                             operation=e["operation"], status=e["status"],
                             started_offset_s=e["started_offset_s"], ended_offset_s=e["ended_offset_s"],
                             inclusive_duration_s=e["duration_s"],
                             exclusive_duration_s=stages.get(str(e["event_id"]), 0) / _NS)
                        for e in events if e["component"] == "stage" and e["timing_kind"] == "wall_interval"]
        comparison = {key: components.get(name, {}).get("wall_duration_s", 0.) for key, name in (
            ("mesh_preparation_s", "mesh_preparation"), ("mesh_file_read_s", "mesh_file_read"),
            ("curobo_setup_s", "curobo_setup"), ("curobo_planning_wall_s", "curobo_planning"),
            ("curobo_initialization_s", "curobo_initialization"), ("curobo_solver_s", "curobo_solver"))}
        explicit_mesh = _union_ns((start, end) for start, end, event in intervals
                                  if event["component"] in {"mesh_preparation", "mesh_file_read"}) / _NS
        mesh_init = _union_ns((start, end) for start, end, event in intervals
                             if event["component"] in {"mesh_preparation", "mesh_file_read", "curobo_initialization"}) / _NS
        solver = comparison["curobo_solver_s"]
        comparison.update(scene_mesh_and_curobo_initialization_s=mesh_init,
            dominant_mesh_or_curobo_solver=("curobo_solver" if solver > mesh_init else
                "scene_mesh_and_curobo_initialization" if mesh_init > solver else None),
            dominant_recorded_component=("curobo_planning" if comparison["curobo_planning_wall_s"] > explicit_mesh else
                "explicit_mesh_processing" if explicit_mesh > comparison["curobo_planning_wall_s"] else None))
        bottleneck = None
        if exclusive:
            name = max(exclusive, key=exclusive.get)
            bottleneck = dict(component=name, total_duration_s=exclusive[name] / _NS,
                              slowest_operation=components.get(name, {}).get("slowest_operation"),
                              basis="exclusive_wall_time")
        stage_labels = {str(row["event_id"]): row["operation"] for row in stage_events}
        stage_totals = {}
        for key, value in stages.items():
            label = stage_labels.get(key, key)
            stage_totals[label] = stage_totals.get(label, 0) + value
        return dict(schema_version=2, started_at=self.started_at, finished_at=self.finished_at, snapshot_at=snapshot_at,
            total_duration_s=total_ns / _NS, status=self.status, error=self.error,
            components=components, comparison=comparison, bottleneck=bottleneck,
            accounting=dict(clock="perf_counter_ns", total_wall_s=total_ns / _NS,
                exclusive_components_s={key: value / _NS for key, value in exclusive.items()},
                exclusive_stages_s={key: value / _NS for key, value in stage_totals.items()},
                accounted_wall_s=sum(exclusive.values()) / _NS,
                unattributed_s=exclusive.get("unattributed", 0) / _NS,
                reconciliation_error_s=(sum(exclusive.values()) - total_ns) / _NS),
            stages=sorted(stage_events, key=lambda row: row["started_offset_s"]),
            events=sorted(events, key=lambda row: (row.get("started_offset_s", float("inf")), row["event_id"])),
            scope=["CLI 从脚本最早可测语句计时，覆盖导入/配置、连接、感知、规划、执行、等待、日志、清理、视频及最终终端输出；直接调用从入口开始。",
                   "不包含 Python 启动到首条计时语句之前、计时报告自身最终写盘及进程退出；终端快照略早于文件终态。",
                   "accounting.exclusive_components_s 与 exclusive_stages_s 各自加总等于 total_duration_s；unattributed 是显式未覆盖时间。",
                   "components.total_duration_s 是同组件实测区间的并集，父子组件仍可能重叠，不能跨组件直接相加。",
                   "没有起止位置的后端报告仅列 reported_duration_s，不冒充实测墙钟、不参与守恒账本。"])

    def _write(self, payload):
        if not self._write_allowed or self.output_dir is None or not self.output_dir.is_dir():
            return
        if not any((self.output_dir / name).exists() for name in (
                "runtime_input.json", "site_input.json", "entry_result.json", "events.jsonl")):
            return
        (self.output_dir / "runtime_timing.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def current_runtime_timing():
    return _CURRENT.get()


def current_timing_event_id():
    stack = _STACK.get()
    return stack[-1][1] if stack and stack[-1][0] is current_runtime_timing() else None


def current_stage_timing_event_id():
    stack = _STACK.get()
    return stack[-1][2] if stack and stack[-1][0] is current_runtime_timing() else None


def record_runtime_duration(component, operation, duration_s, *, metadata=None, started_at_s=None):
    recorder = current_runtime_timing()
    if recorder is not None:
        stack = _STACK.get()
        parent = stack[-1] if stack and stack[-1][0] is recorder else None
        recorder.record(component, operation, duration_s,
                        started_offset_s=None if started_at_s is None else started_at_s - recorder._started,
                        metadata=metadata, parent_id=None if parent is None else parent[1],
                        depth=len(stack), stage_id=None if parent is None else parent[2])


@contextmanager
def measure_runtime(component, operation, *, metadata=None):
    recorder = current_runtime_timing()
    if recorder is None:
        yield None
        return
    started = recorder.clock_ns()
    stack = _STACK.get()
    parent = stack[-1] if stack and stack[-1][0] is recorder else None
    recorder._sequence += 1
    event_id = recorder._sequence
    stage_id = event_id if component == "stage" else parent[2] if parent else None
    token = _STACK.set((*stack, (recorder, event_id, stage_id)))
    recorder._open[event_id] = dict(event_id=event_id, parent_id=None if parent is None else parent[1],
        depth=len(stack), component=str(component), operation=str(operation), stage_id=stage_id,
        timing_kind="wall_interval", started_offset_s=(started - recorder._started_ns) / _NS,
        status="running", **({"metadata": dict(metadata)} if metadata else {}))
    span = {"event_id": event_id}
    status = "success"
    try:
        yield span
    except BaseException:
        status = "failed"
        raise
    finally:
        ended = recorder.clock_ns()
        span["duration_s"] = (ended - started) / _NS
        _STACK.reset(token)
        recorder._open.pop(event_id, None)
        recorder.record(component, operation, span["duration_s"],
            started_offset_s=(started - recorder._started_ns) / _NS,
            ended_offset_s=(ended - recorder._started_ns) / _NS, status=status, metadata=metadata,
            event_id=event_id, parent_id=None if parent is None else parent[1],
            depth=len(stack), stage_id=stage_id)


@contextmanager
def runtime_timing_scope(output_dir=None, *, started_ns=None, started_wall_ns=None, clock_ns=None):
    existing = current_runtime_timing()
    if existing is not None:
        existing.bind_output(output_dir)
        yield existing
        return
    recorder = RuntimeTimingRecorder(output_dir, started_ns=started_ns,
                                    started_wall_ns=started_wall_ns, clock_ns=clock_ns)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    except BaseException as error:
        try:
            recorder.finish(status="failed", error=error)
        except Exception:
            pass  # 旁路统计不能覆盖原操作异常。
        raise
    else:
        try:
            recorder.finish(status="success")
        except Exception:
            pass
    finally:
        _CURRENT.reset(token)


def timed_runtime_call(component, operation=None, *, name_parameter=None, result_duration_key=None):
    """为现有同步函数加墙钟区间；可把完整调用耗时附在字典报告。"""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            name = operation or function.__name__
            if name_parameter is not None:
                value = signature.bind_partial(*args, **kwargs).arguments.get(name_parameter)
                name += ":" + str(getattr(value, "name", value))
            with measure_runtime(component, name) as span:
                result = function(*args, **kwargs)
            if result_duration_key and span is not None and isinstance(result, dict):
                result = {**result, result_duration_key: span["duration_s"]}
            return result
        return wrapped
    return decorate


def timed_runtime(output_parameter):
    """直接入口获得根计时器；已有 CLI 根时复用，不提前结束总计时。"""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            with runtime_timing_scope(bound.arguments[output_parameter]):
                with measure_runtime("realworld", function.__name__):
                    return function(*args, **kwargs)
        return wrapped
    return decorate
