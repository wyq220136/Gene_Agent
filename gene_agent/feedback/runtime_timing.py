"""真机流程的旁路墙钟统计；不参与感知、规划或执行决策。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import math
from pathlib import Path
import time


_CURRENT = ContextVar("realworld_runtime_timing", default=None)
_DETAIL_COMPONENTS = frozenset({"curobo_initialization", "curobo_solver"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class RuntimeTimingRecorder:
    """收集同步流程中的计时事件，并在结束后写入运行目录。"""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.started_at = _timestamp()
        self._started = time.perf_counter()
        self.finished_at = None
        self.total_duration_s = None
        self.status = "running"
        self.error = None
        self.events = []

    def record(self, component: str, operation: str, duration_s: float, *,
               started_offset_s: float | None = None, status: str = "success",
               metadata: dict | None = None) -> None:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0:
            return
        event = {
            "component": str(component),
            "operation": str(operation),
            "duration_s": duration,
            "status": str(status),
        }
        if started_offset_s is not None:
            event["started_offset_s"] = max(0.0, float(started_offset_s))
        if metadata:
            event["metadata"] = dict(metadata)
        self.events.append(event)

    def finish(self, *, status: str, error: BaseException | None = None) -> dict:
        if self.total_duration_s is None:
            self.finished_at = _timestamp()
            self.total_duration_s = time.perf_counter() - self._started
            self.status = str(status)
            if error is not None:
                self.error = f"{type(error).__name__}: {error}"
        payload = self.to_dict()
        self._write(payload)
        return payload

    def to_dict(self) -> dict:
        components = {}
        for event in self.events:
            row = components.setdefault(event["component"], {
                "call_count": 0,
                "total_duration_s": 0.0,
                "max_duration_s": 0.0,
                "slowest_operation": None,
            })
            row["call_count"] += 1
            row["total_duration_s"] += event["duration_s"]
            if event["duration_s"] >= row["max_duration_s"]:
                row["max_duration_s"] = event["duration_s"]
                row["slowest_operation"] = event["operation"]

        primary = {
            name: row for name, row in components.items()
            if name not in _DETAIL_COMPONENTS
        }
        bottleneck = None
        if primary:
            name, row = max(primary.items(), key=lambda item: item[1]["total_duration_s"])
            bottleneck = {
                "component": name,
                "total_duration_s": row["total_duration_s"],
                "slowest_operation": row["slowest_operation"],
            }
        comparison = {
            "mesh_preparation_s": components.get("mesh_preparation", {}).get("total_duration_s", 0.0),
            "mesh_file_read_s": components.get("mesh_file_read", {}).get("total_duration_s", 0.0),
            "curobo_setup_s": components.get("curobo_setup", {}).get("total_duration_s", 0.0),
            "curobo_planning_wall_s": components.get("curobo_planning", {}).get("total_duration_s", 0.0),
            "curobo_initialization_s": components.get("curobo_initialization", {}).get("total_duration_s", 0.0),
            "curobo_solver_s": components.get("curobo_solver", {}).get("total_duration_s", 0.0),
        }
        explicit_mesh = comparison["mesh_preparation_s"] + comparison["mesh_file_read_s"]
        planning_wall = comparison["curobo_planning_wall_s"]
        mesh_and_initialization = explicit_mesh + comparison["curobo_initialization_s"]
        solver = comparison["curobo_solver_s"]
        comparison["scene_mesh_and_curobo_initialization_s"] = mesh_and_initialization
        comparison["dominant_mesh_or_curobo_solver"] = (
            "curobo_solver" if solver > mesh_and_initialization else
            "scene_mesh_and_curobo_initialization" if mesh_and_initialization > solver else
            None
        )
        comparison["dominant_recorded_component"] = (
            "curobo_planning" if planning_wall > explicit_mesh else
            "explicit_mesh_processing" if explicit_mesh > planning_wall else
            None
        )
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_s": self.total_duration_s,
            "status": self.status,
            "error": self.error,
            "components": components,
            "comparison": comparison,
            "bottleneck": bottleneck,
            "events": list(self.events),
            "scope": [
                "total_duration_s 是完整现场执行阶段墙钟时间；CLI记录从cuRobo启动到设备连接、感知、规划、执行、清理及可选视频导出结束。",
                "curobo_initialization 是规划调用内的 solver 初始化，包含 cuRobo 内部场景/mesh 装载，不能与 curobo_planning 相加。",
                "mesh_file_read 可能发生在 curobo_setup 或 curobo_planning 内；各组件用于定位瓶颈，不是互斥分项。",
                "dominant_mesh_or_curobo_solver 比较场景/mesh准备加solver初始化与纯solver调用；初始化还包含机器人配置，不能解释成纯mesh耗时。",
                "dominant_recorded_component 仅比较显式 mesh 处理与 cuRobo 调用墙钟，用于观察端到端调用开销。",
            ],
        }

    def _write(self, payload: dict) -> None:
        # 输出目录尚未被正式入口创建时不落盘，避免污染无效或已有运行目录。
        if not self.output_dir.is_dir():
            return
        if not any((self.output_dir / name).exists() for name in (
                "runtime_input.json", "site_input.json", "entry_result.json", "events.jsonl")):
            return
        path = self.output_dir / "runtime_timing.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def current_runtime_timing() -> RuntimeTimingRecorder | None:
    return _CURRENT.get()


def record_runtime_duration(component: str, operation: str, duration_s: float, *,
                            metadata: dict | None = None) -> None:
    recorder = current_runtime_timing()
    if recorder is not None:
        recorder.record(component, operation, duration_s, metadata=metadata)


@contextmanager
def measure_runtime(component: str, operation: str, *, metadata: dict | None = None):
    recorder = current_runtime_timing()
    if recorder is None:
        yield
        return
    started = time.perf_counter()
    offset = started - recorder._started
    try:
        yield
    except BaseException:
        recorder.record(component, operation, time.perf_counter() - started,
                        started_offset_s=offset, status="failed", metadata=metadata)
        raise
    else:
        recorder.record(component, operation, time.perf_counter() - started,
                        started_offset_s=offset, metadata=metadata)


@contextmanager
def runtime_timing_scope(output_dir: str | Path):
    existing = current_runtime_timing()
    if existing is not None:
        yield existing
        return
    recorder = RuntimeTimingRecorder(output_dir)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    except BaseException as error:
        try:
            recorder.finish(status="failed", error=error)
        except Exception:
            # 旁路统计失败不能覆盖原始现场异常。
            pass
        raise
    else:
        try:
            recorder.finish(status="success")
        except Exception:
            # 日志介质异常不能改变已经完成的任务结果。
            pass
    finally:
        _CURRENT.reset(token)


def timed_runtime(output_parameter: str):
    """让直接调用入口也获得统计；已有外层 scope 时复用同一记录器。"""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            with runtime_timing_scope(bound.arguments[output_parameter]):
                return function(*args, **kwargs)

        return wrapped
    return decorate
