"""按顺序执行同一批子任务函数，并传递上段实际终态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence, TypeVar


State = TypeVar("State")


class StepResult(Protocol[State]):
    step: str
    start_state: State
    end_state: State
    success: bool
    stages: tuple[dict, ...]
    failure: dict | None


@dataclass(frozen=True)
class TaskRun:
    success: bool
    steps: tuple[dict, ...]
    final_state: object
    first_failure: dict | None


def run_steps(
    order: Sequence[str], initial_state: State,
    adapters: Mapping[str, Callable[[State], StepResult[State]]],
) -> TaskRun:
    """独立运行与完整任务共用相同 adapter；失败保留已完成前缀。

    状态类型需要提供 `discontinuity(next_state)` 和 `to_dict()`。
    这里不推断场景状态，也不补跑缺少的前置子任务。
    """
    current = initial_state
    rows: list[dict] = []
    for name in order:
        adapter = adapters.get(name)
        if adapter is None:
            failure = {"step": name, "code": "adapter_unavailable",
                       "message": "缺少可接受上一阶段终态的适配器"}
            rows.append({"step": name, "status": "failed", "failure": failure})
            return TaskRun(False, tuple(rows), current, failure)
        try:
            result = adapter(current)
        except Exception as error:
            failure = {"step": name, "code": "adapter_exception",
                       "message": f"{type(error).__name__}: {error}"}
            rows.append({"step": name, "status": "failed", "failure": failure})
            return TaskRun(False, tuple(rows), current, failure)
        if result.step != name:
            reason = "wrong_step_result"
        else:
            reason = current.discontinuity(result.start_state)
        if reason is not None:
            failure = {"step": name, "code": "state_discontinuity", "message": reason}
            rows.append({"step": name, "status": "failed", "failure": failure})
            return TaskRun(False, tuple(rows), current, failure)
        if not result.success:
            failure = {"step": name, **result.failure}
            row = {"step": name, "status": "failed", "failure": failure,
                   "start_state": result.start_state.to_dict(),
                   "stages": result.stages}
            if getattr(result, "result_status", "planned") == "executed":
                # 实际运动可能已经走了一部分；现场终态不能回退成规划起点。
                row["end_state"] = result.end_state.to_dict()
                rows.append(row)
                return TaskRun(False, tuple(rows), result.end_state, failure)
            rows.append(row)
            return TaskRun(False, tuple(rows), current, failure)
        check = getattr(result, "check", None)
        if check is not None and not check.passed:
            # 后置检查发生在运动之后：失败也必须传回实际执行终态。
            failure = {
                "step": name,
                "code": ("postcondition_unknown" if check.status == "unknown"
                         else "postcondition_failed"),
                "message": check.reason,
            }
            rows.append({
                "step": name, "status": "failed", "failure": failure,
                "start_state": result.start_state.to_dict(),
                "end_state": result.end_state.to_dict(),
                "stages": result.stages, "check": check.to_dict(),
            })
            return TaskRun(False, tuple(rows), result.end_state, failure)
        result_status = getattr(result, "result_status", "planned")
        if result_status not in {"planned", "executed"}:
            failure = {"step": name, "code": "invalid_result_status",
                       "message": f"未知子任务状态：{result_status}"}
            rows.append({"step": name, "status": "failed", "failure": failure})
            return TaskRun(False, tuple(rows), result.end_state, failure)
        row = {"step": name, "status": result_status,
               "start_state": result.start_state.to_dict(),
               "end_state": result.end_state.to_dict(),
               "stages": result.stages}
        if check is not None:
            row["check"] = check.to_dict()
        rows.append(row)
        current = result.end_state
    return TaskRun(True, tuple(rows), current, None)
