"""通用子任务的状态交接和检查结果；防止独立调试与完整任务分叉。"""

from dataclasses import dataclass

from gene_agent.tasks.checks import CheckResult, resolve_unknown
from gene_agent.tasks.runner import run_steps


@dataclass(frozen=True)
class State:
    value: int

    def discontinuity(self, other):
        return None if self.value == other.value else "unexpected_reset"

    def to_dict(self):
        return {"value": self.value}


@dataclass(frozen=True)
class Result:
    step: str
    start_state: State
    end_state: State
    success: bool
    stages: tuple[dict, ...]
    failure: dict | None = None


def test_standalone_and_composed_use_same_step_and_actual_end_state():
    """若组合任务重建默认状态或替换子任务实现，此测试会失败。"""
    def advance(state):
        return Result("advance", state, State(state.value + 1), True,
                      ({"name": "motion", "success": True},))

    standalone = run_steps(("advance",), State(4), {"advance": advance})
    composed = run_steps(("advance", "advance"), State(4), {"advance": advance})

    assert standalone.success and standalone.final_state == State(5)
    assert composed.success and composed.final_state == State(6)
    assert [row["start_state"] for row in composed.steps] == [
        {"value": 4}, {"value": 5},
    ]


def test_discontinuity_stops_before_accepting_step_or_running_following_step():
    """若执行器偷偷从 home 起步，不能拼接成成功任务。"""
    called = []

    def reset(_state):
        return Result("reset", State(0), State(1), True, ({"name": "motion"},))

    def later(_state):
        called.append(True)
        return Result("later", State(1), State(2), True, ({"name": "motion"},))

    report = run_steps(("reset", "later"), State(7), {"reset": reset, "later": later})

    assert not report.success
    assert report.first_failure == {
        "step": "reset", "code": "state_discontinuity", "message": "unexpected_reset",
    }
    assert report.final_state == State(7)
    assert not called


def test_unknown_check_requires_explicit_manual_evidence():
    """自动检查无法判断时，不能直接继续当作成功。"""
    unknown = CheckResult("unknown", "automatic", {"door_angle_deg": None},
                          "门角不可见")
    assert not unknown.passed
    assert resolve_unknown(unknown).status == "unknown"
    accepted = resolve_unknown(unknown, ask=lambda _question: "y")
    assert accepted.passed
    assert accepted.source == "manual_terminal"
    assert accepted.evidence["automatic"]["status"] == "unknown"
    assert accepted.evidence["answer"] == "y"
    rejected = resolve_unknown(unknown, ask=lambda _question: "n")
    assert rejected.status == "fail"
    failed = CheckResult("fail", "automatic", {}, "物理门角未到达")
    assert resolve_unknown(failed, ask=lambda _question: "y") is failed


@dataclass(frozen=True)
class CheckedResult(Result):
    check: CheckResult | None = None
    result_status: str = "planned"


def test_unknown_postcondition_stops_composed_task_without_claiming_success():
    """动作轨迹成功但门角检查不确定时，下一段不得启动。"""
    called = []

    def open_door(state):
        return CheckedResult(
            "open_door", state, State(1), True, ({"name": "push"},),
            None, CheckResult("unknown", "camera", {}, "门被遮挡"),
        )

    def later(_state):
        called.append(True)
        raise AssertionError("未知后置条件不能继续")

    report = run_steps(("open_door", "later"), State(0),
                       {"open_door": open_door, "later": later})

    assert not report.success
    # 运动已发生，失败报告必须保留实测终态，不能回退成旧起点。
    assert report.final_state == State(1)
    assert report.first_failure["code"] == "postcondition_unknown"
    assert report.steps[0]["check"]["source"] == "camera"
    assert not called


def test_successful_postcondition_keeps_source_and_evidence_in_task_record():
    """若成功结果丢掉检查来源，真机与仿真成功等级就无法区分。"""
    def open_door(state):
        return CheckedResult(
            "open_door", state, State(1), True, ({"name": "push"},), None,
            CheckResult("pass", "door_encoder", {"angle_deg": 90.0}),
        )

    report = run_steps(("open_door",), State(0), {"open_door": open_door})

    assert report.success
    assert report.steps[0]["check"] == {
        "status": "pass", "source": "door_encoder",
        "evidence": {"angle_deg": 90.0}, "reason": "",
    }


def test_executed_subtask_is_not_reported_as_visual_planning():
    """真机执行通过时，记录不能标成只有视觉规划。"""
    def open_door(state):
        return CheckedResult(
            "open_door", state, State(1), True, ({"name": "push"},), None,
            CheckResult("pass", "door_encoder", {"angle_deg": 90.0}),
            "executed",
        )

    report = run_steps(("open_door",), State(0), {"open_door": open_door})

    assert report.success
    assert report.steps[0]["status"] == "executed"


def test_failed_execution_keeps_measured_partial_end_state():
    """真机中途停下时，Task 报告必须给出现场终态供人工恢复。"""
    def partial_motion(state):
        return CheckedResult(
            "partial_motion", state, State(3), False,
            ({"name": "push", "status": "stopped"},),
            {"code": "sdk_error", "message": "第 3 个轨迹点失败"},
            None, "executed",
        )

    report = run_steps(("partial_motion",), State(0),
                       {"partial_motion": partial_motion})

    assert not report.success
    assert report.final_state == State(3)
    assert report.steps[0]["end_state"] == {"value": 3}
    assert report.first_failure["code"] == "sdk_error"
