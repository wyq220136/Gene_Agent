"""真机后置条件不得用仿真特权或视觉回放提交。"""

import pytest

from gene_agent.realworld import checks as module
from gene_agent.realworld.checks import resolve_realworld_check
from gene_agent.tasks.checks import CheckResult


@pytest.mark.parametrize("source", [
    "simulation_privileged_weld", "mujoco_contact_geometry_at_planned_waypoints",
    "mixed_kinematic_knob", "legacy_viser_knob_follow",
])
def test_rejects_simulated_success(source):
    with pytest.raises(ValueError, match="仿真"):
        resolve_realworld_check(CheckResult("pass", source, {}, "模拟通过"))


def test_manual_confirmation_can_resolve_unknown():
    result = resolve_realworld_check(
        CheckResult("unknown", "not_configured", {}, "现场检查未配置"),
        ask=lambda prompt: "y",
    )
    assert result.status == "pass" and result.source == "manual_terminal"


def test_manual_answer_keeps_question_answer_and_time(monkeypatch):
    """现场人工判断要能追溯回答内容和实际回答时间。"""
    monkeypatch.setattr(module.time, "time", lambda: 1234.5)
    result = resolve_realworld_check(
        CheckResult("unknown", "not_configured", {}, "现场检查未配置"),
        ask=lambda _prompt: "?", question="是否确实抓稳？[y/n/?] ",
    )
    assert result.status == "unknown"
    assert result.evidence["question"] == "是否确实抓稳？[y/n/?] "
    assert result.evidence["answer"] == "?"
    assert result.evidence["answered_at_unix_ms"] == 1234500


@pytest.mark.parametrize("source", [
    "mujoco_gt", "mujoco_actual_samples", "mujoco_future_backend",
    "prediction", "preview", "arbitrary_unregistered_sensor",
])
def test_new_or_unregistered_source_cannot_become_real_success(source):
    report = {"status": "pass", "source": source, "evidence": {}}
    assert module.has_realworld_success_evidence(report) is False
    with pytest.raises(ValueError, match="来源"):
        resolve_realworld_check(CheckResult("pass", source, {}))


@pytest.mark.parametrize("source", ["manual_terminal", "vision", "observed_door_angle"])
def test_declared_real_source_is_accepted(source):
    check = CheckResult("pass", source, {"source_frame_number": 12})
    assert module.has_realworld_success_evidence(check.to_dict())
    assert resolve_realworld_check(check) is check


def test_predicted_flag_cannot_be_laundered_through_a_real_source_name():
    check = CheckResult("pass", "vision", {"predicted": True})
    assert module.has_realworld_success_evidence(check.to_dict()) is False
    with pytest.raises(ValueError, match="预测"):
        resolve_realworld_check(check)
