"""碗和刷子共用阶段流；已有实测抓取前缀可以检查后续接抬升。"""
import pytest


def test_grasp_scope_includes_check_but_never_attaches_or_lifts():
    from gene_agent.skills.library.pick.skill import run_pick_flow
    b = _backend()
    r = run_pick_flow(b, stop_after='grasp')
    assert r.success
    assert b.calls == ['open', 'pregrasp', 'approach', 'close', 'grasp_check']


def test_post_grasp_flow_requires_check_again_before_attachment():
    from gene_agent.skills.library.pick.skill import run_pick_flow
    b = _backend(held=False)
    r = run_pick_flow(b, resume_after_grasp=True)
    assert not r.success
    assert b.calls == ['grasp_check']


def test_post_grasp_flow_does_not_repeat_gripper_commands():
    from gene_agent.skills.library.pick.skill import run_pick_flow
    b = _backend()
    r = run_pick_flow(b, resume_after_grasp=True)
    assert r.success
    assert b.calls == ['grasp_check', 'establish_hold', 'lift', 'hold_check']


def test_validation_without_execution_is_not_reported_as_executed():
    from gene_agent.skills.library.pick.skill import run_pick_flow
    b = _backend()
    b.execution_enabled = False
    r = run_pick_flow(b, stop_after='pregrasp')
    assert not r.executed


def _backend(held=True):
    from gene_agent.skills.contracts import OperationState, StageOutcome
    from gene_agent.tasks.checks import CheckResult
    class Backend:
        evidence_mode = 'realworld_observation'
        calls = None
        def __init__(self): self.calls = []
        def observe(self): return OperationState(str(len(self.calls)), 'realworld_observation', {})
        def execute(self, stage, start):
            self.calls.append(stage.name)
            check = CheckResult('pass' if held else 'unknown', 'manual_terminal', {})
            return StageOutcome(stage.name, True, start, self.observe(), check)
    return Backend()
