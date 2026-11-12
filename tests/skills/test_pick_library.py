"""推荐 Pick 接口：范围、绑定身份与失败现场。"""
import importlib
import pytest


def test_request_mismatch_is_rejected_before_observation_or_motion():
    from gene_agent.skills.library import pick
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    calls = []
    b = CallbackOperationBackend(observe=lambda: calls.append('observe'),
        perform=lambda n: calls.append(n), evidence_mode='mujoco')
    b.request = pick.PickRequest('cup', 'left')
    with pytest.raises(ValueError, match='绑定'):
        pick.run(b, pick.PickRequest('bowl', 'right'))
    assert calls == []


@pytest.mark.parametrize('scope,expected', [
    ('pregrasp', ['open', 'pregrasp']),
    ('approach', ['open', 'pregrasp', 'approach']),
    ('grasp', ['open', 'pregrasp', 'approach', 'close', 'grasp_check']),
    ('pick', ['open', 'pregrasp', 'approach', 'close', 'grasp_check', 'establish_hold', 'lift', 'hold_check']),
])
def test_public_pick_runs_only_requested_stages(scope, expected):
    from gene_agent.skills.library import pick
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationState
    from gene_agent.tasks.checks import CheckResult
    calls = []
    request = pick.PickRequest('bowl', 'right')
    b = CallbackOperationBackend(
        observe=lambda: OperationState(str(len(calls)), 'mujoco', {}),
        perform=lambda n: calls.append(n) or {'success': True, 'check': CheckResult('pass','contact',{})},
        evidence_mode='mujoco')
    b.request = request
    result = pick.run(b, request, stop_after=scope)
    assert result.success and result.executed
    assert calls == expected
    assert result.skill == 'pick' and result.requested_scope == scope
    assert result.end_state.state_id == str(len(expected))


def test_new_contract_exports_are_the_same_types_as_legacy():
    from gene_agent.skills import contracts
    from gene_agent.skills.contracts import OperationResult, StageOutcome
    from gene_agent.tasks.checks import CheckResult
    assert contracts.OperationResult is OperationResult
    assert contracts.StageOutcome is StageOutcome
    assert contracts.CheckResult is CheckResult


@pytest.mark.parametrize('name', ['grasp', 'pick'])
def test_planning_remains_explicit_and_compatible(name):
    old = importlib.import_module('gene_agent.skills')
    new = importlib.import_module('gene_agent.skills.library.pick.planning')
    cls = 'PickSkill' if name == 'pick' else 'GraspSkill'
    assert getattr(old, cls) is getattr(new, cls)
