"""搬运后的检查必须控制松爪和撤离；两种环境使用同一阶段定义。"""
import pytest

from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult


@pytest.mark.parametrize('operation,retreats', [('insert', ['retreat_1', 'retreat_2']),
                                              ('place', ['retreat'])])
def test_transfer_runs_from_actual_state_and_checks_before_retreat(operation, retreats):
    from gene_agent.skills.library.transfer.skill import run_transfer_flow
    state = OperationState('0', 'mujoco', {})
    calls = []
    def perform(name):
        nonlocal state
        calls.append(name)
        state = OperationState(str(len(calls)), 'mujoco', {})
        return {'success': True, 'check': CheckResult('pass', 'mujoco_contact', {})}
    result = run_transfer_flow(CallbackOperationBackend(
        observe=lambda: state, perform=perform, evidence_mode='mujoco'), operation=operation)
    assert result.success
    assert calls == ['preplace', 'hold_check', 'transfer_path_check', operation,
                     'release', 'detach', 'settle', 'release_check', *retreats, 'final_check']
    assert [s.start.state_id for s in result.stages] == [str(i) for i in range(len(calls))]


@pytest.mark.parametrize('failed', ['hold_check', 'transfer_path_check', 'release_check', 'final_check'])
def test_unknown_transfer_evidence_stops_later_actions(failed):
    from gene_agent.skills.library.transfer.skill import run_transfer_flow
    calls = []
    state = OperationState('actual', 'realworld_observation', {})
    def perform(name):
        calls.append(name)
        return {'success': True, 'check': CheckResult(
            'unknown' if name == failed else 'pass', 'vision', {})}
    result = run_transfer_flow(CallbackOperationBackend(
        observe=lambda: state, perform=perform, evidence_mode='realworld_observation'))
    assert not result.success
    assert result.failure_stage == failed
    assert calls[-1] == failed
    if failed in {'hold_check', 'transfer_path_check'}:
        assert 'release' not in calls
    if failed == 'release_check':
        assert 'retreat_1' not in calls
