"""两个环境共用 Task1 次序、检查闸门和实际终态交接。"""
import pytest
from gene_agent.tasks.policies.microwave_task.policy import MicrowaveTaskPolicy, microwave_task_steps, run_microwave_task1_flow
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.tasks.checks import CheckResult


def test_real_and_sim_share_order_except_declared_environment_transition():
    real = microwave_task_steps(MicrowaveTaskPolicy())
    sim = microwave_task_steps(MicrowaveTaskPolicy(before_reopen_transition=False))
    assert [s.name for s in real if s.name != 'before_reopen_transition'] == [s.name for s in sim]
    assert [s.name for s in real][2:4] == ['pick_bowl', 'insert_bowl']
    assert [s.name for s in real][-3:] == ['place_bowl_on_table', 'before_close_second_transition', 'close_door_second']
    assert 'before_reopen_transition' in [s.name for s in microwave_task_steps(MicrowaveTaskPolicy(include_knob=False))]


@pytest.mark.parametrize('check_status', ['unknown', 'fail'])
def test_common_task_stops_at_bad_check_and_keeps_actual_end(check_status):
    state = OperationState('initial', 'fake_actual', {})
    calls = []
    def perform(name):
        nonlocal state
        calls.append(name)
        state = OperationState(name, 'fake_actual', {})
        return {'success': True, 'check': CheckResult(check_status, 'test', {})}
    backend = CallbackOperationBackend(observe=lambda: state, perform=perform, evidence_mode='test')
    result = run_microwave_task1_flow(backend)
    assert not result.success
    assert calls == ['open_door_first']
    assert result.end_state.state_id == 'open_door_first'


def test_common_task_hands_each_actual_end_to_next_action():
    state = OperationState('initial', 'fake_actual', {})
    calls = []
    def perform(name):
        nonlocal state
        calls.append((name, state.state_id))
        state = OperationState(name, 'fake_actual', {})
        return {'success': True, 'check': CheckResult('pass', 'test', {})}
    result = run_microwave_task1_flow(CallbackOperationBackend(
        observe=lambda: state, perform=perform, evidence_mode='test'))
    assert result.success
    assert calls[1][1] == calls[0][0]
    assert all(s.start == result.stages[i-1].end for i,s in enumerate(result.stages) if i)
