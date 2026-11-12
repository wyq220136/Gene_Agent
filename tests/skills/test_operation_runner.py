from dataclasses import replace

import pytest


def backend(*, source='mujoco', failure=None, check=None, wrong_start=False):
    from gene_agent.skills.contracts import OperationState, StageOutcome
    from gene_agent.tasks.checks import CheckResult
    class Backend:
        evidence_mode = source
        def __init__(self):
            self.state = OperationState('0', source, {})
            self.calls = []
        def observe(self): return self.state
        def execute(self, stage, start):
            self.calls.append(stage.name)
            self.state = OperationState(str(len(self.calls)), source, {'actual': True})
            return StageOutcome(stage.name, stage.name != failure,
                replace(start, state_id='wrong') if wrong_start else start,
                self.state, check=(check or CheckResult('pass', source, {})) if stage.name in {'grasp_check', 'hold_check'} else None)
    return Backend()


def run(b, **kwargs):
    from gene_agent.skills.library.pick.skill import run_pick_flow
    return run_pick_flow(b, **kwargs)


def test_prefix_stops_before_close():
    b = backend()
    result = run(b, stop_after='approach')
    assert result.success
    assert b.calls == ['open', 'pregrasp', 'approach']
    assert result.end_state.values['actual']


def test_failed_stage_returns_actual_end_and_stops():
    b = backend(failure='approach')
    result = run(b)
    assert not result.success
    assert result.failure_stage == 'approach'
    assert b.calls == ['open', 'pregrasp', 'approach']
    assert result.end_state.state_id == '3'


@pytest.mark.parametrize('status', ['fail', 'unknown'])
def test_missing_or_failed_grasp_cannot_lift(status):
    from gene_agent.tasks.checks import CheckResult
    b = backend(check=CheckResult(status, 'no_contact_sensor', {}))
    result = run(b)
    assert not result.success
    assert result.failure_stage == 'grasp_check'
    assert 'lift' not in b.calls


def test_prediction_can_pass_unknown_without_claiming_execution():
    from gene_agent.tasks.checks import CheckResult
    b = backend(source='prediction', check=CheckResult('unknown', 'predicted_grasp', {}))
    result = run(b)
    assert result.success
    assert not result.executed
    assert result.stages[4].check.status == 'unknown'


def test_prediction_cannot_override_known_failure():
    from gene_agent.tasks.checks import CheckResult
    result = run(backend(source='prediction', check=CheckResult('fail', 'collision', {})))
    assert not result.success


def test_mismatched_stage_start_is_rejected():
    result = run(backend(wrong_start=True))
    assert not result.success
    assert result.failure_code == 'state_discontinuity'


def test_shared_sequence_uses_latest_actual_state():
    b = backend()
    result = run(b)
    assert result.success
    assert b.calls == ['open', 'pregrasp', 'approach', 'close', 'grasp_check', 'establish_hold', 'lift', 'hold_check']
    assert [s.start.state_id for s in result.stages] == [str(i) for i in range(8)]


def test_invalid_scope_is_rejected_before_backend_side_effects():
    b = backend()
    with pytest.raises(ValueError): run(b, stop_after='typo')
    assert b.calls == []


@pytest.mark.parametrize('source', ['prediction', 'predicted_after_pick'])
def test_real_execution_rejects_predicted_start_before_any_action(source):
    from gene_agent.skills.contracts import OperationState, OperationStage
    from gene_agent.skills.runtime.runner import run_operation
    b = backend(source='realworld_sdk')
    r = run_operation([OperationStage('open')], backend=b,
                      initial_state=OperationState('0', source, {}))
    assert not r.success
    assert b.calls == []


@pytest.mark.parametrize('source,evidence', [('prediction', {}), ('predicted_grasp', {}), ('vision', {'predicted': True})])
def test_real_execution_rejects_predicted_pass_check(source, evidence):
    from gene_agent.tasks.checks import CheckResult
    b = backend(source='realworld_sdk', check=CheckResult('pass', source, evidence))
    r = run(b)
    assert not r.success
    assert 'lift' not in b.calls


def test_original_failure_survives_callback_runner_and_serialization():
    import json
    import numpy as np
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationStage, OperationState
    from gene_agent.skills.runtime.runner import run_operation
    failure = {'code': 'start_collision', 'source': 'curobo',
               'message': '起点与桌面碰撞', 'report_ref': 'plans/approach.json'}
    calls = []
    state = OperationState('actual', 'mujoco', {})
    def perform(name):
        calls.append(name)
        return {'success': False, 'failure': failure, 'solver': np.ones(2)}
    result = run_operation([OperationStage('approach'), OperationStage('close')],
        backend=CallbackOperationBackend(observe=lambda: state, perform=perform, evidence_mode='mujoco'))
    assert result.failure_code == 'start_collision'
    assert result.failure == failure
    assert calls == ['approach']
    saved = json.loads(json.dumps(result.to_dict()))
    assert saved['failure'] == failure
    assert saved['stage_outcomes'][0]['failure'] == failure
    assert 'solver' not in saved['stage_outcomes'][0]
    assert saved['end_state']['state_id'] == 'actual'


def test_empty_flow_does_not_claim_execution():
    from gene_agent.skills.runtime.runner import run_operation
    result = run_operation([], backend=backend())
    assert result.success
    assert not result.executed


def test_preflight_cannot_advance_into_a_real_motion():
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationStage, OperationState
    from gene_agent.skills.runtime.runner import run_operation
    calls = []
    state = OperationState('observed', 'realworld', {})
    def perform(name):
        calls.append(name)
        return {'success': True}
    result = run_operation([OperationStage('pregrasp'), OperationStage('close')],
        backend=CallbackOperationBackend(observe=lambda: state, perform=perform,
            evidence_mode='realworld', execution_enabled=False))
    assert not result.success
    assert not result.executed
    assert result.failure_code == 'execution_disabled'
    assert calls == ['pregrasp']


def test_observation_failure_after_motion_retains_unknown_and_original_error():
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationStage, OperationState
    from gene_agent.skills.runtime.runner import run_operation
    moved = []
    def observe():
        if moved:
            raise OSError('状态通道断开')
        return OperationState('before', 'realworld', {'joint': .1})
    b = CallbackOperationBackend(observe=observe,
        perform=lambda n: moved.append(n) or {'success': False,
            'failure': {'code': 'execution_timeout', 'source': 'sdk', 'message': '超时'}},
        evidence_mode='realworld')
    result = run_operation([OperationStage('approach'), OperationStage('close')], backend=b)
    assert not result.success
    assert result.failure_code == 'execution_timeout'
    assert result.end_state.source == 'unavailable'
    assert result.end_state.values['last_known_state_id'] == 'before'
    assert result.end_state.values['state_unavailable']
    assert moved == ['approach']

@pytest.mark.parametrize('row,expected,status', [
    ({'success':False,'executed':False,'execution':None},False,'not_executed'),
    ({'success':False,'execution':{'success':False}},True,'partial'),
    ({'success':False},False,'unknown'),
])
def test_execution_fact_comes_from_outcome_not_backend_permission(row, expected, status):
    from gene_agent.skills.contracts import OperationStage, OperationState
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.runtime.runner import run_operation
    b = CallbackOperationBackend(observe=lambda:OperationState('same','mujoco',{}),
        perform=lambda name:row,evidence_mode='mujoco')
    r = run_operation([OperationStage('move')],backend=b)
    assert r.executed is expected
    assert r.to_dict()['execution_status'] == status


def test_execution_before_later_planning_failure_is_retained():
    from gene_agent.skills.contracts import OperationStage, OperationState
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.runtime.runner import run_operation
    rows=iter([{'success':True,'executed':True}, {'success':False,'executed':False}])
    b=CallbackOperationBackend(observe=lambda:OperationState('actual','mujoco',{}),
        perform=lambda n:next(rows),evidence_mode='mujoco')
    r=run_operation([OperationStage('open'),OperationStage('move')],backend=b)
    assert r.executed and r.execution_status == 'partial'
    assert r.to_dict()['stage_outcomes'][1]['execution_status'] == 'not_executed'

@pytest.mark.parametrize('status', ['pass','fail'])
def test_extract_resume_check_alone_is_not_motion(status):
    from gene_agent.skills.library.transfer import resume_extract
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationState,CheckResult
    calls=[]
    def perform(name):
        calls.append(name)
        if name == 'grasp_check':
            return {'success':True,'check':CheckResult(status,'contact',{})}
        return {'success':False,'executed':False}
    r=resume_extract(CallbackOperationBackend(observe=lambda:OperationState('same','mujoco',{}),
        perform=perform,evidence_mode='mujoco'))
    assert not r.executed
    assert r.execution_status == 'not_executed'
    assert calls == (['grasp_check'] if status == 'fail' else ['grasp_check','attach'])


def test_disabled_backend_cannot_hide_explicit_execution_fact():
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationState,OperationStage
    from gene_agent.skills.runtime.runner import run_operation
    b=CallbackOperationBackend(observe=lambda:OperationState('actual','mujoco',{}),
        perform=lambda n:{'success':True,'executed':True},
        evidence_mode='mujoco',execution_enabled=False)
    r=run_operation([OperationStage('move')],backend=b)
    assert r.executed and r.execution_status == 'partial'
    assert r.failure_code == 'execution_disabled'
