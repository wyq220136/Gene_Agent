"""完整 MuJoCo 装配调用共享 Task，子任务消费同一个运行状态。"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from gene_agent.tasks.policies.microwave_task import simulation as module
from gene_agent.tasks.policies.microwave_task.policy import MicrowaveTaskPolicy, microwave_task_steps


def _fake_task(tmp_path, monkeypatch, fail=None):
    calls = []
    data = SimpleNamespace(time=0., qpos=np.array([0., np.pi/4]))
    model = SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[0 if name == 'microwave_door_joint' else 1]))
    runtime = SimpleNamespace(model=model, data=data, metadata={},
        robot_metadata={'joint_groups': {'right': ['j']}},
        replay_sample=lambda **_: None,
        query=lambda **_: {'simulation_attachments': []})
    def advance(name):
        calls.append((name, data.time))
        data.time += 1.
        if name in {'open', 'reopen'}:
            data.qpos[0] = np.pi/2
        elif name == 'close':
            data.qpos[0] = 0.
        if fail == name:
            raise RuntimeError('injected failure')
    class Session:
        def __init__(self, *args, **kwargs):
            self.runtime = runtime
            self.kind = kwargs.get('kind', 'open')
        def run(self):
            advance(self.kind)
            return {'goal_success': True}
        def step(self):
            advance('before_close')
            return {'execution_success': True}
    monkeypatch.setattr(module, 'ExperimentSession', Session)
    monkeypatch.setattr(module, '_prepare_scene', lambda *a: (tmp_path/'door.json', tmp_path/'scene.json'))
    monkeypatch.setattr(module, 'state_id', lambda m,d: f'actual:{d.time}')
    monkeypatch.setattr(module, 'model_fingerprint', lambda m: 'same-model')
    monkeypatch.setattr(module, 'save_state', lambda *a: None)
    monkeypatch.setattr(module, '_release_finished_planner', lambda *a: None)
    def transition(session):
        advance('after_open')
        return {'execution_success': True}
    monkeypatch.setattr(module, '_transition_after_open', transition)
    def close(*args, prepare_only=False):
        assert prepare_only
        return {'session': Session(kind='close')}
    monkeypatch.setattr(module, '_continue_close', close)
    def reopen(*args):
        s = Session(kind='reopen')
        return {'session': s, 'report': s.run()}
    monkeypatch.setattr(module, '_continue_open', reopen)
    def pick(*args, runtime, **kwargs):
        assert runtime is runtime_ref
        advance('pick')
        return {'success': True, 'actual_end': data.time}
    runtime_ref = runtime
    monkeypatch.setattr(module, 'run_experiment', pick)
    def insert(rt, output, recorder, picked, **kwargs):
        assert rt is runtime
        assert picked['actual_end'] == data.time
        advance('insert')
        (output/'physical_insert').mkdir(parents=True, exist_ok=True)
        (output/'physical_insert/insert_result.json').write_text('{}')
        return {'physical_place_success': True}
    monkeypatch.setattr(module, 'run_physical_insert_bowl', insert)
    monkeypatch.setattr(module, '_knob_plan', lambda *a: {'success': True, 'joint_target_angle_rad': np.pi/4})
    def knob(*a):
        advance('knob')
        return {'goal_success': True, 'evidence_mode': 'explicit_mixed'}
    monkeypatch.setattr(module, 'run_planned_knob_execution', knob)
    monkeypatch.setattr(module, 'scale_position_actuator_kp', lambda *a: None)
    def extract(*a, **kw):
        advance('extract_grasp')
        return {'physical_grasp_prefix_success': True}
    monkeypatch.setattr(module, 'run_physical_extract_grasp', extract)
    def retreat(*a, **kw):
        advance('extract_retreat')
        return {'assisted_extract_success': True, 'actual_end': data.time}
    monkeypatch.setattr(module, 'run_assisted_extract_retreat', retreat)
    def place(rt, output, report, **kw):
        assert rt is runtime and report['actual_end'] == data.time
        advance('place')
        return {'physical_table_place_success': True}
    monkeypatch.setattr(module, 'run_assisted_table_place', place)
    monkeypatch.setattr(module, 'audit_push_paths', lambda *a: {'all_planned_linear_tcp': True})
    return calls


def test_physical_task_uses_shared_sequence_without_reset(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    result = module.run_microwave_task1_physical(tmp_path/'run', config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    assert result['goal_success'] is True
    assert result['fully_physical_success'] is False
    assert result['shared_flow']['stages'] == [s.name for s in microwave_task_steps(MicrowaveTaskPolicy(before_reopen_transition=False))]
    assert [name for name,t in calls] == ['open', 'after_open', 'pick', 'insert', 'before_close', 'close', 'knob', 'reopen', 'after_open', 'extract_grasp', 'extract_retreat', 'place', 'before_close', 'close']
    assert [t for name,t in calls] == list(range(len(calls)))
    assert result['subtasks']['pick_insert']['success'] is True
    inserted = next(r for r in result['steps'] if r['step_id'] == 'insert_bowl')
    assert (tmp_path/'run'/inserted['report_ref']).is_file()


def test_task_uses_declared_load_control_for_bowl_phases_only(tmp_path, monkeypatch):
    _fake_task(tmp_path, monkeypatch)
    ratios = []
    monkeypatch.setattr(
        module, 'scale_position_actuator_kp',
        lambda _model, _group, ratio: ratios.append(ratio),
    )
    result = module.run_microwave_task1_physical(
        tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'),
        selection='load_bowl',
    )
    assert result['operation_success'] is True
    assert ratios == [pytest.approx(8.0), pytest.approx(0.125)]
    assert [row['stage'] for row in result['control_transitions']] == ['pick_bowl']


@pytest.mark.parametrize('failure', ['pick', 'insert', 'before_close', 'extract_grasp', 'extract_retreat', 'place'])
def test_physical_task_failure_keeps_actual_prefix_and_does_not_continue(tmp_path, monkeypatch, failure):
    calls = _fake_task(tmp_path, monkeypatch, fail=failure)
    result = module.run_microwave_task1_physical(tmp_path/'run', config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    assert result['goal_success'] is False
    assert result['evidence_mode']=='assisted_pick_extract_and_kinematic_knob'
    assert result['shared_flow']['success'] is False
    assert result['shared_flow']['end_state']['state_id'] == f'actual:{float(len(calls))}'
    assert calls[-1][0] == failure
    assert result['steps'][-1]['status'] == 'failed'
    saved = json.loads((tmp_path/'run/summary.json').read_text())
    assert saved['failure_subtask'] == result['shared_flow']['failure_stage']


def test_load_bowl_prefix_is_local_success_and_stops_before_closing(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'), selection='load_bowl')
    assert [name for name, _ in calls] == ['open', 'after_open', 'pick', 'insert']
    assert result['operation_success'] is True
    assert result['goal_success'] is None
    assert result['requested_scope'] == 'load_bowl'
    assert result['target_steps'] == ['pick_bowl', 'insert_bowl']
    assert result['actual_final']['state_id'] == 'actual:4.0'
    assert result['timing']['prefix_wall_s'] >= 0
    assert result['timing']['target_wall_s'] >= 0


def test_current_insert_without_pick_evidence_does_not_move(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    session = module.MicrowaveSimulationSession(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    session.runtime.data.qpos[0] = np.pi/2
    result = session.run(selection='insert_bowl', preparation='current')
    assert calls == []
    assert result['operation_success'] is False
    assert result['failure']['code'] == 'missing_handoff'
    assert result['actual_final']['state_id'] == 'actual:0.0'


def test_current_pick_rejects_closed_door_and_keeps_state(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'),
        selection='pick_bowl', preparation='current')
    assert not calls
    assert result['failure']['code'] == 'precondition_failed'
    assert result['goal_success'] is None


def test_stale_pick_handoff_is_rejected_before_insert(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    session = module.MicrowaveSimulationSession(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    assert session.run(selection='pick_bowl')['operation_success']
    session.runtime.data.time += .1
    count = len(calls)
    result = session.run(selection='insert_bowl', preparation='current')
    assert len(calls) == count
    assert result['failure']['code'] == 'stale_handoff'


def test_partial_insert_failure_identifies_target_and_keeps_actual_end(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch, fail='insert')
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'), selection='load_bowl')
    assert result['operation_success'] is False
    assert result['goal_success'] is None
    assert result['failure']['step_id'] == 'insert_bowl'
    assert result['failure']['role'] == 'target'
    assert 'injected failure' in result['failure']['message']
    assert result['actual_final']['state_id'] == f'actual:{float(len(calls))}'


@pytest.mark.parametrize('first,second', [
    ('pick_bowl', 'insert_bowl'), ('extract_bowl', 'place_bowl_on_table'),
])
def test_same_state_id_with_changed_model_rejects_handoff(tmp_path, monkeypatch, first, second):
    calls = _fake_task(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'model_fingerprint', lambda model: getattr(model, 'test_fingerprint', 'original'))
    session = module.MicrowaveSimulationSession(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    assert session.run(selection=first)['operation_success']
    count = len(calls)
    session.runtime.model.test_fingerprint = 'changed-control-or-geometry'
    result = session.run(selection=second, preparation='current')
    assert len(calls) == count
    assert result['failure']['code'] == 'incompatible_handoff'


def test_same_session_pick_then_current_insert_uses_actual_state_without_reset(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    session = module.MicrowaveSimulationSession(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    picked = session.run(selection='pick_bowl')
    result = session.run(selection='insert_bowl', preparation='current')
    assert picked['actual_final']['state_id'] == 'actual:3.0'
    assert result['operation_success'] is True
    assert [name for name, _ in calls] == ['open','after_open','pick','insert']
    assert result['shared_flow']['stages'] == ['insert_bowl']
    assert result['prefix_steps'] == []
    assert result['actual_final']['state_id'] == 'actual:4.0'


def test_extract_failure_restores_control_gain_in_wrapper_cleanup(tmp_path, monkeypatch):
    _fake_task(tmp_path, monkeypatch, fail='extract_grasp')
    ratios = []
    monkeypatch.setattr(module, 'scale_position_actuator_kp', lambda m, g, ratio: ratios.append(ratio))
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'), selection='extract_bowl')
    assert result['operation_success'] is False
    assert len(ratios) == 2
    assert ratios[0]*ratios[1] == pytest.approx(1.)


def test_insert_returned_failure_preserves_phase_planner_reason_and_report(tmp_path, monkeypatch):
    _fake_task(tmp_path, monkeypatch)
    def failed(rt, output, recorder, picked, **kw):
        report = {'physical_place_success':False, 'failure_stage':'preplace',
                  'stages': {'preplace': {'planning': {
                      'failure_category':'goal_ik_no_solution', 'status':'IK_FAIL'}}}}
        path = output/'physical_insert/insert_result.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report))
        return report
    monkeypatch.setattr(module, 'run_physical_insert_bowl', failed)
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'), selection='load_bowl')
    assert result['failure']['stage'] == 'preplace'
    assert result['failure']['code'] == 'goal_ik_no_solution'
    assert result['failure']['report_ref'] == 'pick_insert/physical_insert/insert_result.json'
    assert 'IK_FAIL' in result['failure']['message']


def test_session_extract_failure_restores_control_before_return(tmp_path, monkeypatch):
    _fake_task(tmp_path, monkeypatch, fail='extract_grasp')
    ratios = []
    monkeypatch.setattr(module, 'scale_position_actuator_kp', lambda m, g, ratio: ratios.append(ratio))
    session = module.MicrowaveSimulationSession(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    result = session.run(selection='extract_bowl')
    assert not result['operation_success']
    assert len(ratios) == 2
    assert ratios[0]*ratios[1] == pytest.approx(1.)
    assert 'arm_kp_ratio' not in session.context


def test_insert_recorder_error_preserves_original_planner_failure(tmp_path, monkeypatch):
    _fake_task(tmp_path, monkeypatch)
    def failed(rt, output, recorder, picked, **kwargs):
        report = {'physical_place_success': False, 'failure_stage': 'insert',
                  'stages': {'insert': {'planning': {
                      'failure_category': 'linear_path_failed', 'status': 'PATH_FAIL'}}}}
        path = output/'physical_insert/insert_result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        return report
    def finalize():
        raise RuntimeError('video finalization failed')
    from gene_agent.visualization import trajectory
    monkeypatch.setattr(trajectory, 'TrajectoryRecorder',
                        lambda *a, **kw: SimpleNamespace(finalize=finalize))
    monkeypatch.setattr(module, 'run_physical_insert_bowl', failed)
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'),
        selection='load_bowl', record_video=True)
    assert result['failure']['stage'] == 'insert'
    assert result['failure']['code'] == 'linear_path_failed'
    assert result['failure']['report_ref'] == 'pick_insert/physical_insert/insert_result.json'
    assert 'video finalization failed' in result['steps'][-1]['exception']


def test_same_session_extract_then_current_place_uses_retreat_handoff(tmp_path, monkeypatch):
    calls = _fake_task(tmp_path, monkeypatch)
    session = module.MicrowaveSimulationSession(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    extracted = session.run(selection='extract_bowl')
    count = len(calls)
    assert extracted['operation_success']
    placed = session.run(selection='place_bowl_on_table', preparation='current')
    assert placed['operation_success']
    assert [name for name, _ in calls[count:]] == ['place']
    assert placed['shared_flow']['stages'] == ['place_bowl_on_table']
    assert placed['actual_final']['state_id'] != extracted['actual_final']['state_id']


@pytest.mark.parametrize('selection,last', [
    ('open_door','open'), ('close_door','close'), ('rotate_knob','knob'),
    ('pick_bowl','pick'), ('insert_bowl','insert'),
    ('extract_bowl','extract_retreat'), ('place_bowl_on_table','place'), ('unload_bowl','place')])
def test_independent_scope_stops_at_selected_operation(tmp_path, monkeypatch, selection, last):
    calls = _fake_task(tmp_path, monkeypatch)
    result = module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'), selection=selection)
    assert result['operation_success'] is True
    assert result['goal_success'] is None
    assert calls[-1][0] == last
    assert result['requested_scope'] == selection

@pytest.mark.parametrize('selection,scope,producer', [
    ('insert_bowl','release','run_physical_insert_bowl'),
    ('place_bowl_on_table','preplace','run_assisted_table_place'),
    ('rotate_knob','grasp','run_planned_knob_execution'),
])
def test_stage_stop_reaches_only_selected_producer_and_remains_partial(tmp_path, monkeypatch, selection, scope, producer):
    calls=_fake_task(tmp_path, monkeypatch)
    original=getattr(module,producer)
    received=[]
    def paused(*a,stop_after=None,**kw):
        received.append(stop_after)
        result=original(*a,**kw)
        result.update(operation_success=True, goal_success=None, physical_place_success=None,
                      physical_table_place_success=None)
        return result
    monkeypatch.setattr(module,producer,paused)
    result=module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'),selection=selection,stop_after=scope)
    assert received == [scope]
    assert result['status'] == 'partial' and result['operation_success']
    assert result['task_success'] is None and result['goal_success'] is None
    assert result['requested_scope'] == f'{selection}/{scope}'
    assert result['steps'][-1]['step_id'] == selection
    assert result['actual_final']['state_id'] == f'actual:{float(len(calls))}'


def test_extract_grasp_pause_never_attaches_or_retreats(tmp_path, monkeypatch):
    calls=_fake_task(tmp_path, monkeypatch)
    result=module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'),selection='extract_bowl',stop_after='grasp')
    assert result['operation_success'] and result['status'] == 'partial'
    assert calls[-1][0] == 'extract_grasp'
    assert 'extract_retreat' not in [name for name,t in calls]


def test_second_open_preserves_nested_execution_evidence(tmp_path, monkeypatch):
    _fake_task(tmp_path,monkeypatch)
    original=module._continue_open
    def reopen(*a):
        r=original(*a)
        r['report']['shared_flow']={'execution_status':'executed'}
        return r
    monkeypatch.setattr(module,'_continue_open',reopen)
    result=module.run_microwave_task1_physical(tmp_path/'run',
        config=Path('data/task_1/simulation/mujoco_task1_physical.json'))
    row=next(r for r in result['shared_flow']['stage_outcomes'] if r['stage']=='open_door_second')
    assert row['execution_status']=='executed'
