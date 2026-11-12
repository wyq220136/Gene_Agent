"""能力库公开入口必须能限定范围，并保留闭环检查。"""
import pytest

from gene_agent.skills.contracts import OperationState, StageOutcome, CheckResult


class Backend:
    evidence_mode = 'mujoco'
    def __init__(self, failed=None): self.names = []; self.failed = failed
    def observe(self): return OperationState(str(len(self.names)), 'mujoco', {})
    def execute(self, stage, start):
        self.names.append(stage.name)
        check = CheckResult('unknown' if stage.name == self.failed else 'pass', 'measured', {})
        return StageOutcome(stage.name, True, start, self.observe(), check)


@pytest.mark.parametrize('operation', ['insert', 'place'])
def test_transfer_debug_prefix_never_releases_and_full_requires_final(operation):
    from gene_agent.skills.library import transfer
    b = Backend()
    result = getattr(transfer, operation)(b, stop_after=operation)
    assert result.success
    assert b.names[-1] == operation
    assert 'release' not in b.names
    assert result.skill == operation and result.requested_scope == operation
    b = Backend('final_check')
    result = getattr(transfer, operation)(b)
    assert not result.success and result.failure_stage == 'final_check'


def test_extract_grasp_prefix_preserves_actual_end_and_never_attaches():
    from gene_agent.skills.library import transfer
    b = Backend()
    result = transfer.extract(b, stop_after='grasp')
    assert result.success and b.names[-1] == 'grasp_check'
    assert 'attach' not in b.names
    assert result.requested_scope == 'grasp'
    assert result.end_state.state_id == str(len(b.names))


def test_knob_grasp_scope_never_rotates_or_releases():
    from gene_agent.skills.library import knob
    b = Backend()
    result = knob.run(b, target_names=lambda: [
        'open','pregrasp','approach','close','rotate_45','release','retreat'], stop_after='grasp')
    assert result.success and b.names[-1] == 'grasp_check'
    assert not any(n.startswith('rotate_') for n in b.names)
    assert result.skill == 'knob' and result.requested_scope == 'grasp'


@pytest.mark.parametrize('operation', ['insert','place','extract'])
def test_invalid_transfer_scope_is_rejected_before_any_action(operation):
    from gene_agent.skills.library import transfer
    b = Backend()
    with pytest.raises(ValueError): getattr(transfer, operation)(b, stop_after='typo')
    assert b.names == []


def test_door_run_fails_on_unknown_check_before_next_phase():
    from gene_agent.skills.library import door
    from gene_agent.skills.contracts import OperationStage
    b = Backend('pull')
    result = door.run(b, stages=[OperationStage('pull', True),OperationStage('release')])
    assert not result.success and b.names == ['pull']
    assert result.skill == 'door'


def test_sweep_contact_roles_are_supplied_by_adapter():
    import numpy as np
    from gene_agent.skills.library import sweep
    from gene_agent.knowledge.sweep import SweepLayout
    from types import SimpleNamespace
    layout = SimpleNamespace(brush_start_world_xyz=np.array([0.,0.,.1]),
                             brush_end_world_xyz=np.array([.1,0.,.1]))
    contacts = sweep.SweepContacts(tool='squeegee', support='bench', receiver='bin',
                                   gripper_links=('finger_a','finger_b'))
    calls = []
    def move(name, target, kind, duration, *, allowed_contacts):
        calls.append((name, allowed_contacts))
        return {'success':True}
    result = sweep.run(layout, contacts=contacts, initial_brush_z=.1,
        move_stage=move, target_for_brush=lambda xyz:np.eye(4),
        current_brush_xyz=lambda:np.array([.1,0.,.1]), sweep_targets=('block',),
        stop_after='lower_to_contact')
    assert result['success'] and len(calls) == 2
    assert ('finger_a','squeegee') in calls[0][1]
    assert ('squeegee','bench') in calls[1][1]
    assert all('brush' not in pair and 'pika_gripper_left_link' not in pair
               for _,pairs in calls for pair in pairs)
    assert result['requested_scope'] == 'lower_to_contact'


@pytest.mark.parametrize('evidence_mode,success,executed', [
    ('execution', False, False), ('prediction', True, False)])
def test_sweep_never_claims_planned_motion_was_executed(evidence_mode, success, executed):
    import numpy as np
    from types import SimpleNamespace
    from gene_agent.skills.library import sweep
    calls = []
    result = sweep.run(SimpleNamespace(brush_start_world_xyz=np.zeros(3),
        brush_end_world_xyz=np.ones(3)),
        contacts=sweep.SweepContacts('tool','bench','bin',('a','b')),
        initial_brush_z=0., evidence_mode=evidence_mode,
        move_stage=lambda name,*a,**k: calls.append(name) or {
            'success':True,'execution':None,'evidence_mode':'planning_only'},
        target_for_brush=lambda xyz:np.eye(4), current_brush_xyz=lambda:np.zeros(3))
    assert result['success'] is success
    assert result['executed'] is executed
    if evidence_mode == 'execution':
        assert calls == ['move_behind_cube']
        assert result['failure']['code'] == 'motion_not_executed'


def test_transfer_reuses_adapter_failure_without_another_verification():
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.library import transfer
    calls = []
    failure = {'code':'endpoint_timeout','source':'sdk','message':'跟踪超时','report_ref':'motion.json'}
    b = CallbackOperationBackend(observe=lambda:OperationState('actual','realworld',{}),
        perform=lambda n:calls.append(n) or {'success':False}, evidence_mode='realworld',
        failure_details=lambda stage:failure)
    result = transfer.insert(b)
    assert result.failure == failure and result.failure_code == 'endpoint_timeout'
    assert calls == ['preplace']


def test_sweep_scene_adapter_can_stop_a_single_skill_without_scoring_task(tmp_path):
    import numpy as np
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    obj = object.__new__(SweepCubeRun)
    calls = []
    obj.scene = {}; obj.output_dir = tmp_path
    obj.report = {'failure_stage':None,'stages':{}}
    obj.pick = lambda: calls.append('pick') or True
    obj.observe = lambda:{'state_id':'actual','objects':{'brush':{'T_world_object':np.eye(4).tolist()}}}
    obj.begin_sweeping = lambda rotation,order:calls.append('begin')
    obj.sweep = lambda target,label,**kwargs:calls.append((target,kwargs['stop_after'])) or True
    obj._render = lambda name:None
    result = obj.run(selection='sweep', target_name='cube_blue', stop_after='lower_to_contact')
    assert calls == ['pick','begin',('cube_blue','lower_to_contact')]
    assert result['operation_success']
    assert result['requested_scope'] == 'sweep/lower_to_contact'
    assert result['task_success'] is None
    assert result['actual_final']['state_id'] == 'actual'


def test_sweep_first_planning_failure_is_not_execution():
    import numpy as np
    from types import SimpleNamespace
    from gene_agent.skills.library import sweep
    result = sweep.run(SimpleNamespace(brush_start_world_xyz=np.zeros(3),
        brush_end_world_xyz=np.ones(3)), contacts=sweep.SweepContacts('tool','table','bin',('a','b')),
        initial_brush_z=0., move_stage=lambda *a,**k:{'success':False,
            'failure_type':'ik_failed','plan':{'success':False}},
        target_for_brush=lambda xyz:np.eye(4), current_brush_xyz=lambda:np.zeros(3))
    assert not result['success']
    assert not result['executed']


@pytest.mark.parametrize('row,executed', [
    ({'success':True,'executed':False}, False),
    ({'success':False,'execution':{'success':False}}, True),
    ({'success':False,'execution':{},'executed':False}, False),
])
def test_sweep_preserves_explicit_execution_evidence(row, executed):
    import numpy as np
    from types import SimpleNamespace
    from gene_agent.skills.library import sweep
    result = sweep.run(SimpleNamespace(brush_start_world_xyz=np.zeros(3),
        brush_end_world_xyz=np.ones(3)), contacts=sweep.SweepContacts('tool','table','bin',('a','b')),
        initial_brush_z=0., move_stage=lambda *a,**k:row,
        target_for_brush=lambda xyz:np.eye(4), current_brush_xyz=lambda:np.zeros(3))
    assert not result['success']
    assert result['executed'] is executed
