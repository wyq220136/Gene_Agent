"""离线后端替身核对旋钮流程、运动学许可与实际失败；不验证物理效果。"""
from types import SimpleNamespace

import numpy as np
import pytest

import gene_agent.tasks.policies.microwave_task.simulation_knob as module


def _run(tmp_path, monkeypatch, *, contact_turn=False, fail_action=None, raise_action=None, stop_after=None):
    actions = []
    model = SimpleNamespace(joint=lambda name: SimpleNamespace(
        qposadr=[0 if name == 'knob' else 1]))
    data = SimpleNamespace(qpos=np.zeros(2))
    def execute(action, **kwargs):
        actions.append(action)
        if action == 'trajectory' and contact_turn:
            data.qpos[0] = .2
        if action == raise_action:
            data.qpos[0] = .073
            raise RuntimeError('partial movement interrupted')
        return {'success': action != fail_action, 'sim_duration_s': .1}
    runtime = SimpleNamespace(
        model=model, data=data,
        execute_gripper_command=lambda **kw: execute('gripper', **kw),
        execute_hold=lambda **kw: execute('wait', **kw),
        execute_joint_trajectory=lambda **kw: execute('trajectory', **kw))
    monkeypatch.setattr(module, 'state_id', lambda model, data: f'{len(actions)}:{data.qpos.tolist()}')
    monkeypatch.setattr(module.mujoco, 'mj_forward', lambda model, data: None)
    names = ['open', 'pregrasp', 'approach', 'close', 'grasp_settle',
             'rotate_11', 'release', 'release_settle', 'retreat']
    stages = []
    for name in names:
        action = ('gripper_' + name if name in {'open', 'close', 'release'} else
                  'wait' if name.endswith('_settle') else 'linear_tcp')
        stages.append({'name': name, 'action': action, 'gripper_width_m': .04,
                       'visual_angle_commit': name.startswith('rotate_'),
                       'result': {'trajectory': {'joint_names': ['j1'],
                                                'times_s': [0., 1.],
                                                'positions_rad': [[0.], [.1]]}}})
    planned = {'success': True, 'source_format': 'kinematic_plan',
               'input_qpos': [0., 0.], 'joint_name': 'knob',
               'joint_target_angle_rad': .2, 'robot_group': 'right', 'stages': stages}
    return module.run_planned_knob_execution(runtime, planned, tmp_path / 'run', stop_after=stop_after), actions


def test_mixed_knob_retains_unknown_grasp_and_separate_kinematic_permission(tmp_path, monkeypatch):
    result, actions = _run(tmp_path, monkeypatch)
    assert result['goal_success'] and result['kinematic_commit_applied']
    assert not result['physical_rotation_success']
    checks = {s['stage']: s['check'] for s in result['shared_flow']['stage_outcomes']}
    assert checks['grasp_check']['status'] == 'unknown'
    assert checks['kinematic_eligibility_check']['status'] == 'pass'
    assert checks['kinematic_eligibility_check']['evidence']['physical_grasp_verified'] is False
    assert len(actions) == 9


def test_contact_rotation_keeps_original_physical_angle_evidence(tmp_path, monkeypatch):
    result, _ = _run(tmp_path, monkeypatch, contact_turn=True)
    assert result['goal_success'] and result['physical_rotation_success']
    assert not result['kinematic_commit_applied']


def test_motion_failure_stops_before_close_or_angle_commit(tmp_path, monkeypatch):
    result, actions = _run(tmp_path, monkeypatch, fail_action='trajectory')
    assert not result['goal_success'] and not result['execution_success']
    assert actions == ['gripper', 'trajectory']
    assert result['shared_flow']['failure_stage'] == 'pregrasp'
    assert result['shared_flow']['end_state']['state_id'].startswith('2:')
    assert not result['kinematic_commit_applied']


def test_partial_motion_exception_saves_failed_stage_and_actual_endpoint(tmp_path, monkeypatch):
    import json
    result, actions = _run(tmp_path, monkeypatch, raise_action='trajectory')
    assert actions == ['gripper', 'trajectory']
    assert not result['execution_success'] and not result['goal_success']
    assert result['failure_stage'] == 'rotate_knob/pregrasp'
    assert result['shared_flow']['failure_stage'] == 'pregrasp'
    assert result['shared_flow']['end_state']['values']['qpos'][0] == .073
    assert result['final_knob_angle_rad'] == .073
    assert 'partial movement interrupted' in result['first_failure']['traceback']
    assert result['shared_flow']['failure_code'] == 'stage_exception'
    saved = json.loads((tmp_path / 'run/result.json').read_text())
    rows = json.loads((tmp_path / 'run/stages.json').read_text())
    assert saved == result
    assert rows[-1]['stage_id'] == 'rotate_knob/pregrasp'
    assert rows[-1]['end_state_id'] == result['final_state_id']
    assert rows[-1]['execution']['success'] is False
    assert 'RuntimeError' in rows[-1]['traceback']


@pytest.mark.parametrize('stop_after,count', [('open',1),('grasp',4),('rotate',6)])
def test_knob_stage_pause_does_not_release_or_claim_complete(tmp_path, monkeypatch, stop_after, count):
    result, actions = _run(tmp_path, monkeypatch, stop_after=stop_after)
    assert len(actions) == count
    assert result['operation_success']
    assert result['goal_success'] is None
    assert result['failure_stage'] is None
    assert result['requested_scope'] == stop_after
    assert result['actual_final']['state_id'] == result['final_state_id']
