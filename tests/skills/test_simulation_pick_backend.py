import numpy as np
import pytest


@pytest.mark.parametrize('contact_ok', [True, False])
def test_simulation_pick_uses_shared_flow_and_only_attaches_after_contact(tmp_path, contact_ok):
    from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
    class Runtime:
        robot_metadata = {'grippers': {'left': {'mode': 'fixture'}}}
        def __init__(self): self.version = 0; self.commands = []
        def query(self, **kwargs):
            return {'state_id': str(self.version), 'robots': {'left': {'T_world_tcp': np.eye(4).tolist()}},
                    'objects': {'brush': {'T_world_object': np.eye(4).tolist()}}}
        def execute_gripper_command(self, **kwargs):
            assert kwargs['planned_state_id'] == str(self.version)
            self.version += 1; self.commands.append(kwargs['width_m'])
            return {'success': True}
    runtime = Runtime(); motions = []; attachments = []
    def motion(stage, target, state_id):
        assert state_id == str(runtime.version)
        runtime.version += 1; motions.append(stage)
        return {'success': True}
    result = run_pick_skill(runtime, tmp_path,
        candidate={'candidate_id': 'fixture', 'T_world_tcp': np.eye(4)}, motion_stage=motion,
        object_id='brush', group='left', diagnostics=lambda: {'contacts': []},
        evidence_writer=lambda *a: None,
        contact_classifier=lambda *a, **k: {'success': contact_ok},
        after_contact_verified=lambda c: attachments.append(c) or {'enabled': True})
    assert result['success'] == contact_ok
    assert bool(attachments) == contact_ok
    assert result['shared_flow']['stages'][:5] == ['open', 'pregrasp', 'approach', 'close', 'grasp_check']
    assert motions == ['pregrasp', 'approach']
    if not contact_ok:
        assert result['shared_flow']['failure_code'] == 'grasp_contact_rejected'
        assert result['failure']['report_ref'] == 'diagnostics_closed.json'


@pytest.mark.parametrize('offset_xy', [0., .01])
def test_lift_then_hold_uses_measured_object_displacement(tmp_path, offset_xy):
    from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
    class Runtime:
        robot_metadata = {'grippers': {'left': {'mode': 'fixture'}}}
        version = 0
        z = 0.
        x = 0.
        def query(self, **kwargs):
            pose = np.eye(4); pose[2, 3] = self.z
            pose[0, 3] = self.x
            return {'state_id': str(self.version), 'robots': {'left': {'T_world_tcp': pose.tolist()}},
                    'objects': {'brush': {'T_world_object': pose.tolist()}}}
        def execute_gripper_command(self, **kwargs):
            self.version += 1
            return {'success': True}
        def execute_hold(self, **kwargs):
            assert kwargs['planned_state_id'] == str(self.version)
            self.version += 1
            return {'success': True}
    runtime = Runtime()
    def motion(stage, target, state_id):
        assert state_id == str(runtime.version)
        runtime.version += 1
        if stage == 'lift':
            runtime.z = float(target[2, 3]); runtime.x = float(target[0, 3])
        return {'success': True}
    r = run_pick_skill(runtime, tmp_path, candidate={'candidate_id': 'fixture', 'T_world_tcp': np.eye(4)},
        motion_stage=motion, object_id='brush', group='left', lift_m=.045,
        diagnostics=lambda: {'contacts': []}, evidence_writer=lambda *a: None,
        contact_classifier=lambda *a, **k: {'success': True},
        lift_target_builder=lambda actual: actual + np.array([[0, 0, 0, offset_xy],
            [0, 0, 0, 0], [0, 0, 0, .045], [0, 0, 0, 0]]))
    assert r['physical_pick_success']
    assert r['lift_height_m'] == pytest.approx(.045)
    assert r['hold_relative_translation_drift_m'] == pytest.approx(0.)
    assert r['shared_flow']['stages'][-2:] == ['lift', 'hold_check']
    assert runtime.x == pytest.approx(offset_xy)


def test_pick_forwards_stage_gain_to_hold_without_changing_hold_contract(tmp_path):
    from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
    class Runtime:
        robot_metadata = {'grippers': {'left': {'mode': 'fixture'}}}
        version = 0
        z = 0.
        hold_gain = None
        def query(self, **kwargs):
            pose = np.eye(4); pose[2, 3] = self.z
            return {'state_id': str(self.version),
                    'robots': {'left': {'T_world_tcp': pose.tolist()}},
                    'objects': {'brush': {'T_world_object': pose.tolist()}}}
        def execute_gripper_command(self, **kwargs):
            self.version += 1
            return {'success': True}
        def execute_hold(self, **kwargs):
            assert kwargs['planned_state_id'] == str(self.version)
            self.hold_gain = kwargs.get('position_kp_scale')
            self.version += 1
            return {'success': True}
    runtime = Runtime()
    def motion(stage, target, state_id):
        assert state_id == str(runtime.version)
        runtime.version += 1
        if stage == 'lift': runtime.z = float(target[2, 3])
        return {'success': True}
    result = run_pick_skill(runtime, tmp_path,
        candidate={'candidate_id': 'fixture', 'T_world_tcp': np.eye(4)},
        motion_stage=motion, object_id='brush', group='left', lift_m=.08,
        diagnostics=lambda: {'contacts': []}, evidence_writer=lambda *a: None,
        contact_classifier=lambda *a, **k: {'success': True},
        hold_position_kp_scale=2.5)
    assert result['success']
    assert runtime.hold_gain == pytest.approx(2.5)


def test_pick_forwards_arm_hold_gain_only_to_close_command(tmp_path):
    from gene_agent.simulation.pick_backend import run_simulation_pick as run_pick_skill
    class Runtime:
        robot_metadata = {'grippers': {'left': {'mode': 'fixture'}}}
        version = 0
        gripper_gains = []
        def query(self, **kwargs):
            return {'state_id': str(self.version),
                    'robots': {'left': {'T_world_tcp': np.eye(4).tolist()}},
                    'objects': {'brush': {'T_world_object': np.eye(4).tolist()}}}
        def execute_gripper_command(self, **kwargs):
            self.gripper_gains.append(kwargs.get('position_kp_scale'))
            self.version += 1
            return {'success': True}
    runtime = Runtime()
    def motion(*_args):
        runtime.version += 1
        return {'success': True}
    result = run_pick_skill(runtime, tmp_path,
        candidate={'candidate_id': 'fixture', 'T_world_tcp': np.eye(4)},
        motion_stage=motion, object_id='brush', group='left',
        diagnostics=lambda: {'contacts': []}, evidence_writer=lambda *a: None,
        contact_classifier=lambda *a, **k: {'success': True},
        close_position_kp_scale=2.5)
    assert result['success']
    assert runtime.gripper_gains[0] is None
    assert runtime.gripper_gains[1] == pytest.approx(2.5)


@pytest.mark.parametrize('scope', ['pregrasp', 'approach', 'grasp'])
def test_bound_backend_stops_without_opening_or_attaching_after_scope(tmp_path, scope):
    from gene_agent.simulation.pick_backend import make_pick_backend
    from gene_agent.skills.library import pick
    from gene_agent.skills.library.pick.model import GraspSettings
    class Runtime:
        robot_metadata = {'grippers': {'right': {'mode': 'fixture'}}}
        def __init__(self): self.version = 0; self.commands = []
        def query(self, **kwargs):
            return {'state_id': str(self.version), 'robots': {'right': {'T_world_tcp': np.eye(4).tolist()}},
                    'objects': {'bowl': {'T_world_object': np.eye(4).tolist()}}}
        def execute_gripper_command(self, **kwargs):
            self.commands.append(kwargs['width_m']); self.version += 1
            return {'success': True}
    runtime = Runtime(); motions = []; attached = []
    chosen = np.eye(4); chosen[0, 3] = .025
    def motion(name, target, state_id):
        assert state_id == str(runtime.version)
        motions.append((name, target.copy())); runtime.version += 1
        return {'success': True}
    request = pick.PickRequest('bowl', 'right', GraspSettings(), lift=True)
    b = make_pick_backend(runtime, tmp_path, request=request,
        candidate={'candidate_id': 'selected', 'T_world_tcp': np.eye(4)},
        selected_pregrasp_pose=chosen, motion_stage=motion,
        diagnostics=lambda: {'contacts': []}, evidence_writer=lambda *a: None,
        contact_classifier=lambda *a, **k: {'success': True},
        after_contact_verified=lambda *a: attached.append(True))
    assert runtime.commands == []  # 装配不执行动作
    result = pick.run(b, request, stop_after=scope)
    assert result.success
    assert b.report['requested_scope'] == scope
    assert b.report['physical_pick_success'] is False
    assert runtime.commands == ([.08, 0.] if scope == 'grasp' else [.08])
    assert attached == []
    np.testing.assert_array_equal(motions[0][1], chosen)
    assert b.report['actual_final']['state_id'] == str(runtime.version)


def test_resume_rejects_missing_hold_before_any_lift():
    from gene_agent.skills.library import pick
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationState
    from gene_agent.tasks.checks import CheckResult
    calls = []
    b = CallbackOperationBackend(observe=lambda: OperationState('held?', 'mujoco', {}),
        perform=lambda n: calls.append(n) or {'success': False,
            'check': CheckResult('unknown', 'contact', {})}, evidence_mode='mujoco')
    b.request = pick.PickRequest('bowl', 'right')
    result = pick.resume(b, b.request)
    assert not result.success
    assert calls == ['grasp_check']


@pytest.mark.parametrize('motion_result,code,message', [
    ({'success': True, 'plan': {'success': True}, 'execution': None,
      'evidence_mode': 'planning_only'}, 'motion_not_executed', '实际执行'),
    ({'success': False, 'plan': {'success': True, 'status': 'success'},
      'execution': {'success': False, 'failure_type': 'endpoint_timeout',
                    'error': 'joint tracking timeout'}}, 'endpoint_timeout', 'joint tracking timeout'),
])
def test_planning_preview_and_execution_errors_never_become_actual_success(tmp_path, motion_result, code, message):
    from gene_agent.simulation.pick_backend import run_simulation_pick
    class Runtime:
        def __init__(self): self.commands = []
        def query(self, **kwargs):
            return {'state_id': 'actual', 'robots': {'left': {'T_world_tcp': np.eye(4).tolist()}},
                    'objects': {'brush': {'T_world_object': np.eye(4).tolist()}}}
        def execute_gripper_command(self, **kwargs):
            self.commands.append(kwargs['width_m'])
            return {'success': True}
    r = Runtime()
    report = run_simulation_pick(r, tmp_path,
        candidate={'candidate_id': 'chosen', 'T_world_tcp': np.eye(4)},
        motion_stage=lambda *a: motion_result, object_id='brush', group='left',
        diagnostics=lambda: {}, evidence_writer=lambda *a: None)
    assert not report['success']
    assert r.commands == [.08]
    assert report['shared_flow']['failure_code'] == code
    assert message in report['shared_flow']['failure']['message']


def test_adapter_robot_group_mismatch_is_rejected_before_scene_access(tmp_path):
    from gene_agent.simulation.pick_backend import make_pick_backend
    from gene_agent.skills.library import pick
    class Bridge:
        group = 'left'
        def __call__(self, *args):
            raise AssertionError('不能向错误手臂发令')
    with pytest.raises(ValueError, match='机器人组'):
        make_pick_backend(object(), tmp_path, request=pick.PickRequest('bowl','right'),
            candidate={'candidate_id':'chosen','T_world_tcp':np.eye(4)}, motion_stage=Bridge(),
            diagnostics=lambda: {}, evidence_writer=lambda *a: None)
    assert not (tmp_path/'result.json').exists()


def test_gripper_failure_preserves_original_error_before_arm_motion(tmp_path):
    from gene_agent.simulation.pick_backend import run_simulation_pick
    class Runtime:
        def query(self, **kwargs):
            return {'state_id':'actual', 'objects':{}, 'robots':{}}
        def execute_gripper_command(self, **kwargs):
            return {'success':False, 'failure_type':'gripper_timeout', 'error':'gripper stalled'}
    report = run_simulation_pick(Runtime(), tmp_path,
        candidate={'candidate_id':'c','T_world_tcp':np.eye(4)},
        object_id='bowl', group='right', diagnostics=lambda: {}, evidence_writer=lambda *a: None,
        motion_stage=lambda *a: pytest.fail('夹爪失败后不应移动机械臂'))
    assert report['shared_flow']['failure_code'] == 'gripper_timeout'
    assert report['failure']['message'] == 'gripper stalled'
