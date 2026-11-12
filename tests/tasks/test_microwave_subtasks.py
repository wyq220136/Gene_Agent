"""选择子任务时保留原 Task 次序、必要过渡及失败检查。"""
import pytest
from gene_agent.tasks.policies.microwave_task import policy


def resolve(name, **kwargs):
    from gene_agent.tasks.policies.microwave_task.subtasks import resolve_subtask
    return resolve_subtask(name, **kwargs)


def test_load_has_pick_and_insert_and_prefix_keeps_door_transition():
    plan = resolve('load_bowl')
    assert plan.target_steps == ('pick_bowl', 'insert_bowl')
    assert plan.prefix_steps == ('open_door_first', 'after_open_transition')
    assert plan.steps('current') == plan.target_steps
    assert plan.steps('prefix') == plan.prefix_steps + plan.target_steps


def test_unload_keeps_original_second_open_prefix_and_scope():
    plan = resolve('unload_bowl', policy=policy.MicrowaveTaskPolicy(before_reopen_transition=False))
    assert plan.target_steps == ('extract_bowl', 'place_bowl_on_table')
    assert plan.prefix_steps[-2:] == ('open_door_second', 'after_reopen_transition')
    assert 'before_close_first_transition' in plan.prefix_steps
    assert 'before_reopen_transition' not in plan.prefix_steps
    assert 'close_door_second' not in plan.steps('prefix')


def test_standalone_insert_does_not_implicitly_pick_in_current_mode():
    plan = resolve('insert_bowl')
    assert plan.steps('current') == ('insert_bowl',)
    assert plan.prefix_steps[-1] == 'pick_bowl'


def test_all_preserves_exact_existing_sequence():
    p = policy.MicrowaveTaskPolicy(include_knob=False)
    plan = resolve('all', policy=p)
    assert plan.target_steps == tuple(s.name for s in policy.microwave_task_steps(p))
    assert plan.prefix_steps == ()
    with pytest.raises(ValueError, match='旋钮'):
        resolve('rotate_knob', policy=p)


@pytest.mark.parametrize('name,preparation', [('typo','prefix'), ('load_bowl','typo')])
def test_invalid_scope_rejected(name, preparation):
    with pytest.raises(ValueError):
        resolve(name).steps(preparation)


def test_selected_flow_uses_existing_checks_and_actual_end():
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationState
    from gene_agent.tasks.checks import CheckResult
    state = OperationState('start', 'test_actual', {})
    calls = []
    def perform(name):
        nonlocal state
        calls.append(name)
        state = OperationState(name, 'test_actual', {})
        return {'success': True, 'check': CheckResult('fail', 'test', {})}
    backend = CallbackOperationBackend(observe=lambda: state, perform=perform, evidence_mode='actual')
    result = policy.run_microwave_task1_flow(backend, selection='load_bowl', preparation='current')
    assert not result.success
    assert calls == ['pick_bowl']
    assert result.end_state.state_id == 'pick_bowl'


def test_video_of_successful_subtask_does_not_claim_full_task_success(tmp_path):
    import json
    from gene_agent.visualization.video_export import _evidence
    (tmp_path/'summary.json').write_text(json.dumps({
        'task': 'microwave_task1', 'goal_success': None, 'operation_success': True,
        'requested_scope': 'load_bowl', 'evidence_mode': 'simulation_assisted'}))
    result = _evidence(tmp_path, 'actual')
    assert result['task_success'] is None
    assert result['operation_success'] is True
    assert result['requested_scope'] == 'load_bowl'


def test_full_video_respects_failed_final_benchmark_after_successful_operations(tmp_path):
    import json
    from gene_agent.visualization.video_export import _evidence
    (tmp_path/'summary.json').write_text(json.dumps({
        'task':'microwave_task1', 'requested_scope':'all',
        'goal_success':False, 'task_success':False, 'operation_success':True}))
    result = _evidence(tmp_path, 'actual')
    assert result['operation_success'] is True
    assert result['task_success'] is False


def test_shared_insert_adjustment_follows_microwave_y_and_keeps_inputs():
    """仿真/真机的深度及撤离修正应使用同一规则，不能各自组合一次。"""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from gene_agent.tasks.policies.microwave_task.operations.bowl import (
        FixedPlaceTargets, adjust_insert_targets,
    )
    poses = [np.eye(4) for _ in range(5)]
    for i, pose in enumerate(poses):
        pose[:3, 3] = [i * .01, .3, .1]
    geometry = FixedPlaceTargets(*poses)
    microwave = np.eye(4)
    microwave[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    targets, goal = adjust_insert_targets(geometry, microwave,
        depth_offset_m=.02, retreat_rotation_local_deg=(0., -10., 0.))
    for name, original in zip(('preplace', 'insert', 'retreat_1', 'retreat_2'), poses):
        np.testing.assert_allclose(targets[name][:3, 3], original[:3, 3] + [-.02, 0, 0])
    np.testing.assert_allclose(targets['insert'][:3, :3], np.eye(3))
    np.testing.assert_allclose(targets['retreat_2'][:3, :3],
        Rotation.from_euler('y', -10, degrees=True).as_matrix())
    np.testing.assert_allclose(goal[:3, 3], poses[4][:3, 3] + [-.02, 0, 0])
    assert poses[0][0, 3] == 0.  # 不改写 profile/上一轮目标


@pytest.mark.parametrize('angle,expected', [(90., 'pass'), (30., 'fail'), (None, 'unknown'), (float('nan'), 'unknown')])
def test_shared_bowl_access_uses_current_measured_angle(angle, expected):
    import numpy as np
    from gene_agent.tasks.policies.microwave_task.operations.bowl import check_bowl_access
    result = check_bowl_access(None if angle is None else np.deg2rad(angle),
                              target_angle_rad=np.pi/2)
    assert result.status == expected
    assert bool(result.source)


@pytest.mark.parametrize('yaw_deg', [0., 60.])
def test_observed_insert_matches_shared_targets_and_rejects_stale_hold(tmp_path, yaw_deg):
    """真机适配使用与仿真相同目标规则；新观测仍须重新确认夹持。"""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from gene_agent.tasks.policies.microwave_task.operations.bowl import (
        fixed_place_targets, load_object_operation_profile, adjust_insert_targets,
    )
    from gene_agent.realworld.microwave.insert_planning import build_observed_bowl_insert_targets
    xml = tmp_path/'fk.xml'
    xml.write_text('<mujoco><worldbody><site name="tcp" pos=".1 .2 .3"/></worldbody></mujoco>')
    microwave = np.eye(4)
    microwave[:3, :3] = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
    microwave[:3, 3] = [.4, .1, .2]
    tcp = np.eye(4); tcp[:3, 3] = [.1, .2, .3]
    bowl = tcp.copy(); bowl[2, 3] += .05
    scene = {'scene_xml_path': str(xml), 'robot_metadata': {'tcp_sites': {'right':'tcp'}},
             'snapshot': {'snapshot_sha256':'now', 'active_group':'right', 'qpos_snapshot':[]},
             'provenance': {'physics_stepped':False, 'pose_frame_verified':True,
                 'collision_snapshot_sha256':'now', 'source_frame_number':10,
                 'source_timestamp_ms':100., 'aligned_object_poses_world': {
                     'Microwave/Microwave_0':microwave, 'Bowl/Bowl_0':bowl}}}
    picked = {'execute':True, 'status':'executed', 'pick_success':True,
              'post_lift_scene_snapshot_sha256':'now', 'post_lift_source_frame_number':10,
              'post_lift_source_timestamp_ms':100.,
              'postcondition': {'status':'pass', 'source':'vision'}}
    alignment = {'accepted_for_planning':True, 'source':'test_calibration',
                 'T_scene_asset_profile':np.eye(4)}
    observed = build_observed_bowl_insert_targets(scene, picked,
        microwave_frame_alignment=alignment, insert_depth_offset_m=.02,
        retreat_rotation_local_deg=(0., -10., 0.))
    original = fixed_place_targets(microwave,
        load_object_operation_profile('data/task_1','Bowl_0'), np.linalg.inv(tcp)@bowl)
    expected, goal = adjust_insert_targets(original, microwave,
        depth_offset_m=.02, retreat_rotation_local_deg=(0., -10., 0.))
    for name, pose in expected.items():
        np.testing.assert_allclose(observed['keyposes_world_tcp'][name], pose, atol=1e-12)
    np.testing.assert_allclose(observed['desired_placed_object_world'], goal, atol=1e-12)
    scene['provenance'].update(source_frame_number=11, source_timestamp_ms=101.)
    with pytest.raises(ValueError, match='重新确认'):
        build_observed_bowl_insert_targets(scene, picked, microwave_frame_alignment=alignment)


def test_policy_identity_tracks_nested_operation_changes(tmp_path, monkeypatch):
    """拆分之后，运行身份仍必须覆盖真正执行的操作文件。"""
    from gene_agent.tasks import policies
    package = tmp_path/'policies'
    operation = package/'demo_task/operations/bowl.py'
    operation.parent.mkdir(parents=True)
    (package/'demo_task/policy.py').write_text('def run(): pass\n')
    operation.write_text('VALUE = 1\n')
    monkeypatch.setattr(policies, '__file__', str(package/'__init__.py'))
    before = policies.identity('demo_task')['sources_sha256']
    assert 'demo_task/operations/bowl.py' in before
    operation.write_text('VALUE = 2\n')
    after = policies.identity('demo_task')['sources_sha256']
    assert before['demo_task/operations/bowl.py'] != after['demo_task/operations/bowl.py']


@pytest.mark.parametrize('knob,reopen', [(True, True), (False, True), (True, False)])
def test_readable_steps_preserve_named_operations_and_transitions(knob, reopen):
    config = policy.MicrowaveTaskPolicy(include_knob=knob, before_reopen_transition=reopen)
    expected = ['open_door_first', 'after_open_transition', 'pick_bowl', 'insert_bowl',
                'before_close_first_transition', 'close_door_first']
    if knob:
        expected += ['rotate_knob']
    if reopen:
        expected += ['before_reopen_transition']
    expected += ['open_door_second', 'after_reopen_transition', 'extract_bowl',
                 'place_bowl_on_table', 'before_close_second_transition', 'close_door_second']
    assert [step.name for step in policy.iter_task_steps(config)] == expected
    assert tuple(policy.iter_task_steps(config)) == policy.microwave_task_steps(config)


@pytest.mark.parametrize('failure', ['execution', 'unknown'])
def test_partial_insert_stops_full_task_and_returns_actual_state(failure):
    from gene_agent.skills.runtime.operation import CallbackOperationBackend
    from gene_agent.skills.contracts import OperationState
    from gene_agent.tasks.checks import CheckResult
    state = OperationState('start', 'test_actual', {})
    calls = []
    def perform(name):
        nonlocal state
        calls.append(name)
        state = OperationState(name + ':actual', 'test_actual', {'partial': name == 'insert_bowl'})
        failed = name == 'insert_bowl'
        return {'success': not (failed and failure == 'execution'),
                'check': CheckResult('unknown' if failed else 'pass', 'test_actual', {})}
    backend = CallbackOperationBackend(observe=lambda: state, perform=perform, evidence_mode='actual')
    result = policy.run_microwave_task1_flow(backend)
    assert not result.success
    assert calls == ['open_door_first', 'after_open_transition', 'pick_bowl', 'insert_bowl']
    assert result.failure_stage == 'insert_bowl'
    assert result.end_state.state_id == 'insert_bowl:actual'
    assert result.end_state.values['partial']

@pytest.mark.parametrize('selection,stage', [('all','release'),('load_bowl','release'),('insert_bowl','typo'),('rotate_knob','insert')])
def test_stage_stop_rejected_before_scene_creation(selection, stage, tmp_path, monkeypatch):
    import gene_agent.tasks.policies.microwave_task.simulation as m
    monkeypatch.setattr(m,'MicrowaveSimulationSession',lambda *a,**k:pytest.fail('不得构建场景'))
    with pytest.raises(ValueError,match='停止'):
        m.run_microwave_task1_physical(tmp_path,config={},selection=selection,stop_after=stage)
