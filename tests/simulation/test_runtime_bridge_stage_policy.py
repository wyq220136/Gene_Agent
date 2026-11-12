"""在线规划必须在物理执行前应用阶段碰撞边界与安全复核。"""

import json

import numpy as np
import pytest


def test_bridge_rejects_unsafe_target_contact_before_actuator_execution(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module

    observed = {}

    class Runtime:
        model = object()
        data = object()
        robot_metadata = {'joint_groups': {'right': ['joint_1']}}

        def query(self, *, robot_groups):
            assert robot_groups == ('right',)
            return {'state_id': 'initial', 'robots': {'right': {
                'joint_names': ['joint_1'], 'joint_positions_rad': [0.],
            }}}

        def execute_joint_trajectory(self, **_kwargs):
            raise AssertionError('禁止接触未复核通过，不得驱动机械臂')

    def fake_export(_model, _data, _robot, _group, _output, **kwargs):
        observed['disabled'] = kwargs.get('disabled_body_names')
        return {'scene_model': {}, 'scope': 'target contact',
                'snapshot_sha256': 'contact-hash', 'gripper_snapshot': None,
                'mounting_contact_policy': {}}

    class FakePlanner:
        def __init__(self, **_kwargs):
            pass

        def plan_linear_tcp(self, *_args, **_kwargs):
            return {'success': True, 'status': 'success',
                    'joint_names': ['joint_1'], 'positions_rad': [[0.], [.1]],
                    'interpolation_dt_s': .1}

        def export_pre_execution_failure(
            self, positions, _target, *, first_invalid_waypoint,
            diagnostic_label, precheck,
        ):
            observed['failure_export'] = {
                'positions': positions,
                'first_invalid_waypoint': first_invalid_waypoint,
                'diagnostic_label': diagnostic_label,
                'precheck': precheck,
            }
            return {'trace_count': 1, 'kind': 'complete_precheck_path'}

    monkeypatch.setattr(bridge_module, 'export_collision_scene', fake_export)
    monkeypatch.setattr(bridge_module, 'load_realman_motion_config', lambda *_a, **_k: {})
    monkeypatch.setattr(bridge_module, 'SynchronousPlanner', FakePlanner)
    bridge = bridge_module.CuroboRuntimeBridge(
        Runtime(), tmp_path, linear_tcp_stages=('approach',),
        disabled_body_names_for_stage=lambda stage, _observation: ('bowl',)
            if stage == 'approach' else (),
        plan_postcheck_for_stage=lambda _stage, _plan, _snapshot: {
            'success': False, 'forbidden_contacts': [
                {'waypoint_index': 1, 'robot_geom_name': 'palm'}],
        },
    )
    result = bridge('approach', [[1., 0., 0., 0.], [0., 1., 0., 0.],
                                 [0., 0., 1., 0.], [0., 0., 0., 1.]], 'initial')

    assert observed['disabled'] == ('bowl',)
    assert result['success'] is False
    assert result['execution'] is None
    saved = json.loads((tmp_path / 'plans/approach.json').read_text(encoding='utf-8'))
    assert saved['pre_execution_check']['forbidden_contacts'][0]['robot_geom_name'] == 'palm'
    assert saved['failure_category'] == 'collision_postcheck_rejected'
    assert saved['failure_visualization']['kind'] == 'complete_precheck_path'
    assert observed['failure_export']['positions'] == [[0.], [.1]]
    assert observed['failure_export']['first_invalid_waypoint'] == 1


def test_bridge_forwards_live_collision_stop_and_exports_actual_stop_pose(
    tmp_path, monkeypatch,
):
    from gene_agent.motion.curobo import runtime_bridge as bridge_module

    observed = {}

    class Runtime:
        model = object()
        data = object()
        robot_metadata = {'joint_groups': {'right': ['joint_1']}}
        query_count = 0

        def query(self, *, robot_groups):
            assert robot_groups == ('right',)
            self.query_count += 1
            state_id = 'initial' if self.query_count == 1 else 'collision'
            position = 0. if self.query_count == 1 else .06
            return {'state_id': state_id, 'robots': {'right': {
                'joint_names': ['joint_1'], 'joint_positions_rad': [position],
            }}}

        def execute_joint_trajectory(self, **kwargs):
            observed['stop_condition_forwarded'] = callable(kwargs.get('stop_condition'))
            evidence = kwargs['stop_condition'](None, None, {'time_s': .2})
            observed['stop_evidence'] = evidence
            return {
                'success': False,
                'failure_type': 'forbidden_contact_during_execution',
                'failure_step_index': 3,
                'failure_evidence': evidence,
                'end_state_id': 'collision',
            }

    def fake_export(*_args, **_kwargs):
        return {'scene_model': {}, 'scope': 'world',
                'snapshot_sha256': 'hash', 'gripper_snapshot': None,
                'mounting_contact_policy': {}}

    class FakePlanner:
        def __init__(self, **_kwargs):
            pass

        def plan(self, *_args, **_kwargs):
            return {'success': True, 'status': 'success',
                    'joint_names': ['joint_1'], 'positions_rad': [[0.], [.1]],
                    'interpolation_dt_s': .1}

        def export_execution_collision(
            self, positions, _target, *, diagnostic_label, execution_failure,
        ):
            observed['collision_export'] = {
                'positions': positions,
                'diagnostic_label': diagnostic_label,
                'execution_failure': execution_failure,
            }
            return {'trace_count': 1, 'kind': 'actual_collision_stop'}

    monkeypatch.setattr(bridge_module, 'export_collision_scene', fake_export)
    monkeypatch.setattr(bridge_module, 'load_realman_motion_config', lambda *_a, **_k: {})
    monkeypatch.setattr(bridge_module, 'SynchronousPlanner', FakePlanner)
    bridge = bridge_module.CuroboRuntimeBridge(
        Runtime(), tmp_path,
        execution_stop_condition_for_stage=lambda stage, _model, _data, sample: {
            'failure_type': 'forbidden_contact_during_execution',
            'stage': stage, 'sample_time_s': sample['time_s'],
        },
    )
    result = bridge('move', np.eye(4), 'initial')

    assert result['success'] is False
    assert observed['stop_condition_forwarded'] is True
    assert observed['stop_evidence']['stage'] == 'move'
    assert observed['collision_export']['positions'] == [.06]
    assert observed['collision_export']['execution_failure']['failure_step_index'] == 3
    assert result['execution']['failure_visualization']['kind'] == 'actual_collision_stop'


@pytest.mark.parametrize('linear_success', [True, False])
def test_bridge_screens_ordered_postpick_full_paths_then_keeps_first_target(
    tmp_path, monkeypatch, linear_success,
):
    """首个完整路径可行目标被选中后，直线失败也不能偷换下一候选。"""
    from gene_agent.motion.curobo import runtime_bridge as bridge_module

    calls = []

    class Runtime:
        model = object()
        data = object()
        robot_metadata = {'joint_groups': {'right': ['joint_1']}}

        def query(self, *, robot_groups):
            return {'state_id': 'closed', 'robots': {'right': {
                'joint_names': ['joint_1'], 'joint_positions_rad': [0.],
            }}}

        def execute_joint_trajectory(self, **kwargs):
            calls.append(('execute', kwargs['planned_state_id']))
            return {'success': True}

    def fake_export(*_args, **_kwargs):
        return {'scene_model': {}, 'scope': 'attached bowl',
                'snapshot_sha256': 'attached-hash', 'gripper_snapshot': None,
                'mounting_contact_policy': {}}

    class FakePlanner:
        def __init__(self, **_kwargs):
            pass

        def plan(self, _names, _positions, target, **_kwargs):
            x = float(np.asarray(target)[0, 3])
            calls.append(('screen', x))
            return {'success': x != .1, 'status': 'success' if x != .1 else 'goal_ik_no_solution',
                    'joint_names': ['joint_1'], 'positions_rad': [[0.], [.2]],
                    'interpolation_dt_s': .1}

        def plan_linear_tcp(self, _names, _positions, target, **_kwargs):
            x = float(np.asarray(target)[0, 3])
            calls.append(('linear', x))
            return {'success': linear_success,
                    'status': 'success' if linear_success else 'linear_path_failed',
                    'joint_names': ['joint_1'], 'positions_rad': [[0.], [.2]],
                    'interpolation_dt_s': .1}

    monkeypatch.setattr(bridge_module, 'export_collision_scene', fake_export)
    monkeypatch.setattr(bridge_module, 'load_realman_motion_config', lambda *_a, **_k: {})
    monkeypatch.setattr(bridge_module, 'SynchronousPlanner', FakePlanner)
    poses = []
    for x in (.1, .2, .3):
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)

    bridge = bridge_module.CuroboRuntimeBridge(
        Runtime(), tmp_path, linear_tcp_stages=('lift',),
    )
    result = bridge.plan_and_execute_candidates('lift', poses, 'closed')

    assert calls[:3] == [('screen', .1), ('screen', .2), ('linear', .2)]
    assert ('screen', .3) not in calls
    assert result['success'] is linear_success
    assert any(call[0] == 'execute' for call in calls) is linear_success
    saved = json.loads((tmp_path / 'plans/lift.json').read_text(encoding='utf-8'))
    assert saved['selected_candidate_index'] == 1
    assert len(saved['candidate_screen']) == 2
    assert saved['candidate_screen_scene_sha256'] == 'attached-hash'


def test_bridge_uses_explicit_self_collision_snapshot_for_insert(tmp_path, monkeypatch):
    """旧版 insert 忽略 world，但保留自身碰撞，并记录两种独立快照。"""
    from gene_agent.motion.curobo import runtime_bridge as bridge_module

    seen = {}

    class Runtime:
        model = object()
        data = object()
        robot_metadata = {'joint_groups': {'right': ['joint_1']}}

        def query(self, *, robot_groups):
            return {'state_id': 'held', 'robots': {'right': {
                'joint_names': ['joint_1'], 'joint_positions_rad': [0.],
                'T_world_tcp': np.eye(4).tolist(),
            }}}

        def execute_joint_trajectory(self, **_kwargs):
            return {'success': True}

    def fake_export(*_args, **kwargs):
        seen['export_options'] = kwargs
        return {'scene_model': {'mesh': {'microwave': {}}},
                'scope': 'world', 'snapshot_sha256': 'world-hash',
                'gripper_snapshot': {'snapshot_id': 'closed'},
                'mounting_contact_policy': {},
                'transferred_objects': {'bowl': [[0., 0., 0., .01]]}}

    def fake_self_only(source, output, *, attached_object_id=None):
        seen['self_source'] = source['snapshot_sha256']
        seen['self_output'] = output
        seen['attached'] = attached_object_id
        return {'scene_model': {}, 'scope': 'self only',
                'snapshot_sha256': 'self-hash',
                'gripper_snapshot': source['gripper_snapshot'],
                'mounting_contact_policy': {}}

    class FakePlanner:
        def __init__(self, **kwargs):
            seen['planner_scene'] = kwargs['scene_model']
            seen['planner_hash'] = kwargs['scene_snapshot_sha256']

        def plan_linear_tcp(self, *_args, **_kwargs):
            return {'success': True, 'status': 'success',
                    'joint_names': ['joint_1'], 'positions_rad': [[0.], [.1]],
                    'interpolation_dt_s': .1}

    monkeypatch.setattr(bridge_module, 'export_collision_scene', fake_export)
    monkeypatch.setattr(bridge_module, 'export_self_collision_only_snapshot', fake_self_only)
    monkeypatch.setattr(bridge_module, 'load_realman_motion_config',
                        lambda *_a, **_k: {})
    monkeypatch.setattr(bridge_module, 'SynchronousPlanner', FakePlanner)
    bridge = bridge_module.CuroboRuntimeBridge(
        Runtime(), tmp_path, linear_tcp_stages=('insert',),
        self_collision_only_stages=('insert',),
    )
    result = bridge('insert', np.eye(4), 'held')

    assert result['success']
    assert seen['self_source'] == 'world-hash'
    assert seen['planner_scene'] == {}
    assert seen['planner_hash'] == 'self-hash'
    assert seen['attached'] is None
    assert seen['export_options'] == {}
