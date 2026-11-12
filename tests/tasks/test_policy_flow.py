"""独立 policy 维持原操作顺序；失败后不继续执行下一操作。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.tasks.policies.cup_pour_task.policy import run as pour
from gene_agent.tasks.policies.spoon_scoop_task.policy import run as scoop


def context(tmp_path, *, fail=None):
    calls = []
    state = SimpleNamespace(output_dir=tmp_path, object_name="cup", ball_names=("a",) * 5,
                            report_task={"success": False, "skills": {}}, initial_cargo=set())
    def operation(name):
        calls.append(name)
        if name == fail:
            state.report_task["failure_stage"] = name
            return False
        return True
    state._observe_task = lambda _: {"observation": {"objects": {"cup": {"T_world_object": np.eye(4)}}}}
    state.run_pick = lambda: {"success": operation("pick"), "failure_stage": "pick"}
    state._run_pour = lambda: operation("pour")
    state._run_place = lambda: operation("place")
    state._run_scoop = lambda: operation("scoop")
    state._ball_membership = lambda: {"in_cup": set("abcde"), "in_target": set("abcde"), "outside": set()}
    state.cargo_goal_satisfied = lambda t, s: len(t) >= 5 and not s
    state.save_task = lambda: None
    # 新策略只消费操作接口；旧字段留作适配测试数据，不传入 policy。
    for name in ('pick', 'pour', 'place', 'scoop'):
        setattr(state, name, lambda name=name: operation(name))
    state.record_exception = lambda error: state.report_task.update(
        error=str(error), failure_stage='exception')
    def finish(success, scope):
        state.report_task.update(success=success, until_skill=scope)
        return state.report_task
    state.finish = finish
    return state, calls


@pytest.mark.parametrize("policy,steps,stop", [(pour, ["pick", "pour", "place"], "place"),
                                              (scoop, ["pick", "scoop", "pour"], "pour")])
def test_complete_policy_preserves_operation_order(tmp_path, policy, steps, stop):
    state, calls = context(tmp_path)
    assert policy(state, until_skill=stop)["success"]
    assert calls == steps


@pytest.mark.parametrize("policy,stop", [(pour, "place"), (scoop, "pour")])
def test_pick_failure_keeps_prefix_and_stops(tmp_path, policy, stop):
    state, calls = context(tmp_path, fail="pick")
    result = policy(state, until_skill=stop)
    assert not result["success"]
    assert result["failure_stage"] == "pick"
    assert calls == ["pick"]


def test_failed_pour_does_not_release_or_place(tmp_path):
    state, calls = context(tmp_path, fail="pour")
    assert not pour(state)["success"]
    assert calls == ["pick", "pour"]


@pytest.mark.parametrize('policy,terminal,names', [
    (pour, 'place', ['pick', 'pour', 'place']),
    (scoop, 'pour', ['pick', 'scoop', 'pour']),
])
def test_policy_accepts_operations_without_simulation_or_report_internals(policy, terminal, names):
    """策略可以由其他后端提供操作，不能要求 MuJoCo/self 私有状态。"""
    calls = []
    def operation(name):
        calls.append(name)
        return True
    operations = SimpleNamespace(**{n: lambda n=n: operation(n) for n in names})
    operations.finish = lambda success, scope: {'success':success, 'scope':scope}
    operations.record_exception = lambda error: calls.append(type(error).__name__)
    result = policy(operations, until_skill=terminal)
    assert result == {'success':True, 'scope':terminal}
    assert calls == names


def test_common_task_result_distinguishes_local_completion():
    from gene_agent.tasks.results import annotate_task_result
    report = {'success':True, 'status':'completed'}
    annotate_task_result(report, scope='pick', full_scope='place', actual_final={'state_id':'measured'})
    assert report['operation_success'] is True
    assert report['task_success'] is None
    assert report['actual_final']['state_id'] == 'measured'


def test_missing_final_observation_is_not_success():
    from gene_agent.tasks.results import finish_task_report
    def observe():
        raise RuntimeError('state source disconnected')
    report = {'success':True}
    finish_task_report(report, success=True, scope='place', full_scope='place',
                       observe_final=observe, save=lambda: None)
    assert not report['operation_success']
    assert report['status'] == 'unknown'
    assert report['actual_final']['state_unavailable']


def test_task_failure_summary_preserves_original_planner_reason():
    from gene_agent.tasks.results import annotate_task_result
    report = {'success':False, 'status':'failed', 'failure_stage':'upright_30',
              'stages': {'upright_30': {'plan': {
                  'failure_category':'goal_ik_no_solution', 'status':'IK_FAIL'}}}}
    annotate_task_result(report, scope='place', full_scope='place')
    assert report['failure']['stage'] == 'upright_30'
    assert report['failure']['code'] == 'goal_ik_no_solution'
    assert report['failure']['report_ref'] == 'task_result.json'


@pytest.mark.parametrize('policy,stage', [(pour, 'pour'), (scoop, 'scoop')])
def test_partial_operation_exception_preserves_actual_terminal_and_stops(tmp_path, policy, stage):
    from gene_agent.tasks.results import record_task_exception, finish_task_report
    state, calls = context(tmp_path)
    actual = {'state_id': 'initial'}
    def partial():
        actual['state_id'] = 'moved-before-exception'
        calls.append(stage)
        raise RuntimeError('device disconnected after motion')
    setattr(state, stage, partial)
    state.record_exception = lambda error: record_task_exception(state.report_task, error)
    state.finish = lambda success, scope: finish_task_report(
        state.report_task, success=success, scope=scope,
        full_scope='place' if policy is pour else 'pour',
        observe_final=lambda: dict(actual), save=lambda: None)
    result = policy(state)
    assert calls == ['pick', stage]
    assert not result['operation_success']
    assert result['actual_final']['state_id'] == 'moved-before-exception'
    assert 'device disconnected' in result['error']
    assert 'RuntimeError' in result['traceback']


@pytest.mark.parametrize('policy,stop,expected', [(pour, 'pick', ['pick']),
    (pour, 'pour', ['pick', 'pour']), (scoop, 'scoop', ['pick', 'scoop'])])
def test_partial_scope_does_not_execute_later_operation(tmp_path, policy, stop, expected):
    state, calls = context(tmp_path)
    assert policy(state, until_skill=stop)['success']
    assert calls == expected


def test_sweep_policy_uses_public_operations_and_stops_before_release_on_failure():
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    calls = []
    pose = np.eye(4)
    operations = SimpleNamespace(
        pick=lambda: calls.append('pick') or True,
        observe=lambda: {'state_id': 'actual', 'objects': {name: {'T_world_object': pose}
                       for name in ('brush', 'dustpan', 'cube', 'cube_red', 'cube_blue')}},
        begin_sweeping=lambda rotation, order: calls.append('begin'),
        goals=lambda: {'targets': {name: {'success': False} for name in ('cube', 'cube_red', 'cube_blue')}},
        sweep=lambda *args, **kwargs: calls.append('sweep') or False,
        failure=lambda: {'stage': 'cube__sweep', 'screen': {}},
        record_progress=lambda **values: None,
        finish_sweeping=lambda: calls.append('release') or True)
    assert not SweepPolicy(operations).run()
    assert calls == ['pick', 'begin', 'sweep']


def test_sweep_policy_public_operations_success_and_recovery_are_bounded():
    from gene_agent.tasks.policies.sweep_cube_task.policy import SweepPolicy
    calls, completed = [], set()
    poses = {}
    for i, name in enumerate(('brush', 'dustpan', 'cube', 'cube_red', 'cube_blue')):
        pose = np.eye(4); pose[0, 3] = i * .05
        poses[name] = {'T_world_object': pose}
    def goals():
        targets = {name: {'success': name in completed, 'tray_contact': True}
                   for name in ('cube', 'cube_red', 'cube_blue')}
        return {'success': all(v['success'] for v in targets.values()), 'targets': targets}
    def sweep(name, label, **kwargs):
        calls.append(label)
        # 首轮返回动作成功，但红方块后置目标仍不满足，须走原一次恢复。
        if name != 'cube_red' or label.endswith('_retry1'):
            completed.add(name)
        return True
    progress = {}
    operations = SimpleNamespace(pick=lambda: True,
        observe=lambda: {'state_id':'actual', 'objects':poses},
        begin_sweeping=lambda rotation, order: progress.update(order=order),
        goals=goals, sweep=sweep, record_progress=lambda **v: progress.update(v),
        finish_sweeping=lambda: calls.append('release') or True)
    assert SweepPolicy(operations, target_order_policy='farthest_first').run()
    assert progress['order'] == ['cube_blue', 'cube_red', 'cube']
    assert calls == ['cube_blue', 'cube_red', 'cube', 'cube_red_retry1', 'release']
    assert progress['recovery_targets'] == ['cube_red']


def test_long_brush_exposes_shared_policy_without_simulation_resources():
    from gene_agent.tasks.policies.sweep_long_task.policy import run
    calls = []
    operations = SimpleNamespace(pick=lambda: calls.append('pick') or False)
    assert run(operations, alignment_lateral_trigger_m=.045,
               target_order_policy='farthest_first') is False
    assert calls == ['pick']
