"""启动边界：显式选择策略，环境条件不能混入策略参数。"""
import json
from pathlib import Path

import pytest

from gene_agent.tasks.launch import resolve_request, run_benchmark


def test_default_selection_uses_benchmark_input_and_explicit_policy_identity():
    request = resolve_request('cup_pour_task')
    assert request.policy_id == 'cup_pour_task/default'
    assert request.input_path.name == 'pour'
    assert request.environment == {'simulation_weld': False, 'collision_mode': 'full'}
    assert request.parameters == {'until_skill': 'place'}


def test_policy_cannot_be_selected_for_an_incompatible_benchmark():
    with pytest.raises(ValueError, match='兼容'):
        resolve_request('cup_pour_task', policy_id='spoon_scoop_task/default')


@pytest.mark.parametrize('parameters', [{'simulation_weld': True}, {'seed': 123},
                                       {'success_threshold': 0}, {'until_skill': 'invalid'}])
def test_reject_environment_or_unknown_parameters_before_creating_run(tmp_path, parameters):
    output = tmp_path / 'run'
    with pytest.raises(ValueError):
        run_benchmark('cup_pour_task', output, parameters=parameters)
    assert not output.exists()


def test_environment_and_policy_inputs_remain_separate(tmp_path):
    request = resolve_request('spoon_scoop_task', input_path=tmp_path,
                              environment={'simulation_weld': True},
                              parameters={'source_above_z_m': .16})
    assert request.input_path == tmp_path
    assert request.environment['simulation_weld'] is True
    assert request.parameters == {'source_above_z_m': .16, 'until_skill': 'pour'}
    with pytest.raises(ValueError):
        resolve_request('spoon_scoop_task', environment={'source_above_z_m': .16})
    with pytest.raises(ValueError):
        resolve_request('sweep_cube_task', environment={'seed': True})


def test_record_exception_and_preserve_actual_prefix(tmp_path, monkeypatch):
    import gene_agent.tasks.launch as launch
    request = resolve_request('cup_pour_task')
    output = tmp_path / 'attempt'
    def interrupted(req, destination):
        destination.mkdir()
        (destination / 'actual-prefix.txt').write_text('already moved')
        raise RuntimeError('planner rejected')
    monkeypatch.setattr(launch, '_execute', interrupted)
    with pytest.raises(RuntimeError, match='planner rejected'):
        run_benchmark(request.benchmark_id, output)
    manifest = json.loads((output / 'launch.json').read_text())
    assert manifest['status'] == 'exception'
    assert manifest['policy']['id'] == request.policy_id
    assert manifest['benchmark']['id'] == request.benchmark_id
    assert manifest['parameters'] == request.parameters
    assert (output / 'actual-prefix.txt').read_text() == 'already moved'


def test_launch_never_reuses_historical_output(tmp_path):
    output = tmp_path / 'history'
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_benchmark('cup_pour_task', output)
    assert list(output.iterdir()) == []


def test_successful_prefix_is_not_complete_benchmark_success(tmp_path, monkeypatch):
    import gene_agent.tasks.launch as launch
    output = tmp_path / 'prefix'
    def prefix(req, destination):
        destination.mkdir()
        return {'status': 'completed', 'success': True, 'until_skill': 'pick'}
    monkeypatch.setattr(launch, '_execute', prefix)
    result = run_benchmark('cup_pour_task', output, parameters={'until_skill': 'pick'})
    manifest = json.loads((output / 'launch.json').read_text())
    assert manifest['task_success'] is None
    assert manifest['requested_scope'] == 'pick'
    assert manifest['operation_success'] is True
    assert result['benchmark_task_success'] is None
