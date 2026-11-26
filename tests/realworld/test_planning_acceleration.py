"""现有现场入口选择与清理规划资源，不改变观测/执行接线。"""
import json
import pytest
from gene_agent.motion.configuration import motion_configuration
from gene_agent.motion.curobo.optimization import current_optimization
from tools.run.run_microwave_task1_realworld import _planning_acceleration


@pytest.mark.parametrize('group,collision,requested,expected', [
    ('right', 'full', None, 'single_reuse'),
    ('left', 'full', None, 'baseline'),
    ('right', 'robot_only', None, 'baseline'),
    ('right', 'full', 'baseline', 'baseline'),
])
def test_existing_configuration_keeps_compatible_resource_mode(
        tmp_path, group, collision, requested, expected):
    config = {'active_group': group, 'curobo_optimization': requested}
    with motion_configuration({'collision_mode': collision}):
        with _planning_acceleration(config, tmp_path):
            assert (current_optimization() is not None) == (expected == 'single_reuse')
    assert current_optimization() is None
    report = json.loads((tmp_path / 'planning_optimization.json').read_text())
    assert report['mode'] == expected
    if expected == 'single_reuse':
        assert report['closed']


def test_entry_failure_releases_scope_and_preserves_original_error(tmp_path):
    with pytest.raises(ValueError, match='original operation failure'):
        with _planning_acceleration({'active_group': 'right'}, tmp_path):
            owner = current_optimization()
            raise ValueError('original operation failure')
    assert owner.closed and owner.engine is None
    assert current_optimization() is None
    assert json.loads((tmp_path / 'planning_optimization.json').read_text())['closed']
