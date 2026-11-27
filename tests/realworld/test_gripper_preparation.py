import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gene_agent.realworld.microwave import gripper_preparation as prep
from gene_agent.realworld.microwave.task import SUBTASK_NAMES


class Gripper:
    max_width_m = .08
    last_reading = None
    def __init__(self, events, width=.064):
        self.events, self.width = events, width
    def command_width_m(self, width, *, blocking):
        assert width == pytest.approx(.064)
        assert blocking
        self.events.append('command800')
    def read_width_m(self):
        assert 'command800' in self.events
        self.events.append('read')
        return self.width


def test_command_precedes_every_read_and_confirms(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.time, 'sleep', lambda _: None)
    events = []
    report = prep.prepare_right_gripper(Gripper(events), output_dir=tmp_path)
    assert events == ['command800', 'read']
    assert report['status'] == 'confirmed'
    assert json.loads((tmp_path/'right_gripper_preparation.json').read_text())['status'] == 'confirmed'


def test_command_failure_never_reads(tmp_path):
    class Broken(Gripper):
        def command_width_m(self, *args, **kwargs): raise RuntimeError('SDK17')
    events = []
    with pytest.raises(RuntimeError, match='SDK17'):
        prep.prepare_right_gripper(Broken(events), output_dir=tmp_path)
    assert events == []
    assert json.loads((tmp_path/'right_gripper_preparation.json').read_text())['status'] == 'failed'


def test_failed_confirmation_stops_without_retrying_command(tmp_path, monkeypatch):
    clock = iter(range(20))
    monkeypatch.setattr(prep.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(prep.time, 'sleep', lambda _: None)
    events = []
    with pytest.raises(ValueError, match="800"):
        prep.prepare_right_gripper(Gripper(events, .08024), output_dir=tmp_path)
    assert events.count('command800') == 1
    assert json.loads((tmp_path/'right_gripper_preparation.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('subtask', (*SUBTASK_NAMES, 'all'))
@pytest.mark.parametrize('execute', [False, True])
def test_entry_prepares_before_left_connection_and_observer(tmp_path, monkeypatch, execute, subtask):
    from test_live_cli import _full_site_config, load_site_config
    from tools.run import run_microwave_task1_realworld as cli
    events = []
    path, _ = _full_site_config(tmp_path)
    prepare_held = subtask in {'insert_bowl', 'place_bowl_on_table'}
    config = load_site_config(path, subtask=subtask, execute=execute, no_knob=False,
                              prepare_held_bowl=prepare_held)
    monkeypatch.setattr(prep.time, 'sleep', lambda _: None)
    @contextmanager
    def connect(ip, **kwargs):
        events.append('connect_' + kwargs['group'])
        yield SimpleNamespace(robot=object(), gripper=Gripper(events))
    class ObserverReached(Exception): pass
    def observer(**kwargs):
        events.append('observer')
        raise ObserverReached()
    monkeypatch.setattr('gene_agent.realworld.devices.arms.realman.connection.connect_realman', connect)
    monkeypatch.setattr('gene_agent.realworld.microwave.scene.make_live_microwave_observer', observer)
    with pytest.raises(ObserverReached):
        cli.run_site(config, subtask=subtask, output_dir=tmp_path/'run', execute=execute,
                     prepare_held_bowl=prepare_held)
    if execute:
        assert events == ['connect_right', 'command800', 'read', 'connect_left', 'observer']
    else:
        assert events == ['connect_left', 'connect_right', 'observer']


def test_entry_failure_never_reaches_other_devices_or_observer(tmp_path, monkeypatch):
    from test_live_cli import _full_site_config, load_site_config
    from tools.run import run_microwave_task1_realworld as cli
    path, _ = _full_site_config(tmp_path)
    config = load_site_config(path, subtask='pick_bowl', execute=True, no_knob=False)
    class Broken(Gripper):
        def command_width_m(self, *args, **kwargs): raise RuntimeError('SDK17')
    events = []
    @contextmanager
    def connect(ip, **kwargs):
        events.append(kwargs['group'])
        yield SimpleNamespace(robot=object(), gripper=Broken(events))
    monkeypatch.setattr('gene_agent.realworld.devices.arms.realman.connection.connect_realman', connect)
    monkeypatch.setattr('gene_agent.realworld.microwave.scene.make_live_microwave_observer',
                        lambda **kw: pytest.fail('must not construct observer'))
    with pytest.raises(RuntimeError, match='SDK17'):
        cli.run_site(config, subtask='pick_bowl', output_dir=tmp_path/'run', execute=True)
    assert events == ['right']
    report = json.loads((tmp_path/'run/entry_result.json').read_text())
    assert report['failure']['code'] == 'right_gripper_preparation_failed'

