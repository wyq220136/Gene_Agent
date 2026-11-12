"""真机报告只显示已保存的静态观测和测量端点；不连接设备或伪造连续执行。"""
import json
from types import SimpleNamespace

import pytest


def _report(root):
    report = {'execute': True, 'status': 'unknown', 'subtask': 'insert_bowl',
        'stages': [{'stage': 'preplace', 'execution': {'source_format': 'realworld_motion_stage',
            'measured_end': {'joint_names': ['j1'], 'positions_rad': [.2]}}}],
        'shared_flow': {'success': False, 'executed': True, 'stages': ['preplace', 'settle'],
            'end_state': {'source': 'unavailable', 'state_id': 'unavailable:settle',
                          'values': {'state_unavailable': True}},
            'failure_stage': 'settle', 'failure_code': 'stage_failed', 'stage_outcomes': []}}
    (root / 'insert_result.json').write_text(json.dumps(report))
    return report


def test_realworld_report_preserves_unknown_sdk_evidence_without_scene(tmp_path):
    from gene_agent.feedback.realworld_record import load_realworld_record
    original = _report(tmp_path)
    record = load_realworld_record(tmp_path, skill='insert')
    assert record['reports']['insert_result.json'] == original
    assert record['scenes'] == {}
    assert any('scene.xml' in text for text in record['missing'])
    assert record['reports']['insert_result.json']['shared_flow']['end_state']['source'] == 'unavailable'


def test_realworld_report_only_panel_opens_and_labels_static_evidence(tmp_path):
    from gene_agent.feedback.realworld_record import load_realworld_record
    from gene_agent.visualization.realworld_record import RealworldRecordApp
    _report(tmp_path)
    contents = []
    class Control:
        def __init__(self, value): self.value = value
        def on_update(self, callback): pass
    server = SimpleNamespace(gui=SimpleNamespace(
        add_markdown=lambda content: contents.append(content) or SimpleNamespace(content=content),
        add_dropdown=lambda label, options, initial_value: Control(initial_value)),
        scene=SimpleNamespace(set_up_direction=lambda direction: None))
    app = RealworldRecordApp(server, load_realworld_record(tmp_path))
    assert app.scene is None
    text = '\n'.join(str(row) for row in contents)
    assert '静态观测' in text and '实测端点' in text and 'unknown' in text
    assert 'unavailable:settle' in text and 'realworld_motion_stage' in text


def test_realworld_output_requires_saved_scene_instead_of_mujoco_fallback(tmp_path):
    from gene_agent.visualization.saved_run_cli import dispatch_saved_run
    _report(tmp_path)
    with pytest.raises(ValueError, match='scene.xml'):
        dispatch_saved_run('skill', ['--skill', 'insert_bowl', '--run', str(tmp_path),
            '--mode', 'actual', '--output', str(tmp_path / 'should_not_exist.png')])
    assert not (tmp_path / 'should_not_exist.png').exists()


def test_static_observation_uses_saved_qpos_and_source_without_sdk(tmp_path):
    from gene_agent.feedback.realworld_record import load_realworld_record
    from gene_agent.visualization.realworld_record import load_static_scene
    _report(tmp_path)
    folder = tmp_path / 'observations/0000'
    (folder / 'scene').mkdir(parents=True)
    (folder / 'collision').mkdir()
    (folder / 'scene/scene.xml').write_text(
        '<mujoco><worldbody><body name="bowl"><joint type="slide"/>'
        '<geom type="sphere" size=".02"/></body></worldbody></mujoco>')
    provenance = {'physics_stepped': False, 'source_frame_number': 9,
                  'source_timestamp_ms': 900., 'collision_snapshot_sha256': 'scene9'}
    (folder / 'observed_scene_provenance.json').write_text(json.dumps(provenance))
    (folder / 'collision/collision_snapshot.json').write_text(json.dumps({
        'snapshot_sha256': 'scene9', 'qpos_snapshot': [.31]}))
    record = load_realworld_record(tmp_path)
    scene = next(iter(record['scenes'].values()))
    model, data = load_static_scene(scene)
    assert data.qpos[0] == .31
    assert scene['provenance'] == provenance
    assert record['reports']['insert_result.json']['shared_flow']['end_state']['source'] == 'unavailable'
    # 场景仅显示采集第9帧；后续SDK末态缺失，不能把静态画面称作执行终态。
    assert scene['provenance']['source_frame_number'] == 9


def test_saved_planning_targets_are_available_alongside_measurements(tmp_path):
    from gene_agent.feedback.realworld_record import load_realworld_record
    report = _report(tmp_path)
    target = [[1., 0., 0., .4], [0., 1., 0., .2], [0., 0., 1., .3], [0., 0., 0., 1.]]
    (tmp_path / 'targets.json').write_text(json.dumps({'keyposes_world_tcp': {'insert': target}}))
    record = load_realworld_record(tmp_path)
    assert record['targets']['targets.json/keyposes_world_tcp/insert'] == target
    assert record['reports']['insert_result.json'] == report


@pytest.mark.parametrize('filename,extra', [
    ('door_subtask_result.json', {'part': 'open'}),
    ('transition_result.json', {'stage_id': 'before_reopen_return_default_tcp'}),
])
def test_standalone_door_and_transition_summaries_keep_original_labels(tmp_path, filename, extra):
    from gene_agent.feedback.realworld_record import load_realworld_record
    report = {'execute': False, 'status': 'unknown', 'success': False, **extra}
    (tmp_path / filename).write_text(json.dumps(report))
    record = load_realworld_record(tmp_path)
    assert record is not None
    assert record['reports'][filename] == report


@pytest.mark.parametrize('scope,expected', [
    ('object_insert', {'insert_result.json', 'pick_result.json'}),
    ('object_extract', {'extract_result.json', 'table_place_result.json'}),
    ('door_cycle', {'door_subtask_result.json'}),
])
def test_realworld_subtask_scope_excludes_other_saved_operations(tmp_path, scope, expected):
    from gene_agent.feedback.realworld_record import load_realworld_record
    for filename, fields in {
        'insert_result.json': {'subtask': 'insert_bowl'},
        'pick_result.json': {'subtask': 'pick_bowl'},
        'extract_result.json': {'subtask': 'extract_bowl'},
        'table_place_result.json': {'subtask': 'place_bowl_on_table'},
        'door_subtask_result.json': {'part': 'open'},
    }.items():
        (tmp_path / filename).write_text(json.dumps({'execute': False, 'status': 'unknown', **fields}))
    record = load_realworld_record(tmp_path, skill=scope)
    assert set(record['reports']) == expected


def test_realworld_planned_mode_is_not_silently_treated_as_actual(tmp_path):
    from gene_agent.visualization.saved_run_cli import dispatch_saved_run
    _report(tmp_path)
    with pytest.raises(ValueError, match='规划轨迹'):
        dispatch_saved_run('skill', ['--skill', 'insert_bowl', '--run', str(tmp_path),
            '--mode', 'planned', '--output', str(tmp_path / 'not_a_trajectory.png')])


def test_independent_handle_skill_filters_real_door_reports(tmp_path):
    from gene_agent.feedback.realworld_record import load_realworld_record
    for name in ['handle_pull', 'push_open']:
        folder = tmp_path/name
        folder.mkdir()
        (folder/'skill_result.json').write_text(json.dumps({
            'execute': False, 'skill_id': name, 'status': 'validated_prefix'}))
    record = load_realworld_record(tmp_path, skill='handle_pull')
    assert set(record['reports']) == {'handle_pull/skill_result.json'}
