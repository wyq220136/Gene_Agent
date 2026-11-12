"""实际微波炉记录复用只读回放；不能退回在线规划。"""
import json

import numpy as np
import pytest

from gene_agent.visualization.replay import ReplayTimeline
from gene_agent.simulation.skill_viewer import SkillMujocoViewer


def _write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _sample(t, q, state=None):
    return {'sim_time_s': t, 'qpos': [q], 'state_id': state or f's{t}',
            'scene_revision': 2, 'objects': {}, 'robots': {}}


def _run(root):
    model = root / 'scene/source_scene/scene.xml'
    model.parent.mkdir(parents=True)
    model.write_text('<mujoco><worldbody><body name="bowl"><joint name="j" type="slide"/>'
                     '<geom type="sphere" size=".02"/></body></worldbody></mujoco>')
    _write(root, 'summary.json', {'task': 'microwave_task1',
        'source_format': 'mujoco_physical_with_declared_assistance',
        'scene_ref': str(model), 'goal_success': True, 'fully_physical_success': False})
    _write(root, 'table_place/motion_detached/executions/retreat.json', {
        'success': False, 'replay_samples': [_sample(5., .1), _sample(6., .2)]})
    _write(root, 'table_place/result.json', {'mode': 'physical_assisted',
        'skill': 'place_extracted_bowl_on_table', 'physical_table_place_success': False})


def test_actual_task_replays_failed_prefix_and_preserves_full_qpos(tmp_path):
    _run(tmp_path)
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.mode == 'actual'
    assert timeline.view.physical_success is False
    assert timeline.seek(.5).qpos == (.1,)
    assert timeline.seek(timeline.duration_s).qpos == (.2,)
    viewer = SkillMujocoViewer(tmp_path)
    viewer.select(mode='actual')
    viewer.seek(viewer.timeline.duration_s)
    np.testing.assert_allclose(viewer.data.qpos, [.2])
    assert viewer.data.time == 6.
    with pytest.raises(ValueError, match='规划'):
        timeline.select(mode='planned')


def test_actual_skill_subdirectory_finds_same_saved_model(tmp_path):
    _run(tmp_path)
    timeline = ReplayTimeline.load(tmp_path / 'table_place')
    assert timeline.mode == 'actual'
    viewer = SkillMujocoViewer(tmp_path / 'table_place')
    viewer.select(mode='actual')
    np.testing.assert_allclose(viewer.data.qpos, [.1])


def test_planned_preview_samples_are_not_accepted_as_actual(tmp_path):
    _run(tmp_path)
    path = tmp_path / 'table_place/motion_detached/executions/retreat.json'
    _write(tmp_path, str(path.relative_to(tmp_path)), {
        'success': True, 'planned_replay_samples': [_sample(5., .1), _sample(6., .2)]})
    with pytest.raises(ValueError, match='实际.*采样'):
        ReplayTimeline.load(tmp_path)


def test_saved_release_boundary_is_visible_without_fabricating_intermediate_state(tmp_path):
    import mujoco
    from gene_agent.feedback.state import save_state
    _run(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / 'scene/source_scene/scene.xml'))
    data = mujoco.MjData(model)
    data.time = 5.5
    data.qpos[:] = [.15]
    save_state(model, data, tmp_path / 'table_place/states/release.npz')
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.duration_s == pytest.approx(1.)
    assert timeline.seek(.75).qpos == (.15,)
    assert timeline.seek(.75).sim_time_s == 5.5
    assert timeline.seek(.75).sample_quality == 'saved_actual_boundary'


@pytest.mark.parametrize('kind,selection', [('skill', 'place_bowl'), ('task', 'microwave_task1')])
def test_existing_cli_dispatches_actual_record_to_same_read_only_viewer(
    tmp_path, monkeypatch, kind, selection,
):
    from gene_agent.visualization.saved_run_cli import dispatch_saved_run
    from gene_agent.simulation import skill_viewer
    _run(tmp_path)
    captured = {}
    class Viewer(SkillMujocoViewer):
        def save_snapshot(self, output, **options):
            captured.update(frame=self.frame, options=options, qpos=self.data.qpos.copy())
            return {'mode': self.frame.mode}
    monkeypatch.setattr(skill_viewer, 'SkillMujocoViewer', Viewer)
    assert dispatch_saved_run(kind, [f'--{kind}', selection, '--run', str(tmp_path),
        '--mode', 'actual', '--time', '1.', '--output', str(tmp_path / 'snapshot.png')])
    assert captured['frame'].mode == 'actual'
    np.testing.assert_allclose(captured['qpos'], [.2])


def test_extract_contains_grasp_and_retreat_while_pick_keeps_first_pick(tmp_path):
    _run(tmp_path)
    for directory, time_s, qpos in [('pick_insert', 1., .01),
                                    ('extract_grasp', 3., .03),
                                    ('extract_retreat', 4., .04)]:
        _write(tmp_path, directory + '/executions/motion.json', {
            'success': True, 'replay_samples': [_sample(time_s, qpos), _sample(time_s + .1, qpos)]})
    pick = ReplayTimeline.load(tmp_path, skill='pick')
    extract = ReplayTimeline.load(tmp_path, skill='extract')
    assert pick.seek(0.).sim_time_s == 1.
    assert pick.seek(pick.duration_s).sim_time_s == 1.1
    assert extract.seek(0.).sim_time_s == 3.
    assert extract.seek(extract.duration_s).sim_time_s == 4.1
    assert any('extract_grasp' in stage.name for stage in extract.view.stages)
    assert any('extract_retreat' in stage.name for stage in extract.view.stages)


def test_standalone_physical_extract_grasp_is_recognized(tmp_path):
    _run(tmp_path)
    _write(tmp_path, 'extract_grasp/result.json', {
        'mode': 'physical', 'object_id': 'Bowl_0', 'skill': 'extract_bowl_grasp',
        'stages': {'approach': {'execution': {'success': True,
            'replay_samples': [_sample(3., .03), _sample(3.1, .04)]}}}})
    timeline = ReplayTimeline.load(tmp_path / 'extract_grasp', skill='extract')
    assert timeline.mode == 'actual'
    assert timeline.seek(0.).qpos == (.03,)


def test_standalone_door_scope_uses_report_selection_with_arbitrary_folder_name(tmp_path):
    _run(tmp_path)
    _write(tmp_path, 'door_capture/report.json', {
        'source_format': 'mujoco_physical', 'selection': 'microwave_door_open'})
    _write(tmp_path, 'door_capture/stages/handle_pull.json', {
        'execution': {'success': True, 'replay_samples': [_sample(2., .02), _sample(2.1, .03)]}})
    timeline = ReplayTimeline.load(tmp_path / 'door_capture', skill='open_door')
    assert timeline.mode == 'actual'
    assert timeline.seek(0.).sim_time_s == 2.


def test_door_skill_cli_selects_handle_only_not_whole_open_subtask(tmp_path, monkeypatch):
    from gene_agent.visualization.saved_run_cli import dispatch_saved_run
    from gene_agent.simulation import skill_viewer
    _run(tmp_path)
    _write(tmp_path, 'open_first/report.json', {
        'source_format': 'mujoco_physical', 'selection': 'microwave_door_open'})
    for name, time_s in [('handle_pull', 2.), ('push_open', 3.)]:
        _write(tmp_path, f'open_first/stages/{name}_move.json', {
            'execution': {'success': True, 'replay_samples': [_sample(time_s,.02),_sample(time_s+.1,.03)]}})
    selected = []
    class Viewer(SkillMujocoViewer):
        def save_snapshot(self, output, **options):
            selected.extend(s.name for s in self.timeline.view.stages)
            return {}
    monkeypatch.setattr(skill_viewer, 'SkillMujocoViewer', Viewer)
    assert dispatch_saved_run('skill', ['--skill','handle_pull','--run',str(tmp_path),
        '--output',str(tmp_path/'handle.png')])
    assert selected and all('handle_pull' in name for name in selected)
