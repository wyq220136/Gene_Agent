"""失败的桌面任务也能查看保存的真实前缀，不能用计划采样代替。"""
import hashlib
import json

import pytest

from gene_agent.visualization.replay import ReplayTimeline


def make_run(root, *, planned=False):
    scene = root / 'scene.xml'
    scene.write_text('<mujoco><worldbody><body name="cup_green"><joint type="slide"/>'
                     '<geom size=".01" mass=".1"/></body></worldbody></mujoco>')
    sample_key = 'planned_replay_samples' if planned else 'replay_samples'
    report = {'task': 'cup_pour_balls_into_bowl_and_return', 'status': 'failed',
              'success': False, 'cup_name': 'cup_green', 'fully_physical_success': False,
              'scene_xml_path': str(scene),
              'scene_xml_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
              'stages': {'tilt_30': {'success': False, 'execution': {sample_key: [
                  {'sim_time_s': 3., 'qpos': [.1]}, {'sim_time_s': 4., 'qpos': [.2]}]}}}}
    (root / 'task_result.json').write_text(json.dumps(report))


def test_failed_prefix_has_actual_time_and_not_planned_trajectory(tmp_path):
    make_run(tmp_path)
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.mode == 'actual'
    assert timeline.duration_s == pytest.approx(1.)
    assert timeline.seek(1.).qpos == (.2,)
    assert timeline.seek(1.).sim_time_s == 4.
    assert timeline.view.physical_success is False
    with pytest.raises(ValueError):
        timeline.select(mode='planned')


def test_native_replay_rejects_changed_scene(tmp_path):
    make_run(tmp_path)
    (tmp_path / 'scene.xml').write_text('changed')
    with pytest.raises(ValueError, match='哈希'):
        ReplayTimeline.load(tmp_path)


def test_native_replay_never_reads_planned_samples_as_actual(tmp_path):
    make_run(tmp_path, planned=True)
    with pytest.raises(ValueError, match='实际.*采样'):
        ReplayTimeline.load(tmp_path)


def test_video_evidence_does_not_score_successful_prefix_as_full_task(tmp_path):
    make_run(tmp_path)
    report_path = tmp_path / 'task_result.json'
    report = json.loads(report_path.read_text())
    report.update(success=True, until_skill='pick')
    report_path.write_text(json.dumps(report))
    from gene_agent.visualization.video_export import _evidence
    evidence = _evidence(tmp_path, 'actual')
    assert evidence['task_success'] is None
    assert evidence['operation_success'] is True
    assert evidence['requested_scope'] == 'pick'


def test_interrupted_actual_tail_is_not_lost_with_unwritten_stage(tmp_path):
    import mujoco
    make_run(tmp_path)
    live = tmp_path / 'live'
    live.mkdir()
    model = mujoco.MjModel.from_xml_path(str(tmp_path / 'scene.xml'))
    mujoco.mj_saveModel(model, str(live / 'model_0.mjb'), None)
    (live / 'frames.jsonl').write_text(json.dumps(
        {'model': 'model_0.mjb', 'sim_time_s': 4.5, 'qpos': [.3]}) + '\n' + '{"partial":')
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.seek(timeline.duration_s).qpos == (.3,)
    assert timeline.seek(timeline.duration_s).sim_time_s == 4.5
    assert 'unreported_stage' in timeline.seek(timeline.duration_s).stage_id


def test_standalone_pick_prefix_uses_child_actual_samples_and_object_identity(tmp_path):
    make_run(tmp_path)
    report = json.loads((tmp_path/'task_result.json').read_text())
    stages = report.pop('stages')
    report.update(task='scoop_orange', skill='pick_spoon_handle', requested_scope='grasp',
                  execution_backend='mujoco_physical', success=True,
                  simulation_assistance='weld_after_verified_bilateral_inner_finger_contact',
                  pick_result_path=str(tmp_path/'pick/result.json'))
    (tmp_path/'task_result.json').unlink()
    (tmp_path/'result.json').write_text(json.dumps(report))
    (tmp_path/'pick').mkdir()
    (tmp_path/'pick/result.json').write_text(json.dumps({
        'mode':'synchronous_pick_skill', 'instance_id':'spoon_orange',
        'requested_scope':'grasp', 'physical_pick_success':False,
        'contact_success':True, 'stages':stages}))
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.mode == 'actual'
    assert timeline.view.instance_id == 'spoon_orange'
    assert timeline.seek(1.).qpos == (.2,)
    assert timeline.view.physical_success is False
    from gene_agent.visualization.video_export import _evidence
    evidence = _evidence(tmp_path, 'actual')
    assert evidence['evidence_mode'] == 'mujoco_contact'
    assert evidence['task_success'] is None
