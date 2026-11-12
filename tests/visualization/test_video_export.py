"""视频共用实际时间线；运动学渲染与完整 forward 的几何一致。"""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def _actual_run(root):
    folder = root/'scene/source_scene'
    folder.mkdir(parents=True)
    (folder/'scene.xml').write_text('''<mujoco><worldbody>
      <body name="bowl"><joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".03"/><site name="tip" pos="0 0 .1"/>
      <camera name="attached" pos="0 0 1"/></body></worldbody></mujoco>''')
    (root/'summary.json').write_text(json.dumps({'task':'microwave_task1',
        'source_format':'mujoco_physical_with_declared_assistance',
        'scene_ref':str(folder/'scene.xml'),'status':'completed','goal_success':True,
        'evidence_mode':'assisted_pick_extract_and_kinematic_knob','fully_physical_success':False}))
    for name,times in [('extract_retreat',[0., .1]),('table_place',[.4,.5])]:
        out=root/name/'executions';out.mkdir(parents=True)
        (out/'move.json').write_text(json.dumps({'success':True,'replay_samples':[
            {'sim_time_s':t,'qpos':[t],'state_id':str(t),'objects':{},'robots':{}} for t in times]}))


def test_render_only_restore_matches_full_forward_geometry(tmp_path):
    from gene_agent.visualization.video_export import restore_render_state
    _actual_run(tmp_path)
    model=mujoco.MjModel.from_xml_path(str(tmp_path/'scene/source_scene/scene.xml'))
    fast,full=mujoco.MjData(model),mujoco.MjData(model)
    for q in [0.,.2,-.1]:
        frame=SimpleNamespace(qpos=(q,),sim_time_s=1.2)
        restore_render_state(model,fast,frame)
        full.qpos[:]=[q];full.time=1.2;mujoco.mj_forward(model,full)
        for field in ['xpos','xmat','geom_xpos','geom_xmat','site_xpos','cam_xpos','light_xpos']:
            np.testing.assert_allclose(getattr(fast,field),getattr(full,field),atol=1e-12)
        assert fast.time==full.time
    with pytest.raises(ValueError,match='qpos'):
        restore_render_state(model,fast,SimpleNamespace(qpos=(float('nan'),),sim_time_s=0.))


def test_schedule_keeps_gap_last_state_and_short_event():
    from gene_agent.visualization.video_export import frame_schedule
    segments=[SimpleNamespace(start_s=0.,duration_s=.1),
              SimpleNamespace(start_s=.1,duration_s=.001),
              SimpleNamespace(start_s=.101,duration_s=.399)]
    timeline=SimpleNamespace(duration_s=.5,_segments=segments)
    times=frame_schedule(timeline, fps=4)
    assert times[0]==0. and times[-1]==.5
    assert .1 in times
    assert any(.101<=t<.5 for t in times)
    assert times==sorted(set(times))


def test_export_covers_actual_task_and_mixed_evidence_in_one_renderer(tmp_path):
    from gene_agent.visualization.video_export import export_video
    from tools.diagnostics.check_visual_acceptance import decode_frame_count
    _actual_run(tmp_path)
    result=export_video(tmp_path,tmp_path/'video.mp4',cameras=('task','contact'),
        fps=6,width=160,height=120)
    assert result['mode']=='actual'
    assert result['source_duration_s']==pytest.approx(.5)
    assert result['evidence_mode']=='assisted_pick_extract_and_kinematic_knob'
    assert any('extract_retreat' in s for s in result['stages_seen'])
    assert any('table_place' in s for s in result['stages_seen'])
    assert result['frames'][-1]['sim_time_s']==.5
    assert result['frame_count']==decode_frame_count(tmp_path/'video.mp4')
    assert result['size_px']==[320,120]
    assert result['renderer_count']==1
    assert result['timing_s']['total']>0
    assert result['render_state_method']=='kinematics_only'
    assert not list(tmp_path.rglob('*.png'))
    assert not list(tmp_path.glob('*.partial.mp4'))


def test_export_failure_preserves_previous_movie_and_does_not_claim_success(tmp_path):
    from gene_agent.visualization.video_export import export_video
    _actual_run(tmp_path)
    output=tmp_path/'video.mp4';output.write_bytes(b'previous delivery')
    with pytest.raises((ValueError,RuntimeError)):
        export_video(tmp_path,output,cameras=('does_not_exist',),fps=2,width=32,height=32)
    assert output.read_bytes()==b'previous delivery'
    assert not output.with_suffix('.json').exists()
    assert not list(tmp_path.glob('*.partial.mp4'))


def test_legacy_task_collector_includes_new_actual_transfer_records(tmp_path):
    from tools.viewers.recording.record_physical_task1_video import collect_actual_samples
    _actual_run(tmp_path)
    rows=collect_actual_samples(tmp_path)
    assert any('extract_retreat' in row['label'] for row in rows)
    assert any('table_place' in row['label'] for row in rows)
    assert rows[-1]['sim_time_s']==.5


def test_predicted_samples_cannot_be_exported_as_actual(tmp_path):
    from gene_agent.visualization.video_export import export_video
    _actual_run(tmp_path)
    for path in tmp_path.rglob('executions/*.json'):
        row=json.loads(path.read_text())
        row['planned_replay_samples']=row.pop('replay_samples')
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='实际.*采样'):
        export_video(tmp_path,tmp_path/'wrong.mp4')
    assert not (tmp_path/'wrong.mp4').exists()


def test_fast_and_full_forward_export_same_saved_pixels(tmp_path):
    import subprocess
    from gene_agent.visualization.video_export import export_video
    _actual_run(tmp_path)
    videos=[]
    for full in [False,True]:
        video=tmp_path/f'{full}.mp4'
        export_video(tmp_path,video,cameras=('task',),fps=6,width=160,height=120,full_forward=full)
        videos.append(subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','rawvideo',
            '-pix_fmt','rgb24','-'],capture_output=True,check=True).stdout)
    assert videos[0]==videos[1]


def test_cli_video_option_uses_shared_export_not_live_png(tmp_path,monkeypatch):
    from tools.run import run_microwave_task1_physical as cli
    calls=[]
    def run(output,**options):
        calls.append(('run',options['record_video']))
        output.mkdir()
        return {'status':'completed','goal_success':True,'evidence_mode':'mixed','subtasks':{},
                'requested_scope':'all','operation_success':True,'timing':{'wall_s':0.}}
    def export(run,output,**options):
        calls.append(('video',options['mode'],options['cameras']))
        return {'video':str(output),'timing_s':{'total':.1}}
    monkeypatch.setattr(cli,'run_microwave_task1_physical',run)
    monkeypatch.setattr(cli,'export_video',export,raising=False)
    monkeypatch.setattr(cli,'resolve_ffmpeg',lambda:'/usr/bin/ffmpeg')
    monkeypatch.setattr('sys.argv',['run','--output',str(tmp_path/'run'),'--video'])
    cli.main()
    assert calls==[('run',False),('video','actual',('overview','detail','top'))]


def test_cli_exports_recorded_prefix_after_task_exception(tmp_path, monkeypatch):
    from tools.run import run_microwave_task1_physical as cli

    calls = []

    def run(output, **_options):
        output.mkdir()
        calls.append('run')
        raise RuntimeError('collision at actual step')

    def export(run, output, **options):
        calls.append(('video', options['mode'], options['cameras']))
        return {'video': str(output), 'timing_s': {'total': .1}}

    monkeypatch.setattr(cli, 'run_microwave_task1_physical', run)
    monkeypatch.setattr(cli, 'export_video', export)
    monkeypatch.setattr(cli, 'resolve_ffmpeg', lambda: '/usr/bin/ffmpeg')
    monkeypatch.setattr('sys.argv', ['run', '--output', str(tmp_path/'run'), '--video'])
    with pytest.raises(SystemExit) as stopped:
        cli.main()

    assert stopped.value.code == 1
    assert calls == ['run', ('video', 'actual', ('overview', 'detail', 'top'))]
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['task'] == 'microwave_task1'
    assert summary['status'] == 'failed'


def test_visual_resume_identity_tracks_shared_rules_but_not_physical_adapter(tmp_path, monkeypatch):
    """物理接线搬目录后，不应让纯视觉录制无故失去恢复资格。"""
    from gene_agent.visualization.viser_sim import recording
    source = tmp_path/'gene_agent/visualization/viser_sim/recording.py'
    source.parent.mkdir(parents=True)
    source.write_text('VALUE = 1\n')
    physics = tmp_path/'gene_agent/tasks/policies/microwave_task/simulation_knob.py'
    physics.parent.mkdir(parents=True)
    physics.write_text('VALUE = 1\n')
    shared = physics.parent/'operations/door.py'
    shared.parent.mkdir()
    shared.write_text('VALUE = 1\n')
    cli = tmp_path/'tools/viewers/recording/record_visual.py'
    cli.parent.mkdir(parents=True)
    cli.write_text('VALUE = 1\n')
    monkeypatch.setattr(recording, '__file__', str(source))
    before = recording._source_revision()['runtime_sha256']
    physics.write_text('VALUE = 2\n')
    assert recording._source_revision()['runtime_sha256'] == before
    shared.write_text('VALUE = 2\n')
    assert recording._source_revision()['runtime_sha256'] != before


def test_split_uses_encoded_frame_time_after_short_event_holds(tmp_path,monkeypatch):
    from tools.viewers.recording import split_physical_task1_video as cli
    (tmp_path/'task_overview.mp4').write_bytes(b'placeholder')
    prefixes=['open_first/stages/x','pick_insert/stages/x','close_first/stages/x',
              'knob_execution/stages/x','open_second/stages/x',
              'extract_retreat/stages/x','table_place/stages/x','close_second/stages/x']
    frames=[]
    for i,name in enumerate(prefixes):
        for j in [0,1]:
            frames.append({'stage_id':f'{i:02d}_{name}', 'video_time_s':10+i+.2*j,
                           'sim_time_s':100+i+.1*j})
    (tmp_path/'video_manifest.json').write_text(json.dumps({'fps':5,'frames':frames}))
    commands=[]
    monkeypatch.setattr(cli.subprocess,'run',lambda command,**kw:commands.append(command))
    result=cli.split_subtask_clips(tmp_path)
    first=commands[0]
    assert float(first[first.index('-ss')+1])==pytest.approx(10.)
    assert result['clips']['rotate_knob']['source_kind']=='encoded_frame_timeline'
    assert len(commands)==7


@pytest.mark.parametrize('filename,extra',[
    ('result.json',{'mode':'physical_assisted','skill':'extract_bowl_assisted_retreat'}),
    ('insert_result.json',{'fully_physical_pick_insert_success':False,'physical_place_success':True}),
])
def test_independent_skill_keeps_declared_assistance_label(tmp_path,filename,extra):
    from gene_agent.visualization.video_export import _evidence
    (tmp_path/filename).write_text(json.dumps(extra))
    evidence=_evidence(tmp_path,'actual')
    assert evidence['evidence_mode'] in {'simulation_assisted','not_fully_physical'}
    assert evidence['task_success'] is None


def test_legacy_crop_failure_keeps_all_previously_delivered_files(tmp_path,monkeypatch):
    import subprocess
    from tools.viewers.recording import record_physical_task1_video as cli
    (tmp_path/'summary.json').write_text(json.dumps({'status':'completed','goal_success':True}))
    names=['task_video.mp4','task_overview.mp4','task_detail.mp4','video_manifest.json']
    for name in names:(tmp_path/name).write_bytes(b'previous')
    def export(run,output,**kw):
        output.write_bytes(b'new combined')
        return {'video':str(output),'frames':[]}
    def crop(*a,**kw):raise subprocess.CalledProcessError(1,a[0])
    monkeypatch.setattr(cli,'export_video',export)
    monkeypatch.setattr(cli.subprocess,'run',crop)
    with pytest.raises(subprocess.CalledProcessError):cli.record_physical_task1_video(tmp_path)
    for name in names:assert (tmp_path/name).read_bytes()==b'previous'
    assert not list(tmp_path.glob('.task_video.*'))


@pytest.mark.parametrize('task,terminal', [
    ('cup_pour_balls_into_bowl_and_return', 'place'),
    ('scoop_balls_into_other_bowl', 'pour'),
])
@pytest.mark.parametrize('success', [False, True])
def test_normalized_full_scope_video_retains_complete_task_verdict(tmp_path, task, terminal, success):
    """新结果的 full 与旧 until_skill 终点同义；失败完整运行也有明确结论。"""
    from gene_agent.visualization.video_export import _evidence
    report = {'task': task, 'until_skill': terminal, 'success': success,
              'operation_success': success, 'task_success': success,
              'requested_scope': 'full'}
    path = tmp_path/'task_result.json'
    path.write_text(json.dumps(report))
    evidence = _evidence(tmp_path, 'actual')
    assert evidence['requested_scope'] == 'full'
    assert evidence['task_success'] is success
    # 同一个操作接口的前缀成功不能提升为完整任务成功。
    report.update(requested_scope='pick', until_skill='pick', success=True,
                  operation_success=True, task_success=None)
    path.write_text(json.dumps(report))
    partial = _evidence(tmp_path, 'actual')
    assert partial['operation_success'] is True
    assert partial['task_success'] is None
