"""证据记录不能伪造评分，失败前缀和未知诊断必须保留。"""
import json
import pytest


def test_partial_run_retains_events_and_unknown_score(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder, load_native_run
    root=tmp_path/'run';rec=NativeRunRecorder(root,manifest={'episode_id':'one'})
    rec.append_event({'episode_id':'one','stage_id':'approach','state_id':'s1','sim_time_s':.1,'kind':'execution'})
    report=load_native_run(rec.finalize(native_result_ref=None,error=RuntimeError('controller failed')))
    assert report['task_success'] is None and report['status']=='error'
    assert report['error']=='RuntimeError: controller failed'
    assert report['events'][0]['stage_id']=='approach'
    assert (root/'events.jsonl').read_text().endswith('\n')
    with pytest.raises(FileExistsError):NativeRunRecorder(root,manifest={})


def test_native_score_and_raw_hash_verified(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder, load_native_run
    native=tmp_path/'native.json';native.write_text(json.dumps({'success':False,'cancel_reason':'xy_drift'}))
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_event({'episode_id':'one','stage_id':'hit','observed_facts':{'drift_m':.012},'hypotheses':[]})
    result=rec.finalize(native_result_ref=native)
    assert load_native_run(result)['task_success'] is False
    native.write_text('{"success": true}')
    with pytest.raises(ValueError,match='hash'):load_native_run(result)


def test_foreign_episode_event_rejected_before_append(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    with pytest.raises(ValueError):rec.append_event({'episode_id':'other','stage_id':'move'})
    assert not (tmp_path/'run/events.jsonl').exists()


def test_unavailable_diagnostics_explicitly_unknown():
    from gene_agent.simulation.robotwin.diagnostics import RobotwinDiagnostics
    from types import SimpleNamespace
    diag=RobotwinDiagnostics(SimpleNamespace(scene=SimpleNamespace(),info={}))
    row=diag.capture(stage_id='move',state_id='s')
    assert row['contacts']['status']=='unknown' and row['hypotheses']==[] and row['state_id']=='s'


def test_inspection_cli_filters_stage_without_rerunning(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder
    import subprocess,sys
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_event({'episode_id':'one','stage_id':'move','kind':'failure','observed_facts':{'code':'tracking_error'}})
    rec.append_event({'episode_id':'one','stage_id':'close','kind':'execution'})
    rec.finalize(native_result_ref=None)
    result=subprocess.run([sys.executable,'tools/diagnostics/inspect_native_run.py','--run',str(tmp_path/'run'),'--stage','move','--format','json'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['task_success'] is None and [e['stage_id'] for e in report['events']]==['move']


def test_actual_state_event_keeps_episode_and_time_without_duplicate_fields(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder,load_native_run
    from gene_agent.skills.contracts import OperationState
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_state('move',OperationState('actual','native',{'episode_id':'one','sim_time_s':.04,'robots':{}}))
    row=load_native_run(rec.finalize(native_result_ref=None))['events'][0]
    assert row['state_id']=='actual' and row['sim_time_s']==.04 and row['episode_id']=='one'


@pytest.mark.parametrize('raw',[None,'{bad json','{"success": "yes"}'])
def test_unreadable_score_preserves_original_failure_and_events(tmp_path,raw):
    from gene_agent.feedback.native_run import NativeRunRecorder,load_native_run
    score=tmp_path/'score.json'
    if raw is not None:score.write_text(raw)
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_event({'episode_id':'one','stage_id':'move','kind':'failure'})
    result=load_native_run(rec.finalize(native_result_ref=score,error=RuntimeError('original failure')))
    assert result['task_success'] is None and result['status']=='error'
    assert result['error']=='RuntimeError: original failure'
    assert result['recording_errors'] and result['events'][0]['stage_id']=='move'


def test_unavailable_end_state_records_unknown_without_fabricating_pose(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder,load_native_run
    from gene_agent.skills.contracts import OperationState
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_state('end',OperationState('unavailable:s','unavailable',{'state_unavailable':True,'error':'native observation lost'}))
    row=load_native_run(rec.finalize(native_result_ref=None))['events'][0]
    assert row['episode_id']=='one' and row['kind']=='state_unavailable'
    assert row['state_unavailable'] and 'robots' not in row and 'sim_time_s' not in row


def test_evaluation_index_keeps_native_media_and_missing_episode_explicit(tmp_path):
    from gene_agent.feedback.native_run import index_native_evaluation
    native=tmp_path/'native/task';native.mkdir(parents=True)
    (native/'episode0.mp4').write_bytes(b'native media bytes')
    (native/'_result.txt').write_text('0/1')
    path=index_native_evaluation(tmp_path)
    result=json.loads(path.read_text())
    assert result['episodes']==[] and result['task_success'] is None
    assert {a['kind'] for a in result['native_artifacts']}=={'video','native_result'}
    assert all(a['sha256'] for a in result['native_artifacts'])


def test_native_identity_links_video_counter_and_diagnostic_seed_only():
    from gene_agent.simulation.robotwin.diagnostics import native_episode_identity
    from types import SimpleNamespace
    env=SimpleNamespace(ep_num=8,test_num=3,guidance_seed=100007,eval_video_path='/tmp/native')
    row=native_episode_identity(env)
    assert row['native_episode_ordinal']==8 and row['native_seed']==100007
    assert row['native_video_ref']=='/tmp/native/episode3.mp4'


def test_finalize_writes_development_summary_without_inventing_timing(tmp_path):
    from gene_agent.feedback.native_run import NativeRunRecorder
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_event({'episode_id':'one','stage_id':'align','kind':'planning','state_id':'s',
        'planning':{'request_id':'native/align','success':False,'failure':{'kind':'goal_ik_no_solution'}}})
    rec.finalize(native_result_ref=None)
    summary=json.loads((rec.root/'planning_summary.json').read_text())
    assert summary['recorded_planning_s'] is None
    text=(rec.root/'development_summary.md').read_text()
    assert 'align' in text and 'inspect_native_run.py' in text and '未知' in text
