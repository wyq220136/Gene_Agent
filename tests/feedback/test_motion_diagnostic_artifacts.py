import json
from gene_agent.feedback.motion_diagnostics import save_motion_diagnostic, validate_diagnostic_bundle


def test_bundle_hash_rejects_changed_replay_configuration(tmp_path):
    from gene_agent.motion.curobo.diagnostics import digest
    import hashlib,sys
    module=tmp_path/'module.py';module.write_text('# dependency')
    replay={'joint_names':['a'],'scene_snapshot_sha256':'s','scene_scope':'test','scene_model':{}}
    request={'scene':{'snapshot_sha256':'s','scope':'test'}}
    snapshot={'snapshot_sha256':'s','scope':'test','scene_model':{}}
    bundle=dict(replay_config=replay,replay_config_sha256=digest(replay),request=request,request_sha256=digest(request),
        snapshot=snapshot,snapshot_payload_sha256=digest(snapshot),environment=dict(curobo_module=str(module),python=sys.version,executable=sys.executable),
        asset_hashes={str(module):hashlib.sha256(module.read_bytes()).hexdigest()})
    validate_diagnostic_bundle(bundle)
    bundle['replay_config']['joint_names']=['b']
    import pytest
    with pytest.raises(ValueError,match='hash'):validate_diagnostic_bundle(bundle)


def test_saving_missing_diagnostic_does_not_invent_replay(tmp_path):
    event={'stage_id':'align','planning':{'success':False,'failure':{'kind':'goal_ik_no_solution'}}}
    path=save_motion_diagnostic(tmp_path,event)
    saved=json.loads(path.read_text())
    assert saved['diagnostic'] is None
    assert saved['summary']['unknowns']
    assert saved['media']['actual']['status']=='unknown'


def test_recheck_refuses_incomplete_dependency_manifest():
    from gene_agent.motion.curobo.diagnostics import digest
    import pytest
    replay={'joint_names':['a']}
    with pytest.raises(ValueError,match='缺少'):
        validate_diagnostic_bundle({'replay_config':replay,'replay_config_sha256':digest(replay)})


def test_different_targets_in_same_state_keep_separate_failure_artifacts(tmp_path):
    import numpy as np
    event={'episode_id':'e','stage_id':'align','state_id':'s',
           'planning':{'success':False,'failure':{'kind':'goal_ik_no_solution'}},
           'target_world_tcp':np.eye(4).tolist()}
    first=save_motion_diagnostic(tmp_path,event)
    second_goal=np.eye(4);second_goal[0,3]=.1
    second=save_motion_diagnostic(tmp_path,{**event,'target_world_tcp':second_goal.tolist()})
    assert first!=second
    assert json.loads(first.read_text())['target_world_tcp']==event['target_world_tcp']
    assert json.loads(second.read_text())['target_world_tcp']==second_goal.tolist()


def test_retries_of_same_pose_keep_separate_failure_artifacts(tmp_path):
    event={'episode_id':'e','stage_id':'align','state_id':'s',
        'planning':{'request_id':'same','success':False,'failure':{'kind':'goal_ik_no_solution'},
                    'metrics':{'planning_timing':{'call_id':'first'}}}}
    first=save_motion_diagnostic(tmp_path,event)
    event['planning']['metrics']['planning_timing']['call_id']='second'
    second=save_motion_diagnostic(tmp_path,event)
    assert first!=second
    assert json.loads(first.read_text())['planning']['metrics']['planning_timing']['call_id']=='first'


def test_inspection_does_not_offer_viewer_for_summary_only_artifact(tmp_path):
    import subprocess,sys
    event={'episode_id':'e','stage_id':'align','state_id':'s','kind':'planning',
           'planning':{'success':False,'failure':{'kind':'goal_ik_no_solution'}}}
    (tmp_path/'events.jsonl').write_text(json.dumps(event)+'\n')
    save_motion_diagnostic(tmp_path,event)
    result=subprocess.run([sys.executable,'tools/diagnostics/inspect_native_run.py','--run',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'view_motion_diagnostics.py --bundle' not in result.stdout
    assert '未采集' in result.stdout
