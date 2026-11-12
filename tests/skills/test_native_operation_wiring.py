"""检查实际组合根能在 Pick 成功后交给 Hit。"""
import json
from contextlib import nullcontext
from types import SimpleNamespace
from gene_agent.tasks.policies.novel_tool_task import execution, pick_backend, hit_backend
from gene_agent.skills.library import pick, hit


def test_successful_pick_enters_hit(tmp_path, monkeypatch):
    candidate=tmp_path/'candidate.json'
    candidate.write_text(json.dumps({'group':'left','hit_parameters':{}}))
    session=SimpleNamespace(describe=lambda:{"api_available":True})
    monkeypatch.setattr(execution,'ProcessPlanningSession',lambda *a,**k:nullcontext(session))
    backend=object()
    monkeypatch.setattr(pick_backend,'make_pick_backend',lambda **k:backend)
    monkeypatch.setattr(pick,'run',lambda *a,**k:SimpleNamespace(success=True))
    constructed=[]
    monkeypatch.setattr(hit_backend,'NativeHitBackend',lambda *a,**k:constructed.append((a,k)) or backend)
    hit_options=[]
    monkeypatch.setattr(hit,'run',lambda b,**k:hit_options.append(k) or 'hit_result')
    result=execution.run_operation('hit',runtime=None,robot={'joint_groups':['left']},
        start=SimpleNamespace(values={'episode_id':'test'}),
        recorder=SimpleNamespace(append_event=lambda e:None),
        options={'gene_candidate':str(candidate),'_scene_provider':None,'_target_body':'target'},output=tmp_path)
    assert result=='hit_result'
    assert constructed[0][0][0] is backend
    assert hit_options==[{'withdraw':False}]
    candidate.write_text(json.dumps({'group':'left','hit_parameters':{'withdraw_distance_m':.1}}))
    execution.run_operation('hit',runtime=None,robot={'joint_groups':['left']},
        start=SimpleNamespace(values={'episode_id':'test'}),
        recorder=SimpleNamespace(append_event=lambda e:None),
        options={'gene_candidate':str(candidate),'_scene_provider':None,'_target_body':'target'},output=tmp_path)
    assert hit_options[-1]=={'withdraw':True}


def test_other_arm_only_after_unexecuted_candidate_screen_failure(tmp_path,monkeypatch):
    candidate=tmp_path/'candidate.json'
    candidate.write_text(json.dumps({'group':'left','search_other_group':True}))
    session=SimpleNamespace(describe=lambda:{"api_available":True})
    monkeypatch.setattr(execution,'ProcessPlanningSession',lambda *a,**k:nullcontext(session))
    monkeypatch.setattr(pick_backend,'make_pick_backend',lambda **k:k['request'].group)
    calls=[]
    failure_code=['no_feasible_grasp_candidate']
    def run(backend,*a,**k):
        calls.append(backend)
        return SimpleNamespace(success=backend=='right',failure_code=failure_code[0])
    monkeypatch.setattr(pick,'run',run)
    kwargs=dict(runtime=None,robot={'joint_groups':['left','right']},start=SimpleNamespace(values={'episode_id':'test'}),recorder=SimpleNamespace(append_event=lambda e:None),options={'gene_candidate':str(candidate),'_scene_provider':None},output=tmp_path)
    assert execution.run_operation('pick',**kwargs).success
    assert calls==['left','right']
    calls.clear();failure_code[0]='path_not_found'
    assert not execution.run_operation('pick',**kwargs).success
    assert calls==['left']


def test_wrong_planner_dependency_fails_before_candidate_or_execution(tmp_path,monkeypatch):
    import pytest
    session=SimpleNamespace(describe=lambda:{'api_available':False,'curobo_path':'old/curobo','python':'native-python'})
    monkeypatch.setattr(execution,'ProcessPlanningSession',lambda *a,**k:nullcontext(session))
    with pytest.raises(RuntimeError,match='old/curobo'):
        execution.run_operation('hit',runtime=None,robot={},start=SimpleNamespace(values={'episode_id':'e'}),
            recorder=SimpleNamespace(append_event=lambda e:None),options={},output=tmp_path)
