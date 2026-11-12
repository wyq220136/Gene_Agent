"""统计真实记录语义：失败不能丢，batch和lease不能重复计时。"""
import json
from copy import deepcopy
import pytest
from gene_agent.feedback.planning_performance import summarize_planning_run


def result(name, duration, *, success=True, lease='lease1', setup=2., call='call1', size=1):
    return dict(request_id=name,operation='ik' if size>1 else 'plan',success=success,duration_s=duration,
        metrics={'planner_cache':{'acquisition_id':lease,'setup_duration_s':setup,'cache_hit':True},
                 'planning_timing':{'call_id':call,'batch_size':size}})


def test_new_call_log_counts_failed_attempts_and_ignores_duplicated_event_results(tmp_path):
    first=result('native/align',3.,success=False)
    second=result('native/align',.5,lease='lease2',setup=.1,call='call2')
    calls=[dict(call_id='a',operation='plan_to',results=[first],roundtrip_wall_s=5.2),
           dict(call_id='b',operation='plan_to',results=[second],roundtrip_wall_s=.7)]
    (tmp_path/'planning_calls.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in calls))
    (tmp_path/'events.jsonl').write_text(json.dumps({'kind':'planning','stage_id':'align','planning':first})+'\n')
    s=summarize_planning_run(tmp_path)
    assert s['source']=='planning_calls'
    assert s['failed_result_count']==1 and s['call_count']==2
    assert s['backend_duration_s']==3.5 and s['pool_setup_s']==2.1
    assert s['recorded_planning_s']==5.6
    assert s['request_roundtrip_s']==pytest.approx(5.9)


def test_legacy_batch_counts_setup_once_and_preserves_separate_acquisitions(tmp_path):
    a=[result(f'candidate/right/{i}',.25,size=4) for i in range(4)]
    path=result('candidate/right/0/pregrasp',1.,lease='lease2',setup=.2,call='call2')
    (tmp_path/'candidate_screen_right.json').write_text(json.dumps({'results':a,'path_screens':[{'pregrasp':path}]}))
    s=summarize_planning_run(tmp_path)
    assert s['backend_duration_s']==2. and s['pool_setup_s']==2.2
    assert s['call_count']==2 and s['result_count']==5
    assert s['request_roundtrip_s'] is None


def test_missing_times_and_truncated_lines_report_incomplete_coverage(tmp_path):
    (tmp_path/'planning_calls.jsonl').write_text(json.dumps({'call_id':'a','operation':'plan_to','error':'worker failed','roundtrip_wall_s':3.})+'\n{unfinished')
    s=summarize_planning_run(tmp_path)
    assert s['request_roundtrip_s']==3.
    assert s['recorded_planning_s'] is None
    assert s['error_call_count']==1 and s['warnings']


def test_parent_wait_not_added_to_nested_solver_time(tmp_path):
    r=result('native/lift',.5,setup=.2)
    (tmp_path/'events.jsonl').write_text(json.dumps({'kind':'planning','stage_id':'lift','planning':r,'planning_wall_s':1.,'scene_snapshot_s':.2})+'\n')
    s=summarize_planning_run(tmp_path)
    assert s['recorded_planning_s']==.7
    assert s['action_planning_wait_s']==1.


def test_stage_filter_also_filters_candidate_paths(tmp_path):
    events=[dict(kind='planning',stage_id='lift',planning=result('native/lift',1.)),
            dict(kind='planning',stage_id='align',planning=result('native/align',2.,lease='lease2',setup=0.,call='c2'))]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    s=summarize_planning_run(tmp_path,stage='align')
    assert s['backend_duration_s']==2. and s['result_count']==1
