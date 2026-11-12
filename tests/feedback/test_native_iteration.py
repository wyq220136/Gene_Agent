"""同案例回归必须实际测到目标阶段；阶段改善不能冒充完整任务成功。"""
import json
import subprocess
import sys

from gene_agent.feedback.native_run import NativeRunRecorder


def recorded(root,*,seed=42,stage='strike',success=False,execution='executed',task=False,failure_code='blocked'):
    rec=NativeRunRecorder(root,manifest={'episode_id':root.name,'benchmark':'hit_fixture',
        'embodiment':'arm','native_identity':{'native_seed':seed}})
    rec.append_event({'episode_id':root.name,'stage_id':stage,'kind':'stage_outcome',
        'success':success,'execution_status':execution,
        'failure':None if success else {'code':failure_code}})
    score=root/'native_score.json';score.write_text(json.dumps({'success':task}))
    rec.finalize(native_result_ref=score)
    return root


def test_earlier_failure_does_not_test_physical_strike_hypothesis(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    before=recorded(tmp_path/'before')
    after=recorded(tmp_path/'after',stage='align',execution='not_executed')
    result=compare_stage_runs(before,after,stage='strike',hypothesis='减速改善首触',require_executed=True)
    assert result['verdict']=='not_tested'
    assert result['case_match'] is True and not result['after']['reached']
    assert result['hypothesis_verdict']=='requires_observable_review'


def test_planning_only_does_not_test_physical_execution(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a'),recorded(tmp_path/'b',execution='not_executed'),
        stage='strike',hypothesis='抓法可抵抗接触',require_executed=True)
    assert result['after']['reached'] and not result['after']['executed']
    assert result['verdict']=='not_tested'


def test_observed_snapshot_without_execution_status_is_not_motion_evidence(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    before=recorded(tmp_path/'a')
    rec=NativeRunRecorder(tmp_path/'b',manifest={'episode_id':'b','benchmark':'hit_fixture',
        'embodiment':'arm','native_identity':{'native_seed':42}})
    rec.append_event({'episode_id':'b','stage_id':'strike','kind':'actual_sample','state_id':'static'})
    rec.finalize(native_result_ref=None)
    result=compare_stage_runs(before,rec.root,stage='strike',hypothesis='接触过程改善',require_executed=True)
    assert result['verdict']=='not_tested' and not result['after']['executed']


def test_different_seed_is_not_same_case_evidence(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a'),recorded(tmp_path/'b',seed=43,success=True),
        stage='strike',hypothesis='原失败修复')
    assert result['case_match'] is False and result['verdict']=='not_comparable'


def test_stage_pass_does_not_change_native_task_failure(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a'),recorded(tmp_path/'b',success=True,task=False),
        stage='strike',hypothesis='击打阶段改善')
    assert result['verdict']=='stage_passed' and result['after']['task_success'] is False


def test_partial_native_success_preserves_incomplete_stage(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a',stage='withdraw'),
        recorded(tmp_path/'b',stage='withdraw',execution='partial',task=True),
        stage='withdraw',hypothesis='有界目标可完成原撤离',require_executed=True)
    assert result['verdict']=='native_task_succeeded_stage_partial'
    assert result['after']['completed'] is False and result['after']['task_success'] is True


def test_native_success_with_failed_postcondition_is_not_partial_execution(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a'),
        recorded(tmp_path/'b',execution='executed',task=True,failure_code='tool_postcondition_failed'),
        stage='strike',hypothesis='改善夹持')
    assert result['verdict']=='native_task_succeeded_stage_failed'
    assert result['after']['execution_status']=='executed'


def test_native_success_termination_is_not_failed_tool_postcondition(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a',stage='withdraw',execution='not_executed'),
        recorded(tmp_path/'b',stage='withdraw',execution='executed',task=True,failure_code='native_terminated'),
        stage='withdraw',hypothesis='实际撤离通过原评分')
    assert result['verdict']=='native_task_succeeded_stage_terminated'
    assert result['after']['completed'] is False


def test_comparison_reports_changed_failure_without_claiming_fix(tmp_path):
    from gene_agent.feedback.native_run import compare_stage_runs
    result=compare_stage_runs(recorded(tmp_path/'a',execution='not_executed',failure_code='goal_ik_no_solution'),
        recorded(tmp_path/'b',failure_code='tool_postcondition_failed'),stage='strike',hypothesis='替代目标改善IK')
    assert result['verdict']=='failure_changed' and result['failure_changed'] is True
    assert result['after']['task_success'] is False


def test_inspection_cli_outputs_same_case_hypothesis_evidence(tmp_path):
    before=recorded(tmp_path/'before');after=recorded(tmp_path/'after',success=True)
    result=subprocess.run([sys.executable,'tools/diagnostics/inspect_native_run.py',
        '--run',str(after),'--compare-with',str(before),'--stage','strike',
        '--hypothesis','同一输入下改进执行','--require-executed','--format','json'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['verdict']=='stage_passed' and report['case_match'] is True


def test_summary_exposes_effective_ik_policy_and_iteration_entry(tmp_path):
    rec=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'})
    rec.append_event({'episode_id':'one','stage_id':'pregrasp','kind':'planning','state_id':'s',
        'ik_policy':{'mode':'bounded_approximate','budget':{'max_position_error_m':.01,'max_rotation_error_rad':.05}},
        'planning':{'success':False,'failure':{'kind':'goal_ik_no_solution'}}})
    rec.finalize(native_result_ref=None)
    text=(rec.root/'development_summary.md').read_text()
    assert 'bounded_approximate' in text and '0.01' in text and '--compare-with' in text
