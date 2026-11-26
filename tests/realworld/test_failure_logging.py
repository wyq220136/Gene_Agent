"""每条候选分支、未执行阶段和旁路错误均有诚实的日志语义。"""
from copy import deepcopy
import json
import numpy as np
import pytest
from gene_agent.feedback.realworld_diagnostics import candidate_inventory,RealworldRunDiagnostics,inspect_realworld_run,print_realworld_console_summary
from gene_agent.motion.recording import current_motion_recording
from gene_agent.motion.model import JointState,MotionRequest,MotionResult,MotionOperation,PlanningSceneRef,PoseTarget,MotionFailure,MotionFailureKind


def candidate(name):return {'candidate_id':name,'T_world_tcp':np.eye(4).tolist(),'parameters':{'azimuth_rad':.2}}

def attempt(name,stage,success,request_id=None):
    return dict(candidate_id=name,stage=stage,request={'request_id':request_id or name+':'+stage},
        result=dict(success=success,failure=None if success else {'kind':'goal_ik_no_solution','summary':'not converged','details':{}}))


def test_all_candidates_and_actual_branch_results_are_listed_without_false_path_failures():
    plan={'attempts':[attempt('a','final_grasp_ik',False),attempt('b','final_grasp_ik',True),
        attempt('b','pregrasp_ik',False,'b:branch0'),attempt('b','pregrasp_ik',True,'b:branch1')],
        'selected_candidate_id':'b','stages':[{'name':s,'status':'planned'} for s in ('open','pregrasp','approach','close')]}
    report=candidate_inventory([candidate(s) for s in 'abc'],plan,candidate_order=['b','a','c'])
    rows={r['candidate_id']:r for r in report['candidates']}
    assert report['generated_count']==3
    assert rows['a']['first_rejection']['stage']=='final_grasp_ik'
    assert all(p['reason']=='earlier_stage_rejected' for p in rows['a']['not_reached'])
    assert rows['b']['status']=='selected' and rows['b']['first_rejection'] is None
    assert len(rows['b']['phases'])==3
    assert rows['c']['status']=='not_evaluated'


def test_posture_checks_retain_numeric_condition_and_do_not_disappear():
    checks=[attempt('a','pregrasp_ik',True,'a:p0'),{'candidate_id':'a','stage':'pregrasp_posture_check',
        'request_id':'a:p0','result':{'success':False,'reason_code':'pregrasp_joint_travel_exceeded','maximum_joint_travel_rad':2.1,'limit_rad':1.5}}]
    row=candidate_inventory([candidate('a')],{'attempts':checks})['candidates'][0]
    assert row['status']=='rejected'
    assert row['first_rejection']['stage']=='pregrasp_posture_check'
    assert row['first_rejection']['postcheck_details']['maximum_joint_travel_rad']==2.1


def request(name='a:final_ik'):
    return MotionRequest(name,MotionOperation.IK,'right',JointState(('q',),(0.,)),
        PoseTarget('world_tcp',np.eye(4)),PlanningSceneRef(0,'snapshot','offline'),seed=123)


def test_recorded_failures_are_independent_and_context_restores_on_exception(tmp_path):
    error=RuntimeError('original stage error')
    with pytest.raises(RuntimeError,match='original'):
        with RealworldRunDiagnostics(tmp_path/'run',options={'mode':'off'}) as recorder:
            assert current_motion_recording() is recorder
            for name in ['a:final_ik','a:final_ik']:
                req=request(name)
                result=MotionResult(req.request_id,MotionOperation.IK,False,'curobo',req.planner_mode,.1,
                    failure=MotionFailure(MotionFailureKind.GOAL_IK_NO_SOLUTION,'not converged'))
                recorder.record([req],[result],snapshot={})
            raise error
    assert current_motion_recording() is None
    value=inspect_realworld_run(tmp_path/'run')
    assert len(value['events'])==2
    assert len({e['diagnostic_ref'] for e in value['events']})==2
    assert (tmp_path/'run/failure_analysis.md').is_file()
    assert json.loads((tmp_path/'run/initial_scene/manifest.json').read_text())['status']=='initial_scene_unavailable'


def test_recording_exception_cannot_replace_original_task_exception(tmp_path,monkeypatch):
    import gene_agent.feedback.realworld_diagnostics as module
    with pytest.raises(RuntimeError,match='task failed'):
        with RealworldRunDiagnostics(tmp_path/'run',options={'mode':'off'}):
            monkeypatch.setattr(module,'write_json',lambda *a,**kw: (_ for _ in ()).throw(OSError('disk full')))
            raise RuntimeError('task failed')
    assert current_motion_recording() is None


def test_diagnostic_helper_failure_preserves_event_and_raw_result(tmp_path,monkeypatch):
    import gene_agent.feedback.realworld_diagnostics as module
    with RealworldRunDiagnostics(tmp_path/'run',options={'mode':'off'}) as recorder:
        monkeypatch.setattr(module,'save_motion_diagnostic',lambda *a,**kw: (_ for _ in ()).throw(OSError('detail unavailable')))
        req=request()
        result=MotionResult(req.request_id,MotionOperation.IK,False,'curobo',req.planner_mode,.1,
            failure=MotionFailure(MotionFailureKind.GOAL_IK_NO_SOLUTION,'original failure'))
        recorder.record([req],[result],snapshot={})
    value=inspect_realworld_run(tmp_path/'run')
    assert len(value['events'])==1
    assert value['events'][0]['success'] is False
    assert 'detail unavailable' in value['events'][0]['diagnostic_error']
    assert json.loads((tmp_path/'run'/value['events'][0]['result_ref']).read_text())['failure']['summary']=='original failure'


def test_failed_enter_does_not_leak_recording_context(tmp_path,monkeypatch):
    import gene_agent.feedback.realworld_diagnostics as module
    monkeypatch.setattr(module,'write_json',lambda *a,**kw: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        with RealworldRunDiagnostics(tmp_path/'run',options={'mode':'off'}):pass
    assert current_motion_recording() is None


def test_legacy_log_inspection_is_read_only_and_preserves_unknowns(tmp_path):
    path=tmp_path/'candidate_selection.json'
    payload={'candidate_order':['a','b'],'plan':{'attempts':[attempt('a','final_grasp_ik',False),attempt('b','final_grasp_ik',False)]}}
    path.write_text(json.dumps(payload));before=path.read_bytes()
    value=inspect_realworld_run(tmp_path,candidate='b',stage='final_grasp_ik')
    assert value['inventories'][0]['legacy_extracted']
    assert [r['candidate_id'] for r in value['inventories'][0]['candidates']]==['b']
    assert len(value['events'])==1
    assert path.read_bytes()==before
    assert not (tmp_path/'initial_scene').exists()


def test_read_only_inspector_does_not_import_gpu_or_robot_sdk(tmp_path):
    import subprocess,sys
    root=__import__('pathlib').Path(__file__).resolve().parents[2]
    code="from tools.diagnostics.inspect_realworld_run import main; main(['--run',"+repr(str(tmp_path))+",'--format','json']); import sys; assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules; assert not any('realman.connection' in n for n in sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_invalid_model_does_not_discard_available_raw_perception(tmp_path):
    from test_scene_archive import toy_scene
    from gene_agent.realworld.scene_archive import verify_scene_archive
    scene=toy_scene(tmp_path);scene['scene_xml_path']=str(tmp_path/'missing.xml')
    with RealworldRunDiagnostics(tmp_path/'run',options={'mode':'off'}) as recorder:
        recorder.capture_initial(scene)
    _,manifest=verify_scene_archive(tmp_path/'run/initial_scene')
    assert manifest['status']=='initial_scene_unavailable'
    assert manifest['perception_status']=='captured'
    assert all(manifest['initial_frame_files'])


def test_console_summary_exposes_underlying_candidate_failure_and_viser_command(tmp_path,capsys):
    selection=tmp_path/'grasp/selection/selection.json';selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({'candidate_order':['a'],'candidate_parameters':{'candidate_targets':[
        {'candidate_id':'a','T_world_tcp':np.eye(4).tolist(),'table_clamp_delta_sdk_z_m':.1}]},
        'plan':{'attempts':[{'candidate_id':'a','stage':'final_grasp_ik','request':{},
            'result':{'success':False,'failure':{'kind':'backend_error','backend_status':'AttributeError',
                'summary':"'Tensor' object has no attribute 'robot_spheres'"}}}]}}))
    report={'status':'failed','pick_success':False,'first_failure':{
        'code':'grasp_selection_failed','message':'原 PickSkill 没有生成完整的抓取前缀'}}

    command=print_realworld_console_summary(tmp_path,report,subtask='pick_bowl',execute=False,collision_mode='full')
    output=capsys.readouterr().out

    assert '候选=1' in output
    assert "robot_spheres" in output
    assert '已保存=0' in output and '没有生成关节轨迹' in output
    assert 'grasp_selection_failed' in output
    assert 'view_realworld_scene.py' in command and str(tmp_path) in command
