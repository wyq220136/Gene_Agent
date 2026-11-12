"""在线显示只消费已落盘的记录，不改变任务结果。"""
import json
from pathlib import Path


def test_task_live_view_keeps_task_page_and_task_log():
    """实际执行前端函数，防止查看器抢走 Task 日志或失败被就绪提示覆盖。"""
    import shutil, subprocess
    import pytest
    if not shutil.which('node'):pytest.skip('需要 node 验证网页状态')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements={};const $=id=>elements[id] ||= {hidden:true,textContent:'',getAttribute(){return ''},removeAttribute(){}};
const state={epoch:1,task:'microwave'}, calls=[];
const context={state,$,console,toast(){},showPage(p){calls.push(p)},
 async launch(action){state.job=action;return {id:action,inputs:{output_run:'runs/test'}}},
 async loadJob(id){calls.push('log:'+id)},async refreshRuns(){},
 async api(url){return url.includes('id=run')?{state:'failed',log:'bowl: outside_table_support'}:
 {state:'running',viewer_ready:true,inputs:{viewer_url:'http://localhost:1234'}}},states:{failed:'失败'}};
vm.createContext(context);vm.runInContext(fs.readFileSync('gene_agent/workbench/static/planning_pages.js','utf8'),context);
(async()=>{
 await vm.runInContext("beginLiveRun('run')",context);
 assert(calls.includes('results'));assert(!calls.includes('skills'));
 assert.equal(state.job,'run');assert.equal(state.taskViewJob,'live_viewer');
 assert(calls.includes('log:run'));
 await vm.runInContext('refreshPlanningPages()',context);
 assert($('#task-preview-status').textContent.includes('outside_table_support'));
 assert($('#task-preview-status').textContent.includes('失败'));
 // 初态可能在查看器加载前就失败；绑定查看器不能清空已经收到的失败。
 await vm.runInContext("attachPureViewer({id:'live_viewer'},'task')",context);
 assert($('#task-preview-status').textContent.includes('outside_table_support'));
 state.visualPending=null;calls.length=0;
 await vm.runInContext("beginLiveRun('run',{preview_skill:'pick_bowl'})",context);
 assert(calls.includes('skills'));assert.equal(state.skillViewJob,'live_viewer');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run(['node','-e',script],check=True,cwd=Path(__file__).resolve().parents[2])


def test_atomic_plan_checkpoint_preserves_failed_prefix(tmp_path):
    from gene_agent.feedback.live_record import publish_plan
    report={'record_type':'pure_visual_plan','task':'sweep','status':'planning','stages':[{'name':'pregrasp','success':True}]}
    publish_plan(tmp_path,report)
    assert json.loads((tmp_path/'live/plan.json').read_text())['stages']==report['stages']
    report.update(status='planning_failed',failure_stage='approach',planning_success=False)
    publish_plan(tmp_path,report)
    result=json.loads((tmp_path/'live/plan.json').read_text())
    assert result['stages'][0]['success'] and result['failure_stage']=='approach'
    assert json.loads((tmp_path/'live/manifest.json').read_text())['status']=='planning_failed'


def test_tail_never_consumes_incomplete_frame(tmp_path):
    from gene_agent.feedback.live_record import JsonlTail
    p=tmp_path/'frames.jsonl';p.write_text('{"sim_time_s":0}\n{"sim_time_s":')
    tail=JsonlTail(p)
    assert tail.read()==[{'sim_time_s':0}]
    with p.open('a') as f:f.write('1}\n')
    assert tail.read()==[{'sim_time_s':1}]
    assert tail.read()==[]


def test_physics_recording_uses_exact_samples_and_finishes_failed(tmp_path):
    from types import SimpleNamespace
    import mujoco
    from gene_agent.feedback.live_record import live_physics_recording,record_actual_sample,JsonlTail
    run=tmp_path/'run'
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);runtime=SimpleNamespace(model=model,data=data)
    with live_physics_recording(run,enabled=True):
        first={'qpos':data.qpos.tolist(),'sim_time_s':float(data.time)}
        record_actual_sample(runtime,first)
        for _ in range(8):mujoco.mj_step(model,data)
        final={'qpos':data.qpos.tolist(),'sim_time_s':float(data.time)}
        record_actual_sample(runtime,final)
        (run/'result.json').write_text(json.dumps({'status':'failed','success':False,'failure_stage':'contact'}))
    rows=JsonlTail(run/'live/frames.jsonl').read()
    assert [r['qpos'] for r in rows]==[first['qpos'],final['qpos']]
    frozen=mujoco.MjModel.from_binary_path(str(run/'live'/rows[0]['model']))
    assert frozen.nq==model.nq
    result=json.loads((run/'live/manifest.json').read_text())
    assert result['status']=='failed' and result['task_success'] is False and result['sample_count']==2


def test_live_viewer_route_does_not_require_final_report(tmp_path):
    from gene_agent.workbench.catalog import build_command
    import sys
    cmd=build_command(tmp_path,sys.executable,'sweep','live_viewer',{'run':'runs/future'},tmp_path/'viewer')
    assert 'gene_agent.visualization.live_run' in cmd
    assert not (tmp_path/'runs/future/result.json').exists()


def test_actual_viewer_forwards_frozen_model_before_building_visuals(tmp_path,monkeypatch):
    import mujoco,numpy as np
    from types import SimpleNamespace
    from gene_agent.visualization.live_run import ActualStreamApp
    from gene_agent.visualization.pick_bowl import scene
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>')
    (tmp_path/'live').mkdir();mujoco.mj_saveModel(model,str(tmp_path/'live/model.mjb'),None)
    data=mujoco.MjData(model)
    def build(server,frozen,state,*args):
        assert np.linalg.det(state.geom_xmat[0].reshape(3,3))>0.99
        return SimpleNamespace(set_collision_visible=lambda value:None,update_visuals=lambda value:None)
    monkeypatch.setattr(scene,'PickBowlScene',build)
    app=ActualStreamApp.__new__(ActualStreamApp)
    from threading import RLock
    app._lock=RLock()
    app.run=tmp_path;app.server=None;app.model_name=None;app.renderer=None;app.last_index=-1
    app.frames=[{'model':'model.mjb','qpos':data.qpos.tolist(),'sim_time_s':0.}];app.times=[0.]
    app.set_camera=lambda *args:None
    app.seek(0.)
    assert app.last_index==0


def test_interrupted_plan_stays_discoverable_without_final_report(tmp_path):
    from gene_agent.feedback.live_record import publish_plan
    from gene_agent.workbench.records import describe_run
    run=tmp_path/'runs/interrupted'
    report={'record_type':'pure_visual_plan','task':'sweep','status':'planning','success':None,
            'stages':[{'name':'pregrasp','success':True}]}
    publish_plan(run,report)
    row=describe_run(tmp_path,'runs/interrupted')
    assert row['live_available'] and row['record_type']=='pure_visual_plan'
    assert row['stages']==report['stages'] and row['success'] is None
    assert row['reports']==['runs/interrupted/live/plan.json']


def test_physics_exception_is_not_hidden_as_stopped(tmp_path):
    import mujoco,pytest
    from types import SimpleNamespace
    from gene_agent.feedback.live_record import live_physics_recording,record_actual_sample
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
    runtime=SimpleNamespace(model=model,data=mujoco.MjData(model))
    with pytest.raises(RuntimeError):
        with live_physics_recording(tmp_path,enabled=True):
            record_actual_sample(runtime,{'qpos':[],'sim_time_s':0.})
            raise RuntimeError('planner crashed')
    result=json.loads((tmp_path/'live/manifest.json').read_text())
    assert result['status']=='failed' and result['error']=='RuntimeError: planner crashed'
    assert result['sample_count']==1 and result['task_success'] is None
