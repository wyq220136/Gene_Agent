"""本地工作台的任务事实、路径与进程生命周期回归。"""
import json
from pathlib import Path
import sys
import time

import pytest

from gene_agent.workbench.catalog import build_command, safe_path
from gene_agent.workbench.records import describe_run
from gene_agent.workbench.jobs import JobManager


def test_paths_reject_escape_and_symlink(tmp_path):
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs/link').symlink_to('/etc')
    for name in ['../etc/passwd', 'runs/link/passwd']:
        with pytest.raises(ValueError):
            safe_path(tmp_path, name, ('runs',))


def test_command_is_whitelisted_and_no_shell(tmp_path):
    with pytest.raises(ValueError):
        build_command(tmp_path, sys.executable, 'unknown', 'run', {}, tmp_path/'out')
    with pytest.raises(ValueError):
        build_command(tmp_path, sys.executable, 'sweep', 'run', {'seed': ';touch /tmp/x'}, tmp_path/'out')


def test_failed_task_stays_failed(tmp_path):
    p=tmp_path/'runs/a';p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps({'task':'sweep_cube_into_dustpan','success':False,
        'status':'failed','failure_stage':'approach','planner_backend':'curobo'}))
    row=describe_run(tmp_path, 'runs/a')
    assert row['success'] is False
    assert row['failure_stage']=='approach'
    assert row['planner_backend']=='curobo'


def test_broken_report_is_visible(tmp_path):
    p=tmp_path/'runs/b';p.mkdir(parents=True)
    (p/'result.json').write_text('{')
    row=describe_run(tmp_path,'runs/b')
    assert row['success'] is None
    assert row['errors']


def test_assisted_success_stays_explicit(tmp_path):
    p=tmp_path/'runs/assisted';p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps({'success':True,'simulation_assistance':'weld_after_contact'}))
    row=describe_run(tmp_path,'runs/assisted')
    assert row['simulation_assistance']=='weld_after_contact'
    assert '辅助' in row['evidence_mode']


def wait_done(manager, jid):
    for _ in range(150):
        row=manager.get(jid)
        if row['state'] in {'completed','failed','canceled','interrupted'}:return row
        time.sleep(.02)
    raise AssertionError('job did not finish')


def test_job_records_real_returncode(tmp_path):
    manager=JobManager(tmp_path)
    job=manager.submit('test',{},[[sys.executable,'-c','print("evidence");raise SystemExit(3)']])
    row=wait_done(manager,job['id'])
    assert row['state']=='failed'
    assert row['steps'][0]['returncode']==3
    assert 'evidence' in manager.log(job['id'])
    manager.close()


def test_cancel_only_own_process(tmp_path):
    manager=JobManager(tmp_path)
    job=manager.submit('test',{},[[sys.executable,'-c','import time;time.sleep(60)']])
    time.sleep(.1)
    manager.cancel(job['id'])
    assert wait_done(manager,job['id'])['state']=='canceled'
    manager.close()


def test_restart_marks_unfinished_interrupted(tmp_path):
    p=tmp_path/'runs/workbench/jobs/old';p.mkdir(parents=True)
    (p/'job.json').write_text(json.dumps({'id':'old','state':'running','created_at':0}))
    manager=JobManager(tmp_path)
    assert manager.get('old')['state']=='interrupted'
    manager.close()


def test_second_server_cannot_take_running_jobs(tmp_path):
    first=JobManager(tmp_path)
    with pytest.raises(ValueError,match='已有工作台'):JobManager(tmp_path)
    first.close()
    second=JobManager(tmp_path);second.close()


def test_distribution_does_not_wait_for_simulation_or_viewers(tmp_path):
    manager=JobManager(tmp_path)
    sleeper=[[sys.executable,'-c','import time;time.sleep(60)']]
    try:
        manager.submit('run',{},sleeper)
        for _ in range(4):
            manager.submit('scene_viewer',{},sleeper,background=True)
        preview=manager.submit('distribution',{},[[sys.executable,'-c','print("preview ready")']])
        assert wait_done(manager,preview['id'])['state']=='completed'
        assert 'preview ready' in manager.log(preview['id'])
    finally:
        manager.close()


def test_formal_simulation_remains_serial(tmp_path):
    manager=JobManager(tmp_path)
    try:
        first=manager.submit('run',{},[[sys.executable,'-c','import time;time.sleep(60)']])
        second=manager.submit('run',{},[[sys.executable,'-c','print("second")']])
        time.sleep(.1)
        assert manager.get(second['id'])['state']=='queued'
        manager.cancel(first['id'])
        assert wait_done(manager,second['id'])['state']=='completed'
    finally:
        manager.close()


def test_visual_mode_routes_and_never_exports_physics_video(tmp_path):
    scene=tmp_path/'data/scene';scene.mkdir(parents=True)
    options={'scene':'data/scene','execution_mode':'visual','preview_skill':'pick_brush'}
    cmd=build_command(tmp_path,sys.executable,'sweep','run',options,tmp_path/'out')
    assert 'plan_visual.py' in cmd[1] and '--skill' in cmd
    assert not any('mujoco' in part for part in cmd)
    with pytest.raises(ValueError):
        build_command(tmp_path,sys.executable,'sweep','run_video',options,tmp_path/'out')
    for invalid in [float('nan'),float('inf'),46]:
        with pytest.raises(ValueError):
            build_command(tmp_path,sys.executable,'sweep','visual_plan',{**options,'tilt':invalid},tmp_path/'out')


def test_knowledge_preview_is_independent_and_rejects_unknown_selection(tmp_path):
    (tmp_path/'data/scene').mkdir(parents=True)
    options={'scene':'data/scene','preview_knowledge':'brush_grasp'}
    cmd=build_command(tmp_path,sys.executable,'sweep','knowledge_preview',options,tmp_path/'out')
    assert 'gene_agent.visualization.pure_planning.cli' in cmd and '--knowledge' in cmd
    assert '--run' not in cmd
    with pytest.raises(ValueError):
        build_command(tmp_path,sys.executable,'sweep','knowledge_preview',{**options,'preview_knowledge':'bad'},tmp_path/'out')


def test_visual_planning_success_is_not_task_success(tmp_path):
    p=tmp_path/'runs/pure';p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps({'record_type':'pure_visual_plan','success':None,
        'task_success':None,'planning_success':True,'status':'planned'}))
    row=describe_run(tmp_path,'runs/pure')
    assert row['success'] is None and row['planning_success'] is True
    assert row['record_type']=='pure_visual_plan'


@pytest.mark.parametrize('task',['sweep','microwave'])
@pytest.mark.parametrize('execution',['visual','physical'])
@pytest.mark.parametrize('collision',['full','robot_only'])
def test_execution_and_collision_are_independent(tmp_path,task,execution,collision):
    (tmp_path/'data/scene').mkdir(parents=True)
    cmd=build_command(tmp_path,sys.executable,task,'run',
        {'scene':'data/scene','execution_mode':execution,'collision_mode':collision},tmp_path/'out')
    assert cmd[cmd.index('--collision-mode')+1]==collision
    assert ('plan_visual.py' in cmd[1])==(execution=='visual')
