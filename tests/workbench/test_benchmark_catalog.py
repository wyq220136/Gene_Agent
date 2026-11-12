"""工作台必须逐一对应 benchmark；不能把未接入功能路由到另一任务。"""
from pathlib import Path
import sys
import pytest
from benchmarks.catalog import list_benchmarks,load_definition
from gene_agent.workbench.catalog import TASKS,build_command

ROOT=Path(__file__).resolve().parents[2]


def test_every_benchmark_is_registered_once():
    assert {r['benchmark_id'] for r in TASKS.values()}==set(list_benchmarks())
    assert len(TASKS)==5
    for r in TASKS.values():
        definition=load_definition(r['benchmark_id'])
        assert (r.get('scene') or r.get('config'))==definition['scene']['input']


@pytest.mark.parametrize('task,entry',[('sweep_long','run_sweep_long.py'),('cup_pour','run_cup_pour_task.py'),('spoon_scoop','run_spoon_scoop_task.py')])
def test_new_task_routes_actual_cli_and_scene(tmp_path,task,entry):
    cmd=build_command(ROOT,sys.executable,task,'run',{'collision_mode':'robot_only','randomize':False},tmp_path/'new')
    assert Path(cmd[1]).name==entry
    assert cmd[cmd.index('--collision-mode')+1]=='robot_only'
    assert '--live' in cmd
    preview=build_command(ROOT,sys.executable,task,'scene_viewer',{},tmp_path/'view')
    assert 'view_benchmark.py' in preview[1] and TASKS[task]['benchmark_id'] in preview
    with pytest.raises(ValueError,match='纯视觉'):
        build_command(ROOT,sys.executable,task,'run',{'execution_mode':'visual'},tmp_path/'bad')


@pytest.mark.parametrize('task',['sweep_long','cup_pour','spoon_scoop'])
def test_new_task_description_does_not_use_microwave_adapter(task):
    from gene_agent.workbench.scene_inputs import description
    info=description(ROOT,task)
    assert info['task']==task and info['viewable'] and not info['editable']
    assert info['objects']


def test_full_tabletop_report_takes_precedence_over_pick(tmp_path):
    import json
    from gene_agent.workbench.records import describe_run
    run=tmp_path/'runs/cup';run.mkdir(parents=True)
    (run/'task_result.json').write_text(json.dumps({'task':'cup_pour_into_bowl','status':'failed','success':False,'failure_stage':'pour'}))
    (run/'result.json').write_text(json.dumps({'status':'completed','success':True}))
    row=describe_run(tmp_path,'runs/cup')
    assert row['task']=='cup_pour_into_bowl' and row['success'] is False and row['failure_stage']=='pour'


def test_scene_viewer_accepts_separate_robot_metadata(tmp_path,monkeypatch):
    import json,viser
    from types import SimpleNamespace
    from tools.viewers import view_reconstructed_tabletop_scene as viewer
    from gene_agent.visualization.pick_bowl import scene
    (tmp_path/'robot').mkdir()
    (tmp_path/'robot/robot_metadata.json').write_text(json.dumps({'profile':'separate_robot'}))
    monkeypatch.setattr(viewer,'load_scene',lambda path:({},None,None))
    noop=lambda *args,**kwargs:None
    server=SimpleNamespace(initial_camera=SimpleNamespace(),scene=SimpleNamespace(set_up_direction=noop,set_background_image=noop,add_frame=noop),gui=SimpleNamespace(add_markdown=noop))
    monkeypatch.setattr(viser,'ViserServer',lambda **kwargs:server)
    captured=[]
    monkeypatch.setattr(scene,'PickBowlScene',lambda s,m,d,robot,snapshot:captured.append(robot))
    def stop(*args):raise KeyboardInterrupt
    monkeypatch.setattr(viewer.time,'sleep',stop)
    viewer.view_viser(tmp_path,port=8098)
    assert captured==[{'profile':'separate_robot'}]


@pytest.mark.parametrize('task',['sweep','microwave','sweep_long','cup_pour','spoon_scoop'])
def test_run_preview_preserves_selected_initialization(tmp_path,task):
    cmd=build_command(ROOT,sys.executable,task,'run_preview',{'seed':317,'randomize':task not in {'cup_pour','spoon_scoop'},'execution_mode':'physical'},tmp_path/'preview')
    assert 'run_scene_preview' in ' '.join(cmd)
    assert cmd[cmd.index('--seed')+1]=='317'
    assert cmd[cmd.index('--task')+1]==task
    assert '--edit' not in cmd


def test_long_preview_random_seed_matches_official_initializer(tmp_path):
    import mujoco,numpy as np
    from gene_agent.workbench.run_scene_preview import build_physical
    from gene_agent.tasks.policies.sweep_long_task.simulation import prepare_long_brush_scene
    from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
    source=ROOT/TASKS['sweep_long']['scene']
    model,data,_=build_physical('sweep_long',source,tmp_path/'preview',seed=317,randomize=True)
    official=prepare_long_brush_scene(source,tmp_path/'official',layout_seed=317)
    official_run=SweepCubeRun(official,tmp_path/'official',prepared_output_dir=True)
    assert model.nq==official_run.model.nq
    np.testing.assert_array_equal(data.qpos,official_run.data.qpos)
