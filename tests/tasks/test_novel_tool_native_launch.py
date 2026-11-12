"""外部套件身份和子进程边界不依赖仿真器/GPU。"""
import json
from pathlib import Path
import subprocess
import sys
import yaml
import pytest


def source(tmp_path):
    root=tmp_path/'external';bench=root/'env/RoboTwin/benchmarks/novel_tool';bench.mkdir(parents=True)
    (bench/'hit_tools.yaml').write_text(yaml.safe_dump({'version':1,'groups':{'train':[{'id':'hammer_train','task':'beat_nail_hammer','model_id':0}]}}))
    return root


def test_selected_instance_identity_and_unknown_rejection(tmp_path):
    from benchmarks.novel_tool_task.source import resolve_source
    root=source(tmp_path)
    result=resolve_source(root,family='hit',instance='hammer_train')
    assert result['entry']=={'id':'hammer_train','task':'beat_nail_hammer','model_id':0}
    assert result['group']=='train' and len(result['suite_sha256'])==64
    with pytest.raises(ValueError):resolve_source(root,family='hit',instance='absent')


def test_inspect_cli_runs_without_importing_native_engine(tmp_path):
    root=source(tmp_path);out=tmp_path/'inspect'
    command=[sys.executable,'tools/run/run_novel_tool.py','--external-root',str(root),'--family','hit','--instance','hammer_train','--mode','inspect','--output',str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    record=json.loads((out/'launch.json').read_text())
    assert record['source']['entry']['model_id']==0
    assert record['requested_scope']=='inspect' and record['task_success'] is None
    assert subprocess.run(command,capture_output=True).returncode!=0


def test_native_evaluator_command_keeps_cwd_policy_and_original_protocol(tmp_path):
    from benchmarks.novel_tool_task.source import native_command
    root=source(tmp_path);output=tmp_path/'run';output.mkdir()
    request={'external_root':str(root),'entry':{'task':'beat_nail_hammer','model_id':0}}
    command,cwd,env=native_command(request,output,python=sys.executable,test_num=2)
    assert command[1].endswith('script/eval_policy.py')
    cfg=yaml.safe_load((output/'deploy.yml').read_text())
    assert cfg['expert_check'] is True and cfg['task_setup']=='shared' and cfg['tool_placement']=='fixed'
    assert cfg['tool_model_id']==0 and cfg['test_num']==2
    assert cfg['policy_name']=='gene_agent.tasks.policies.novel_tool_task.deploy_policy'
    assert cwd==root/'env/RoboTwin'
    assert env['TOOL_USE_REPORT_ROOT']==str(output)


def test_native_bootstrap_rejects_foreign_benchmark_namespace(tmp_path):
    from gene_agent.simulation.robotwin.session import bootstrap_native
    root=source(tmp_path)
    # 本测试进程已加载本项目benchmarks，禁止混装两套同名题目包。
    with pytest.raises(RuntimeError,match='benchmarks'):
        bootstrap_native(root)


def test_policy_lifecycle_loads_without_engines_and_resets_episode(tmp_path):
    from gene_agent.tasks.policies.novel_tool_task.deploy_policy import get_model,reset_model
    model=get_model({'gene_output_root':str(tmp_path),'gene_operation':'hold'})
    assert model.episode_done is False
    model.episode_done=True
    reset_model(model)
    assert model.episode_done is False


def test_pick_requires_explicit_own_candidate_before_native_launch(tmp_path):
    command=[sys.executable,'tools/run/run_novel_tool.py','--external-root',str(tmp_path/'absent'),'--family','hit','--instance','hammer_train','--mode','evaluate','--operation','pick','--output',str(tmp_path/'out')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==2 and 'candidate' in result.stderr
    assert not (tmp_path/'out').exists()


def test_policy_options_exclude_setup_seed_and_expert_data():
    from gene_agent.tasks.policies.novel_tool_task.deploy_policy import get_model
    model=get_model({'gene_operation':'motion','gene_output_root':'/tmp/output','seed':42,'task_setup':'shared',
                     'expert_case_manifest':'secret','setting':{'pose':'hidden'},'gene_planner_python':'python'})
    assert model.options=={'gene_operation':'motion','gene_output_root':'/tmp/output','gene_planner_python':'python'}
