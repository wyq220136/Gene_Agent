"""只读定位外部套件并构造原 evaluator 命令，不加载环境或设备。"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import yaml


def resolve_source(external_root: Path, *, family: str, instance: str) -> dict:
    """按源套件选实例，模型编号和任务名不另建第二份清单。"""
    if family not in {'hit','pour','sweep'}:raise ValueError('未知操作族')
    root=Path(external_root).expanduser().resolve()
    suite=root/f'env/RoboTwin/benchmarks/novel_tool/{family}_tools.yaml'
    data=yaml.safe_load(suite.read_text())
    matches=[(group,row) for group,rows in data['groups'].items() for row in rows if row['id']==instance]
    if len(matches)!=1:raise ValueError(f'套件实例不存在或重复: {instance}')
    group,entry=matches[0]
    if type(entry.get('model_id')) is not int or entry['model_id']<0:raise ValueError('实例模型编号无效')
    rw=root/'env/RoboTwin'
    sources=[suite,rw/'script/eval_policy.py',rw/'script/task_setup.py',rw/'envs/robot/robot.py',
             rw/'envs/_base_task.py',rw/f'envs/{entry["task"]}.py',rw/'task_config/novel_tool_clean.yml',
             rw/'benchmarks/novel_tool/registry.py',rw/f'benchmarks/novel_tool/{family}_suite.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in sources}
    return dict(external_root=str(root),family=family,entry=dict(entry),group=group,
                suite=str(suite),suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
                suite_version=data['version'],native_source_hashes=hashes)


def native_environment(external_root, output):
    """子进程使用原benchmarks；Gene_Agent只有具名package参与导入。"""
    root=Path(external_root).resolve(); rw=root/'env/RoboTwin'
    project=Path(__file__).resolve().parents[2]
    env=os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPATH']=os.pathsep.join([str(rw),str(rw/'script'),str(root/'src'),str(project)])
    env['TOOL_USE_REPORT_ROOT']=str(Path(output).resolve())
    return env


def native_command(request, output, *, python, test_num):
    """原 evaluator 拥有reset/筛选/预算/评分，配置只负责选择本项目policy。"""
    output=Path(output).resolve();root=Path(request['external_root']);rw=root/'env/RoboTwin'
    cfg=dict(policy_name='gene_agent.tasks.policies.novel_tool_task.deploy_policy',
             task_name=request['entry']['task'],tool_model_id=request['entry']['model_id'],
             task_config=request.get('task_config','novel_tool_clean'),task_setup='shared',
             ckpt_setting='none',seed=int(request.get('seed',0)),instruction_type='unseen',
             expert_check=True,test_num=int(test_num),eval_result_root=str(output/'native'),
             gene_output_root=str(output/'episodes'),gene_operation=request.get('operation','hold'),
             gene_planner_python=request.get('planner_python'),gene_planner_pythonpath=request.get('planner_pythonpath'),
             gene_candidate=request.get('candidate'),gene_stop_after=request.get('stop_after','pregrasp'),
             gene_save_motion_requests=bool(request.get('save_motion_requests',False)),
             gene_diagnostic_mode=request.get('diagnostic_mode','off'),gene_diagnostic_budget_s=request.get('diagnostic_budget_s',5.))
    if request.get('family','hit')=='hit':cfg['tool_placement']='fixed'
    # case cache 可显式提供，原native入口继续验证case身份/recipe，不在此绕过。
    if request.get('expert_case_manifest'):cfg['expert_case_manifest']=request['expert_case_manifest']
    path=output/'deploy.yml';path.write_text(yaml.safe_dump(cfg,allow_unicode=True))
    command=[str(python),str(rw/'script/eval_policy.py'),'--config',str(path),'--test_num',str(test_num)]
    return command,rw,native_environment(root,output)
