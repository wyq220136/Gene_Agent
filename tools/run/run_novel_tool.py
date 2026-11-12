#!/usr/bin/env python3
"""在外部 Novel Tool 原环境查看场景或评测 Gene_Agent policy。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.novel_tool_task.source import resolve_source,native_command,native_environment


def main():
    """输出目录独占，原生异常/日志保留，inspect不初始化引擎。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-root',type=Path,required=True)
    parser.add_argument('--family',choices=('hit','pour','sweep'),required=True)
    parser.add_argument('--instance',required=True)
    parser.add_argument('--mode',choices=('inspect','scene','evaluate'),default='inspect')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',default=sys.executable,help='原生RoboTwin解释器')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--test-num',type=int,default=1)
    parser.add_argument('--operation',choices=('hold','motion','pick','hit'),default='hold',help='基础能力、共享Pick或完整Pick+Hit')
    parser.add_argument('--candidate',type=Path,help='本项目Concept/Knowledge生成的抓取候选JSON')
    parser.add_argument('--stop-after',choices=('pregrasp','approach','grasp','pick'),default='pregrasp')
    parser.add_argument('--planner-python',help='隔离的Gene_Agent规划解释器')
    parser.add_argument('--planner-pythonpath',help='Gene_Agent所需cuRobo源码/依赖路径，绝不注入原生进程')
    parser.add_argument('--expert-case-manifest',type=Path)
    parser.add_argument('--viewer',action='store_true',help='scene模式启动原生交互viewer')
    parser.add_argument('--save-motion-requests',action='store_true',help='保存完整规划请求供离线重规划，含成功和完整batch')
    parser.add_argument('--diagnostic-mode', choices=('off','on_failure'),default='off')
    parser.add_argument('--diagnostic-budget-s',type=float,default=5.)
    args=parser.parse_args()
    from gene_agent.motion.curobo.diagnostics import DiagnosticOptions
    DiagnosticOptions(mode=args.diagnostic_mode,budget_s=args.diagnostic_budget_s)
    if args.test_num<=0:parser.error('--test-num必须为正数')
    if args.mode=='evaluate' and args.operation in {'pick','hit'} and args.candidate is None:parser.error('pick需要--candidate，不能从原专家抓取点取答案')
    source=resolve_source(args.external_root,family=args.family,instance=args.instance)
    out=args.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    request=dict(source,seed=args.seed,test_num=args.test_num,operation=args.operation,viewer=args.viewer,
                 planner_python=args.planner_python,planner_pythonpath=args.planner_pythonpath,
                 candidate=str(args.candidate.resolve()) if args.candidate else None,stop_after=args.stop_after,
                 diagnostic_mode=args.diagnostic_mode,diagnostic_budget_s=args.diagnostic_budget_s,save_motion_requests=args.save_motion_requests)
    if args.expert_case_manifest:request['expert_case_manifest']=str(args.expert_case_manifest.resolve())
    project=Path(__file__).resolve().parents[2]
    code_dirs=['gene_agent/knowledge','gene_agent/robots','gene_agent/motion','gene_agent/simulation','gene_agent/skills',
               'gene_agent/tasks/policies/novel_tool_task','benchmarks/novel_tool_task']
    files={p for folder in code_dirs for p in (project/folder).rglob('*.py')}
    files.update([Path(__file__).resolve(),project/'gene_agent/feedback/native_run.py',project/'gene_agent/feedback/planning_performance.py'])
    code_hashes={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    record=dict(code_hashes=code_hashes,launcher_python=sys.executable,source=source,requested_scope=args.mode,task_success=None,request=request,status='inspected')
    launch=out/'launch.json';launch.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    if args.mode=='inspect':
        print(json.dumps(record,ensure_ascii=False,indent=2));return
    if args.mode=='scene':
        command=[args.python,'-m','gene_agent.simulation.robotwin.session','--request',str(launch),'--output',str(out)]
        cwd=Path(source['external_root'])/'env/RoboTwin';env=native_environment(source['external_root'],out)
    else:command,cwd,env=native_command(request,out,python=args.python,test_num=args.test_num)
    record.update(command=command,cwd=str(cwd),status='running')
    launch.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    with (out/'native.log').open('w') as log:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if args.mode=='evaluate':
        from gene_agent.feedback.native_run import index_native_evaluation
        try:record['evaluation_index_ref']=str(index_native_evaluation(out))
        except Exception as exc:record['recording_errors']=[f'index: {type(exc).__name__}: {exc}']
    record.update(status='completed' if result.returncode==0 else 'error',returncode=result.returncode)
    launch.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(f'{args.mode}: {record["status"]}; 原生记录 {out}')
    if args.mode=='evaluate':
        import shlex
        print('检查：python tools/diagnostics/inspect_native_run.py --run '+shlex.quote(str(out)))
        print('每episode的development_summary.md包含耗时、失败与下一步命令。')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
