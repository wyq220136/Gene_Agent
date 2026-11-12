#!/usr/bin/env python3
"""只读原生运行证据，可按阶段定位事实、未知项和媒体引用。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gene_agent.feedback.native_run import load_native_run, read_planning_failures
from gene_agent.feedback.motion_diagnostics import find_diagnostic_bundles


def main():
    """不加载引擎、不重跑policy、不推断缺失的失败原因。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--stage')
    parser.add_argument("--episode")
    parser.add_argument("--format", choices=("summary", "json"), default="summary")
    parser.add_argument("--details", action="store_true")
    parser.add_argument('--performance',action='store_true',help='只读规划耗时统计，包含失败与候选筛选')
    parser.add_argument('--recheck',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--planner-python',default=sys.executable)
    parser.add_argument('--planner-pythonpath')
    parser.add_argument('--budget-s',type=float,default=5.)
    parser.add_argument('--compare-with',type=Path,help='原失败episode目录；与--run的同一阶段作只读比较')
    parser.add_argument('--hypothesis',help='本轮最小修改要验证的猜想；工具不自动声称因果已证实')
    parser.add_argument('--require-executed',action='store_true',help='物理/接触猜想要求两次都实际执行目标阶段')
    args=parser.parse_args()
    if args.compare_with:
        if not args.stage or not args.hypothesis:parser.error('--compare-with需要--stage和--hypothesis')
        if args.recheck or args.performance:parser.error('同案例比较不能与--recheck/--performance混用')
        from gene_agent.feedback.native_run import compare_stage_runs
        report=compare_stage_runs(args.compare_with,args.run,stage=args.stage,hypothesis=args.hypothesis,require_executed=args.require_executed)
        if args.output:
            with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        if args.format=='json':print(json.dumps(report,ensure_ascii=False,indent=2))
        else:
            print(f"{report['verdict']}: {args.stage}; 同案例={report['case_match']}")
            print('原失败：'+json.dumps(report['before'],ensure_ascii=False))
            print('本次：'+json.dumps(report['after'],ensure_ascii=False))
            print('猜想仍需结合预期观测判断；阶段改善不等于完整任务或分布成功。')
        return
    if args.recheck:
        if args.output is None or not args.episode or not args.stage:
            parser.error('--recheck需要--episode、--stage和新的--output')
        import subprocess, os
        root=args.run/'episodes'/args.episode if (args.run/'episodes').exists() else args.run
        files=[]
        for path in find_diagnostic_bundles(root,stage=args.stage):
            content=json.loads(path.read_text())
            if content.get('stage_id')==args.stage and content.get('diagnostic'):files.append(path)
        if len(files)!=1:parser.error('未找到唯一完整诊断现场；旧记录只支持摘要，不能静默重建')
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(filter(None,[args.planner_pythonpath,str(Path(__file__).resolve().parents[2])]))
        raise SystemExit(subprocess.call([args.planner_python,'-m','gene_agent.motion.curobo.diagnostics',
            '--bundle',str(files[0].resolve()),'--output',str(args.output.resolve()),'--budget-s',str(args.budget_s)],env=env))
    from gene_agent.feedback.planning_performance import summarize_planning_run, performance_markdown
    roots = sorted((args.run/"episodes").glob("*")) if (args.run/"episodes").exists() else [args.run]
    roots = [root for root in roots if root.is_dir() and (not args.episode or root.name == args.episode)]
    if args.performance:
        reports=[dict(episode=root.name,performance=summarize_planning_run(root,stage=args.stage)) for root in roots]
        if args.format=='json':print(json.dumps(reports,ensure_ascii=False,indent=2))
        else:
            for report in reports:print(report['episode']+': '+performance_markdown(report['performance']))
        return
    if args.format == "summary":
        roots = sorted((args.run/"episodes").glob("*")) if (args.run/"episodes").exists() else [args.run]
        for root in roots:
            if args.episode and root.name != args.episode:
                continue
            if not (root/"events.jsonl").exists():
                continue
            print(performance_markdown(summarize_planning_run(root,stage=args.stage)))
            failures = list(read_planning_failures(root, stage=args.stage))
            print(f"{root.name}: {len(failures)} 个匹配的规划失败（不代表任务评分）")
            for row in failures if args.details else failures[:3]:
                print(f"阶段 {row['stage_id']} / state {row['state_id']}")
                print(row["markdown"])
                print(f"证据：{row['events_ref']}\n场景：{row['snapshot_ref'] or '缺失'}")
                refs=[]
                for path in find_diagnostic_bundles(root,stage=row['stage_id']):
                    meta=json.loads(path.read_text())
                    if (meta.get('stage_id')==row['stage_id'] and meta.get('state_id')==row['state_id']
                        and meta.get('target_world_tcp')==row.get('target_world_tcp')
                        and meta.get('planning',{}).get('request_id')==row['planning'].get('request_id')
                        and meta.get('planning',{}).get('metrics',{}).get('planning_timing',{}).get('call_id')
                            ==row['planning'].get('metrics',{}).get('planning_timing',{}).get('call_id')):refs.append(str(path))
                print('诊断工件：'+(', '.join(refs) if refs else '旧记录未采集球级现场'))
                if refs:
                    import shlex
                    artifact=json.loads(Path(refs[0]).read_text())
                    if artifact.get('diagnostic'):
                        print('查看：python tools/viewers/view_motion_diagnostics.py --bundle '+shlex.quote(refs[0]))
                    else:
                        print('球级现场未采集；此工件仅有失败摘要，不能直接打开碰撞viewer。')
                    media=artifact.get('media',{})
                    print('现场图：'+str(media.get('actual',{}).get('path',media.get('actual',{}).get('reason','未知'))))
                replay=row['planning'].get('metrics',{}).get('replay_request_ref')
                if replay:
                    import shlex
                    print('重规划：python tools/diagnostics/replay_motion_request.py --bundle '+shlex.quote(replay)+' --output <新目录>')
                if args.details:
                    print(json.dumps(row["planning"], ensure_ascii=False, indent=2))
            if len(failures)>3 and not args.details:
                print(f'另有{len(failures)-3}个失败，--details查看全部')
        return
    index=args.run/'evaluation_index.json' if args.run.is_dir() else args.run
    if index.name=='evaluation_index.json' and index.exists():
        report=json.loads(index.read_text())
        report['episode_reports']=[load_native_run(Path(p)) for p in report['episodes'] if not args.episode or Path(p).parent.name==args.episode]
        if args.stage:
            for episode in report['episode_reports']:
                episode['events']=[e for e in episode['events'] if e['stage_id']==args.stage]
    else:
        report=load_native_run(args.run)
        if args.stage:report['events']=[e for e in report['events'] if e['stage_id']==args.stage]
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
