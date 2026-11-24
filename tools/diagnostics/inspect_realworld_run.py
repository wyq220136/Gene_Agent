"""只读列出真机逐候选/逐阶段失败；无需 SDK、仿真或 GPU。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--stage')
    parser.add_argument('--details', action='store_true', help='列出全部请求及原始结果引用')
    parser.add_argument('--format', choices=['text','json'], default='text')
    args=parser.parse_args(argv)
    from gene_agent.feedback.realworld_diagnostics import inspect_realworld_run
    value=inspect_realworld_run(args.run,candidate=args.candidate,stage=args.stage)
    if args.format=='json':
        print(json.dumps(value,ensure_ascii=False,indent=2));return 0
    print(value['scope'])
    print('规划请求：',value['counts'])
    if value.get('exception'):print('运行异常：',value['exception']['error'],'；完整栈见 diagnostic_exception.json')
    for failure in value.get('operation_failures',[]):
        item=failure['first_failure']
        print('操作首个失败：',item.get('code') or item.get('kind'),item.get('message') or item.get('summary'),'report=',failure['report_ref'])
    if not value.get('terminal_reports'):print('未记录任务终态；规划通过不能视为任务成功')
    for error in value.get('recording_errors',[]):print('记录不完整：',error)
    for inventory in value['inventories']:
        print(f"候选生成 {inventory['generated_count']} / 进入筛选 {inventory['selection_count']}；{inventory['counts']}；{inventory['ref']}")
        if args.details or args.candidate:
            for row in inventory['candidates']:
                first=row['first_rejection'] or {}
                print(row['candidate_id'],row['status'],'首个拒绝阶段=',first.get('stage'))
                for phase in row['phases']:
                    summary=phase.get('summary') or {}
                    print(' ',phase['stage'],phase['request_id'],'success=',phase['success'],
                          summary.get('conclusion'),
                          'position_m=',summary.get('best_position_error_m'),
                          'rotation_rad=',summary.get('best_rotation_error_rad'))
    failed=[e for e in value['events'] if not e.get('success')]
    for row in (value['events'] if args.details else failed[:3]):
        summary=row.get('summary',{})
        print(row['request_id'],'success=',row['success'],summary.get('conclusion',''),
              'detail=',row.get('diagnostic_ref'),'result=',row['result_ref'],
              'replay=',row.get('replay_request_ref'))
        if not row['success']:
            for fact in summary.get('diagnostic_facts',[]):print(' ',fact)
            for unknown in summary.get('unknowns',[]):print(' 未知：',unknown)
    if len(failed)>3 and not args.details:print(f'另有 {len(failed)-3} 个失败；--details 查看全部')
    print('初始现场包：',Path(args.run)/value['initial_scene_ref'])
    for error in value['errors']:print(error)
    return 0


if __name__=='__main__':raise SystemExit(main())
