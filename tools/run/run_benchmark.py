"""显式选择 benchmark 与兼容 policy；沿用已有仿真任务算法。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.catalog import list_benchmarks
from gene_agent.tasks.launch import resolve_request, run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=list_benchmarks(), required=True)
    parser.add_argument('--policy', help='显式策略 ID，例如 cup_pour_task/default')
    parser.add_argument('--input', type=Path, help='场景目录或旧微波炉复合配置')
    parser.add_argument('--environment', type=Path, help='场景/仿真条件 JSON 对象')
    parser.add_argument('--parameters', type=Path, help='策略参数 JSON 对象')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true', help='只检查输入选择，不启动 GPU 或设备')
    args = parser.parse_args()
    def load(path):
        return json.loads(path.read_text(encoding='utf-8')) if path else None
    options = dict(policy_id=args.policy, input_path=args.input,
                   environment=load(args.environment), parameters=load(args.parameters))
    if args.inspect:
        request = resolve_request(args.benchmark, **options)
        print(json.dumps(vars(request), default=str, ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error('执行时必须指定新的 --output 目录')
    result = run_benchmark(args.benchmark, args.output, **options)
    success = result.get('goal_success', result.get('success'))
    print(f"status={result.get('status')} scope={result['requested_scope']} "
          f"task_success={result['benchmark_task_success']} operation_success={success}；记录：{args.output}")
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
