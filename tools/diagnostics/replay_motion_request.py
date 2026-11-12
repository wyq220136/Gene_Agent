#!/usr/bin/env python3
"""从冻结请求独立重跑cuRobo，保存冷/热结果；不加载或执行仿真。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='必须不存在的新目录')
    parser.add_argument('--warm-repeats',type=int,default=1)
    parser.add_argument('--allow-code-change',action='store_true',help='仅允许项目代码变化并记录diff哈希，资产和依赖仍严格验证')
    args=parser.parse_args()
    from gene_agent.motion.curobo.diagnostics import replan_bundle
    try:
        path=replan_bundle(args.bundle,args.output,warm_repeats=args.warm_repeats,allow_code_change=args.allow_code_change)
    except (ValueError,FileExistsError) as exc:
        parser.exit(2,str(exc)+'\n')
    print(f'离线规划结果（无物理执行）：{path}')


if __name__=='__main__':main()
