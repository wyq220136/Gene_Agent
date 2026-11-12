#!/usr/bin/env python3
"""在 scoop_orange / scoop_pink 场景从新 reset 执行抓勺 Skill。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 与现有直接运行的 CLI 一致，从源码根目录加载 benchmark 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_agent.tasks.policies.spoon_scoop_task.pick_backend import SpoonPickRun


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=Path("data/realworld/my_scene_0/scenes/scoop_orange"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-simulation-weld", action="store_true",
                        help="关闭接触后的仿真辅助夹持，检验纯 MuJoCo 物理持握")
    parser.add_argument("--grasp-handle-x-offset-m", type=float, default=0.,
                        help="沿勺子局部 X 微调抓柄位置，范围 ±0.015 m")
    parser.add_argument('--stop-after', choices=('pick', 'pregrasp', 'approach', 'grasp'),
                        default='pick', help='只执行所选阶段前缀，保留实际现场；pick 为完整请求')
    args = parser.parse_args()
    result = SpoonPickRun(args.scene, args.output,
                          simulation_weld=not args.no_simulation_weld,
                          grasp_handle_x_offset_m=args.grasp_handle_x_offset_m).run(stop_after=args.stop_after)
    print(args.output.resolve())
    print(f"scope={result.get('requested_scope', args.stop_after)} status={result['status']} success={result['success']} "
          f"failure_stage={result.get('failure_stage')}")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
