#!/usr/bin/env python3
"""RGB + 米制 mesh + 用户要求 → 有界 Agent 功能 Concept 标注。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gene_agent.perception_modules.concept_agent import CodexCliAgent, annotate_concept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="物体所在场景的 RGB")
    parser.add_argument("--mesh", type=Path, required=True, help="object frame、米制参考 mesh")
    parser.add_argument("--prompt", required=True, help="用户对功能 part 的要求")
    parser.add_argument("--output-dir", type=Path, required=True, help="新运行目录，不能已存在")
    parser.add_argument("--review-rounds", type=int, default=1, choices=(0, 1, 2),
                        help="默认复核一次；0 最快，最多 2 次")
    parser.add_argument("--model", help="可选 Codex CLI 模型；默认使用本机配置")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    agent = CodexCliAgent(args.output_dir.expanduser().resolve(), model=args.model,
                          timeout_seconds=args.timeout_seconds)
    result = annotate_concept(args.rgb, args.mesh, args.prompt, args.output_dir,
                              agent=agent, review_rounds=args.review_rounds)
    record = json.loads((args.output_dir / "annotation_run.json").read_text(encoding="utf-8"))
    print(f"Concept: {result}")
    print(f"叠图: {args.output_dir / 'overlay_final.png'}")
    print(f"总耗时: {record['elapsed_seconds']:.1f} 秒；模型轮数: {len(record['rounds'])}")
    print("继续手工调整：PYTHONPATH=. python tools/editors/edit_concept.py "
          f"--mesh {args.mesh} --concept {result}")


if __name__ == "__main__":
    main()
