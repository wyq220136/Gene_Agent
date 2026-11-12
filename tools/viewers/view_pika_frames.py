"""查看 Pika 离线坐标报告与原始 mesh；不连接真机。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "runs/pika_frames/audit/report.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--output", type=Path, help="静态截图目录；默认在报告旁的 previews 下")
    args = parser.parse_args()
    if args.render_only:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from gene_agent.visualization.pika_frames import render_audit, serve_audit
    if args.render_only:
        paths = render_audit(args.report, args.output or args.report.resolve().parent / "previews")
        print("\n".join(str(path) for path in paths))
    else:
        serve_audit(args.report, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
