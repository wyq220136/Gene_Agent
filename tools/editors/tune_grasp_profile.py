"""启动 Knowledge 抓取候选的 Viser 人工检查 demo。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.visualization.grasp_tuner.app import GraspTunerApp
from gene_agent.visualization.grasp_tuner.session import SUPPORTED_PROFILES


ROOT = Path(__file__).resolve().parents[2]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Viser 监听地址")
    parser.add_argument("--port", type=int, default=8080, help="Viser 监听端口")
    parser.add_argument("--output", type=Path, default=None, help="保存目标；默认 archive/task_1/configs/grasp_profiles.tuned.json")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default="task_1_bowl_0",
        help="初始 Knowledge profile；启动后也可在 GUI 中切换",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    server = viser.ViserServer(host=args.host, port=args.port)
    app = GraspTunerApp(server, ROOT, args.profile, output_path=args.output)
    url = f"http://{args.host}:{args.port}"
    print(f"[grasp-tuner] 已就绪：{url}", flush=True)
    print(f"[grasp-tuner] {args.profile}: {app.candidate_count} candidates", flush=True)
    print("[grasp-tuner] 服务会持续运行；打开上面的地址，Ctrl-C 退出。", flush=True)
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[grasp-tuner] 已退出")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
