"""打开一个只读 Motion DebugBundle Viser 页面。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.feedback import load_motion_debug_bundle
from gene_agent.visualization.motion_debug import MotionDebugApp


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--view", choices=("overview", "target", "collision"), default="overview"
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    bundle = load_motion_debug_bundle(args.bundle)
    server = viser.ViserServer(host=args.host, port=args.port)
    app = MotionDebugApp(server, bundle, initial_view=args.view)
    print(f"[motion-debug] URL: http://{args.host}:{args.port}")
    print(f"[motion-debug] loaded: run={bundle.run_id}; attempts={len(bundle.attempts)}")
    print(f"[motion-debug] nodes={len(app.node_names)}; Ctrl-C 退出")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[motion-debug] 已退出")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
