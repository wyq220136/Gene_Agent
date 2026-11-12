"""打开已保存的抓碗规划 run；浏览本身不调用 GPU 规划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import viser

from gene_agent.visualization.pick_bowl.app import PickBowlPlanningApp


def find_default_planning_run(project_root: Path) -> Path:
    """选择最近一次有完整场景且规划成功的抓碗 run，供交互编辑与重规划。"""
    candidates = []
    for report in (project_root / "runs").glob("*/pick_bowl_result.json"):
        run = report.parent
        if not (run / "source_scene/scene.xml").is_file():
            continue
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("plan_success") is True:
            candidates.append((report.stat().st_mtime, run))
    if not candidates:
        raise FileNotFoundError("找不到可编辑的抓碗规划 run；请用 --run 指定目录")
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="规划 run；省略时选最近一次规划成功的完整场景")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--view", choices=("overview", "bowl", "gripper"), default="overview")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--candidate-id", help="初始选中的抓取候选，便于固定截图和复核失败")
    parser.add_argument("--trajectory-source", choices=("抓取目标", "Skill pregrasp", "Skill approach", "TCP 直线探针"),
                        default="抓取目标", help="初始回放的规划阶段")
    args = parser.parse_args()
    run = args.run or find_default_planning_run(Path(__file__).resolve().parents[2])
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = PickBowlPlanningApp(server, run, seed=args.seed, view=args.view,
                                  show_collision=args.show_collision,
                                  initial_candidate_id=args.candidate_id,
                                  trajectory_source=args.trajectory_source)
        print(f"[pick-bowl] loaded: run={run.resolve()}; revision={app.case.revision}; view={args.view}", flush=True)
        print(f"[pick-bowl] URL: http://{args.host}:{args.port}", flush=True)
        while True:
            time.sleep(.25)
    except KeyboardInterrupt:
        print("[pick-bowl] 已退出", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
