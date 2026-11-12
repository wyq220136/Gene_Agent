"""在 Viser 中检查微波炉把手 grasp/拉门、门板推门和旋钮 Knowledge。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene
from gene_agent.visualization.grasp_tuner.microwave_app import IK_BACKENDS, MODES, MicrowaveKnowledgeApp
from gene_agent.visualization.grasp_tuner.feedback import read_tuner_state


ROOT = Path(__file__).resolve().parents[2]


def resolve_run(explicit_run: Path | None, restored: dict | None) -> Path:
    """默认 Knowledge 入口现场生成基础场景，不依赖未归档的历史 runs。

    用户显式指定的 run 或恢复清单必须严格复用原目录；拼错路径时不能
    悄悄创建一个不同的场景，否则参数和截图将失去可追溯性。
    """
    if restored:
        return Path(restored["run"])
    if explicit_run is not None:
        return explicit_run
    default = ROOT / "runs/pick-bowl-planning-final"
    if not (default / "case_meta.json").is_file():
        if default.exists():
            raise RuntimeError(f"默认 Knowledge 场景目录不完整：{default}")
        prepare_pick_bowl_scene(default, seed=0)
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="已有的场景 run；省略时自动建立默认场景")
    parser.add_argument("--mode", choices=MODES, help="默认沿用恢复清单中的模式，否则为 handle_grasp")
    parser.add_argument("--waypoint", type=int, help="路径起始 waypoint；-1 表示终点")
    parser.add_argument("--ik-backend", choices=IK_BACKENDS, help="初始 IK 求解器；界面中也可切换")
    parser.add_argument("--restore-state", type=Path, help="从截图 JSON 清单恢复参数、候选、waypoint 和相机")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--state-out", type=Path)
    args = parser.parse_args()
    restored = read_tuner_state(args.restore_state, tool="microwave_knowledge") if args.restore_state else None
    mode = args.mode or (restored or {}).get("mode") or "handle_grasp"
    run = resolve_run(args.run, restored)
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    app = MicrowaveKnowledgeApp(server, ROOT, run, mode=mode,
                                state_path=args.state_out, initial_waypoint=args.waypoint or 0,
                                restored=restored, ik_backend=args.ik_backend)
    print(f"[microwave-knowledge] 已就绪：{mode}；http://127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
