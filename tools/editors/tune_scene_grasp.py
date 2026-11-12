"""在现有 MuJoCo GT run 的完整场景中调节单个物体的抓取 Knowledge。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.visualization.grasp_tuner.scene_app import SceneGraspTunerApp
from gene_agent.visualization.grasp_tuner.feedback import read_tuner_state


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="已保存的 MuJoCo GT run；恢复截图时默认使用截图原 run")
    parser.add_argument("--target", choices=("bowl",), default="bowl")
    parser.add_argument("--view", choices=("bowl", "gripper", "overview"), default=None)
    parser.add_argument("--output", type=Path, default=None, help="task-owned tuned profile 输出路径")
    parser.add_argument("--state-out", type=Path, default=None, help="实时界面状态 JSON；供 Codex 截取当前调参结果")
    parser.add_argument("--restore-state", type=Path, help="读取实时状态或截图清单，恢复有效调参结果")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--check-ik", action="store_true", help="启动后检查全部候选的纯位姿 IK，便于自动截图")
    args = parser.parse_args()
    restored = read_tuner_state(args.restore_state, tool="scene") if args.restore_state else None
    run = args.run or (Path(restored["run"]) if restored else ROOT / "runs/pick-bowl-planning-final")
    if not (run / "case_meta.json").is_file():
        parser.error(f"--run 不是可加载的 GT 场景：{run}")
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = SceneGraspTunerApp(server, ROOT, run, target=args.target, view=args.view or (restored or {}).get("view", "bowl"), output_path=args.output, state_path=args.state_out, restored=restored)
        if args.check_ik or (restored is not None and restored.get("ik_reachable") is not None):
            app._check_ik()
            if restored is not None and restored.get("ik_reachable") is not None:
                actual = None if app.ik_results is None else [item.reachable for item in app.ik_results]
                if actual != restored["ik_reachable"]:
                    raise ValueError("恢复后纯 IK 结果与截图不一致；请检查机器人模型或求解器版本")
        print(f"[scene-grasp-tuner] 已就绪：http://{args.host}:{args.port}", flush=True)
        print(f"[scene-grasp-tuner] run={run.resolve()} target={args.target} candidates={len(app.state.candidates)}", flush=True)
        print("[scene-grasp-tuner] 服务会持续运行；打开上面的地址，Ctrl-C 退出。", flush=True)
        while True:
            time.sleep(.25)
    except KeyboardInterrupt:
        print("[scene-grasp-tuner] 已退出", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
