"""用 AffordToolAct 风格 Viser 查看真机操作实际输入，不连接硬件或运行规划。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="单次 run、initial_scene、manifest.json 或含唯一 initial_scene 的 tar.gz")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--check-only", action="store_true", help="仅打印实际加载的快照和位姿，不启动服务")
    args = parser.parse_args(argv)
    from gene_agent.visualization.realworld_scene import scene_bundle, load_observation, observation_summary, RealworldSceneApp
    from gene_agent.visualization.afford_scene import create_scene_server
    with scene_bundle(args.run) as bundle:
        scene, model, data, rgbd, warning = load_observation(bundle)
        print(json.dumps(observation_summary(scene, rgbd, warning), ensure_ascii=False, indent=2), flush=True)
        if args.check_only:
            return 0
        server = create_scene_server(host=args.host, port=args.port)
        try:
            RealworldSceneApp(server, scene, model, data, rgbd, warning)
            print(f"真机冻结输入 Viser: http://{args.host}:{args.port}（只读，无机器人连接）", flush=True)
            while True:
                time.sleep(.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
