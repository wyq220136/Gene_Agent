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
    parser.add_argument("--selection", help="显式指定抓碗 selection.json；默认在 run 内自动发现")
    parser.add_argument("--no-pick-debug", action="store_true", help="旧版界面不加载抓碗候选/轨迹")
    parser.add_argument("--check-only", action="store_true", help="仅打印实际加载的快照和位姿，不启动服务")
    parser.add_argument("--follow", action="store_true", help="每两秒刷新运行中新增的已落盘观测")
    parser.add_argument("--max-points", type=int, default=60000, help="每帧显示点数上限")
    parser.add_argument("--legacy-view", action="store_true", help="旧版抓碗候选/失败定位界面")
    args = parser.parse_args(argv)
    from gene_agent.visualization.realworld_scene import scene_bundle, load_observation, observation_summary, RealworldSceneApp
    from gene_agent.visualization.realworld_pick_debug import load_pick_debug, pick_debug_summary
    from gene_agent.visualization.afford_scene import create_scene_server
    with scene_bundle(args.run, include_run=not args.legacy_view) as bundle:
        scene, model, data, rgbd, warning = load_observation(bundle)
        debug = None if args.no_pick_debug else load_pick_debug(
            args.run, selection=args.selection,
            scene_snapshot_sha256=scene["snapshot"].get("snapshot_sha256"))
        summary = observation_summary(scene, rgbd, warning)
        summary["pick_debug"] = pick_debug_summary(debug)
        from gene_agent.visualization.realworld_timeline import RealworldTimeline
        timeline = RealworldTimeline(bundle, scene, max_points=args.max_points)
        summary["perception_replay"] = timeline.summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.check_only:
            return 0
        server = create_scene_server(host=args.host, port=args.port)
        try:
            if args.legacy_view:
                app = RealworldSceneApp(server, scene, model, data, rgbd, warning, debug)
            else:
                from gene_agent.visualization.realworld_dashboard import RealworldDashboard
                app = RealworldDashboard(server, scene, model, data, rgbd, warning, timeline, follow=args.follow)
            print(f"真机统一回放 Viser: http://{args.host}:{args.port}（只读，无机器人连接/重规划）", flush=True)
            while True:
                time.sleep(.05)
                app.tick()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
