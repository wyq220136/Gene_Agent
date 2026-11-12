"""在线检查一个门 Skill：可编辑场景、重新规划并播放连续轨迹。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log
from gene_agent.motion import PlannerMode
from gene_agent.visualization.skill_inspector.live_door_app import LiveDoorApp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=("handle_pull", "push_open", "push_close"),
                        default="push_close")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--planner-mode", choices=tuple(mode.value for mode in PlannerMode),
                        help="默认读取任务配置；development 用于诊断性能模式本身")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = LiveDoorApp(server, config_path=args.config,
                          mode="skill", skill_id=args.skill,
                          planner_mode=args.planner_mode)
        configure_curobo_run_log(app.session.output_dir)
        print(f"[door-skill] {args.skill}：http://{args.host}:{args.port}", flush=True)
        print(f"[door-skill] 本次记录：{app.session.output_dir}", flush=True)
        while True:
            app.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        print("[door-skill] 已退出", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
