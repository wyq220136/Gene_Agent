"""在线纯视觉 Skill/Task 入口共用的命令行和 Viser 生命周期。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log
from gene_agent.visualization.viser_sim.app import (
    DOOR_SKILLS, DOOR_TASKS, DoorVisualApp,
)


def main(mode: str) -> None:
    if mode not in {"skill", "task"}:
        raise ValueError("mode 必须是 skill 或 task")
    options = DOOR_SKILLS if mode == "skill" else DOOR_TASKS
    parser = argparse.ArgumentParser(
        description="在线纯视觉检查门 Skill" if mode == "skill"
        else "在线纯视觉检查门 Task",
    )
    parser.add_argument(f"--{mode}", choices=tuple(options), default=next(iter(options)))
    parser.add_argument("--config", type=Path, help="默认 data/task_1/skills/door_skill_debug.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092 if mode == "skill" else 8093)
    args = parser.parse_args()
    selection = getattr(args, mode)
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = DoorVisualApp(
            server, mode=mode, initial_selection=selection, config_path=args.config,
        )
        configure_curobo_run_log(app.session.output_dir)
        print(f"[{mode}] {selection}：http://{args.host}:{args.port}", flush=True)
        print(f"[{mode}] 本次记录：{app.session.output_dir}", flush=True)
        while True:
            app.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        print(f"[{mode}] 已退出", flush=True)
    finally:
        server.stop()


__all__ = ["main"]
