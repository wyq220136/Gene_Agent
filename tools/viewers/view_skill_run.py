"""在 Viser 中回放已保存的 Skill run。在线门调试请用 view_door_skill.py。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import viser

from gene_agent.visualization.skill_inspector import SkillInspectorApp


def find_default_run(project_root: Path, skill: str | None = None) -> Path:
    """按所选 Skill 找最近一次完整记录，免去复制长目录名。

    Pick 仍要求物理抓起成功。Insert 目前只有运动学规划记录，因此
    单独按规划成功筛选，界面会继续明确显示尚未验证物理接触。
    """
    if skill == "insert":
        candidates = []
        for report in (project_root / "runs").glob("**/skill_insert_bowl_result.json"):
            run = report.parent
            if not (run / "source_scene/scene.xml").is_file():
                continue
            try:
                result = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if result.get("success") is True:
                candidates.append((report.stat().st_mtime, run))
        if not candidates:
            raise FileNotFoundError("没有成功的放碗规划记录；请用 --run 指定目录")
        return max(candidates, key=lambda item: (item[0], str(item[1])))[1]
    required = {"open", "pregrasp", "approach", "close", "lift", "hold"}
    candidates = []
    for report in (project_root / "runs").glob("*/result.json"):
        run = report.parent
        if not (run / "model/scene.xml").is_file():
            continue
        try:
            result = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (result.get("success") is True and result.get("physical_pick_success") is True
                and required <= set(result.get("stages", {}))):
            candidates.append((report.stat().st_mtime, run))
    if not candidates:
        raise FileNotFoundError("没有可用的完整抓碗 Skill 运行记录；请用 --run 指定目录")
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="回放目录；省略时选择最近的完整抓碗记录")
    parser.add_argument("--skill", choices=("pick", "insert"),
                        help="同一场景含多个 Skill 时选择要检查的规划")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--stage", help="阶段 ID；省略时播放完整 Skill")
    parser.add_argument("--mode", choices=("planned", "actual"))
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--hide-visual", action="store_true", help="仅看碰撞代理时隐藏 visual 外观")
    parser.add_argument("--time", type=float, default=0., help="初始回放时间，单位秒")
    args = parser.parse_args()
    run = args.run or find_default_run(Path(__file__).resolve().parents[2], args.skill)
    print(f"[skill-inspector] 正在检查：{run}", flush=True)
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        app = SkillInspectorApp(server, run, stage_id=args.stage, skill=args.skill,
                                mode=args.mode, show_collision=args.show_collision,
                                show_visual=not args.hide_visual,
                                initial_time_s=args.time)
        print(f"[skill-inspector] URL: http://{args.host}:{args.port}", flush=True)
        while True:
            app.tick()
            time.sleep(.03)
    except KeyboardInterrupt:
        print("[skill-inspector] 已退出", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
