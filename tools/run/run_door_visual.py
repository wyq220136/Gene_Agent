"""无 GUI 地运行同一门 Skill 在线纯视觉会话，供 Codex 取结构化反馈。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gene_agent.tasks.policies.microwave_task.planning.door_session import DoorDebugSession, configure_curobo_run_log
from gene_agent.motion import PlannerMode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--planner-mode", choices=tuple(mode.value for mode in PlannerMode),
                        help="覆盖任务配置中的 planner 模式")
    parser.add_argument("--skill", choices=("handle_pull", "push_open", "push_close", "door_cycle"),
                        default="door_cycle", help="可单独调试一个门 Skill")
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.targets_only == args.plan:
        parser.error("请选择 --targets-only 或 --plan")
    options = {}
    if args.config:
        options["config_path"] = args.config
    if args.planner_mode:
        options["planner_mode"] = args.planner_mode
    session = DoorDebugSession(**options)
    configure_curobo_run_log(session.output_dir)
    output = session.output_dir / "knowledge_targets.json"
    session.export_targets(output)
    if args.targets_only:
        print(json.dumps({"kind": "knowledge_targets_only", "phase_count": len(session.phases),
                          "output": str(output)}, ensure_ascii=False))
        return
    selected = None if args.skill == "door_cycle" else args.skill
    timeline_report = session.plan_sequence(selected)
    segment_count = (len(session.timeline.segments) if session.timeline is not None else
                     int(timeline_report.get("completed_segments", 0)))
    report = {
        "kind": "planned_kinematic_timeline",
        "success": bool(timeline_report["success"]),
        "selection": args.skill,
        "segment_count": segment_count,
        "duration_s": timeline_report.get("duration_s"),
        "first_failure": timeline_report.get("first_failure"),
        "results": session.results,
        "targets_ref": str(output),
    }
    result_path = session.output_dir / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure = report["first_failure"] or {}
    print(json.dumps({"success": report["success"], "segments": report["segment_count"],
                      "first_failure": None if not failure else {
                          "phase_id": failure.get("phase_id"),
                          "failure_category": failure.get("failure_category"),
                          "failure_summary": failure.get("failure_summary"),
                      },
                      "output": str(result_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
