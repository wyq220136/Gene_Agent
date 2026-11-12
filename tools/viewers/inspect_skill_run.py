"""无需启动GPU或Viser，检查单次Skill各阶段与状态链。"""

import argparse
import json
from pathlib import Path

from gene_agent.feedback.skill_run import load_skill_run, write_skill_run_index
from gene_agent.visualization.replay import ReplayTimeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", help="指定阶段ID；省略时检查完整Skill")
    parser.add_argument("--mode", choices=("planned", "actual"), help="检查规划轨迹或实际采样")
    parser.add_argument("--time", type=float, default=0.0, help="所选时间轴上的秒数")
    args = parser.parse_args()
    view = load_skill_run(args.run)
    path = write_skill_run_index(args.run)
    timeline = ReplayTimeline.load(args.run)
    mode = args.mode or timeline.mode
    timeline.select(args.stage, mode=mode)
    frame = timeline.seek(args.time)
    print(json.dumps({
        "skill": view.skill, "instance_id": view.instance_id,
        "physical_success": view.physical_success,
        "source_format": view.source_format,
        "stages": [{"id": item.stage_id, "operation": item.operation,
                    "planning": item.planning_status, "execution": item.execution_status}
                   for item in view.stages],
        "index": str(path),
        "timeline_duration_s": timeline.duration_s,
        "frame": {
            "stage_id": frame.stage_id, "mode": frame.mode,
            "sample_time_s": frame.sample_time_s, "sample_quality": frame.sample_quality,
            "state_id": frame.state_id, "collision_snapshot_ref": frame.collision_snapshot_ref,
            "path_kind": frame.path_kind,
            "attachment_object_id": (frame.attachment or {}).get("object_instance_id"),
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
