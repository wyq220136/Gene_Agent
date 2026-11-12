#!/usr/bin/env python3
"""连续执行杯把抓取、向碗内倾倒和放回杯子。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 与现有直接运行的 CLI 一致，从源码根目录加载 benchmark 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_agent.tasks.policies.cup_pour_task.simulation import CupPourTaskRun


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=Path("data/realworld/my_scene_0/scenes_tilt50_yaw360_reachable/pour"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until-skill", choices=("pick", "pour", "place"),
                        default="place")
    parser.add_argument("--simulation-weld", action="store_true",
                        help="仅在实测杯把双指接触后启用仿真辅助夹持")
    parser.add_argument('--collision-mode',choices=('full','robot_only'),default='full')
    parser.add_argument('--live',action='store_true',help='记录实际 qpos 供网页在线显示')
    args = parser.parse_args()
    from gene_agent.motion.configuration import motion_configuration
    from gene_agent.feedback.live_record import live_physics_recording
    with motion_configuration({'collision_mode':args.collision_mode}), live_physics_recording(args.output,enabled=args.live):
        task_run = CupPourTaskRun(args.scene, args.output,
                                simulation_weld=args.simulation_weld)
        task_run.runtime.replay_sample(object_names=(task_run.object_name,),robot_groups=(task_run.group,))
        result=task_run.run_task(until_skill=args.until_skill)
    print(args.output.resolve())
    print(f"status={result['status']} success={result['success']} "
          f"failure_stage={result.get('failure_stage')}")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
