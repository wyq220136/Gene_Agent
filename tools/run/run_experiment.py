"""在线执行一个微波炉物理实验，并保存逐阶段实际状态与诊断。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_agent.tasks.policies.microwave_task.simulation_door.session import ExperimentSession

DEFAULT_CONFIG = ROOT / "data/task_1/simulation/door_physical_open.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("handle_pull", "push_open", "push_close",
                                                "microwave_door_open", "microwave_door_close",
                                                "microwave_door_cycle"),
                        default="microwave_door_open")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--until-stage", help="执行到此阶段后暂停；结果标记 partial")
    parser.add_argument("--resume", type=Path, help="从上一轮保存的真实物理检查点继续")
    parser.add_argument("--parameters", type=Path,
                        help="阶段边界参数补丁 JSON；只影响尚未执行的阶段")
    parser.add_argument("--probe-next", action="store_true",
                        help="只规划并检查下一运动阶段，不执行 MuJoCo 物理")
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / f"microwave-physics-{datetime.now():%Y%m%d-%H%M%S}"
    session = ExperimentSession(args.selection, "physics", args.config, args.seed,
                                output_dir=output)
    if args.resume is not None:
        session.restore_checkpoint(args.resume)
    if args.parameters is not None:
        patch = json.loads(args.parameters.read_text(encoding="utf-8"))
        session.apply_parameters(patch)
    if args.probe_next:
        if args.until_stage is not None:
            parser.error("--probe-next 不能与 --until-stage 同时使用")
        result = session.probe_next()
        frozen = result.get("frozen_start_door_path_probe") or {}
        print(json.dumps({
            "status": result["status"],
            "stage_id": result["stage_id"],
            "planning_success": result["planning_success"],
            "failure_category": result.get("failure_category"),
            "frozen_start_first_contact_order": frozen.get("first_contact_order"),
            "first_forbidden_sample_index": frozen.get("first_forbidden_sample_index"),
            "first_pad_target_sample_index": frozen.get("first_pad_target_sample_index"),
            "report": str(output / "probe_next.json"),
        }, ensure_ascii=False))
        if not result["planning_success"]:
            raise SystemExit(1)
        return
    report = session.run(until_stage=args.until_stage)
    checkpoint = None
    if report["status"] == "partial":
        checkpoint = session.save_checkpoint(output / "checkpoint")
    print(json.dumps({
        "status": report["status"],
        "goal_success": report["goal_success"],
        "actual_door_angle_deg": round(
            report["actual_final"]["door_angle_rad"] * 180 / 3.141592653589793, 2
        ),
        "last_stage": report["stages"][-1] if report["stages"] else None,
        "report": str(output / "report.json"),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    }, ensure_ascii=False))
    if report["status"] == "failed":
        raise SystemExit(1)
    if report["status"] == "partial":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
