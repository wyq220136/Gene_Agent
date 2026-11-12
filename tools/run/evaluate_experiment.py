"""从同一个 MuJoCo 检查点顺序比较少量微波炉参数补丁。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_agent.tasks.policies.microwave_task.simulation_door.evaluation import evaluate_parameter_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True,
                        help="所有候选共享的真实物理检查点")
    parser.add_argument("--cases", type=Path, required=True,
                        help="包含 cases 列表的 UTF-8 JSON")
    parser.add_argument("--until-stage", required=True,
                        help="每个候选运行到此阶段后停止")
    parser.add_argument("--selection", choices=("handle_pull", "push_open", "push_close",
                                                "microwave_door_open", "microwave_door_close",
                                                "microwave_door_cycle"),
                        default="microwave_door_open")
    parser.add_argument("--config", type=Path, default=ROOT /
                        "data/task_1/visualization/door_visual_open_close_verified_xy_052_031.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        parser.error("候选文件须为 {\"cases\": [...]} JSON")
    output = args.output or ROOT / "runs" / f"microwave-parameter-eval-{datetime.now():%Y%m%d-%H%M%S}"
    summary = evaluate_parameter_cases(
        selection=args.selection, config=args.config, seed=args.seed,
        checkpoint=args.resume, output_dir=output,
        until_stage=args.until_stage, cases=payload["cases"],
    )
    print(json.dumps({
        "case_count": summary["case_count"],
        "reached_until_stage_count": summary["reached_until_stage_count"],
        "goal_success_count": summary["goal_success_count"],
        "summary": str(Path(output).resolve() / "summary.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
