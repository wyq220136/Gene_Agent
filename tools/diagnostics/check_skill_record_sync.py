"""检查一次 Skill 实际录像是否与完整 MuJoCo 状态采样对齐。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_run(run_dir: Path) -> dict:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "visual" / "manifest.json").read_text(encoding="utf-8"))
    stages = {}
    errors = []
    for stage_name, video_stage in manifest["stages"].items():
        stage = result["stages"].get(stage_name)
        if stage is None:
            errors.append(f"{stage_name}: 执行报告缺少阶段")
            continue
        execution = stage.get("execution") or stage
        samples = execution.get("replay_samples", [])
        by_state = {sample["state_id"]: sample for sample in samples}
        frames = video_stage.get("frames", [])
        unmatched = []
        for frame in frames:
            sample = by_state.get(frame["state_id"])
            if sample is None or abs(float(sample["sim_time_s"]) - float(frame["sim_time_s"])) > 1e-7:
                unmatched.append(frame["index"])
                continue
            if not sample.get("qpos") or result["instance_id"] not in sample.get("objects", {}):
                unmatched.append(frame["index"])
        if not samples or not frames or unmatched:
            errors.append(f"{stage_name}: 缺少完整样本/录像帧，或帧未匹配 {unmatched}")
        stages[stage_name] = {
            "video_frames": len(frames),
            "replay_samples": len(samples),
            "unmatched_frames": unmatched,
            "start_state_matches": bool(samples) and samples[0]["state_id"] == execution.get("start_state_id"),
            "end_state_matches": bool(samples) and samples[-1]["state_id"] == execution.get("end_state_id"),
        }
        if not stages[stage_name]["start_state_matches"] or not stages[stage_name]["end_state_matches"]:
            errors.append(f"{stage_name}: 阶段边界 state ID 与采样不一致")
    video = run_dir / "visual" / manifest["combined_video"]["path"]
    if not video.is_file():
        errors.append("缺少完整 Skill 视频")
    return {"success": not errors, "run": str(run_dir.resolve()), "stages": stages, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_run(args.run)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
