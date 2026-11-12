"""从 fresh reset 在线执行微波炉测试任务，输出每段报告和实际状态。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from gene_agent.tasks.policies.microwave_task.simulation import run_microwave_task1_physical
from gene_agent.visualization.video_export import export_video, resolve_ffmpeg
from gene_agent.tasks.policies.microwave_task.subtasks import SUBTASK_NAMES, REPLAY_SKILLS, validate_stage_stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subtask', choices=('all', *SUBTASK_NAMES), default='all',
                        help='独立操作/组合；all 保持完整 Task')
    parser.add_argument('--preparation', choices=('prefix', 'current'), default='prefix',
                        help='prefix 从 fresh reset 执行必要前缀；current 不补动作，检查当前前置条件')
    parser.add_argument("--config", type=Path,
                        default=ROOT / "data/task_1/simulation/mujoco_task1_physical.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / f"microwave-task1-physical-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--video", action="store_true",
                        help="记录完整实际状态流，并在正常或异常结束后导出全景+近景+俯视 MP4")
    parser.add_argument('--seed',type=int,help='可复现的摆位 seed')
    parser.add_argument('--collision-mode', choices=['full', 'robot_only'],
                        help='覆盖配置 motion.collision_mode；robot_only 跳过场景避障与规划 attachment')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--randomize',action='store_true')
    mode.add_argument('--fixed-placement',action='store_true')
    parser.add_argument('--live',action='store_true',help='追加实际 qpos 供在线查看')
    parser.add_argument("--stop-after", help="单项 Skill 阶段停止；如 insert_bowl 的 release、rotate_knob 的 grasp")
    args = parser.parse_args()
    try:
        validate_stage_stop(args.subtask, args.stop_after)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"运行目录已存在：{output}")
    # 录像依赖在任何物理动作前验证；避免任务执行完成后才暴露可执行文件
    # 或动态库缺失，并把无关基础设施问题误记为操作失败。
    if args.video:
        resolve_ffmpeg()
    result = None
    task_error = None
    video_error = None
    video = None
    try:
        from gene_agent.feedback.live_record import live_physics_recording
        # --video 必须从动作第一帧开始记录；异常退出时 context manager 仍会
        # 封口实际状态流，随后用已产生的前缀导出到真实停止点。
        with live_physics_recording(output, enabled=args.live or args.video):
            result = run_microwave_task1_physical(
                output, config=args.config, record_video=False, seed=args.seed,
                collision_mode=args.collision_mode, selection=args.subtask,
                preparation=args.preparation, stop_after=args.stop_after,
                randomize=(True if args.randomize else
                           (False if args.fixed_placement else None)),
            )
    except Exception as error:
        task_error = error
        # 用户只看一行失败摘要；完整 Python traceback 和已完成阶段报告
        # 留在同一个 run，Codex 可直接定位具体阶段而不依赖终端滚屏。
        output.mkdir(parents=True, exist_ok=True)
        (output / "task_exception.txt").write_text(
            traceback.format_exc(), encoding="utf-8")
        summary_path = output / "summary.json"
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.is_file() else {})
        summary.setdefault("task", "microwave_task1")
        summary.setdefault("source_format", "mujoco_physical_exception_prefix")
        summary["status"] = "failed"
        summary["exception"] = {"type": type(error).__name__,
                                "message": str(error),
                                "traceback_ref": "task_exception.txt"}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)
                                + "\n", encoding="utf-8")
    if args.video:
        # 录像是反馈，不改变原始任务结论；任务异常也先尝试导出已执行前缀。
        try:
            video = export_video(output, output / "video/task.mp4", mode="actual",
                                 cameras=("overview", "detail", "top"),
                                 profile="microwave_task1",
                                 skill=REPLAY_SKILLS.get(args.subtask))
        except Exception as error:
            video_error = error
            (output / "video_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    if task_error is not None:
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps({
            "status": "failed",
            "failure_subtask": summary.get("failure_subtask"),
            "error": str(task_error),
            "report": str(summary_path),
            "video": None if video is None else video["video"],
            "video_error": None if video_error is None else str(video_error),
        }, ensure_ascii=False))
        raise SystemExit(1) from None
    if video_error is not None:
        print(json.dumps({"task_status": result["status"],
                          "goal_success": result["goal_success"],
                          "video_status": "failed", "error": str(video_error),
                          "traceback_ref": str(output / "video_error.txt")},
                         ensure_ascii=False))
        raise SystemExit(2) from None
    print(json.dumps({
        "status": result["status"], "goal_success": result["goal_success"],
        "requested_scope": result["requested_scope"], "operation_success": result["operation_success"],
        "failure": result.get("failure"), "timing": result["timing"],
        "evidence_mode": result["evidence_mode"],
        "subtasks": {key: value["success"] for key, value in result["subtasks"].items()},
        "report": str(output / "summary.json"),
        "video": None if video is None else video["video"],
    }, ensure_ascii=False))
    if not result["operation_success"] or (args.subtask == "all" and not result["goal_success"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
