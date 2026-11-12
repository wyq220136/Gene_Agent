"""核对纯视觉迁移的 8 个 Skill、6 个子任务和完整 Task 视频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


SKILLS = (
    "handle_pull", "push_open", "push_close", "pick_bowl", "insert_bowl",
    "extract_bowl", "place_bowl", "rotate_knob",
)
SUBTASKS = (
    "microwave_door_open", "microwave_door_close", "microwave_door_cycle",
    "microwave_object_insert", "microwave_object_extract", "microwave_knob_rotate",
)
TASK = "microwave_task1"


def decode_frame_count(movie: Path) -> int:
    """逐帧解码真实视频，返回完成解码的帧数。

    ffprobe 只读取容器头；即使视频中间损坏，它也可能报告正常时长。
    这里让 ffmpeg 消费完整视频，并用 progress 中的最后一帧核对清单。
    """
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2",
         "-i", str(movie), "-map", "0:v:0", "-f", "null", "-",
         "-progress", "pipe:1", "-nostats"],
        capture_output=True, text=True, check=False,
    )
    frames = [int(line.partition("=")[2]) for line in result.stdout.splitlines()
              if line.startswith("frame=") and line.partition("=")[2].strip().isdigit()]
    if result.returncode or not frames or frames[-1] <= 0:
        raise RuntimeError("MP4 逐帧解码失败：" + (result.stderr.strip() or "没有解码帧"))
    return frames[-1]


def inspect(root: Path) -> dict:
    """同时检查 MP4 可解码、运行成功清单与录像实际覆盖的阶段。"""
    rows = []
    errors = []
    for name in (*SKILLS, *SUBTASKS, TASK):
        cameras = ("overview", "gripper_side") if name in SKILLS else ("overview",)
        for camera in cameras:
            movie = root / name / f"{camera}.mp4"
            manifest_path = movie.with_suffix(".json")
            label = f"{name}/{camera}"
            if not movie.is_file() or movie.stat().st_size <= 1024 or not manifest_path.is_file():
                errors.append(f"{label}: 视频或清单缺失/为空")
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                run = manifest["run"]
                stages = run["stages"]
                frame_ids = manifest["stage_ids_by_frame"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"{label}: 清单损坏：{exc}")
                continue
            if (manifest.get("source") != "online_viser_visual_run"
                    or manifest.get("camera") != camera
                    or run.get("selection_id") != name
                    or run.get("success") is not True
                    or not stages or not all(stage.get("success") is True for stage in stages)
                    or len(frame_ids) < 2):
                errors.append(f"{label}: 不是该项的完整在线成功运行")
                continue
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(movie)],
                capture_output=True, text=True, check=False,
            )
            try:
                duration = float(probe.stdout.strip())
            except ValueError:
                duration = 0.
            if probe.returncode or duration <= 0.:
                errors.append(f"{label}: MP4 无法解码")
                continue
            try:
                decoded_frames = decode_frame_count(movie)
            except RuntimeError as exc:
                errors.append(f"{label}: {exc}")
                continue
            if decoded_frames != len(frame_ids):
                errors.append(f"{label}: 视频实际解码 {decoded_frames} 帧，清单记录 {len(frame_ids)} 帧")
                continue
            # 零时长夹爪命令不一定占据视频采样帧，故只报告覆盖数量，
            # 不把此类离散事件错误判作规划轨迹缺失。
            rows.append({
                "name": name, "camera": camera, "video": str(movie),
                "stage_count": len(stages), "frame_count": len(frame_ids),
                "decoded_frames": decoded_frames,
                "stage_ids_seen": len(set(frame_ids)), "duration_s": duration,
                "planning_duration_s": sum(float(stage.get("planning_duration_s") or 0.)
                                           for stage in stages),
            })
    return {"success": not errors and len(rows) == 23,
            "units_expected": 15, "videos_expected": 23,
            "videos_verified": len(rows), "errors": errors, "videos": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2]
                        / "artifacts/viser_acceptance")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.root.resolve())
    output = args.output or args.root / "acceptance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已验收 {report['videos_verified']}/{report['videos_expected']} 个视频；"
          f"报告：{output}")
    for error in report["errors"]:
        print("失败：" + error)
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
