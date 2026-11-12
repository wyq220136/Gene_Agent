"""从已录制的完整实际视频裁出七段可独立检查的子任务。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.viewers.recording.record_physical_task1_video import collect_actual_samples


def split_subtask_clips(run: Path) -> dict:
    run = Path(run).resolve()
    full = run / "task_overview.mp4"
    if not full.is_file():
        raise FileNotFoundError("请先录制完整任务全景视频")
    manifest_path=run/'video_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    frames=manifest.get('frames') or []
    samples=[] if frames else collect_actual_samples(run)
    task_start=samples[0]['sim_time_s'] if samples else None
    def label(value):
        return re.sub(r'^\d+_', '', value)
    groups = {
        "open_first": ("open_first/",),
        "pick_insert": ("pick_insert",),
        "close_first": ("close_first/",),
        "rotate_knob": ("rotate_knob", "knob_execution"),
        "open_second": ("open_second/",),
        "extract_place": ("extract_grasp", "extract_retreat", "table_place"),
        "close_second": ("close_second/",),
    }
    directory = run / "clips"
    directory.mkdir(exist_ok=True)
    rows = {}
    for name, labels in groups.items():
        if frames:
            selected=[row for row in frames if label(row['stage_id']).startswith(labels)]
            if len(selected)<2:raise ValueError(f'子任务 {name} 缺少实际视频帧')
            start=min(row['video_time_s'] for row in selected)
            duration=max(row['video_time_s'] for row in selected)-start+1/manifest['fps']
            source_times=[row['sim_time_s'] for row in selected]
            source_kind='encoded_frame_timeline'
        else:
            source_times=[sample['sim_time_s'] for sample in samples
                          if label(sample['label']).startswith(labels)]
            if len(source_times)<2:raise ValueError(f'子任务 {name} 缺少实际 MuJoCo 视频采样')
            start=max(0.,min(source_times)-task_start-.08)
            duration=max(source_times)-min(source_times)+.16
            source_kind='legacy_sim_time_timeline'
        path = directory / f"{name}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
            "-i", str(full), "-t", f"{duration:.3f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", str(path),
        ], check=True, capture_output=True, text=True)
        rows[name] = {"video": str(path), "sim_start_s": min(source_times),
                      "sim_end_s": max(source_times), "source_kind": source_kind,
                      "video_start_s": start, "video_duration_s": duration}
    manifest = {"source_video": str(full), "source_kind": "actual_mujoco_replay",
                "clips": rows}
    (run / "subtask_clips.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = split_subtask_clips(args.run)
    print(json.dumps({key: row["video"] for key, row in result["clips"].items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
