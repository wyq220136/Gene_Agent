#!/usr/bin/env python3
"""用抓勺 run 的实际 MuJoCo qpos 录制 MP4，并保存验收帧。"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def _samples(value, label=""):
    if isinstance(value, dict):
        rows = value.get("replay_samples")
        if isinstance(rows, list):
            for sample in rows:
                if "sim_time_s" in sample and "qpos" in sample:
                    yield {"sim_time_s": float(sample["sim_time_s"]),
                           "qpos": sample["qpos"], "stage": label}
        for key, item in value.items():
            if key != "replay_samples":
                yield from _samples(item, f"{label}/{key}" if label else key)
    elif isinstance(value, list):
        for item in value:
            yield from _samples(item, label)


def record_video(run_dir: str | Path, *, output: str | Path | None = None,
                 fps: int = 15) -> Path:
    run = Path(run_dir).resolve()
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    pick = json.loads((run / "pick/result.json").read_text(encoding="utf-8"))
    if not result.get("success") or not (
            pick.get("simulation_assisted_pick_success") or
            pick.get("physical_pick_success")):
        raise ValueError("run 尚未达到抬升保持成功条件")
    grip_label = ("assisted after bilateral contact"
                  if pick.get("simulation_assisted_pick_success")
                  else "physical grasp, no weld")
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    scene_xml = Path(result["scene_xml_path"])
    if hashlib.sha256(scene_xml.read_bytes()).hexdigest() != result["scene_xml_sha256"]:
        raise ValueError("场景 XML 已改变，不能将旧 qpos 误渲染为本次运行")
    unique = {round(row["sim_time_s"], 7): row for row in _samples(pick)}
    frames = sorted(unique.values(), key=lambda row: row["sim_time_s"])
    if len(frames) < 2:
        raise ValueError("run 缺少实际 MuJoCo 执行采样")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if any(np.asarray(row["qpos"]).shape != (model.nq,) for row in frames):
        raise ValueError("回放 qpos 与运行场景不符")
    destination = Path(output).resolve() if output else run / "spoon_pick_actual.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 145., -28., .57
    camera.lookat[:] = [.38, -.15, .09]
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", str(destination)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    times = [row["sim_time_s"] for row in frames]
    duration_s = times[-1] - times[0]
    count = max(2, int(np.ceil(duration_s * fps)) + 1)
    stage_frames = run / "video_frames"
    stage_frames.mkdir(exist_ok=True)
    # 保存真实执行采样中与阶段结束最近的帧；不重造抓取姿态。
    close_end = float(pick["stages"]["close"]["replay_samples"][-1]["sim_time_s"])
    targets = {"initial": times[0], "closed": close_end, "held": times[-1]}
    saved = set()
    try:
        with mujoco.Renderer(model, width=width, height=height) as renderer:
            for index in range(count):
                at = min(times[0] + index / fps, times[-1])
                row = frames[max(0, bisect.bisect_right(times, at) - 1)]
                data.qpos[:] = row["qpos"]
                data.time = row["sim_time_s"]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 39), fill=(25, 33, 46))
                draw.text((12, 12),
                          f"MuJoCo actual | {row['stage'][:35]} | {at:.2f} s | {grip_label}",
                          fill="white")
                encoder.stdin.write(np.asarray(frame).tobytes())
                for label, instant in targets.items():
                    if label not in saved and at >= instant:
                        frame.save(stage_frames / f"{label}.png")
                        saved.add(label)
    except Exception:
        encoder.kill()
        encoder.wait()
        destination.unlink(missing_ok=True)
        raise
    encoder.stdin.close()
    error = encoder.stderr.read().decode("utf-8", errors="replace")
    if encoder.wait() != 0:
        raise RuntimeError(f"视频编码失败：{error}")
    manifest = {"source_run": str(run), "video": str(destination),
                "source": "actual_mujoco_qpos_nearest_sample",
                "sample_count": len(frames), "frame_count": count,
                "fps": fps, "duration_s": duration_s,
                "simulation_assistance": result["simulation_assistance"],
                "physical_pick_success": bool(pick.get("physical_pick_success")),
                "acceptance_frames": {key: str(stage_frames / f"{key}.png")
                                      for key in saved}}
    (run / "video_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    print(record_video(args.run, output=args.output, fps=args.fps))


if __name__ == "__main__":
    main()
