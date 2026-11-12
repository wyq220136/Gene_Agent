#!/usr/bin/env python3
"""由清扫 run 保存的实际 MuJoCo qpos 帧生成视频。"""

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
                 fps: int = 12, manifest_path: str | Path | None = None) -> Path:
    run = Path(run_dir).resolve()
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    pick = json.loads((run / "pick/result.json").read_text(encoding="utf-8"))
    if result.get("status") not in {"completed", "failed"}:
        raise ValueError("run 尚未结束")
    rows = list(_samples(pick, "pick")) + list(_samples(result["stages"], "task"))
    unique = {round(row["sim_time_s"], 7): row for row in rows}
    frames = sorted(unique.values(), key=lambda item: item["sim_time_s"])
    if len(frames) < 2:
        raise ValueError("run 缺少实际 MuJoCo 回放采样")
    scene_xml = Path(result.get("scene_xml_path") or
                     "data/realworld/brush_cube_dustpan_0/scene.xml").resolve()
    if hashlib.sha256(scene_xml.read_bytes()).hexdigest() != result["scene_xml_sha256"]:
        raise ValueError("run 的 scene.xml 哈希与当前场景不一致，不能录制旧尺寸回放")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if any(np.asarray(row["qpos"]).shape != (model.nq,) for row in frames):
        raise ValueError("回放 qpos 与场景模型不匹配")
    destination = Path(output).resolve() if output else run / "sweep_actual.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 145, -30, .95
    camera.lookat[:] = [.44, 0, .10]
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", str(destination)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    times = [row["sim_time_s"] for row in frames]
    frame_count = max(2, int(np.ceil((times[-1] - times[0]) * fps)) + 1)
    try:
        with mujoco.Renderer(model, width=width, height=height) as renderer:
            for index in range(frame_count):
                at = min(times[0] + index / fps, times[-1])
                row = frames[max(0, bisect.bisect_right(times, at) - 1)]
                data.qpos[:] = row["qpos"]
                data.time = row["sim_time_s"]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 39), fill=(25, 33, 46))
                draw.text((12, 12), f"MuJoCo actual | {row['stage'][:45]} | {at:.2f} s | assisted brush grip",
                          fill="white")
                encoder.stdin.write(np.asarray(image).tobytes())
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
                "sample_count": len(frames), "frame_count": frame_count,
                "fps": fps, "duration_s": times[-1] - times[0],
                "simulation_assistance": result["simulation_assistance"]}
    manifest_destination = Path(manifest_path) if manifest_path else run / "video_manifest.json"
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--manifest", type=Path, help="清单另存路径；用于只读导出历史 run")
    args = parser.parse_args()
    print(record_video(args.run, output=args.output, fps=args.fps, manifest_path=args.manifest))


if __name__ == "__main__":
    main()
