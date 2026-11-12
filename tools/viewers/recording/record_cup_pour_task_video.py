#!/usr/bin/env python3
"""把杯子 Skill/Task 的实际 MuJoCo qpos 回放录成视频和验收帧。"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from gene_agent.feedback.tabletop.cup_run import load_replay


SELECTIONS = {"pick": "cup_pick", "pour": "cup_pour",
              "place": "cup_place", "full": "cup_pour_into_bowl"}


def record(run_dir: str | Path, *, segment: str = "full", fps: int = 15) -> Path:
    if segment not in SELECTIONS or fps <= 0:
        raise ValueError("segment 必须为 pick/pour/place/full，fps 必须为正数")
    run = Path(run_dir).resolve()
    task, samples, scene = load_replay(run, SELECTIONS[segment])
    model = mujoco.MjModel.from_xml_path(str(scene))
    if any(len(row["qpos"]) != model.nq for row in samples):
        raise ValueError("回放 qpos 与本次场景不一致")
    width, height = 960, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    destination = run / f"cup_{segment}_actual.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "22", "-pix_fmt", "yuv420p", str(destination)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 145., -30.
    camera.distance = .58 if segment == "pick" else .75
    camera.lookat[:] = ([.35, -.16, .12] if segment == "pick" else [.34, -.01, .12])
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    times = [row["sim_time_s"] for row in samples]
    duration = times[-1] - times[0]
    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    frames_dir = run / "video_frames" / segment
    frames_dir.mkdir(parents=True, exist_ok=True)
    milestones = {"initial": samples[0], "final": samples[-1]}
    for name in ("approach", "lift", "carry_raise", "carry_above",
                 "pour_approach", "tilt_30", "tilt_50",
                 "pour_lower_50", "hold_50_after_lower", "tilt_60",
                 "pour_lower_1", "tilt_90", "pour_lower_2", "tilt_105",
                 "hold_90_after_lower", "pour_settle",
                 "pour_raise", "clear_bowl", "upright_settle_15",
                 "upright_settle_0",
                 "return_turn_upright",
                 "return_above", "place_descend",
                 "release_open", "retreat_low", "retreat_high"):
        matching = [row for row in samples if row["stage"] == name]
        if matching:
            milestones[name] = matching[-1]
    captured = set()
    try:
        with mujoco.Renderer(model, width=width, height=height) as renderer:
            def draw(row):
                data.qpos[:] = row["qpos"]
                data.time = row["sim_time_s"]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                ink = ImageDraw.Draw(frame)
                ink.rectangle((0, 0, width, 38), fill=(25, 33, 46))
                ink.text((12, 12),
                         f"Actual MuJoCo | {row['stage']} | {row['sim_time_s']:.2f} s | "
                         f"{task['simulation_assistance']}", fill="white")
                return frame

            for index in range(frame_count):
                at = min(times[0] + index / fps, times[-1])
                sample = samples[max(0, bisect.bisect_right(times, at) - 1)]
                encoder.stdin.write(np.asarray(draw(sample)).tobytes())
                for name, row in milestones.items():
                    if name not in captured and at >= row["sim_time_s"]:
                        draw(row).save(frames_dir / f"{name}.png")
                        captured.add(name)
    except Exception:
        encoder.kill()
        encoder.wait()
        destination.unlink(missing_ok=True)
        raise
    encoder.stdin.close()
    error = encoder.stderr.read().decode("utf-8", errors="replace")
    if encoder.wait() != 0:
        raise RuntimeError(f"视频编码失败：{error}")
    manifest = {"source": "actual_mujoco_qpos_nearest_sample",
                "source_run": str(run), "segment": segment,
                "video": str(destination), "fps": fps,
                "duration_s": duration, "sample_count": len(samples),
                "frame_count": frame_count,
                "acceptance_frames": {name: str(frames_dir / f"{name}.png")
                                      for name in sorted(captured)}}
    (run / f"video_{segment}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--segment", choices=tuple(SELECTIONS), default="full")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    print(record(args.run, segment=args.segment, fps=args.fps))


if __name__ == "__main__":
    main()
