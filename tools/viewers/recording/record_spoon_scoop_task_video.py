#!/usr/bin/env python3
"""录制抓勺、舀球、倾倒的 MuJoCo 实际执行；支持单 Skill 或整条 Task。"""

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


PICK_STAGES = ("pregrasp", "approach", "close", "lift", "hold")
SCOOP_STAGES = ("source_above", "source_pitch", "source_entry", "source_sweep",
                "source_lift_low", "source_level", "source_hold")
POUR_STAGES = ("target_above", "target_pretilt_hold", "target_tilt_30",
               "target_tilt_45", "target_tilt_60", "target_tilt_75",
               "target_settle")


def _stage_samples(stage: str, value: dict) -> list[dict]:
    rows = value.get("execution", {}).get("replay_samples", ()) if value.get("execution") else ()
    if not rows:
        rows = value.get("replay_samples", ())
    return [{"sim_time_s": float(row["sim_time_s"]), "qpos": row["qpos"],
             "stage": stage} for row in rows if "sim_time_s" in row and "qpos" in row]


def _collect(run: Path, segment: str) -> tuple[dict, list[dict]]:
    task = json.loads((run / "task_result.json").read_text(encoding="utf-8"))
    pick = json.loads((run / "pick/result.json").read_text(encoding="utf-8"))
    if not pick.get("success"):
        raise ValueError("抓勺 Skill 尚未成功")
    stages = []
    if segment in {"pick", "full"}:
        stages.extend((name, pick["stages"].get(name, {})) for name in PICK_STAGES)
    if segment in {"scoop", "full"}:
        if not task.get("skills", {}).get("scoop", {}).get("success"):
            raise ValueError("盛球 Skill 尚未成功")
        stages.extend((name, task["stages"].get(name, {})) for name in SCOOP_STAGES)
    if segment in {"pour", "full"}:
        if segment == "pour" and not task.get("skills", {}).get("pour", {}).get("success"):
            raise ValueError("倾倒 Skill 尚未成功")
        stage_names = (POUR_STAGES if "target_tilt_30" in task["stages"] else
                       ("target_above", "target_pretilt_hold", "target_tilt",
                        "target_settle"))
        stages.extend((name, task["stages"].get(name, {})) for name in stage_names)
    rows = [row for name, value in stages for row in _stage_samples(name, value)]
    unique = {round(row["sim_time_s"], 7): row for row in rows}
    return task, sorted(unique.values(), key=lambda row: row["sim_time_s"])


def record(run_dir: str | Path, *, segment: str = "full", fps: int = 15) -> Path:
    if segment not in {"pick", "scoop", "pour", "full"}:
        raise ValueError("segment 必须是 pick、scoop、pour 或 full")
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    run = Path(run_dir).resolve()
    task, samples = _collect(run, segment)
    if len(samples) < 2:
        raise ValueError("缺少实际 MuJoCo 执行 qpos 采样")
    scene_xml = Path(task["scene_xml_path"])
    if hashlib.sha256(scene_xml.read_bytes()).hexdigest() != task["scene_xml_sha256"]:
        raise ValueError("场景 XML 与运行记录不一致")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if any(np.asarray(row["qpos"]).shape != (model.nq,) for row in samples):
        raise ValueError("qpos 与本次场景不一致")
    destination = run / f"spoon_{segment}_actual.mp4"
    width, height = 960, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    lookat, distance = {
        "pick": ([.38, -.21, .09], .58),
        "scoop": ([.42, -.09, .09], .50),
        "pour": ([.42, .13, .11], .50),
        "full": ([.39, -.02, .12], .85),
    }[segment]
    camera.azimuth, camera.elevation, camera.distance = 145., -30., distance
    camera.lookat[:] = lookat
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-pix_fmt", "yuv420p", str(destination)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    times = [row["sim_time_s"] for row in samples]
    duration_s = times[-1] - times[0]
    frame_count = max(2, int(np.ceil(duration_s * fps)) + 1)
    frames_dir = run / "video_frames" / segment
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = {"initial": samples[0], "final": samples[-1]}
    if segment in {"scoop", "full"}:
        for name in ("source_sweep", "source_hold"):
            rows = _stage_samples(name, task["stages"].get(name, {}))
            if rows:
                capture[name] = rows[-1]
    if segment in {"pour", "full"}:
        for name in ("target_above", "target_tilt_30", "target_tilt_45",
                     "target_tilt_60", "target_tilt_75", "target_settle"):
            rows = _stage_samples(name, task["stages"].get(name, {}))
            if rows:
                capture[name] = rows[-1]
    saved = set()
    try:
        with mujoco.Renderer(model, width=width, height=height) as renderer:
            def draw_sample(row: dict) -> Image.Image:
                data.qpos[:] = row["qpos"]
                data.time = row["sim_time_s"]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 39), fill=(25, 33, 46))
                draw.text((12, 12),
                          f"Actual MuJoCo | {row['stage']} | "
                          f"{row['sim_time_s']:.2f} s | {task['simulation_assistance']}",
                          fill="white")
                return frame

            for index in range(frame_count):
                at = min(times[0] + index / fps, times[-1])
                sample = samples[max(0, bisect.bisect_right(times, at) - 1)]
                frame = draw_sample(sample)
                encoder.stdin.write(np.asarray(frame).tobytes())
                for label, captured in capture.items():
                    if label not in saved and at >= captured["sim_time_s"]:
                        # 阶段边界可能与下一段首帧同时间；从该阶段原始
                        # MuJoCo 采样重绘，避免验收图被下一阶段覆盖。
                        draw_sample(captured).save(frames_dir / f"{label}.png")
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
    if segment in {"pour", "full"}:
        data.qpos[:] = samples[-1]["qpos"]
        mujoco.mj_forward(model, data)
        target = data.body(task["target_bowl"]).xpos
        top_camera = mujoco.MjvCamera()
        top_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        top_camera.azimuth, top_camera.elevation, top_camera.distance = 180., -86., .36
        top_camera.lookat[:] = target + np.array([0., 0., .035])
        with mujoco.Renderer(model, width=720, height=720) as renderer:
            renderer.update_scene(data, camera=top_camera, scene_option=option)
            Image.fromarray(renderer.render()).save(frames_dir / "target_top.png")
            visual_name = f"{task['tool_name']}_visual"
            visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, visual_name)
            if visual_id >= 0:
                model.geom_rgba[visual_id, 3] = 0.
                renderer.update_scene(data, camera=top_camera, scene_option=option)
                Image.fromarray(renderer.render()).save(frames_dir / "target_top_tool_hidden.png")
                saved.add("target_top_tool_hidden")
        saved.add("target_top")
    manifest = {"segment": segment, "source_run": str(run),
                "video": str(destination), "source": "actual_mujoco_qpos_nearest_sample",
                "sample_count": len(samples), "frame_count": frame_count,
                "fps": fps, "duration_s": duration_s,
                "simulation_assistance": task["simulation_assistance"],
                "acceptance_frames": {key: str(frames_dir / f"{key}.png") for key in saved}}
    (run / f"video_{segment}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--segment", choices=("pick", "scoop", "pour", "full"),
                        default="full")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    print(record(args.run, segment=args.segment, fps=args.fps))


if __name__ == "__main__":
    main()
