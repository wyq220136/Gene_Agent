"""从真实 MuJoCo 执行采样录制门 Skill 或连续门循环视频。"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gene_agent.visualization.style import camera_presets, configure_camera
from gene_agent.visualization.grasp_tuner.microwave_scene import set_preview_microwave_pose


def _reports(run: Path) -> list[dict]:
    online = run / "report.json"
    if online.is_file():
        report = json.loads(online.read_text(encoding="utf-8"))
        if report.get("source_format") != "mujoco_physical":
            raise ValueError("在线实验报告必须来自 MuJoCo 实际执行")
        return [report]
    cycle = run / "door_cycle_result.json"
    if cycle.is_file():
        paths = [run / "open/physical_door_result.json",
                 run / "close/physical_door_result.json"]
    else:
        paths = [run / "physical_door_result.json"]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("缺少门物理执行记录；不能从规划预览伪造视频")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _apply_run_scene_pose(model: mujoco.MjModel, run: Path) -> None:
    """录像模型复原运行时编辑的微波炉及垫台摆位，不能只复原 qpos。

    微波炉和垫台是 world 下的固定 body；实验会直接更新编译模型的
    body pos/quat。这些值不在逐帧 qpos 中。若录像只加载原场景 XML，
    扰动实验的门与夹爪会显示在不同的摆位，视觉反馈就会误导调参。
    """
    config_path = Path(run) / "config_initial.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pose = config.get("microwave_pose_world")
    if pose is None:
        return
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    set_preview_microwave_pose(
        SimpleNamespace(model=model, data=data), np.asarray(pose, dtype=float)
    )


def _samples(run: Path, reports: list[dict]) -> list[dict]:
    directories = ([run / "open", run / "close"] if len(reports) == 2 else [run])
    samples: list[dict] = []
    for directory, report in zip(directories, reports):
        if report.get("source_format") == "mujoco_physical" and "goal_success" in report:
            for stage_id in report["stages"]:
                path = directory / "stages" / f"{stage_id.replace('/', '_')}.json"
                stage = json.loads(path.read_text(encoding="utf-8"))
                execution = stage.get("execution") or {}
                for sample in execution.get("replay_samples", []):
                    row = {"sim_time_s": float(sample["sim_time_s"]),
                           "qpos": sample["qpos"], "stage": stage_id.split("/")[-1],
                           "skill": stage_id.split("/")[0]}
                    if samples and abs(samples[-1]["sim_time_s"] - row["sim_time_s"]) < 1e-9:
                        samples[-1] = row
                    else:
                        samples.append(row)
            continue
        for stage in report["stages"]:
            execution = json.loads((directory / stage["execution_ref"]).read_text(
                encoding="utf-8"))
            for sample in execution["replay_samples"]:
                row = {"sim_time_s": float(sample["sim_time_s"]),
                       "qpos": sample["qpos"], "stage": stage["name"],
                       "skill": report["skill"]}
                if samples and abs(samples[-1]["sim_time_s"] - row["sim_time_s"]) < 1e-9:
                    samples[-1] = row
                else:
                    samples.append(row)
    if len(samples) < 2 or any(a["sim_time_s"] >= b["sim_time_s"]
                               for a, b in zip(samples, samples[1:])):
        raise ValueError("实际执行采样时间必须连续递增")
    return samples


def record_physical_door_video(run: str | Path, *, cameras: tuple[str, ...] =
                               ("task", "gripper_side"), fps: int = 24,
                               width: int = 640, height: int = 480,
                               geometry: str = "visual") -> dict:
    """每帧取最近的真实 qpos；不插值门角或生成虚构物理状态。"""
    run = Path(run).resolve()
    if fps <= 0 or min(width, height) <= 0 or width % 2 or height % 2:
        raise ValueError("fps 和偶数画面宽高必须为正")
    if geometry not in {"visual", "collision"}:
        raise ValueError("geometry 只能为 visual 或 collision")
    reports = _reports(run)
    samples = _samples(run, reports)
    first = reports[0]
    scene = Path(first["scene_ref"])
    if not scene.is_file():
        raise FileNotFoundError(f"保存的 MuJoCo 场景不存在：{scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    _apply_run_scene_pose(model, run)
    if geometry == "collision":
        # 物理模型把很多 collision geom 的 alpha 设为 0，正常画面不显示。
        # 只改此处独立加载的录像模型颜色，不改实验模型或接触形状。
        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) != 3:
                continue
            model.geom_rgba[geom_id, 3] = max(float(model.geom_rgba[geom_id, 3]), .18)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name in {"microwave_door_panel_collision", "microwave_handle_grip_collision"}:
                model.geom_rgba[geom_id] = (1., .42, .12, .48)
    data = mujoco.MjData(model)
    robot_path = (scene.parent / "robot/robot_metadata.json"
                  if "planned_run" not in first else
                  Path(first["planned_run"]) / "source_scene/robot/robot_metadata.json")
    robot = json.loads(robot_path.read_text(encoding="utf-8"))
    group = first.get("robot_group") or next(iter(first["actual_final"]["robots"]))
    if any(np.asarray(row["qpos"]).shape != (model.nq,) for row in samples):
        raise ValueError("实际 qpos 与保存的场景维度不一致")
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    start = samples[0]["sim_time_s"]
    stop = samples[-1]["sim_time_s"]
    nframes = max(2, int(np.ceil((stop - start) * fps)) + 1)
    times = [row["sim_time_s"] for row in samples]
    encoders = {}
    outputs = {}
    option = mujoco.MjvOption()
    if geometry == "collision":
        # 本场景资产把所有真实接触代理放在 geom group 3；只显示这一
        # 层可以看到 MuJoCo 实际拿来碰撞的形状，不用 visual mesh 猜。
        option.geomgroup[:] = 0
        option.geomgroup[3] = 1
    else:
        option.geomgroup[3] = 0
    try:
        for name in cameras:
            key = name if geometry == "visual" else f"{name}_collision"
            path = run / f"physical_door_{key}.mp4"
            command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                       "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "22",
                       "-pix_fmt", "yuv420p", str(path)]
            encoders[key] = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            outputs[key] = str(path)
        with mujoco.Renderer(model, width=width, height=height) as renderer:
            for index in range(nframes):
                query_time = min(start + index / fps, stop)
                sample_index = max(0, bisect.bisect_right(times, query_time) - 1)
                sample = samples[sample_index]
                data.qpos[:] = sample["qpos"]
                data.time = sample["sim_time_s"]
                mujoco.mj_forward(model, data)
                presets = camera_presets(model, data, "microwave_door",
                                         robot_metadata=robot, active_group=group)
                for name in cameras:
                    if name not in presets:
                        raise ValueError(f"未知相机 {name}；可用：{sorted(presets)}")
                    params = dict(presets[name])
                    if name == "gripper_side":
                        # 门体原点在铰链，侧视镜头需要对准真实把手与指垫。
                        handle = np.asarray(data.site("microwave_handle_grasp").xpos)
                        pads = [np.asarray(data.geom(p).xpos)
                                for p in robot["grippers"][group]["pad_geom_names"]]
                        midpoint = np.mean(pads, axis=0)
                        is_inside_push = sample["skill"] == "push_open"
                        # 推开门的接触点位于门内侧；仍从外侧看会被整块门板
                        # 遮住。夹爪侧视随实际 Skill 选择门内/外侧，并始终
                        # 对准两指中心；开门抓把手与关门仍保留外侧机位。
                        params["lookat"] = (midpoint if is_inside_push else
                                            0.45 * handle + 0.55 * midpoint).tolist()
                        # 通用夹爪侧视会按机器人根部决定正反；门板可能恰好挡在
                        # 选中的这一侧。这里把视线固定到柜体外侧，再与双指开合
                        # 轴正交，保证画面能看到指垫和把手的接触区域。
                        opening = pads[0] - pads[1]
                        side = np.cross(opening, [0.0, 0.0, 1.0])
                        side[2] = 0.0
                        outward = handle - np.asarray(data.body("microwave").xpos)
                        outward[2] = 0.0
                        if np.linalg.norm(side) < 1e-8:
                            side = outward.copy()
                        if np.dot(side, outward) < 0:
                            side = -side
                        if is_inside_push:
                            side = -side
                        side /= np.linalg.norm(side)
                        side[2] = 0.65 if is_inside_push else 0.25
                        side /= np.linalg.norm(side)
                        params["distance"] = 0.42 if is_inside_push else 0.34
                        params["azimuth"] = float(np.degrees(np.arctan2(-side[1], -side[0])))
                        params["elevation"] = float(-np.degrees(np.arcsin(side[2])))
                    camera = configure_camera(mujoco.MjvCamera(), params)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.rounded_rectangle((8, 8, min(width - 8, 550), 61),
                                           radius=6, fill=(22, 30, 42))
                    angle = float(data.qpos[model.joint("microwave_door_joint").qposadr[0]])
                    draw.text((18, 17), f"MUJOCO {geometry.upper()} | {sample['skill']} / {sample['stage']}",
                              fill="white")
                    draw.text((18, 39), f"door {np.degrees(angle):.1f} deg | t={sample['sim_time_s']:.2f} s",
                              fill=(220, 235, 245))
                    key = name if geometry == "visual" else f"{name}_collision"
                    assert encoders[key].stdin is not None
                    encoders[key].stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
    except Exception:
        for encoder in encoders.values():
            encoder.kill()
            encoder.wait()
        for path in outputs.values():
            Path(path).unlink(missing_ok=True)
        raise
    for name, encoder in encoders.items():
        assert encoder.stdin is not None and encoder.stderr is not None
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"相机 {name} 视频编码失败：{error}")
    manifest_path = run / "physical_door_video_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prior_videos = previous.get("videos", {}) if (previous.get("source_run") == str(run)
                    and previous.get("fps") == fps and previous.get("frame_count") == nframes) else {}
    manifest = {
        "schema_version": 1, "source_run": str(run),
        "videos": {**prior_videos, **outputs},
        "frame_count": nframes, "fps": fps, "duration_s": stop - start,
        "sample_policy": "previous_saved_actual_sample_no_interpolation",
        "physical_success": all(report.get("goal_success", report.get("success", False))
                                for report in reports),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["task", "gripper_side"])
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--geometry", choices=("visual", "collision"), default="visual")
    args = parser.parse_args()
    print(json.dumps(record_physical_door_video(
        args.run, cameras=tuple(args.cameras), fps=args.fps,
        geometry=args.geometry,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
