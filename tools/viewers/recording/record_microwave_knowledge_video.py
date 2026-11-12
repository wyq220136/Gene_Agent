"""录制 Viser 中逐角推门 Knowledge，并保存每帧可复现的状态清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import viser
from PIL import Image, ImageDraw, ImageFont

from tools.viewers.capture.capture_knowledge_viser import _scene_material_ready, _wait_ui
from gene_agent.visualization.grasp_tuner.feedback import read_tuner_state
from gene_agent.visualization.grasp_tuner.microwave_app import MicrowaveKnowledgeApp
from gene_agent.visualization.motion_debug.capture import DEFAULT_CHROME, DEFAULT_WINDOW_SIZE, _free_port, _stop, _validate_png
from gene_agent.visualization.pick_bowl.capture import _devtools_page, _save_cdp_screenshot


ROOT = Path(__file__).resolve().parents[3]


def _annotate_frame(source: Path, output: Path, *, mode: str, angle_deg: float,
                    reachable: bool, position_error_m: float, rotation_error_deg: float) -> None:
    """在视频帧上直接标明 IK 结论；原始 Viser 截图仍单独保存。"""
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.rounded_rectangle((18, 18, 925, 78), radius=10, fill=(27, 35, 47))
    label = (f"{mode}  {angle_deg:.0f} deg  |  IK {'PASS' if reachable else 'FAIL'}"
             f"  |  TCP {position_error_m * 1000:.1f} mm, {rotation_error_deg:.1f} deg")
    draw.text((35, 33), label, font=font, fill=(100, 225, 155) if reachable else (255, 137, 119))
    image.save(output)


def record(output: Path, *, run: Path | None = None, restore_state: Path | None = None,
           push_mode: str | None = None, seconds_per_waypoint: float = .8,
           closeup: bool = False) -> dict:
    """以同一 Viser 客户端截取每个真实 waypoint；不插值、不声称执行成功。"""
    if not 0.1 <= seconds_per_waypoint <= 10:
        raise ValueError("每个 waypoint 的展示时间必须在 0.1–10 秒之间")
    restored = read_tuner_state(restore_state, tool="microwave_knowledge") if restore_state else None
    if restored and restored["mode"] != "door_push":
        raise ValueError("恢复清单必须来自 door_push 模式")
    if push_mode is not None and push_mode not in {"right_open", "right_close"}:
        raise ValueError("当前活动臂只支持 right_open 或 right_close 回放")
    source_run = Path(restored["run"]) if restored else run or ROOT / "runs/pick-bowl-planning-final"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output.with_suffix(".frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    state_path = frames_dir / "current.state.json"
    log_path = output.with_suffix(".log")
    port, chrome_port = _free_port(), _free_port()
    server = viser.ViserServer(host="127.0.0.1", port=port)
    browser = None
    try:
        app = MicrowaveKnowledgeApp(server, ROOT, source_run, mode="door_push",
                                    state_path=state_path, restored=restored)
        if push_mode is not None and app.push_mode.value != push_mode:
            # 与界面的模式下拉框共用切换逻辑；它同时恢复 Task1 的起止角，
            # 而非仅把开门轨迹倒序播放。
            app.push_mode.value = push_mode
            app._on_push_mode()
        # 多数推门候选的名义首选纯 IK 未求得；视频必须同时画出目标夹爪，
        # 否则只见机械臂最近尝试姿态，无法检查接触知识本身。
        app.target_ghost.value = True
        if closeup:
            app.overview.value = False
            path = app.knowledge.door_path
            start = path.waypoints[0].tcp_pose_world[:3, 3]
            end = path.waypoints[-1].tcp_pose_world[:3, 3]
            center = ((start + end) / 2).copy()
            app.camera = {"position": (center + [-.32, .48, .26]).tolist(),
                          "look_at": center.tolist(), "up": [0., 0., 1.], "fov": .95}
        # 录制前检查整段纯 IK；回放时只切换已经计算好的结果，
        # 不让求解耗时改变各 waypoint 的展示长度。
        app._regenerate(check_all=True)
        count = len(app.knowledge.door_path.waypoints)
        with tempfile.TemporaryDirectory(prefix="gene-door-knowledge-chrome-") as profile, \
                log_path.open("w", encoding="utf-8") as log:
            command = [str(DEFAULT_CHROME), "--headless=new", "--no-sandbox",
                       "--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=gl-egl",
                       "--disable-dev-shm-usage", "--remote-allow-origins=*",
                       "--force-device-scale-factor=1",
                       f"--window-size={DEFAULT_WINDOW_SIZE[0]},{DEFAULT_WINDOW_SIZE[1]}",
                       f"--user-data-dir={profile}", f"--remote-debugging-port={chrome_port}",
                       f"http://127.0.0.1:{port}/"]
            browser = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=os.environ.copy())
            socket_url = _devtools_page(chrome_port, 25.)
            _wait_ui(socket_url, "微波炉 Knowledge 调参")
            # 校准 URDF 的门板包含多个 mesh；必须等待材质上传，
            # 否则首帧会把白色占位模型误当作最终推门画面。
            time.sleep(12.)
            records = []
            for index in range(count):
                app.show_waypoint(index)
                for client in server.get_clients().values():
                    client.flush()
                    client.get_render(height=8, width=8, transport_format="png", timeout=10.)
                frame = frames_dir / f"frame_{index:03d}.png"
                _save_cdp_screenshot(socket_url, frame)
                _validate_png(frame, DEFAULT_WINDOW_SIZE)
                if index == 0 and not closeup and not _scene_material_ready(frame, require_gripper=True, view="microwave"):
                    raise RuntimeError("首帧的微波炉或夹爪材质尚未加载，停止录制以免误导检查")
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not state.get("input_valid") or state["waypoint_index"] != index:
                    raise RuntimeError(f"第 {index} 帧图像与 Knowledge 状态不同步")
                angle_deg = float(state["path"]["waypoints"][index]["angle_rad"] * 180 / 3.141592653589793)
                video_frame = frames_dir / f"video_frame_{index:03d}.png"
                _annotate_frame(
                    frame, video_frame, mode=state["tuner_values"]["door_push_mode"],
                    angle_deg=angle_deg, reachable=bool(state["ik_reachable"][index]),
                    position_error_m=float(state["ik_position_error_m"][index]),
                    rotation_error_deg=float(state["ik_rotation_error_deg"][index]),
                )
                records.append({
                    "waypoint_index": index,
                    "angle_deg": angle_deg,
                    "ik_reachable": state["ik_reachable"][index],
                    "ik_position_error_m": state["ik_position_error_m"][index],
                    "ik_rotation_error_deg": state["ik_rotation_error_deg"][index],
                    "tcp_pose_world": state["selected_tcp_pose_world"],
                    "parameters": state["selected_grasp_parameters"],
                    "frame": str(frame),
                    "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    "video_frame": str(video_frame),
                })
            _stop(browser)
            browser = None
        # 重复静态 waypoint 帧即可表达离散知识；不存在仿真运动或
        # 关节轨迹插值。以稳定的 24 fps 编码，便于各平台播放。
        command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                   f"{1 / seconds_per_waypoint:.8f}", "-i", str(frames_dir / "video_frame_%03d.png"),
                   "-c:v", "libx264", "-r", "24", "-pix_fmt", "yuv420p", str(output)]
        subprocess.run(command, check=True)
        first_state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": 1, "source": "viser_microwave_knowledge",
            "claim": "逐角 Knowledge 运动学预览；纯 IK，不检查碰撞、路径连续性或真实执行",
            "video": str(output), "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_run": str(source_run.resolve()), "scene_revision": first_state["scene_revision"],
            "push_mode": first_state["tuner_values"]["door_push_mode"],
            "scene_snapshot_sha256": first_state["scene_snapshot_sha256"],
            "tuner_values": first_state["tuner_values"], "microwave_pose_world": first_state["microwave_pose_world"],
            "camera": first_state["camera"], "seconds_per_waypoint": seconds_per_waypoint,
            "waypoints": records,
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        if browser is not None:
            _stop(browser)
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--restore-state", type=Path,
                        help="复现已保存的 door_push 滑块、摆位、选中候选与相机")
    parser.add_argument("--push-mode", choices=("right_open", "right_close"),
                        help="指定 Task1 推门模式；right_close 使用关门参数与 85°→0° 起止角")
    parser.add_argument("--seconds-per-waypoint", type=float, default=.8)
    parser.add_argument("--closeup", action="store_true", help="固定门板近景；默认全景含机械臂")
    args = parser.parse_args()
    result = record(args.output, run=args.run, restore_state=args.restore_state,
                    push_mode=args.push_mode,
                    seconds_per_waypoint=args.seconds_per_waypoint, closeup=args.closeup)
    print(f"推门 Knowledge 视频：{result['video']}")
    print(f"逐帧反馈：{Path(result['video']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
