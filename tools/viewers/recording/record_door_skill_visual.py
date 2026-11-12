"""从在线 Skill 检查器录制完整 Knowledge 目标动画，不运行 MuJoCo 动力学。"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from websockets.sync.client import connect
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gene_agent.visualization.motion_debug.capture import _free_port, DEFAULT_CHROME
from gene_agent.visualization.pick_bowl.capture import _devtools_page
from gene_agent.visualization.skill_inspector.live_door_app import LiveDoorApp
from gene_agent.tasks.policies.microwave_task.planning.door_session import configure_curobo_run_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--skill", choices=("door_cycle", "handle_pull", "push_open", "push_close"),
                        default="door_cycle")
    parser.add_argument("--camera", choices=("overview", "contact", "gripper_side"),
                        default="overview")
    parser.add_argument("--source", choices=("targets", "planned"), default="targets",
                        help="targets 录 Knowledge 关键点；planned 先运行 cuRobo 再录连续轨迹")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = output.with_suffix(".frames")
    frames.mkdir(parents=True, exist_ok=True)
    viser_port, chrome_port = _free_port(), _free_port()
    while chrome_port == viser_port:
        chrome_port = _free_port()
    server = viser.ViserServer(host="127.0.0.1", port=viser_port)
    app = LiveDoorApp(
        server,
        config_path=args.config,
        mode="task" if args.skill == "door_cycle" else "skill",
        skill_id="push_close" if args.skill == "door_cycle" else args.skill,
    )
    configure_curobo_run_log(app.session.output_dir)
    app.camera.value = {"overview": "全景", "contact": "门接触",
                        "gripper_side": "夹爪侧面"}[args.camera]
    app._set_camera()
    command = [str(DEFAULT_CHROME), "--headless=new", "--no-sandbox",
               "--disable-gpu-sandbox", "--enable-unsafe-swiftshader",
               "--use-gl=angle", "--use-angle=swiftshader", "--disable-dev-shm-usage",
               "--remote-allow-origins=*", "--window-size=1500,900",
               f"--remote-debugging-port={chrome_port}",
               f"--user-data-dir=/tmp/door-record-chrome-{chrome_port}",
               f"http://127.0.0.1:{viser_port}"]
    browser = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = _devtools_page(chrome_port, 30.)
        with connect(page, open_timeout=10., close_timeout=5.) as websocket:
            identifier = 0
            def call(method, params):
                nonlocal identifier
                identifier += 1
                websocket.send(json.dumps({"id": identifier, "method": method, "params": params}))
                while True:
                    response = json.loads(websocket.recv(timeout=30.))
                    if response.get("id") == identifier:
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        return response["result"]
            deadline = time.monotonic() + 40.
            while time.monotonic() < deadline:
                body = call("Runtime.evaluate", {"expression": "document.body.innerText",
                                                 "returnByValue": True})["result"].get("value", "")
                if "在线纯视觉调试" in body:
                    break
                time.sleep(.2)
            else:
                raise TimeoutError("浏览器未加载在线门 Skill 控件")
            call("Runtime.evaluate", {"expression": """
                Array.from(document.querySelectorAll('button')).find(
                  b => (b.parentElement?.innerText || '').includes('Software WebGL rendering detected')
                )?.click()
            """})
            time.sleep(2.)
            if args.source == "planned":
                selected = None if args.skill == "door_cycle" else args.skill
                report = app.session.plan_sequence(selected)
                if not report["success"]:
                    raise RuntimeError(f"连续规划失败：{report.get('first_failure')}")
                app.timeline.max = max(app.session.timeline.duration_s, .01)
                app.timeline.value = 0.0
                app.result_status.content = (
                    f"**规划成功：** {report['completed_segments']} 个阶段，"
                    f"墙钟总耗时 {report['elapsed_wall_s']:.3f} s；"
                    "正在播放连续轨迹。"
                )
                sample_times = np.linspace(
                    0., app.session.timeline.duration_s,
                    max(2, int(np.ceil(app.session.timeline.duration_s * args.fps)) + 1),
                )
                stage_ids = []
                for frame_index, sample_time in enumerate(sample_times):
                    app.timeline_time_s = float(sample_time)
                    sample = app.session.seek_timeline(sample_time)
                    # 录制器直接驱动与在线界面同一个时间轴；同步滑块和
                    # 阶段索引，避免视频右栏显示“尚未规划”或停在第 0 帧。
                    app.syncing = True
                    try:
                        app.timeline.value = float(sample_time)
                        app.stage.value = sample.segment_index
                    finally:
                        app.syncing = False
                    app._show_timeline_frame()
                    stage_ids.append(sample.stage_id)
                    time.sleep(.03)
                    png = base64.b64decode(call("Page.captureScreenshot", {"format": "png"})["data"])
                    (frames / f"frame_{frame_index:03d}.png").write_bytes(png)
                    print(f"[door-record] {frame_index+1}/{len(sample_times)} {sample.stage_id}", flush=True)
                kind = "planned_kinematic_timeline"
            else:
                chosen = list(app.scope_indices)
                if not chosen:
                    raise ValueError("选中 Skill 没有可录制阶段")
                stage_ids = [app.session.phases[i].phase_id for i in chosen]
                for frame_index, stage_index in enumerate(chosen):
                    app.syncing = True
                    app.stage.value = chosen.index(stage_index)
                    app.syncing = False
                    app._show_target()
                    time.sleep(.20)
                    png = base64.b64decode(call("Page.captureScreenshot", {"format": "png"})["data"])
                    (frames / f"frame_{frame_index:03d}.png").write_bytes(png)
                    print(f"[door-record] {frame_index+1}/{len(chosen)} {app.session.phases[stage_index].phase_id}", flush=True)
                kind = "knowledge_targets_only"
        manifest = {"kind": kind, "source": "live_door_app",
                    "stage_ids": stage_ids,
                    "config": app.session.config, "camera": args.camera,
                    "physical_execution": False}
        output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", str(frames / "frame_%03d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(output)], check=True)
        print(output, flush=True)
    finally:
        browser.terminate()
        try:
            browser.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            browser.kill()
        server.stop()


if __name__ == "__main__":
    main()
