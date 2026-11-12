"""从在线 Viser 检查器当前 VisualRun 录制纯视觉视频。"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from websockets.sync.client import connect

from gene_agent.visualization.motion_debug.capture import DEFAULT_CHROME, _free_port
from gene_agent.visualization.pick_bowl.capture import _devtools_page


CAMERA_LABELS = {"overview": "前上方", "gripper_side": "夹爪侧面"}


def _source_revision() -> dict:
    """把代码版本写入验收清单；工作树未提交时明确标记，避免误认视频可精确复现。"""
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
        check=False,
    )
    # 未提交工作树只记录 HEAD 无法判断代码是否变化。记录纯视觉规划、
    # 场景和录制实际依赖的文件；真机 SDK/现场代码不参与当前纯视觉入口，
    # 修改它们不应迫使 --resume 重新规划并录制全部 Viser 视频。
    digest = hashlib.sha256()
    sources = sorted(path for path in (root / "gene_agent").rglob("*.py")
                     if "realworld" not in path.relative_to(root / "gene_agent").parts
                     and not (path.parent.name in {"simulation_bowl", "simulation_door"}
                              and path.parent.parent.name == "microwave_task")
                     and not (path.parent.name == "microwave_task"
                              and path.name in {"simulation.py", "simulation_knob.py"}))
    sources.append(root / "tools/viewers/recording/record_visual.py")
    sources.extend(sorted(path for path in (root / "data/task_1").rglob("*.json")
                          if "realworld" not in path.name))
    sources.extend(sorted(path for path in (root / "assets").rglob("*") if path.is_file()))
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(source.read_bytes())
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "runtime_sha256": digest.hexdigest(),
    }


def record_visual_run(app, output: Path, *, viser_port: int, camera: str = "overview",
                      fps: int = 12, width: int = 1500, height: int = 900,
                      software_renderer: bool = False) -> dict:
    """只逐帧 seek 当前 Viewer 的同一条规划结果，不另建演示动画。"""
    if app.playback is None or app.controller.run is None or not app.controller.run.success:
        raise ValueError("只可录制当前在线界面已成功规划的完整运行")
    if camera not in CAMERA_LABELS or fps < 1 or min(width, height) < 480:
        raise ValueError("相机、帧率或画面尺寸无效")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_name(output.stem + ".part.mp4")
    started = time.perf_counter()
    # 浏览器仍须保留独立用户目录，但截图直接送给 ffmpeg，避免逐帧写入
    # 1500x900 PNG 再读取。画面时间、相机和阶段仍来自原 VisualRun。
    frame_workspace = tempfile.TemporaryDirectory(prefix="gene-visual-frames-")
    frames_dir = Path(frame_workspace.name)
    chrome_port = _free_port()
    while chrome_port == viser_port:
        chrome_port = _free_port()
    renderer_args = (["--enable-unsafe-swiftshader", "--use-gl=angle",
                      "--use-angle=swiftshader"] if software_renderer else
                     ["--enable-gpu", "--use-gl=angle", "--use-angle=gl"])
    browser = subprocess.Popen([
        str(DEFAULT_CHROME), "--headless=new", "--no-sandbox",
        "--disable-gpu-sandbox", *renderer_args, "--disable-dev-shm-usage",
        "--remote-allow-origins=*", f"--window-size={width},{height}",
        f"--remote-debugging-port={chrome_port}",
        f"--user-data-dir={frames_dir / 'chrome'}",
        f"http://127.0.0.1:{viser_port}",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    encoder = None
    try:
        page = _devtools_page(chrome_port, 30.)
        # 帧里有高分辨率机器人 mesh；CDP 的 base64 PNG 可能超过默认 1 MiB。
        with connect(page, open_timeout=10., close_timeout=5.,
                     max_size=None) as websocket:
            identifier = 0

            def call(method: str, params: dict) -> dict:
                nonlocal identifier
                identifier += 1
                websocket.send(json.dumps({"id": identifier, "method": method,
                                           "params": params}))
                while True:
                    response = json.loads(websocket.recv(timeout=30.))
                    if response.get("id") == identifier:
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        return response["result"]

            deadline = time.monotonic() + 30.
            while time.monotonic() < deadline:
                body = call("Runtime.evaluate", {
                    "expression": "document.body.innerText", "returnByValue": True,
                })["result"].get("value", "")
                if "在线纯视觉" in body and "检查" in body:
                    break
                time.sleep(.2)
            else:
                raise TimeoutError("浏览器没有加载在线 Viser 检查器")
            call("Runtime.evaluate", {"expression": """
                Array.from(document.querySelectorAll('button')).find(
                  b => (b.parentElement?.innerText || '').includes('Software WebGL rendering detected')
                )?.click()
            """})
            call("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": 1,
                "mobile": False,
            })
            app.playing = False
            app.camera.value = CAMERA_LABELS[camera]
            app._set_camera()
            time.sleep(1.)
            duration = app.playback.duration_s
            times = np.linspace(0., duration, max(2, int(np.ceil(duration * fps)) + 1))
            stage_ids = []
            encoder = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe",
                "-vcodec", "mjpeg", "-framerate", str(fps), "-i", "pipe:0",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                str(partial_output),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
               stderr=subprocess.PIPE)
            capture_started = time.perf_counter()
            timings = {"show_s": 0., "flush_s": 0., "wait_s": 0.,
                       "screenshot_s": 0., "encode_feed_s": 0.}
            for index, sample_time in enumerate(times):
                step_started = time.perf_counter()
                app.time_s = float(sample_time)
                app._show_frame()
                stage_ids.append(app.playback.sample(sample_time).stage_id)
                timings["show_s"] += time.perf_counter() - step_started
                # 等待服务端状态送达浏览器并完成至少一次绘制；否则视频会
                # 出现阶段信息已经更新、机器人仍停在上一帧的错位。
                flush_started = time.perf_counter()
                for client in app.server.get_clients().values():
                    client.flush()
                timings["flush_s"] += time.perf_counter() - flush_started
                flush_done = time.perf_counter()
                time.sleep(.08)
                timings["wait_s"] += time.perf_counter() - flush_done
                screenshot_started = time.perf_counter()
                jpeg = base64.b64decode(call("Page.captureScreenshot", {
                    "format": "jpeg", "quality": 85,
                })["data"])
                timings["screenshot_s"] += time.perf_counter() - screenshot_started
                assert encoder.stdin is not None
                feed_started = time.perf_counter()
                encoder.stdin.write(jpeg)
                timings["encode_feed_s"] += time.perf_counter() - feed_started
                if (index + 1 == len(times) or index % max(1, len(times) // 10) == 0
                        or index and stage_ids[-1] != stage_ids[-2]):
                    print(f"[record] {index + 1}/{len(times)} {stage_ids[-1]}", flush=True)
            capture_wall_s = time.perf_counter() - capture_started
        assert encoder.stdin is not None and encoder.stderr is not None
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"Viser 视频编码失败：{error}")
        partial_output.replace(output)
        manifest = {
            "source": "online_viser_visual_run",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": _source_revision(),
            "camera": camera,
            "fps": fps,
            "width": width,
            "height": height,
            "capture_format": "jpeg_stream_quality_85",
            "chrome_renderer": "software" if software_renderer else "opengl",
            "capture_wall_s": capture_wall_s,
            "capture_timings_s": timings,
            "recording_wall_s": time.perf_counter() - started,
            "visual_only": True,
            "video": str(output),
            "stage_ids_by_frame": stage_ids,
            "config": app.session.config,
            "planning_duration_s": sum(
                stage.planning_duration_s or 0. for stage in app.controller.run.stages
            ),
            "run": app.controller.run.manifest(),
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        partial_output.unlink(missing_ok=True)
        browser.terminate()
        try:
            browser.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            browser.kill()
        frame_workspace.cleanup()


__all__ = ["record_visual_run", "CAMERA_LABELS"]
