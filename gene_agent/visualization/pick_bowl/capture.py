"""保存 pick bowl Viser 截图及与规划结果对应的清单。"""

from __future__ import annotations

import hashlib
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.request import urlopen

from PIL import Image
from websockets.sync.client import connect

from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case
from gene_agent.visualization.motion_debug.capture import (
    DEFAULT_CHROME, DEFAULT_WINDOW_SIZE, _free_port, _stop, _validate_png,
)

from .app import camera_presets


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_capture_manifest(
    run_dir: Path, output_dir: Path, image_paths: dict[str, Path],
    cameras: dict[str, dict[str, Any]], candidate_id: str | None = None,
    trajectory_source: str = "抓取目标",
) -> dict[str, Any]:
    """清单把每张图绑定到本轮碰撞快照、真实规划报告和固定相机。"""
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    report_path = run / "pick_bowl_result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    views: dict[str, Any] = {}
    for name, path in image_paths.items():
        if name not in cameras:
            raise ValueError(f"未知截图相机：{name}")
        image_path = Path(path).resolve()
        with Image.open(image_path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"截图不是 PNG：{image_path}")
            size = list(image.size)
        views[name] = {
            "path": image_path.relative_to(output).as_posix(),
            "png_sha256": _sha256(image_path),
            "size_px": size,
            "camera": cameras[name],
        }
    manifest = {
        "schema_version": 1,
        "task": "pick_bowl_planning_inspection",
        "run_dir": str(run),
        "scene_snapshot_sha256": report["snapshot_sha256"],
        "report_sha256": _sha256(report_path),
        "selected_candidate_id": candidate_id or report.get("selected_candidate_id"),
        "trajectory_source": trajectory_source,
        "views": views,
    }
    (output / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _wait_marker(process: subprocess.Popen, log_path: Path, marker: str,
                 timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if marker in log:
            return
        if process.poll() is not None:
            raise RuntimeError(f"Viser/Chrome 提前退出：{log_path}\n{log[-2000:]}")
        time.sleep(.1)
    raise TimeoutError(f"等待 {marker} 超时；查看 {log_path}")


def _devtools_page(port: int, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/json", timeout=1.) as response:
                pages = json.load(response)
            for page in pages:
                if page.get("type") == "page" and page.get("webSocketDebuggerUrl"):
                    return str(page["webSocketDebuggerUrl"])
        except OSError:
            pass
        time.sleep(.1)
    raise TimeoutError("Chrome DevTools 页面未启动")


def _save_cdp_screenshot(websocket_url: str, output: Path) -> None:
    """在场景 render 确认之后才截屏，避免 Chrome 静态截图提前关闭 client。"""
    # PNG 经 CDP 的 JSON/base64 包装后可能超过 websockets 默认 1 MiB
    # 消息上限，尤其是包含细节丰富的机器人与夹爪侧视图时。截图是本机
    # Chrome 返回的单条消息，允许完整接收，避免误报为 Viser 渲染失败。
    with connect(websocket_url, open_timeout=10., close_timeout=5.,
                 max_size=None) as socket:
        def call(identifier: int, method: str, params: dict) -> dict:
            socket.send(json.dumps({"id": identifier, "method": method, "params": params}))
            deadline = time.monotonic() + 15.
            while time.monotonic() < deadline:
                message = json.loads(socket.recv(timeout=max(.1, deadline - time.monotonic())))
                if message.get("id") != identifier:
                    continue
                if "error" in message:
                    raise RuntimeError(f"Chrome DevTools {method} 失败：{message['error']}")
                return message["result"]
            raise TimeoutError(f"Chrome DevTools 没有返回 {method}")

        # headless Chrome 的窗口外框会吃掉约 139px；显式覆盖页面 viewport。
        call(1, "Emulation.setDeviceMetricsOverride", {
            "width": DEFAULT_WINDOW_SIZE[0], "height": DEFAULT_WINDOW_SIZE[1],
            "deviceScaleFactor": 1, "mobile": False,
        })
        result = call(2, "Page.captureScreenshot", {
            "format": "png", "captureBeyondViewport": False, "fromSurface": True,
        })
        output.write_bytes(base64.b64decode(result["data"]))


def _capture_one(run: Path, output: Path, view: str, chrome_path: Path,
                 candidate_id: str | None = None,
                 trajectory_source: str = "抓取目标") -> Path:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    log_path = output / f"{view}.log"
    image_path = output / f"{view}.png"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    camera_view = "bowl" if view == "collision" else view
    command = [
        sys.executable, "-u", str(ROOT / "tools/viewers/view_pick_bowl_planning.py"),
        "--run", str(run), "--host", "127.0.0.1", "--port", str(port),
        "--view", camera_view,
    ]
    if view == "collision":
        command.append("--show-collision")
    if candidate_id is not None:
        command.extend(("--candidate-id", candidate_id))
    command.extend(("--trajectory-source", trajectory_source))
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_marker(server, log_path, "[pick-bowl] loaded:", 45.)
            with urlopen(url, timeout=5.) as response:
                if response.status != 200:
                    raise RuntimeError(f"Viser HTTP {response.status}")
            with tempfile.TemporaryDirectory(prefix="gene-agent-pick-bowl-chrome-") as profile:
                debug_port = _free_port()
                args = [
                    str(chrome_path), "--headless=new", "--no-sandbox",
                    "--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=gl-egl",
                    "--disable-dev-shm-usage", "--remote-allow-origins=*",
                    "--force-device-scale-factor=1",
                    f"--window-size={DEFAULT_WINDOW_SIZE[0]},{DEFAULT_WINDOW_SIZE[1]}",
                    f"--user-data-dir={profile}",
                    f"--remote-debugging-port={debug_port}", url,
                ]
                browser = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    devtools_url = _devtools_page(debug_port, 15.)
                    _wait_marker(browser, log_path, f"[pick-bowl] scene-ready: view={camera_view};", 30.)
                    _save_cdp_screenshot(devtools_url, image_path)
                finally:
                    _stop(browser)
            _validate_png(image_path, DEFAULT_WINDOW_SIZE)
        finally:
            _stop(server)
    return image_path


def capture_pick_bowl_planning(
    run_dir: Path, output_dir: Path, *, chrome_path: Path = DEFAULT_CHROME,
    views: tuple[str, ...] = ("overview", "bowl", "gripper", "collision"),
    candidate_id: str | None = None,
    trajectory_source: str = "抓取目标",
) -> dict[str, Any]:
    """复用已保存规划，按固定相机截全景和碗近景；不重跑 GPU。"""
    run = Path(run_dir).resolve()
    case = load_pick_bowl_case(run)
    if candidate_id is not None and candidate_id not in {item["candidate_id"] for item in case.candidates}:
        raise ValueError(f"未知抓取候选：{candidate_id}")
    selected = candidate_id or (case.report or {}).get("selected_candidate_id")
    cameras = camera_presets(case, candidate_id=selected)
    cameras["collision"] = {**cameras["bowl"], "collision_overlay": True}
    if set(views) - set(cameras):
        raise ValueError(f"未知视角：{set(views) - set(cameras)}")
    if trajectory_source not in ("抓取目标", "Skill pregrasp", "Skill approach", "TCP 直线探针"):
        raise ValueError(f"未知轨迹阶段：{trajectory_source}")
    chrome = Path(chrome_path)
    if not chrome.is_file():
        raise FileNotFoundError(f"找不到 Chrome：{chrome}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = {name: _capture_one(run, output, name, chrome, candidate_id,
                                  trajectory_source) for name in views}
    return write_capture_manifest(run, output, images, cameras, candidate_id,
                                  trajectory_source)


__all__ = ["write_capture_manifest", "capture_pick_bowl_planning"]
