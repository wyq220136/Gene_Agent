"""启动 Viser grasp tuner，以固定 headless Chrome 视口保存一张 PNG。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
CHROME = Path("/usr/bin/google-chrome")
PROFILES = ("task_1_bowl_0", "task_1_knob_0", "trifold_handle")
WINDOW_SIZE = (1440, 1000)
STARTUP_TIMEOUT_S = 30.0
CHROME_TIMEOUT_S = 40.0
SCENE_SETTLE_VIRTUAL_MS = 6000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="task_1_bowl_0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 PNG 路径；默认写入 runs/viser_capture/after/{bowl,knob}.png",
    )
    parser.add_argument("--port", type=int, default=8080, help="临时 Viser server 使用的本机端口")
    return parser.parse_args()


def _output_path(profile: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    object_name = {"task_1_bowl_0": "bowl", "task_1_knob_0": "knob", "trifold_handle": "handle"}[profile]
    return ROOT / "runs" / "viser_capture" / "after" / f"{object_name}.png"


def _wait_for_server(
    process: subprocess.Popen[bytes], log_path: Path, url: str, profile: str, port: int
) -> None:
    """同时等待 scene 构建日志与 HTTP 服务，避免只检查到空闲端口。"""
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    scene_marker = f"[grasp-tuner] {profile}:"
    last_error = "尚未收到 HTTP 响应"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Viser server 提前退出，exit={process.returncode}；日志：{log_path}"
            )

        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            log_text = ""

        # tune_grasp_profile 在 GraspTunerApp 完成 scene 构建后才打印候选数量。
        scene_constructed = scene_marker in log_text and " candidates" in log_text
        if scene_constructed:
            endpoints = re.findall(r"http://127\.0\.0\.1:(\d+)", log_text)
            if endpoints and int(endpoints[0]) != port:
                raise RuntimeError(
                    f"Viser 实际监听端口 {endpoints[0]} 与请求端口 {port} 不一致；"
                    f"日志：{log_path}。请为本次截图选择空闲端口。"
                )
            try:
                with urlopen(url, timeout=0.75) as response:
                    if response.status == 200:
                        page = response.read(4096).lower()
                        if b"html" in page:
                            return
                        last_error = "HTTP 已响应，但首页内容不是 HTML"
            except (OSError, URLError) as error:
                last_error = str(error)

        time.sleep(0.1)

    raise TimeoutError(
        f"等待 Viser scene/HTTP 超时（{STARTUP_TIMEOUT_S:.0f}s）：{last_error}；日志：{log_path}"
    )


def _assert_port_available(port: int) -> None:
    """避免 Viser 遇到占用端口后自动换端口，进而误连到旧 server。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        try:
            port_probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise SystemExit(f"端口 {port} 已被占用；请指定空闲端口：{error}") from error


def _wait_for_scene_client(log_path: Path) -> None:
    """确认 Chrome 已连上 Viser，且 server 已发送带固定 camera 的场景。"""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            if "[grasp-tuner] scene-ready: client camera fixed" in log_path.read_text(
                encoding="utf-8", errors="replace"
            ):
                return
        except FileNotFoundError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"Chrome 未连接 Viser scene，未观察到相机就绪标记；日志：{log_path}")


def _validate_png(path: Path) -> None:
    """检查 Chrome 产物是完整 PNG，且窗口像素尺寸没有被系统缩放。"""
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Chrome 没有写出有效 PNG：{path}")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != WINDOW_SIZE:
        raise RuntimeError(
            f"PNG 尺寸应为 {WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}，实际为 {width}x{height}：{path}"
        )


def _terminate_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def main() -> None:
    args = _arguments()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port 必须在 1–65535 之间")
    if not CHROME.is_file():
        raise SystemExit(f"找不到固定截图程序：{CHROME}")
    _assert_port_available(args.port)

    output_path = _output_path(args.profile, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = output_path.with_name(f"{output_path.stem}.{stamp}.log")
    url = f"http://127.0.0.1:{args.port}/"

    server_command = [
        sys.executable,
        "-u",
        str(ROOT / "tools" / "editors" / "tune_grasp_profile.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--profile",
        args.profile,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)

    with log_path.open("w", encoding="utf-8") as log_file:
        server = subprocess.Popen(
            server_command,
            cwd=ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, log_path, url, args.profile, args.port)
            with tempfile.TemporaryDirectory(prefix="gene-agent-viser-chrome-") as profile_dir:
                chrome_command = [
                    str(CHROME),
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=1",
                    f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}",
                    "--run-all-compositor-stages-before-draw",
                    f"--virtual-time-budget={SCENE_SETTLE_VIRTUAL_MS}",
                    "--timeout=20000",
                    f"--user-data-dir={profile_dir}",
                    f"--screenshot={output_path}",
                    url,
                ]
                result = subprocess.run(
                    chrome_command,
                    cwd=ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=CHROME_TIMEOUT_S,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"headless Chrome exit={result.returncode}；详细输出：{log_path}"
                )
            _wait_for_scene_client(log_path)
            _validate_png(output_path)
        finally:
            _terminate_server(server)

    print(f"截图：{output_path}")
    print(f"日志：{log_path}")
    print(f"窗口：{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}；profile：{args.profile}")


if __name__ == "__main__":
    main()
