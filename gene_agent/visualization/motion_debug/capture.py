"""以固定浏览器和相机捕获 Motion DebugBundle。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image

from gene_agent.feedback import MotionDebugBundle, save_motion_debug_bundle

from .cameras import camera_presets


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROME = Path("/usr/bin/google-chrome")
DEFAULT_WINDOW_SIZE = (1440, 1000)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_server(process: subprocess.Popen, log_path: Path, url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "尚未收到 HTTP 响应"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Viser server 提前退出，exit={process.returncode}；日志：{log_path}"
            )
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if "[motion-debug] loaded:" in text:
            try:
                with urlopen(url, timeout=0.75) as response:
                    if response.status == 200 and b"html" in response.read(4096).lower():
                        return
            except (OSError, URLError) as error:
                last_error = str(error)
        time.sleep(0.1)
    raise TimeoutError(f"等待 Motion Viser 超时：{last_error}；日志：{log_path}")


def _wait_for_scene_ready(
    process: subprocess.Popen, log_path: Path, view: str, timeout_s: float = 8.0
) -> None:
    marker = f"[motion-debug] scene-ready: view={view};"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if marker in text:
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"Chrome 在浏览器 render acknowledgement 前退出，exit={process.returncode}；"
                f"日志：{log_path}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"Chrome 未收到 Motion Viser 场景；日志：{log_path}")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _validate_png(path: Path, window_size: tuple[int, int]) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != window_size:
                raise RuntimeError(
                    f"截图应为 {window_size[0]}x{window_size[1]} PNG，实际为 "
                    f"{image.format} {image.size}：{path}"
                )
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Chrome 没有写出有效 PNG：{path}") from error


def chrome_arguments(
    chrome_path: Path,
    profile_dir: Path,
    output_path: Path,
    url: str,
    window_size: tuple[int, int],
) -> list[str]:
    """固定本地 GPU WebGL 参数，避免强制 software renderer 遮挡调试画面。"""
    return [
        str(chrome_path),
        "--headless=new",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
        "--enable-gpu",
        "--use-angle=gl-egl",
        "--disable-dev-shm-usage",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={window_size[0]},{window_size[1]}",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
        "--timeout=20000",
        f"--user-data-dir={profile_dir}",
        f"--screenshot={output_path}",
        url,
    ]


def _capture_view(
    bundle_path: Path,
    view: str,
    output_path: Path,
    log_path: Path,
    *,
    chrome_path: Path,
    window_size: tuple[int, int],
    startup_timeout_s: float,
    chrome_timeout_s: float,
) -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "tools" / "viewers" / "view_motion_debug.py"),
        "--bundle",
        str(bundle_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--view",
        view,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, log_path, url, startup_timeout_s)
            with tempfile.TemporaryDirectory(prefix="gene-agent-motion-chrome-") as profile:
                chrome = subprocess.Popen(
                    chrome_arguments(
                        chrome_path, Path(profile), output_path, url, window_size
                    ),
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    # ready 只有在 Viser client 返回实际 WebGL render 后才会写入日志。
                    _wait_for_scene_ready(chrome, log_path, view, startup_timeout_s)
                    returncode = chrome.wait(timeout=chrome_timeout_s)
                except BaseException:
                    _stop(chrome)
                    raise
            if returncode != 0:
                raise RuntimeError(f"headless Chrome exit={returncode}；日志：{log_path}")
            _validate_png(output_path, window_size)
        finally:
            _stop(server)


def write_capture_manifest(
    output_dir: str | Path,
    bundle: MotionDebugBundle,
    bundle_path: str | Path,
    image_paths: dict[str, Path],
    *,
    window_size: tuple[int, int],
) -> dict[str, object]:
    output = Path(output_dir)
    bundle_path = Path(bundle_path)
    presets = camera_presets(bundle)
    views: dict[str, object] = {}
    for name, path in image_paths.items():
        resolved = Path(path)
        if name not in presets:
            raise ValueError(f"manifest 包含未知相机：{name}")
        _validate_png(resolved, window_size)
        views[name] = {
            "path": resolved.relative_to(output).as_posix(),
            "png_sha256": _sha256(resolved),
            "camera": presets[name].to_dict(),
        }
    manifest = {
        "schema_version": 1,
        "run_id": bundle.run_id,
        "bundle_path": bundle_path.relative_to(output).as_posix(),
        "bundle_sha256": _sha256(bundle_path),
        "window_size": list(window_size),
        "views": views,
    }
    _write_json(output / "capture_manifest.json", manifest)
    return manifest


def capture_motion_debug(
    bundle: MotionDebugBundle,
    output_dir: str | Path,
    *,
    views: tuple[str, ...] = ("overview", "target", "collision"),
    chrome_path: str | Path = DEFAULT_CHROME,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
    startup_timeout_s: float = 30.0,
    chrome_timeout_s: float = 40.0,
) -> dict[str, object]:
    """保存 bundle 后按固定视角截图；失败时 bundle 和日志仍保留。"""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundle_path = save_motion_debug_bundle(output / "bundle.json", bundle)
    chrome = Path(chrome_path)
    if not chrome.is_file():
        raise FileNotFoundError(f"找不到用于 Motion Viser 截图的 Chrome：{chrome}")
    unknown = set(views) - set(camera_presets(bundle))
    if unknown:
        raise ValueError(f"未知截图视角：{sorted(unknown)}")
    images: dict[str, Path] = {}
    for view in views:
        image_path = output / f"{view}.png"
        _capture_view(
            bundle_path,
            view,
            image_path,
            output / f"{view}.log",
            chrome_path=chrome,
            window_size=window_size,
            startup_timeout_s=startup_timeout_s,
            chrome_timeout_s=chrome_timeout_s,
        )
        images[view] = image_path
    return write_capture_manifest(
        output, bundle, bundle_path, images, window_size=window_size
    )


__all__ = [
    "DEFAULT_CHROME",
    "DEFAULT_WINDOW_SIZE",
    "capture_motion_debug",
    "chrome_arguments",
    "write_capture_manifest",
]
