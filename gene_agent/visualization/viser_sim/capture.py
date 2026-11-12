"""在线 Viser 页面通用截图；等待真实客户端完成一帧渲染再保存。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable

from gene_agent.visualization.motion_debug.capture import (
    DEFAULT_CHROME, DEFAULT_WINDOW_SIZE, _free_port, _stop, _validate_png,
)
from gene_agent.visualization.pick_bowl.capture import (
    _devtools_page, _save_cdp_screenshot,
)


def capture_viser_page(
    server, output: str | Path, *, timeout_s: float = 30.,
    ready_check: Callable[[Path], bool] | None = None,
) -> Path:
    """截取和用户看到的同一个 Viser 页面，不生成第二套演示画面。"""
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    browser = None
    with tempfile.TemporaryDirectory(prefix="gene-viser-capture-") as profile, \
            (path.parent / f".{path.stem}.chrome.log").open("w", encoding="utf-8") as log:
        chrome_port = _free_port()
        command = [
            str(DEFAULT_CHROME), "--headless=new", "--no-sandbox",
            "--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=gl-egl",
            "--disable-dev-shm-usage", "--remote-allow-origins=*",
            "--force-device-scale-factor=1",
            f"--window-size={DEFAULT_WINDOW_SIZE[0]},{DEFAULT_WINDOW_SIZE[1]}",
            f"--user-data-dir={profile}", f"--remote-debugging-port={chrome_port}",
            f"http://127.0.0.1:{server.get_port()}/",
        ]
        browser = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy())
        try:
            socket_url = _devtools_page(chrome_port, timeout_s)
            deadline = time.monotonic() + timeout_s
            last_error = "Viser 客户端尚未连接"
            while time.monotonic() < deadline:
                clients = list(server.get_clients().values())
                if not clients:
                    time.sleep(.2)
                    continue
                try:
                    for client in clients:
                        client.flush()
                        client.get_render(height=8, width=8, transport_format="png", timeout=5.)
                    break
                except Exception as error:
                    last_error = f"Viser 客户端尚未完成渲染：{error}"
                    time.sleep(.3)
            else:
                raise TimeoutError(last_error)
            # get_render 只证明一帧可用，不保证较大的 gripper mesh 已
            # 上传到浏览器。截图完成后可按场景特征核验，未就绪就重试。
            time.sleep(1.)
            while time.monotonic() < deadline:
                _save_cdp_screenshot(socket_url, path)
                _validate_png(path, DEFAULT_WINDOW_SIZE)
                if ready_check is None or ready_check(path):
                    return path
                time.sleep(.5)
            raise TimeoutError(f"Viser 截图未通过场景就绪检查：{path}")
        finally:
            _stop(browser)


__all__ = ["capture_viser_page"]
