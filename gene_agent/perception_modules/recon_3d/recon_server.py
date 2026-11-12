"""SAM3D RGB-D 常驻重建服务。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PERCEPTION_DIR = os.path.dirname(_MODULE_DIR)
if _PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, _PERCEPTION_DIR)

from gene_agent.perception_modules.recon_3d.recon_core import Sam3DReconstructor  # noqa: E402


_REPO_ROOT = PROJECT_ROOT


def _resolve_repo_path(path: str):
    """将相对路径解释为项目根目录下的路径。"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


class ReconService:
    """保存默认配置并串行处理 GPU 重建请求。"""

    REQUIRED_FIELDS = (
        "rgb_path",
        "depth_path",
        "mask_path",
        "intrinsics_path",
        "output_path",
    )

    def __init__(self, config_path=None, device="cuda", reconstructor=None):
        self.config_path = config_path
        self.device = device
        self.reconstructor = reconstructor or Sam3DReconstructor(
            config_path, device, load_on_init=False,
        )
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        """返回当前 SAM3D 模型是否驻留在内存。"""
        return bool(getattr(self.reconstructor, "is_loaded", True))

    def reconstruct(self, payload):
        """执行一次 RGB-D 重建请求。"""
        if not isinstance(payload, dict):
            raise ValueError("请求 JSON 必须是 object")
        missing = [field for field in self.REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"请求缺少字段: {', '.join(missing)}")

        with self._lock:
            start_time = time.time()
            from gene_agent.perception_modules.recon_3d.recon_core import (
                _load_depth,
                _load_image,
                _load_intrinsics,
            )

            rgb_path = _resolve_repo_path(payload["rgb_path"])
            depth_path = _resolve_repo_path(payload["depth_path"])
            mask_path = _resolve_repo_path(payload["mask_path"])
            intrinsics_path = _resolve_repo_path(payload["intrinsics_path"])
            result = self.reconstructor.reconstruct(
                rgb=_load_image(Path(rgb_path), mode="RGB"),
                depth=_load_depth(Path(depth_path)),
                mask=_load_image(Path(mask_path), mode="L"),
                intrinsics=_load_intrinsics(Path(intrinsics_path)),
                output_path=_resolve_repo_path(payload["output_path"]),
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
                intrinsics_path=intrinsics_path,
                depth_scale=float(payload.get("depth_scale", 0.001)),
                seed=int(payload.get("seed", 42)),
                simplify_mesh=bool(payload.get("simplify_mesh", False)),
                visualize=bool(payload.get("visualize", False)),
                use_foundationpose=bool(payload.get("use_foundationpose", False)),
                pose_server=str(payload.get("pose_server", "http://127.0.0.1:18081")),
            )
            result.update({"ok": True, "elapsed": time.time() - start_time})
            return result

    def load_model(self):
        """在锁内加载 SAM3D 模型，已经加载时保持幂等。"""
        with self._lock:
            self.reconstructor.ensure_loaded()
            return {"ok": True, "model_loaded": self.model_loaded}

    def unload_model(self):
        """在锁内卸载 SAM3D 模型，已经卸载时保持幂等。"""
        with self._lock:
            self.reconstructor.unload_model()
            return {"ok": True, "model_loaded": self.model_loaded}


class ReconRequestHandler(BaseHTTPRequestHandler):
    """将 HTTP 请求转换为 ReconService 调用。"""

    service: ReconService | None = None

    def _send_json(self, status, body):
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "recon_3d_server",
                    "model_loaded": self.service.model_loaded,
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        if self.path == "/model/load":
            try:
                self._send_json(200, self.service.load_model())
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if self.path == "/model/unload":
            try:
                self._send_json(200, self.service.unload_model())
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if self.path != "/reconstruct":
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.service.reconstruct(payload)
            self._send_json(200, result)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, fmt, *args):
        print(f"[recon_3d_server] {self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="SAM3D RGB-D 常驻重建服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ReconRequestHandler.service = ReconService(args.config, args.device)
    server = ThreadingHTTPServer((args.host, args.port), ReconRequestHandler)
    print(f"recon_3d_server 已启动: http://{args.host}:{args.port}")
    print("GET /health, POST /reconstruct, POST /model/load, POST /model/unload（首次请求加载模型）")
    server.serve_forever()


if __name__ == "__main__":
    main()
