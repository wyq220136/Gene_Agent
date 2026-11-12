"""FoundationPose 首帧注册与同步 tracking HTTP 服务。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import argparse
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import numpy as np


_POSE_DIR = Path(__file__).resolve().parent
_PERCEPTION_DIR = _POSE_DIR.parent
_REPO_ROOT = PROJECT_ROOT
for path in (_POSE_DIR, _PERCEPTION_DIR, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .batch_est_pose import load_pose_inputs, load_rgb_image, load_depth_image, load_camera_parameters
from .fp_pose_est.pose_est import FoundationPoseBackend


def _resolve_repo_path(path: str | os.PathLike) -> str:
    """把相对路径解释为仓库根目录下的路径。"""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return str(candidate.resolve())


def _default_camera_client_factory(endpoint: str):
    """惰性创建 camera server 客户端。"""
    from gene_agent.perception_modules.camera.camera_client import CameraClient

    return CameraClient(endpoint)


@dataclass
class SessionRecord:
    """server 保存的单物体 tracking 会话。"""

    session: object
    camera_serial: str | None
    image_shape: tuple[int, int]
    intrinsic: np.ndarray
    camera_id: str


class PoseService:
    """管理共享 FoundationPose 后端和多个独立 tracking session。"""

    REQUIRED_REGISTER_FIELDS = (
        "session_id",
        "rgb_path",
        "depth_path",
        "mask_path",
        "mesh_path",
        "camera_path",
    )

    def __init__(
        self,
        *,
        backend=None,
        device: str = "cuda:0",
        camera_endpoint: str = "tcp://127.0.0.1:18082",
        camera_client_factory: Callable[[str], object] | None = None,
        input_loader: Callable[..., object] = load_pose_inputs,
    ):
        self.backend = backend or FoundationPoseBackend(device=device)
        self.camera_endpoint = camera_endpoint
        self.camera_client_factory = (
            camera_client_factory or _default_camera_client_factory
        )
        self.input_loader = input_loader
        self.sessions: dict[str, SessionRecord] = {}
        self._camera_clients: dict[str, object] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _session_id(payload: dict) -> str:
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        return session_id

    def register(self, payload: dict) -> dict:
        """读取首帧数据，执行 register 并保存 session。"""
        missing = [
            field for field in self.REQUIRED_REGISTER_FIELDS if field not in payload
        ]
        if missing:
            raise ValueError(f"注册请求缺少字段: {missing}")
        session_id = self._session_id(payload)
        paths = {
            name: _resolve_repo_path(payload[name])
            for name in self.REQUIRED_REGISTER_FIELDS
            if name.endswith("_path")
        }

        with self._lock:
            if session_id in self.sessions:
                raise ValueError(
                    f"tracking session 已存在: {session_id}，请先调用 stop"
                )
            inputs = self.input_loader(**paths)
            session = self.backend.register(
                mesh=inputs.mesh,
                rgb=inputs.rgb,
                depth=inputs.depth,
                mask=inputs.mask,
                intrinsic=inputs.intrinsic,
            )
            serial = inputs.camera_parameters.get("serial")
            camera_id = str(payload.get("camera_id", "0"))
            self.sessions[session_id] = SessionRecord(
                session=session,
                camera_serial=serial,
                image_shape=inputs.rgb.shape[:2],
                intrinsic=np.asarray(inputs.intrinsic, dtype=np.float64).copy(),
                camera_id=camera_id,
            )

        return {
            "ok": True,
            "session_id": session_id,
            "camera_id": camera_id,
            "T_camera_object": session.pose.tolist(),
        }

    def _camera(self, camera_id: str):
        """按 camera_id 复用对应的实时相机 client。"""
        if camera_id not in self._camera_clients:
            self._camera_clients[camera_id] = self.camera_client_factory(
                self.camera_endpoint
            )
        return self._camera_clients[camera_id]

    def track(self, payload: dict) -> dict:
        """从 camera server 读取最新帧，并同步执行一次 track_one。"""
        session_id = self._session_id(payload)
        with self._lock:
            if session_id not in self.sessions:
                raise KeyError(f"tracking session 不存在: {session_id}")
            record = self.sessions[session_id]
            packet = self._camera(record.camera_id).get_frame()
            if record.camera_serial and packet.serial != record.camera_serial:
                raise ValueError(
                    f"相机序列号不一致: register={record.camera_serial}, "
                    f"tracking={packet.serial}"
                )
            if (
                packet.rgb.shape[:2] != record.image_shape
                or packet.depth.shape != record.image_shape
            ):
                raise ValueError(
                    "tracking RGB-D 尺寸不一致: "
                    f"register={record.image_shape}, "
                    f"rgb={packet.rgb.shape[:2]}, depth={packet.depth.shape}"
                )
            tracking_intrinsic = np.asarray(
                packet.intrinsic_matrix,
                dtype=np.float64,
            )
            if (
                tracking_intrinsic.shape != (3, 3)
                or not np.allclose(
                    tracking_intrinsic,
                    record.intrinsic,
                    rtol=1e-4,
                    atol=1e-3,
                )
            ):
                raise ValueError("tracking 相机内参与首帧相机内参不一致")
            pose = record.session.track(
                rgb=packet.rgb,
                depth=packet.depth,
                intrinsic=tracking_intrinsic,
            )

        return {
            "ok": True,
            "session_id": session_id,
            "camera_id": record.camera_id,
            "T_camera_object": pose.tolist(),
            "camera_serial": packet.serial,
            "frame_number": int(packet.frame_number),
            "timestamp_ms": float(packet.timestamp_ms),
        }

    def track_frame(self, payload: dict) -> dict:
        """从显式 RGB-D 文件跟踪，供离线仿真相机使用。"""
        session_id = self._session_id(payload)
        paths = {key: _resolve_repo_path(payload[key]) for key in
                 ("rgb_path", "depth_path", "camera_path")}
        camera = load_camera_parameters(paths["camera_path"])
        rgb = load_rgb_image(paths["rgb_path"])
        depth = load_depth_image(paths["depth_path"],
                                 depth_unit=str(camera.get("depth_unit", "millimeter")))
        intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
        with self._lock:
            if session_id not in self.sessions:
                raise KeyError(f"tracking session 不存在: {session_id}")
            record = self.sessions[session_id]
            if rgb.shape[:2] != record.image_shape or depth.shape != record.image_shape:
                raise ValueError("tracking RGB-D 尺寸与 register 不一致")
            if not np.allclose(intrinsic, record.intrinsic, rtol=1e-4, atol=1e-3):
                raise ValueError("tracking 相机内参与 register 不一致")
            pose = record.session.track(rgb=rgb, depth=depth, intrinsic=intrinsic)
        return {"ok": True, "session_id": session_id,
                "camera_id": record.camera_id, "T_camera_object": pose.tolist()}

    def stop(self, payload: dict) -> dict:
        """释放一个 tracking session。"""
        session_id = self._session_id(payload)
        with self._lock:
            if session_id not in self.sessions:
                raise KeyError(f"tracking session 不存在: {session_id}")
            del self.sessions[session_id]
        return {"ok": True, "session_id": session_id}

    def unload_model(self) -> dict:
        """所有 tracking session 结束后才允许释放共享模型。"""
        with self._lock:
            if self.sessions:
                raise ValueError(f"仍有 {len(self.sessions)} 个活动 session，不能卸载模型")
            self.backend.unload_model()
            return {"ok": True, "model_loaded": bool(self.backend.is_loaded)}

    def health(self) -> dict:
        return {
            "ok": True,
            "service": "pose_server",
            "session_count": len(self.sessions),
            "model_loaded": bool(getattr(self.backend, "is_loaded", True)),
            "camera_endpoint": self.camera_endpoint,
        }

    def close(self) -> None:
        """关闭持久 camera client 并释放 session 引用。"""
        with self._lock:
            self.sessions.clear()
            for client in self._camera_clients.values():
                client.close()
            self._camera_clients.clear()


class PoseRequestHandler(BaseHTTPRequestHandler):
    service: PoseService | None = None

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, self.service.health())
            return
        self._send_json(404, {"ok": False, "error": "未知接口"})

    def do_POST(self):
        routes = {
            "/pose/register": self.service.register,
            "/pose/track": self.service.track,
            "/pose/track_frame": self.service.track_frame,
            "/pose/stop": self.service.stop,
            "/model/unload": lambda payload: self.service.unload_model(),
        }
        operation = routes.get(self.path)
        if operation is None:
            self._send_json(404, {"ok": False, "error": "未知接口"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求 JSON 必须是 object")
            self._send_json(200, operation(payload))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[pose_server] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundationPose register/tracking server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    args = parser.parse_args()

    service = PoseService(
        device=args.device,
        camera_endpoint=args.camera_endpoint,
    )
    PoseRequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), PoseRequestHandler)
    print(f"pose server 已启动: http://{args.host}:{args.port}")
    print("POST /pose/register, /pose/track, /pose/stop, /model/unload; GET /health")
    try:
        server.serve_forever()
    finally:
        service.close()
        server.server_close()


if __name__ == "__main__":
    main()
