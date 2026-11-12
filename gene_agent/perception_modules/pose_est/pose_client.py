"""Pose server 的可复用客户端和命令行工具。"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


class PoseClient:
    """封装 register、track 和 stop 三个同步接口。"""

    def __init__(self, server: str = "http://127.0.0.1:18081", timeout=None):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.server + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                message = detail
            raise RuntimeError(f"pose server HTTP {exc.code}: {message}") from exc
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "pose server 请求失败"))
        return result

    def register(
        self,
        *,
        session_id: str,
        rgb_path,
        depth_path,
        mask_path,
        mesh_path,
        camera_path,
        camera_id: str = "0",
    ) -> dict:
        """注册首帧并在 server 保存 tracking session。"""
        return self._post("/pose/register", {
            "session_id": session_id,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
            "mesh_path": str(mesh_path),
            "camera_path": str(camera_path),
            "camera_id": str(camera_id),
        })

    def track(self, session_id: str) -> dict:
        """请求 server 使用最新相机帧执行一次 tracking。"""
        return self._post("/pose/track", {"session_id": session_id})

    def track_frame(self, session_id: str, *, rgb_path, depth_path, camera_path) -> dict:
        """使用显式离线帧跟踪，不连接真机 camera server。"""
        return self._post("/pose/track_frame", {
            "session_id": session_id,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "camera_path": str(camera_path),
        })

    def stop(self, session_id: str) -> dict:
        """释放 server 中的 tracking session。"""
        return self._post("/pose/stop", {"session_id": session_id})

    def unload_model(self) -> dict:
        """所有 session 已停止时释放 FoundationPose GPU 权重。"""
        return self._post("/model/unload", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 pose server")
    parser.add_argument("--server", default="http://127.0.0.1:18081")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--session-id", required=True)
    register_parser.add_argument("--rgb-path", required=True)
    register_parser.add_argument("--depth-path", required=True)
    register_parser.add_argument("--mask-path", required=True)
    register_parser.add_argument("--mesh-path", required=True)
    register_parser.add_argument("--camera-path", required=True)
    register_parser.add_argument("--camera-id", default="0")

    for command in ("track", "stop"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--session-id", required=True)

    args = parser.parse_args()
    client = PoseClient(args.server)
    if args.command == "register":
        result = client.register(
            session_id=args.session_id,
            rgb_path=args.rgb_path,
            depth_path=args.depth_path,
            mask_path=args.mask_path,
            mesh_path=args.mesh_path,
            camera_path=args.camera_path,
            camera_id=args.camera_id,
        )
    elif args.command == "track":
        result = client.track(args.session_id)
    else:
        result = client.stop(args.session_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
