"""Camera server 客户端和独立取流检查工具。"""

from __future__ import annotations

import argparse

from .live_rgbd_protocol import FramePacket, decode_frame


class CameraClient:
    def __init__(self, endpoint: str, timeout_ms: int = 6000, context=None):
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("CameraClient 需要在相机环境中安装 pyzmq") from error
        self._zmq = zmq
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context = context or self._zmq.Context()
        self._owns_context = context is None
        self.socket = None
        self._connect()
        self.last_frame_number = -1

    def _connect(self):
        """创建新的 REQ socket；超时后的旧 socket 不能继续使用。"""
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.endpoint)

    def _reset_socket(self):
        self.socket.close()
        self._connect()

    def _request_json(self, command: str) -> dict:
        try:
            self.socket.send_json({"command": command})
            result = self.socket.recv_json()
        except self._zmq.Again as exc:
            self._reset_socket()
            raise TimeoutError(f"camera server 请求超时: {command}") from exc
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "camera server 请求失败"))
        return result

    def health(self) -> dict:
        return self._request_json("health")

    def info(self) -> dict:
        return self._request_json("info")

    def get_frame(self) -> FramePacket:
        try:
            self.socket.send_json({
                "command": "frame",
                "last_frame_number": self.last_frame_number,
            })
            parts = self.socket.recv_multipart()
        except self._zmq.Again as exc:
            self._reset_socket()
            raise TimeoutError("camera server 取帧超时") from exc
        if len(parts) == 1:
            result = __import__("json").loads(parts[0].decode("utf-8"))
            raise RuntimeError(result.get("error", "camera server 取帧失败"))
        packet = decode_frame(parts)
        self.last_frame_number = packet.frame_number
        return packet

    def close(self):
        self.socket.close()
        if self._owns_context:
            self.context.term()


def main():
    parser = argparse.ArgumentParser(description="检查 camera server RGB-D 数据")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    client = CameraClient(args.endpoint)
    try:
        print(client.health())
        print(client.info())
        for _ in range(args.frames):
            packet = client.get_frame()
            print(
                f"frame={packet.frame_number}, rgb={packet.rgb.shape}, "
                f"depth={packet.depth.shape}, timestamp_ms={packet.timestamp_ms:.3f}"
            )
    finally:
        client.close()


if __name__ == "__main__":
    main()
