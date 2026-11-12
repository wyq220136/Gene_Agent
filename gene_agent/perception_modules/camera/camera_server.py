"""通过 ZeroMQ 提供 RealSense 最新 RGB-D 帧。"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time

import zmq

from gene_agent.perception_modules.camera.live_rgbd_protocol import LatestFrameBuffer, encode_frame
from gene_agent.perception_modules.camera.realsense_stream import RealSenseStream


def validate_max_fps(value: float) -> float:
    """Camera server 对外响应频率限制为 0 到 10 FPS。"""
    value = float(value)
    if not 0 < value <= 10:
        raise ValueError("max_fps 必须满足 0 < max_fps <= 10")
    return value


class CameraServer:
    """采集线程持续排空设备帧，服务线程只返回最新帧。"""

    def __init__(self, stream, *, endpoint: str, max_fps: float, context=None):
        self.stream = stream
        self.endpoint = endpoint
        self.max_fps = validate_max_fps(max_fps)
        self.context = context
        self.buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()
        self.capture_error: Exception | None = None
        self._last_response_time = 0.0

    def _capture_loop(self):
        try:
            while not self.stop_event.is_set():
                self.buffer.publish(self.stream.read())
        except Exception as exc:
            self.capture_error = exc
            self.buffer.fail(exc)

    def _wait_rate_limit(self):
        interval = 1.0 / self.max_fps
        remaining = interval - (time.monotonic() - self._last_response_time)
        if remaining > 0:
            self.stop_event.wait(remaining)
        self._last_response_time = time.monotonic()

    def serve_forever(self):
        context = self.context or zmq.Context()
        owns_context = self.context is None
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        capture_thread = None

        try:
            socket.bind(self.endpoint)
            capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            capture_thread.start()
            # 服务宣告就绪前必须先有一帧，避免 info 请求与采集线程竞态。
            # stream.read() 允许最多 3 次各 5 秒的短暂无帧超时；
            # 服务就绪等待必须覆盖这段恢复窗口。
            self.buffer.wait_for_newer(-1, timeout=20.0)
            print(f"camera server 已启动: {self.endpoint}")
            while not self.stop_event.is_set():
                if not socket.poll(200):
                    continue
                raw_request = socket.recv()
                try:
                    request = json.loads(raw_request.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("请求 JSON 必须是 object")
                    command = request.get("command")
                    if command == "health":
                        socket.send_json({
                            "ok": self.capture_error is None,
                            "service": "camera_server",
                            "error": (
                                str(self.capture_error)
                                if self.capture_error is not None
                                else None
                            ),
                        })
                    elif command == "info":
                        packet = self.buffer.get_latest()
                        socket.send_json({
                            "ok": True,
                            "serial": packet.serial,
                            "shape": list(packet.depth.shape),
                            "intrinsic_matrix": packet.intrinsic_matrix.tolist(),
                            "max_fps": self.max_fps,
                        })
                    elif command == "frame":
                        self._wait_rate_limit()
                        last_frame = int(request.get("last_frame_number", -1))
                        packet = self.buffer.wait_for_newer(last_frame, timeout=5.0)
                        socket.send_multipart(encode_frame(packet))
                    else:
                        socket.send_json({"ok": False, "error": "未知命令"})
                except Exception as exc:
                    socket.send_json({"ok": False, "error": str(exc)})
        finally:
            self.stop_event.set()
            if capture_thread is not None:
                capture_thread.join(timeout=16)
            self.stream.close()
            socket.close()
            if owns_context:
                context.term()


def main():
    parser = argparse.ArgumentParser(description="RealSense RGB-D camera server")
    parser.add_argument("--serial", required=True, help="RealSense 序列号")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stream-fps", type=int, default=30)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    # 在占用 RealSense 设备前先拒绝非法参数。
    validate_max_fps(args.max_fps)

    stream = RealSenseStream(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.stream_fps,
        warmup_frames=args.warmup_frames,
    )
    server = CameraServer(stream, endpoint=args.endpoint, max_fps=args.max_fps)

    def stop_server(*_):
        print("\n收到退出信号，正在停止 camera server...")
        server.stop_event.set()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    server.serve_forever()


if __name__ == "__main__":
    main()
