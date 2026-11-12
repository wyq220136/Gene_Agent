"""实时 RGB-D 帧的进程间传输协议和最新帧缓存。"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FramePacket:
    """一组已经对齐的 RGB-D 数据及其相机元数据。"""

    rgb: np.ndarray
    depth: np.ndarray
    frame_number: int
    timestamp_ms: float
    serial: str
    intrinsic_matrix: np.ndarray

    def validate(self) -> None:
        """校验帧能否安全传给分割和位姿跟踪模块。"""
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb 必须是 HxWx3 的 uint8 数组")
        if self.depth.dtype != np.float32 or self.depth.ndim != 2:
            raise ValueError("depth 必须是 HxW 的 float32 数组，单位为米")
        if self.rgb.shape[:2] != self.depth.shape:
            raise ValueError("rgb 和 depth 的高宽必须一致")
        if np.asarray(self.intrinsic_matrix).shape != (3, 3):
            raise ValueError("intrinsic_matrix 必须是 3x3 矩阵")


def encode_frame(packet: FramePacket) -> list[bytes]:
    """把一帧编码为 ZeroMQ 可直接发送的多段二进制消息。"""
    packet.validate()
    rgb = np.ascontiguousarray(packet.rgb)
    depth = np.ascontiguousarray(packet.depth)
    intrinsic = np.ascontiguousarray(packet.intrinsic_matrix, dtype=np.float64)
    header = {
        "schema_version": SCHEMA_VERSION,
        "frame_number": int(packet.frame_number),
        "timestamp_ms": float(packet.timestamp_ms),
        "serial": str(packet.serial),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "color_order": "RGB",
        "depth_unit": "meter",
    }
    return [
        json.dumps(header, ensure_ascii=False).encode("utf-8"),
        memoryview(rgb).tobytes(),
        memoryview(depth).tobytes(),
        memoryview(intrinsic).tobytes(),
    ]


def decode_frame(parts: list[bytes]) -> FramePacket:
    """从多段二进制消息恢复一帧，并再次执行结构校验。"""
    if len(parts) != 4:
        raise ValueError(f"RGB-D 帧消息必须有 4 段，实际为 {len(parts)} 段")
    header = json.loads(parts[0].decode("utf-8"))
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的帧协议版本: {header.get('schema_version')}")

    rgb_shape = tuple(header["rgb_shape"])
    depth_shape = tuple(header["depth_shape"])
    rgb = np.frombuffer(parts[1], dtype=np.uint8).reshape(rgb_shape).copy()
    depth = np.frombuffer(parts[2], dtype=np.float32).reshape(depth_shape).copy()
    intrinsic = np.frombuffer(parts[3], dtype=np.float64).reshape(3, 3).copy()
    packet = FramePacket(
        rgb=rgb,
        depth=depth,
        frame_number=int(header["frame_number"]),
        timestamp_ms=float(header["timestamp_ms"]),
        serial=str(header["serial"]),
        intrinsic_matrix=intrinsic,
    )
    packet.validate()
    return packet


class LatestFrameBuffer:
    """只保存最新一帧，慢消费者不会造成历史帧积压。"""

    def __init__(self):
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._error: Exception | None = None

    def publish(self, packet: FramePacket) -> None:
        packet.validate()
        with self._condition:
            self._latest = packet
            self._error = None
            self._condition.notify_all()

    def fail(self, error: Exception) -> None:
        """唤醒所有消费者，并让后续取帧返回真实采集错误。"""
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def get_latest(self) -> FramePacket:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"相机采集失败: {self._error}") from self._error
            if self._latest is None:
                raise RuntimeError("相机尚未产生有效帧")
            return self._latest

    def wait_for_newer(self, last_frame_number: int, timeout: float) -> FramePacket:
        """等待帧号变化；超时后明确报错，不返回重复帧。"""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._error is not None
                    or (
                        self._latest is not None
                        and self._latest.frame_number != last_frame_number
                    )
                ),
                timeout=timeout,
            )
            if not ready:
                raise TimeoutError("等待新的相机帧超时")
            if self._error is not None:
                raise RuntimeError(f"相机采集失败: {self._error}") from self._error
            return self._latest
