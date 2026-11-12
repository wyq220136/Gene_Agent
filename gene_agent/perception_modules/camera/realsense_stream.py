"""RealSense 对齐 RGB-D 流封装。"""

from __future__ import annotations

import numpy as np

from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket


class RealSenseStream:
    """独占一台 RealSense，并返回深度已对齐到 RGB 的最新帧。"""

    def __init__(
        self,
        *,
        serial: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 30,
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境缺少 pyrealsense2，请在 RealSense 环境启动 camera server"
            ) from exc

        self._rs = rs
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._closed = True
        try:
            self.profile = self.pipeline.start(config)
            self._closed = False
            self.align = rs.align(rs.stream.color)

            color_stream = self.profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.intrinsic_matrix = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())

            # RealSense 刚启动时偶尔会丢第一批帧。累计允许 6 次 SDK 超时；
            # 若持续无帧，保留实际流配置和设备信息供排查 USB/带宽问题。
            remaining_timeouts = 6
            for _ in range(warmup_frames):
                _, used_timeouts = self._wait_for_frames(
                    timeout_ms=5000,
                    max_timeouts=remaining_timeouts,
                    phase="预热",
                )
                remaining_timeouts -= used_timeouts
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms: int = 5000) -> FramePacket:
        """阻塞读取一帧，并统一转换为 RGB 与米制 float32 depth。"""
        frames, _ = self._wait_for_frames(
            timeout_ms=timeout_ms,
            max_timeouts=3,
            phase="采集",
        )
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense 返回了不完整的 RGB-D 帧")

        bgr = np.asanyarray(color_frame.get_data())
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        depth = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * self.depth_scale
        )
        return FramePacket(
            rgb=rgb,
            depth=np.ascontiguousarray(depth),
            frame_number=int(color_frame.get_frame_number()),
            timestamp_ms=float(color_frame.get_timestamp()),
            serial=self.serial,
            intrinsic_matrix=self.intrinsic_matrix.copy(),
        )

    def _wait_for_frames(self, *, timeout_ms: int, max_timeouts: int, phase: str):
        """只重试 SDK 的无帧超时；其他设备错误立即向上报告。"""
        timeouts = 0
        while True:
            try:
                return self.pipeline.wait_for_frames(timeout_ms=timeout_ms), timeouts
            except RuntimeError as exc:
                if "frame didn't arrive within" not in str(exc).lower():
                    raise
                timeouts += 1
                if timeouts >= max_timeouts:
                    raise RuntimeError(self._timeout_diagnostic(phase, timeouts, timeout_ms)) from exc
                print(
                    f"[RealSense {self.serial}] {phase}等待帧超时 "
                    f"({timeouts}/{max_timeouts})，继续等待...",
                    flush=True,
                )

    def _timeout_diagnostic(self, phase: str, timeouts: int, timeout_ms: int) -> str:
        """持续无帧时输出实际请求配置，避免只留下 SDK 的 5000ms 报错。"""
        usb_type = "未知"
        try:
            device = self.profile.get_device()
            info = self._rs.camera_info.usb_type_descriptor
            if device.supports(info):
                usb_type = device.get_info(info)
        except (AttributeError, RuntimeError):
            pass
        if self.width <= 640 and self.height <= 480 and self.fps <= 15:
            next_step = (
                "已使用较低带宽流；请用 RealSense Viewer 单独测试该相机的彩色和深度流，"
                "再检查 USB 连接、设备占用和系统日志。"
            )
        else:
            next_step = (
                "可按设备支持的流配置尝试 --width 640 --height 480 --stream-fps 15，"
                "并检查 USB 连接与其他程序是否占用相机。"
            )
        return (
            f"RealSense {self.serial} {phase}持续无帧：{timeouts} 次各等待 "
            f"{timeout_ms}ms；请求流 {self.width}x{self.height}@{self.fps} "
            f"(depth z16 + color bgr8)，USB type={usb_type}。"
            "请用 rs-enumerate-devices -s 核对设备和流配置；"
            f"{next_step}"
        )

    def close(self) -> None:
        if not self._closed:
            self.pipeline.stop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
