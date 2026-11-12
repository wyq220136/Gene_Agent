"""并发记录多台 RealSense 的 RGB-D 帧。

该入口复用项目的 :class:`RealSenseStream`，因此输出与 camera server
保持一致：RGB 是 RGB ``uint8``，深度是已经对齐到 RGB 的米制
``float32``。每台相机独立占用一个采集线程和一个 RealSense pipeline。

示例::

    PYTHONPATH=perception_modules python -m camera.record_dual_frames \
        --out-dir data/recorded --fps 5 --duration 10

输出结构::

    <out_dir>/<serial>/rgb/<timestamp>.png
    <out_dir>/<serial>/depth/<timestamp>.npy
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket
from gene_agent.perception_modules.camera.realsense_stream import RealSenseStream


DEFAULT_SERIALS = ["323522063237", "323522063015"]


def make_timestamp(now: float) -> str:
    """根据墙上时钟生成可排序且不易重名的文件名。"""
    return time.strftime("%Y%m%d_%H%M%S_", time.localtime(now)) + (
        f"{int((now % 1) * 1e6):06d}"
    )


def save_frame_packet(
    packet: FramePacket,
    rgb_dir: str | Path,
    depth_dir: str | Path,
    timestamp: str,
) -> tuple[Path, Path]:
    """按项目约定保存一帧 RGB-D，并返回两个文件路径。"""
    packet.validate()
    rgb_dir = Path(rgb_dir)
    depth_dir = Path(depth_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = rgb_dir / f"{timestamp}.png"
    depth_path = depth_dir / f"{timestamp}.npy"
    # FramePacket.rgb 已经是 RGB；不使用 cv2，避免引入额外的颜色顺序转换。
    Image.fromarray(np.ascontiguousarray(packet.rgb)).save(rgb_path)
    np.save(depth_path, np.ascontiguousarray(packet.depth, dtype=np.float32))
    return rgb_path, depth_path


class FrameRecorder(threading.Thread):
    """单台相机的采集、限频和落盘线程。"""

    def __init__(
        self,
        serial: str,
        out_dir: str | Path,
        width: int,
        height: int,
        stream_fps: int,
        save_fps: float,
        warmup: int,
        stop_event: threading.Event,
        barrier: threading.Barrier | None,
        realsense_type: str = "D415",
        max_err: int = 30,
    ):
        super().__init__(daemon=True, name=f"rec-{serial}")
        self.serial = serial
        self.out_dir = Path(out_dir)
        self.width = width
        self.height = height
        self.stream_fps = stream_fps
        self.save_fps = save_fps
        self.warmup = warmup
        self.stop_event = stop_event
        self.barrier = barrier
        # 保留源脚本参数，当前 RealSenseStream 由设备序列号决定型号。
        self.realsense_type = realsense_type
        self.max_err = max_err

        self.stream: RealSenseStream | None = None
        self.rgb_dir: Path | None = None
        self.depth_dir: Path | None = None
        self.saved = 0

    def start_stream(self) -> bool:
        """启动并预热相机；失败只跳过本相机。"""
        try:
            self.stream = RealSenseStream(
                serial=self.serial,
                width=self.width,
                height=self.height,
                fps=self.stream_fps,
                warmup_frames=self.warmup,
            )
            camera_dir = self.out_dir / self.serial
            self.rgb_dir = camera_dir / "rgb"
            self.depth_dir = camera_dir / "depth"
            self.rgb_dir.mkdir(parents=True, exist_ok=True)
            self.depth_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{self.serial}] 相机已启动，输出目录: {camera_dir}")
            return True
        except Exception as exc:
            print(f"[{self.serial}] 启动失败: {exc}（该相机将被跳过）")
            self.close()
            return False

    def run(self) -> None:
        """等待多相机同步屏障，然后持续读取并按保存频率落盘。"""
        if self.stream is None or self.rgb_dir is None or self.depth_dir is None:
            return
        if self.barrier is None:
            raise RuntimeError("录制线程缺少多相机同步屏障")

        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            print(f"[{self.serial}] 屏障已被打破，停止录制")
            return

        interval = 1.0 / self.save_fps
        last_save = 0.0
        errors = 0
        print(f"[{self.serial}] 开始录制（save_fps={self.save_fps}）")

        while not self.stop_event.is_set():
            try:
                packet = self.stream.read()
            except Exception as exc:
                errors += 1
                print(
                    f"[{self.serial}] 取帧错误({errors}/{self.max_err}): {exc}"
                )
                if errors >= self.max_err:
                    print(f"[{self.serial}] 连续错误过多，停止本相机线程")
                    break
                continue
            errors = 0

            now = time.time()
            if now - last_save < interval:
                continue
            last_save = now
            save_frame_packet(
                packet,
                self.rgb_dir,
                self.depth_dir,
                make_timestamp(now),
            )
            self.saved += 1

        print(f"[{self.serial}] 录制结束，共保存 {self.saved} 帧")

    def close(self) -> None:
        """停止并释放本相机的 pipeline。"""
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="多台 RealSense 并发连续抓取 RGB-D 帧，保存 PNG + npy"
    )
    parser.add_argument(
        "--serials", type=str, nargs="+", default=DEFAULT_SERIALS,
        help="一个或多个相机序列号（默认两台 D415）",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/frames",
        help="输出根目录，自动创建 <serial>/rgb 与 <serial>/depth",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stream-fps", type=int, default=30)
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="人为控制的保存频率，应不高于 --stream-fps",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="录制时长（秒）；<=0 表示录到 Ctrl+C",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--hardware-reset", action="store_true",
        help="兼容旧入口；当前 RealSenseStream 不执行硬件复位",
    )
    parser.add_argument("--realsense-type", type=str, default="D415")
    parsed = parser.parse_args(args)
    if parsed.stream_fps <= 0:
        parser.error("--stream-fps 必须大于 0")
    if parsed.fps <= 0:
        parser.error("--fps 必须大于 0")
    if parsed.width <= 0 or parsed.height <= 0:
        parser.error("--width 和 --height 必须大于 0")
    if parsed.warmup < 0:
        parser.error("--warmup 不能小于 0")
    return parsed


def main(args=None) -> None:
    options = parse_args(args)
    if options.fps > options.stream_fps:
        print(
            f"警告: 保存频率 --fps={options.fps} 大于流帧率 "
            f"--stream-fps={options.stream_fps}，实际保存频率受流帧率限制"
        )

    stop_event = threading.Event()

    def stop_recording(*_):
        print("\n收到退出信号，正在停止录制...")
        stop_event.set()

    signal.signal(signal.SIGINT, stop_recording)
    signal.signal(signal.SIGTERM, stop_recording)

    recorders = []
    for serial in options.serials:
        recorder = FrameRecorder(
            serial=serial,
            out_dir=options.out_dir,
            width=options.width,
            height=options.height,
            stream_fps=options.stream_fps,
            save_fps=options.fps,
            warmup=options.warmup,
            stop_event=stop_event,
            barrier=None,
            realsense_type=options.realsense_type,
        )
        if recorder.start_stream():
            recorders.append(recorder)

    if not recorders:
        print("没有成功启动任何相机，退出。")
        return

    barrier = threading.Barrier(len(recorders))
    for recorder in recorders:
        recorder.barrier = barrier
        recorder.start()

    print(
        f"已启动 {len(recorders)} 台相机；"
        + (
            "按 Ctrl+C 停止。"
            if options.duration <= 0
            else f"将录制 {options.duration} 秒。"
        )
    )
    deadline = time.time() + options.duration if options.duration > 0 else None
    try:
        while not stop_event.is_set():
            if deadline is not None and time.time() >= deadline:
                print("达到设定录制时长，停止。")
                stop_event.set()
                break
            if all(not recorder.is_alive() for recorder in recorders):
                break
            time.sleep(0.05)
    finally:
        stop_event.set()
        for recorder in recorders:
            recorder.join(timeout=10)
        for recorder in recorders:
            recorder.close()

    print("=== 录制汇总 ===")
    for recorder in recorders:
        print(
            f"  [{recorder.serial}] 保存 {recorder.saved} 帧 -> "
            f"{options.out_dir}/{recorder.serial}"
        )


if __name__ == "__main__":
    main()
