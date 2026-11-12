#!/usr/bin/env python3
"""用一台 RealSense 拍一帧 RGB-D，保存为场景构建器的输入。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket
from gene_agent.perception_modules.camera.realsense_stream import RealSenseStream
from gene_agent.perception_modules.interactive_realworld_perception import (
    CapturePaths, save_capture,
)


CAPTURE_FILES = ("rgb.png", "depth.png", "camera.json")


def capture_frame(packet: FramePacket, output_dir: str | Path) -> CapturePaths:
    """保存对齐的一帧；拒绝覆盖旧 RGB、深度或相机参数。"""
    directory = Path(output_dir).expanduser().resolve()
    existing = [directory / name for name in CAPTURE_FILES if (directory / name).exists()]
    if existing:
        raise FileExistsError(
            "采集目录已有文件，避免混合不同时刻的帧: "
            + ", ".join(str(path) for path in existing)
        )
    return save_capture(packet, directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="RealSense 序列号")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="新的采集目录，如 data/realworld/my_scene_0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stream-fps", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    if min(args.width, args.height, args.stream_fps) <= 0 or args.warmup_frames < 0:
        parser.error("宽、高、帧率须为正数，预热帧数不能为负数")
    existing = [args.output_dir / name for name in CAPTURE_FILES
                if (args.output_dir / name).exists()]
    if existing:
        parser.error("采集目录已有文件，不能覆盖: " + ", ".join(map(str, existing)))

    with RealSenseStream(
        serial=args.serial, width=args.width, height=args.height,
        fps=args.stream_fps, warmup_frames=args.warmup_frames,
    ) as stream:
        packet = stream.read()
    paths = capture_frame(packet, args.output_dir)
    print(f"相机 {packet.serial}，帧号 {packet.frame_number}，时间戳 {packet.timestamp_ms:.3f} ms")
    print(f"RGB: {paths.rgb}")
    print(f"深度（uint16 毫米，对齐 RGB）: {paths.depth}")
    print(f"内参与采集信息: {paths.camera}")


if __name__ == "__main__":
    main()
