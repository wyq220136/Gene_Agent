"""通过 PoseClient 请求 server，并实时显示 tracking pose。"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np


_VIS_DIR = Path(__file__).resolve().parent
_POSE_DIR = _VIS_DIR.parent
_PERCEPTION_DIR = _POSE_DIR.parent
for path in (_POSE_DIR, _PERCEPTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ..batch_est_pose import (
    load_camera_parameters,
    load_depth_image,
    load_mesh,
    load_rgb_image,
)
from ..pose_client import PoseClient
from .live_pose_viser import LivePoseViser


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose server 实时 tracking 可视化")
    parser.add_argument("--pose-server", default="http://127.0.0.1:18081")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--depth-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--camera-path", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--viser-host", default="0.0.0.0")
    parser.add_argument("--viser-port", type=int, default=8080)
    args = parser.parse_args()
    if not 0 < args.fps <= 10:
        raise ValueError("fps 必须满足 0 < fps <= 10")

    pose_client = PoseClient(args.pose_server)
    camera_parameters = load_camera_parameters(args.camera_path)
    initial_rgb = load_rgb_image(args.rgb_path)
    initial_depth = load_depth_image(
        args.depth_path,
        depth_unit=str(camera_parameters.get("depth_unit", "millimeter")),
    )
    mesh = load_mesh(args.mesh_path)
    visualizer = LivePoseViser(
        mesh=mesh,
        host=args.viser_host,
        port=args.viser_port,
        max_points=30000,
        max_depth=3.0,
        trajectory_length=100,
    )
    stopped = False

    def request_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    registered = False
    try:
        response = pose_client.register(
            session_id=args.session_id,
            rgb_path=args.rgb_path,
            depth_path=args.depth_path,
            mask_path=args.mask_path,
            mesh_path=args.mesh_path,
            camera_path=args.camera_path,
        )
        registered = True
        pose = np.asarray(response["T_camera_object"], dtype=np.float64)
        visualizer.set_origin_from_pose(pose)
        visualizer.update_pose(pose, frame_number=0, elapsed_ms=0.0)
        # 场景固定显示注册首帧；实时 RGB-D 只由 pose server 读取，避免双客户端不同步。
        visualizer.update_point_cloud(
            initial_rgb,
            initial_depth,
            camera_parameters["intrinsic_matrix"],
        )
        print(f"viser 页面: http://{args.viser_host}:{args.viser_port}")

        interval = 1.0 / args.fps
        while not stopped:
            if "stop" in visualizer.poll_actions():
                break
            started_at = time.perf_counter()
            response = pose_client.track(args.session_id)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            pose = np.asarray(response["T_camera_object"], dtype=np.float64)
            visualizer.update_pose(
                pose,
                frame_number=int(response["frame_number"]),
                elapsed_ms=elapsed_ms,
            )
            remaining = interval - (time.perf_counter() - started_at)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if registered:
            try:
                pose_client.stop(args.session_id)
            except Exception as exc:
                print(f"释放 tracking session 失败: {exc}")
        visualizer.close()


if __name__ == "__main__":
    main()
