"""一键执行 task 的 camera、segmentation 和 FoundationPose 流程。

三个模型 server 需要提前启动；本脚本只负责检查服务并串联客户端调用。
默认流程为：camera server 取任务初始帧 → segmentation server 生成 mask →
pose server 为每个物体执行首帧 FoundationPose 估计。
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import DEFAULT_TEST_DIR, PROJECT_ROOT

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

_PERCEPTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
for _path in (_REPO_ROOT, _PERCEPTION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gene_agent.perception_modules.camera import CameraClient
from gene_agent.perception_modules.pose_est.task_pose_est import estimate_task_poses
from gene_agent.perception_modules.segmentation.task_segmentation import segment_task_objects


def _check_http_server(server: str, name: str) -> None:
    """检查 HTTP server 是否已经提供 health 接口。"""
    try:
        with urllib.request.urlopen(
            server.rstrip("/") + "/health",
            timeout=3,
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"{name} 未就绪: {server}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"{name} 未就绪: {result}")


def _check_camera_server(endpoint: str) -> None:
    """检查 camera server 是否已经启动。"""
    client = CameraClient(endpoint)
    try:
        result = client.health()
    finally:
        client.close()
    if not result.get("ok"):
        raise RuntimeError(f"camera server 未就绪: {result}")


def run_task_perception(
    task_dir: str | Path = DEFAULT_TEST_DIR,
    categories: str | Iterable[str] = "all",
    *,
    config_path: str | Path = "gene_agent/perception_modules/segmentation/category_config.py",
    camera_endpoint: str = "tcp://127.0.0.1:18082",
    seg_server: str = "http://127.0.0.1:18080",
    pose_server: str = "http://127.0.0.1:18081",
    camera_timeout_ms: int = 6000,
    side: str = "left",
    use_microwave_door_plane_yaw: bool = True,
    visualize: bool = False,
) -> dict:
    """串联一次任务初始感知、分割和首帧 pose 估计。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    _check_camera_server(camera_endpoint)
    _check_http_server(seg_server, "segmentation server")
    _check_http_server(pose_server, "pose server")

    print(f"开始处理任务: {task_dir}")
    segmentation_results = segment_task_objects(
        task_dir,
        categories,
        config_path=config_path,
        camera_endpoint=camera_endpoint,
        seg_server=seg_server,
        camera_timeout_ms=camera_timeout_ms,
    )
    pose_results = estimate_task_poses(
        task_dir,
        categories,
        server=pose_server,
        side=side,
        use_microwave_door_plane_yaw=use_microwave_door_plane_yaw,
        visualize=visualize,
    )
    return {
        "task_dir": str(task_dir),
        "segmentation": segmentation_results,
        "pose": pose_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键执行 task 的 camera、segmentation 和 FoundationPose 流程"
    )
    parser.add_argument("--task-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument(
        "--config",
        default="gene_agent/perception_modules/segmentation/category_config.py",
        help="类别配置 .py 或 .json 文件",
    )
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--seg-server", default="http://127.0.0.1:18080")
    parser.add_argument("--pose-server", default="http://127.0.0.1:18081")
    parser.add_argument("--camera-timeout-ms", type=int, default=6000)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--no-microwave-door-plane-yaw",
        dest="use_microwave_door_plane_yaw",
        action="store_false",
        help="关闭 Microwave handle pose 后的门板平面 yaw 修正",
    )
    parser.set_defaults(use_microwave_door_plane_yaw=True)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    result = run_task_perception(
        args.task_dir,
        args.categories,
        config_path=args.config,
        camera_endpoint=args.camera_endpoint,
        seg_server=args.seg_server,
        pose_server=args.pose_server,
        camera_timeout_ms=args.camera_timeout_ms,
        side=args.side,
        use_microwave_door_plane_yaw=args.use_microwave_door_plane_yaw,
        visualize=args.visualize,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    segmentation_ok = all(item.get("ok") for item in result["segmentation"])
    pose_ok = all(item.get("ok") for item in result["pose"])
    raise SystemExit(0 if segmentation_ok and pose_ok else 1)


if __name__ == "__main__":
    main()
