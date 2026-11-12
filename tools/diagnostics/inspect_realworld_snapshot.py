"""只读检查历史或新采集的真机感知位姿，不连接机器人。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gene_agent.realworld.perception import load_observed_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="只读检查 FoundationPose 场景坐标转换")
    parser.add_argument("--task-dir", default="data/task_1")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--objects", nargs="+", required=True,
                        help="实例 ID，例如 Bowl/Bowl_0 Microwave/Microwave_0")
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--max-age-s", type=float)
    parser.add_argument("--require-matched-frame", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scene = load_observed_scene(
        args.task_dir, side=args.side, object_ids=args.objects,
        camera_id=args.camera_id, max_age_s=args.max_age_s,
        require_matched_frame=args.require_matched_frame,
    )
    report = {
        "source_format": "read_only_perception_snapshot",
        "side_sdk_base": scene.side,
        "camera_id": scene.camera_id,
        "frame_number": scene.frame_number,
        "timestamp_ms": scene.timestamp_ms,
        "pose_frame_verified": scene.pose_frame_verified,
        "T_sdk_camera": scene.T_sdk_camera.tolist(),
        "object_poses_sdk": {name: pose.tolist()
                             for name, pose in scene.object_poses_sdk.items()},
        "sources": dict(scene.sources),
        "claim": "只验证文件和坐标组合；不表示感知准确、场景碰撞可用或真机可执行",
    }
    contents = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents, encoding="utf-8")
    print(contents, end="")


if __name__ == "__main__":
    main()
